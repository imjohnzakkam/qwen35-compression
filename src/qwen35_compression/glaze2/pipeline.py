"""Glaze v2 end to end for one variant: Fisher pass, allocation, rounding, export, manifest."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from qwen35_compression.config import ExperimentConfig, Glaze2Config, VariantConfig
from qwen35_compression.glaze2.allocate import allocate, build_units, uniform
from qwen35_compression.glaze2.data import load_blocks
from qwen35_compression.glaze2.export import plan_bytes, write_export
from qwen35_compression.glaze2.fisher import fisher_pass
from qwen35_compression.glaze2.rounding import RoundingSettings, quantize_layers


def directory_bytes(directory: Path) -> int:
    """Bytes of an export as the size comparison counts them: every file in the directory,
    the run manifest excepted (it is written after the comparison)."""
    from qwen35_compression.export import MANIFEST_NAME

    return sum(
        p.stat().st_size for p in directory.iterdir() if p.is_file() and p.name != MANIFEST_NAME
    )


def language_linears(text_model: nn.Module) -> dict[str, nn.Linear]:
    """The decoder layers' Linears: what AutoRound quantizes (248 in Qwen3.5-4B)."""
    return {
        name: module
        for name, module in text_model.named_modules()
        if isinstance(module, nn.Linear) and name.startswith("layers.")
    }


def rounding_settings(settings: Glaze2Config) -> RoundingSettings:
    return RoundingSettings(
        blocks_per_batch=settings.blocks_per_batch,
        max_iters=settings.max_iters,
        min_iters=settings.min_iters,
        eval_every=settings.eval_every,
        lr=settings.lr,
        seed=settings.seed,
    )


def load_teacher(config: ExperimentConfig) -> tuple[nn.Module, nn.Module, Any, Path, str]:
    """BF16 text model (frozen, vision tower dropped), the full model, processor, snapshot, rev."""
    from qwen35_compression.models import download_model, load_resolved_model

    snapshot, revision = download_model(config)
    model, processor = load_resolved_model(f"{config.model.id}@{revision}", config)
    model.requires_grad_(False)
    model.eval()
    if getattr(model.model, "visual", None) is not None:
        del model.model.visual
    teacher = model.model.language_model
    try:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(teacher, recurse=True)
    except ImportError:
        pass
    return teacher, model, processor, Path(snapshot), revision


def quantize(
    config: ExperimentConfig, variant: VariantConfig, log: Callable[[str], None] = print
) -> tuple[Path, dict[str, Any]]:
    from qwen35_compression.export import write_export_manifest

    settings = variant.glaze2
    if settings is None:
        raise ValueError(f"{variant.name} is not a glaze2 variant")
    output_dir = config.paths.outputs / variant.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output_dir}")
    target_dir = config.paths.outputs / settings.byte_target
    if not target_dir.exists():
        raise FileNotFoundError(f"byte target export missing: {target_dir}")
    target_bytes = directory_bytes(target_dir)
    calibration = load_blocks(settings.calibration_blocks)
    held_out = load_blocks(settings.held_out_blocks)

    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    teacher, model, _, snapshot, revision = load_teacher(config)
    head = model.get_output_embeddings().weight
    linears = language_linears(teacher)
    log(f"glaze2: {len(linears)} linears, {len(calibration)} + {len(held_out)} blocks")

    factors, weights = fisher_pass(
        teacher, head, calibration.ids, linears, settings.fisher_chunk_tokens, settings.seed
    )
    units = build_units({name: m.weight for name, m in linears.items()}, factors)
    budget = None
    if settings.allocate:
        shapes = {name: tuple(m.weight.shape) for name, m in linears.items()}
        plan = plan_bytes(snapshot, shapes)
        budget = plan.allocation_budget(target_bytes, settings.margin_bytes)
        if settings.budget_fraction is not None:
            budget = min(budget, int(settings.budget_fraction * plan.language_base))
        allocation = allocate(units, budget)
        log(f"glaze2 allocation: {allocation.summary()}, +{allocation.extra_bytes:,} bytes")
    else:
        allocation = uniform(units)

    quantized, records = quantize_layers(
        teacher,
        calibration.ids,
        held_out.ids,
        allocation,
        list(linears),
        weights,
        rounding_settings(settings),
        log=log,
    )
    export = write_export(snapshot, output_dir, quantized, settings.allocate, log)
    export["bytes_over_target"] = export["total_bytes"] - target_bytes
    # Only the allocation spends the target's bytes; at uniform 4-bit g128 (the same layout as
    # AutoRound's) a few KB of config either way is not a size difference.
    if settings.allocate and export["total_bytes"] > target_bytes:
        raise ValueError(
            f"export is {export['total_bytes']:,} bytes, over the {target_bytes:,} budget"
        )
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    record = {
        "byte_target": {"variant": settings.byte_target, "bytes": target_bytes},
        "export": export,
        "allocation": {
            "enabled": settings.allocate,
            "budget": budget,
            "extra_bytes": allocation.extra_bytes,
            "predicted_saving": allocation.predicted_saving,
            "options": allocation.summary(),
            "choice": {name: option.name for name, option in allocation.choice.items()},
        },
        "layers": [
            {
                "index": r.index,
                "iterations": r.iterations,
                "best_iteration": r.best_iteration,
                "held_out_rtn": r.dev_loss_rtn,
                "held_out_best": r.dev_loss_best,
            }
            for r in records
        ],
        "data": {
            "calibration_blocks": len(calibration),
            "held_out_blocks": len(held_out),
            "calibration_tokens": calibration.tokens_by_domain(),
        },
    }
    manifest = write_export_manifest(
        output_dir,
        config,
        variant,
        revision,
        time.perf_counter() - started,
        peak,
        extra={"glaze2": record},
    )
    return output_dir, manifest
