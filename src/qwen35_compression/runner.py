from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

from qwen35_compression.calibration import build_calibration_dataset
from qwen35_compression.config import CalibrationConfig, ExperimentConfig, VariantConfig
from qwen35_compression.export import write_export_manifest
from qwen35_compression.feature1 import require_calibration_lock
from qwen35_compression.models import load_resolved_model, resolve_revision
from qwen35_compression.multimodal import require_multimodal_lock
from qwen35_compression.quantization.recipes import build_recipe


def blocks_dataset(path: Path) -> tuple[Any, Any]:
    """Packed calibration blocks written by Glaze v2's data stage, as AutoRound takes them."""
    import torch
    from datasets import Dataset

    from qwen35_compression.glaze2.data import load_blocks

    blocks = load_blocks(path)
    if not len(blocks):
        raise ValueError(f"no calibration blocks in {path}")
    dataset = Dataset.from_list(
        [{"input_ids": [ids], "attention_mask": [[1] * len(ids)]} for ids in blocks.ids]
    )

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("compression calibration requires batch size 1")
        return {key: torch.as_tensor(value) for key, value in batch[0].items()}

    return dataset, collate


def quantize(config: ExperimentConfig, variant: VariantConfig) -> tuple[Path, dict[str, Any]]:
    if variant.method == "bf16":
        raise ValueError(
            "BF16 is evaluated from the pinned source checkpoint and is not re-exported"
        )
    if variant.method == "glaze":
        # Glaze refines its init's export instead of quantizing the BF16 model.
        from qwen35_compression.glaze.train import refine

        return refine(config, variant)
    if variant.method == "glaze2":
        # Glaze v2 quantizes from BF16 with its own data, allocation and rounding.
        from qwen35_compression.glaze2.pipeline import quantize as glaze2_quantize

        return glaze2_quantize(config, variant)
    calibration = config.calibration
    if variant.requires_multimodal_calibration:
        multimodal = config.multimodal_calibration
        if multimodal is None:
            raise ValueError("variant requires multimodal_calibration")
        require_multimodal_lock(multimodal)
        calibration = CalibrationConfig(
            path=multimodal.path,
            num_samples=multimodal.num_samples,
            max_sequence_length=config.calibration.max_sequence_length,
            seed=multimodal.seed,
        )
    elif calibration.lock_path is not None:
        require_calibration_lock(calibration)

    import torch
    from llmcompressor import oneshot

    revision = resolve_revision(config)
    model, processor = load_resolved_model(f"{config.model.id}@{revision}", config)
    # AutoRound stacks every sample's cached inputs into one tensor, so it needs one length:
    # the same conversations, packed into max_sequence_length blocks.
    pack = variant.method == "autoround"
    if variant.calibration_blocks is not None:
        dataset, collator = blocks_dataset(variant.calibration_blocks)
    else:
        dataset, collator = build_calibration_dataset(processor, calibration, pack=pack)
    if variant.calibration_samples is not None:
        dataset = dataset.select(range(min(variant.calibration_samples, len(dataset))))
    recipe = build_recipe(variant)
    output_dir = config.paths.outputs / variant.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    oneshot(
        model=model,
        recipe=recipe,
        dataset=dataset,
        max_seq_length=calibration.max_sequence_length,
        num_calibration_samples=len(dataset),
        data_collator=collator,
    )
    model.save_pretrained(output_dir, safe_serialization=True, save_compressed=True)
    processor.save_pretrained(output_dir)
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    manifest = write_export_manifest(
        output_dir,
        config,
        variant,
        revision,
        elapsed,
        peak_memory,
    )

    del model, processor, dataset, recipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir, manifest
