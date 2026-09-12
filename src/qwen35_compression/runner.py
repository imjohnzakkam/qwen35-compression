from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

from qwen35_compression.calibration import build_calibration_dataset
from qwen35_compression.config import ExperimentConfig, VariantConfig
from qwen35_compression.export import write_export_manifest
from qwen35_compression.models import load_resolved_model, resolve_revision
from qwen35_compression.quantization.recipes import build_recipe


def quantize(config: ExperimentConfig, variant: VariantConfig) -> tuple[Path, dict[str, Any]]:
    if variant.method == "bf16":
        raise ValueError(
            "BF16 is evaluated from the pinned source checkpoint and is not re-exported"
        )

    import torch
    from llmcompressor import oneshot

    revision = resolve_revision(config)
    model, processor = load_resolved_model(f"{config.model.id}@{revision}", config)
    dataset, collator = build_calibration_dataset(processor, config.calibration)
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
        max_seq_length=config.calibration.max_sequence_length,
        num_calibration_samples=config.calibration.num_samples,
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
