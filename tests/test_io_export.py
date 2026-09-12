import json
from pathlib import Path

import pytest

from qwen35_compression.config import load_config
from qwen35_compression.export import verify_export, write_export_manifest
from qwen35_compression.io import read_jsonl

ROOT = Path(__file__).parent.parent


def test_phase0_fixtures_are_valid_and_sized() -> None:
    config = load_config(ROOT / "configs/phase0.yaml")
    calibration = read_jsonl(config.calibration.path)
    evaluation = read_jsonl(config.evaluation.path)
    assert len(calibration) == config.calibration.num_samples
    assert len(evaluation) >= config.evaluation.max_samples


def test_export_manifest_detects_mutation(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/phase0.yaml")
    variant = config.variant("gptq_w4a16_g128")
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"format": "pack-quantized"}}),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    write_export_manifest(tmp_path, config, variant, "revision", 1.0, 100)
    verify_export(tmp_path, variant)
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_export(tmp_path, variant)
