from pathlib import Path

import pytest

from qwen35_compression.config import load_config

ROOT = Path(__file__).parent.parent


def test_phase0_is_locked_to_small_model() -> None:
    config = load_config(ROOT / "configs/phase0.yaml")
    assert config.model.id == "Qwen/Qwen3.5-0.8B"
    assert [variant.name for variant in config.variants] == [
        "bf16",
        "int8_w8a8",
        "gptq_w4a16_g128",
        "awq_w4a16_g128",
    ]


def test_unknown_variant_lists_choices() -> None:
    config = load_config(ROOT / "configs/phase0.yaml")
    with pytest.raises(ValueError, match="unknown variant"):
        config.variant("not-real")


def test_config_paths_are_absolute() -> None:
    config = load_config(ROOT / "configs/phase0.yaml")
    assert config.calibration.path.is_absolute()
    assert config.paths.outputs.is_absolute()
