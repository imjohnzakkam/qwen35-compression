"""Glaze v2's controls: the AutoScheme allocation, its recipe, and the one-GPU driver."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from qwen35_compression.config import load_config
from qwen35_compression.glaze2.quant import Option


def _load(name: str):
    sys.path.insert(0, "scripts")
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _group(bits: int, group_size: int, targets: list[str]) -> dict:
    weights = {"num_bits": bits, "group_size": group_size, "symmetric": True, "type": "int"}
    return {"weights": weights, "targets": targets}


PREFIX = "model.language_model.layers.0"


def test_reference_allocation_reads_the_language_model_only() -> None:
    allocate = _load("autoscheme_allocate")
    config = {
        "quantization_config": {
            "config_groups": {
                "group_0": _group(4, 32, [f"{PREFIX}.mlp.gate_proj", f"{PREFIX}.mlp.up_proj"]),
                "group_1": _group(8, 128, [f"{PREFIX}.mlp.down_proj", "model.visual.blocks.0.fc"]),
            }
        }
    }
    assert allocate.reference_allocation(config) == {
        f"{PREFIX}.mlp.gate_proj": Option(4, 32),
        f"{PREFIX}.mlp.up_proj": Option(4, 32),
        f"{PREFIX}.mlp.down_proj": Option(8, 128),
    }
    config["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] = 16
    with pytest.raises(ValueError, match="unexpected reference scheme"):
        allocate.reference_allocation(config)


def test_budget_is_counted_as_autoscheme_counts_it() -> None:
    allocate = _load("autoscheme_allocate")
    # 4 x 256 at 4-bit g128: 4,096 code bits, 8 groups of a 16-bit scale and a 4-bit zero point.
    assert allocate.autoscheme_bits((4, 256), Option(4, 128)) == 4 * 1024 + 8 * 20
    shapes = {"a": (4, 256), "b": (4, 256)}
    mixed = {"a": Option(4, 128), "b": Option(8, 128)}
    expected = (4 * 1024 + 8 * 20 + 8 * 1024 + 8 * 24) / 2048
    assert allocate.average_bits(mixed, shapes) == pytest.approx(expected)
    # Export bytes: codes plus one BF16 scale per group, no zero point.
    assert allocate.export_bytes(mixed, shapes) == 512 + 16 + 1024 + 16


def test_autoscheme_choices_map_to_export_names_and_the_menu() -> None:
    allocate = _load("autoscheme_allocate")
    layer_config = {
        "model.language_model.layers.3.mlp.down_proj": {"bits": 8, "group_size": 128, "sym": True},
        "model.layers.4.self_attn.q_proj": {"bits": 4, "group_size": 32, "sym": True},
        "model.visual.blocks.0.attn.qkv": {"bits": 16, "group_size": 128},
        "model.language_model.layers.5.mlp.up_proj": {"bits": 16, "group_size": 128},
        "mtp.layers.0.mlp.up_proj": {"bits": 4, "group_size": 128, "sym": True},
        "lm_head": {"bits": 16},
    }
    assert allocate.from_layer_config(layer_config) == {
        "model.language_model.layers.3.mlp.down_proj": Option(8, 128),
        "model.language_model.layers.4.self_attn.q_proj": Option(4, 32),
    }
    layer_config["model.language_model.layers.3.mlp.down_proj"]["group_size"] = 16
    with pytest.raises(ValueError, match="outside the menu"):
        allocate.from_layer_config(layer_config)


def test_fused_units_must_share_an_option() -> None:
    allocate = _load("autoscheme_allocate")
    names = [f"{PREFIX}.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj")]
    allocate.check_units(dict.fromkeys(names, Option(4, 64)))
    with pytest.raises(ValueError, match="mixed options"):
        allocate.check_units({**dict.fromkeys(names, Option(4, 64)), names[0]: Option(8, 128)})


def test_allocation_recipe_tunes_each_linear_at_its_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen35_compression.quantization.recipes import build_recipe

    module = types.ModuleType("llmcompressor.modifiers.autoround")
    module.AutoRoundModifier = lambda **kwargs: types.SimpleNamespace(**kwargs)
    for name in ("llmcompressor", "llmcompressor.modifiers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "llmcompressor.modifiers.autoround", module)
    variants = yaml.safe_load(Path("configs/variants/feature1.yaml").read_text())["variants"]
    variant = next(v for v in variants if v["name"] == "autoround_autoscheme_w4a16")
    allocation = tmp_path / "allocation.json"
    allocation.write_text(
        json.dumps(
            {
                "options": {
                    "w4g128": [f"{PREFIX}.mlp.down_proj"],
                    "w4g32": [f"{PREFIX}.mlp.gate_proj", f"{PREFIX}.mlp.up_proj"],
                    "w4g64": [],
                    "w8g128": [f"{PREFIX}.linear_attn.out_proj"],
                }
            }
        )
    )
    variant["allocation"] = str(allocation)
    (tmp_path / "variants.yaml").write_text(yaml.safe_dump({"variants": [variant]}))
    raw = yaml.safe_load(Path("configs/feature1.yaml").read_text())
    raw["variants_path"] = str(tmp_path / "variants.yaml")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    config = load_config(tmp_path / "config.yaml")
    (modifier,) = build_recipe(config.variant("autoround_autoscheme_w4a16"))
    groups = modifier.config_groups
    assert list(groups) == ["group_w4g128", "group_w4g32", "group_w8g128"]
    layout = {
        name: (g.weights.num_bits, g.weights.group_size, g.weights.strategy, g.targets)
        for name, g in groups.items()
    }
    assert layout["group_w4g32"] == (
        4,
        32,
        "group",
        [f"{PREFIX}.mlp.gate_proj", f"{PREFIX}.mlp.up_proj"],
    )
    assert layout["group_w8g128"][:3] == (8, 128, "group")
    assert modifier.iters == 200


def test_only_autoround_variants_take_an_allocation(tmp_path: Path) -> None:
    variants = yaml.safe_load(Path("configs/variants/feature1.yaml").read_text())["variants"]
    gptq = next(v for v in variants if v["name"] == "gptq_w4a16_g128")
    gptq["allocation"] = "allocation.json"
    (tmp_path / "variants.yaml").write_text(yaml.safe_dump({"variants": [gptq]}))
    raw = yaml.safe_load(Path("configs/feature1.yaml").read_text())
    raw["variants_path"] = str(tmp_path / "variants.yaml")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="only autoround variants may set allocation"):
        load_config(tmp_path / "config.yaml")


def test_controls_driver_pilots_both_then_runs_each_in_full() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_glaze_controls.py", "--dry-run", "--code-revision", "abc"],
        check=True,
        capture_output=True,
        text=True,
    )
    steps = json.loads(result.stdout)
    assert list(steps) == [
        "blocks:full",
        "blocks:pilot",
        "pilot:data:quantize",
        "pilot:allocation:allocate",
        "pilot:allocation:quantize",
        "pilot:allocation:vllm_check",
        "full:data:quantize",
        "full:data:suite",
        "full:allocation:allocate",
        "full:allocation:quantize",
        "full:allocation:suite",
        "evaluate",
    ]
    config = load_config("configs/feature1.yaml")
    data = config.variant("autoround_glaze2_data_w4a16_g128")
    scheme = config.variant("autoround_autoscheme_w4a16")
    glaze = config.variant("glaze2_w4a16_g128").glaze2
    # Both controls tune on Glaze v2's own calibration blocks, as many as AutoRound's cache holds.
    assert data.calibration_blocks == scheme.calibration_blocks == glaze.calibration_blocks
    assert data.calibration_samples == scheme.calibration_samples == 128
    assert data.allocation is None and scheme.allocation is not None
    for name in ("autoround_glaze2_data_pilot_w4a16_g128", "autoround_autoscheme_pilot_w4a16"):
        assert config.variant(name).autoround_iters < 200
    evaluate = steps["evaluate"]
    assert evaluate[evaluate.index("--variants") + 1].split(",") == [
        "autoround_w4a16_g128",
        "autoround_glaze2_data_w4a16_g128",
        "autoround_autoscheme_w4a16",
        "glaze2_w4a16_g128",
    ]
    suite = steps["full:allocation:suite"]
    assert suite[suite.index("--suite") + 1] == "configs/evaluation/feature1_controls.yaml"
    assert "--text-only" in suite and "--skip-bootstrap" in suite
    full = steps["full:allocation:allocate"]
    assert full[full.index("--nsamples") + 1] == "16"
    assert full[full.index("--reference") + 1].endswith("outputs/feature1/glaze2_w4a16_g128")


def test_controls_suite_matches_the_instruct_protocol() -> None:
    from qwen35_compression.feature1 import load_benchmark_suite

    controls = load_benchmark_suite(Path("configs/evaluation/feature1_controls.yaml"))
    full = load_benchmark_suite(Path("configs/evaluation/feature1.yaml"))
    assert controls.text_tasks == ("minerva_math500", "ifeval")
    assert not controls.vision_tasks
    for field in ("fewshot", "max_model_len", "seed", "enable_thinking", "generation"):
        assert getattr(controls, field) == getattr(full, field)


def test_a_failed_control_skips_only_its_own_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    import qwen35_compression.drift as drift
    import qwen35_compression.models as models

    driver = _load("run_glaze_controls")
    ran: list[list[str]] = []

    def run_logged(command: list[str], log: Path, root: Path) -> float:
        ran.append(command)
        if "autoround_autoscheme_pilot_w4a16" in command and "scripts/quantize.py" in command:
            raise subprocess.CalledProcessError(1, command)
        return 0.0

    monkeypatch.setattr(driver, "run_logged", run_logged)
    monkeypatch.setattr(driver, "check_blocks", lambda config, stage: {"stage": stage})
    monkeypatch.setattr(models, "download_model", lambda config: (tmp_path, "rev"))
    monkeypatch.setattr(drift, "copy_published", lambda repo, target: "published")
    args = argparse.Namespace(
        config=Path("configs/feature1.yaml"),
        output=tmp_path / "results",
        skip_bootstrap=True,
        code_revision="abc",
    )
    manifest = driver.study(args)
    assert manifest["status"] == "failed"
    assert manifest["steps"]["pilot:allocation:quantize"]["status"] == "failed"
    assert manifest["skipped"] == [
        "pilot:allocation:vllm_check",
        "full:allocation:allocate",
        "full:allocation:quantize",
        "full:allocation:suite",
    ]
    assert manifest["steps"]["full:data:suite"]["status"] == "passed"
    evaluate = ran[-1]
    assert evaluate[evaluate.index("--variants") + 1].split(",") == [
        "autoround_w4a16_g128",
        "autoround_glaze2_data_w4a16_g128",
        "glaze2_w4a16_g128",
    ]
