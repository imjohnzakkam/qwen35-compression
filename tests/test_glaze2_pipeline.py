"""Glaze v2 in the pipeline: config validation, the proxy driver, the gate, and a tiny run."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from glaze_tiny import WIDE, random_blocks, tiny_teacher

from qwen35_compression.config import load_config
from qwen35_compression.glaze2.data import PackedBlocks, save_blocks

PROXY = "configs/glaze2_proxy.yaml"
FULL = "glaze2_w4a16_g128"
UNIFORM = "glaze2_uniform_w4a16_g128"


def _load(name: str):
    sys.path.insert(0, "scripts")
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate_script = _load("glaze2_evaluate")


def _config_with(tmp_path: Path, change, paths: dict | None = None) -> Path:
    raw = yaml.safe_load(Path(PROXY).read_text(encoding="utf-8"))
    variants = yaml.safe_load(Path(raw["variants_path"]).read_text(encoding="utf-8"))
    change({v["name"]: v for v in variants["variants"]})
    (tmp_path / "variants.yaml").write_text(yaml.safe_dump(variants), encoding="utf-8")
    raw["variants_path"] = str(tmp_path / "variants.yaml")
    if paths:
        raw["paths"] = paths
    path = tmp_path / "configs" / "proxy.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------- configuration


def test_proxy_configs_describe_phase_one() -> None:
    config = load_config(PROXY)
    assert config.model.id == "Qwen/Qwen3.5-0.8B"
    full, uniform = config.variant(FULL).glaze2, config.variant(UNIFORM).glaze2
    assert full is not None and uniform is not None
    assert full.allocate and not uniform.allocate
    assert full.byte_target == uniform.byte_target == "autoround_w4a16_g128"
    data = config.variant("autoround_glaze2_data_w4a16_g128")
    assert data.calibration_blocks == full.calibration_blocks
    pilot = load_config("configs/glaze2_proxy_pilot.yaml")
    assert pilot.paths.outputs != config.paths.outputs
    assert pilot.variant(FULL).glaze2.max_iters < full.max_iters
    # The study runs AutoRound as published; only the pilot cuts its steps.
    for name in ("autoround_w4a16_g128", "autoround_glaze2_data_w4a16_g128"):
        assert config.variant(name).autoround_iters == 200
        assert pilot.variant(name).autoround_iters < 200


def test_autoround_recipe_takes_its_step_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    from qwen35_compression.quantization.recipes import build_recipe

    # llm-compressor lives in the GPU environment only; a stand-in records the arguments.
    module = types.ModuleType("llmcompressor.modifiers.autoround")
    module.AutoRoundModifier = lambda **kwargs: types.SimpleNamespace(**kwargs)
    for name in ("llmcompressor", "llmcompressor.modifiers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "llmcompressor.modifiers.autoround", module)
    pilot = load_config("configs/glaze2_proxy_pilot.yaml")
    (modifier,) = build_recipe(pilot.variant("autoround_w4a16_g128"))
    assert modifier.iters == 10


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda v: v[FULL].pop("glaze2"), "glaze2 settings"),
        (lambda v: v[FULL]["glaze2"].update(typo=1), "unknown glaze2 settings"),
        (lambda v: v[FULL]["glaze2"].pop("held_out_blocks"), "held_out_blocks"),
        (lambda v: v[FULL]["glaze2"].update(byte_target="bf16"), "byte_target"),
        (lambda v: v[FULL].update(group_size=64), "W4A16 g128"),
        (lambda v: v[FULL].update(ignore=["lm_head"]), "same layers"),
        (lambda v: v[FULL]["glaze2"].update(max_iters=0), "max_iters"),
        (lambda v: v[FULL]["glaze2"].update(min_iters=500), "out of range"),
        (lambda v: v[FULL]["glaze2"].update(margin_bytes=-1), "margin_bytes"),
        (lambda v: v[FULL]["glaze2"].update(budget_fraction=0), "budget_fraction"),
        (lambda v: v["bf16"].update(calibration_blocks="x.jsonl"), "calibration_blocks"),
        (lambda v: v["bf16"].update(autoround_iters=10), "autoround_iters"),
        (lambda v: v["autoround_w4a16_g128"].update(autoround_iters=0), "autoround_iters"),
        (lambda v: v["autoround_w4a16_g128"].update(glaze2={}), "glaze2 settings"),
    ],
)
def test_glaze2_settings_are_validated(tmp_path: Path, change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_config_with(tmp_path, change))


# ---------------------------------------------------------------- driver and gate


def test_proxy_dry_run_answers_once_then_pilots_before_the_study() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_glaze2_proxy.py", "--dry-run", "--code-revision", "abc"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert list(plan) == ["answers", "pilot", "full"]
    assert "--limit" not in plan["answers"]
    shared = plan["answers"][plan["answers"].index("--output") + 1]
    for stage in ("pilot", "full"):
        assert list(plan[stage]) == [
            "blocks",
            "quantize:autoround_w4a16_g128",
            "quantize:autoround_glaze2_data_w4a16_g128",
            "quantize:glaze2_uniform_w4a16_g128",
            "quantize:glaze2_w4a16_g128",
            "evaluate",
            "vllm_check",
        ]
        blocks = plan[stage]["blocks"]
        assert blocks[blocks.index("--answers") + 1] == shared
    full = plan["full"]["blocks"]
    assert full[full.index("--calibration-blocks") + 1] == "512"
    pilot = plan["pilot"]["blocks"]
    assert "glaze2_proxy_pilot" in pilot[pilot.index("--output-dir") + 1]


def _scores(a: float, b: float, c: float, d: float, chat_d: float = 0.010) -> dict:
    def entry(in_domain: float, chat: float) -> dict:
        return {"in_domain": {"mean_kl": in_domain}, "chat": {"mean_kl": chat}}

    return {
        evaluate_script.BASELINE: entry(a, 0.010),
        evaluate_script.DATA_ONLY: entry(b, 0.010),
        evaluate_script.UNIFORM: entry(c, 0.010),
        evaluate_script.FULL: entry(d, chat_d),
    }


def test_gate_needs_fifteen_percent_without_hurting_chat_or_size() -> None:
    sizes = {name: 100 for name in evaluate_script.VARIANTS}
    passed = evaluate_script.gate(_scores(0.040, 0.038, 0.036, 0.032), sizes)
    assert passed["decision"] == "go" and passed["in_domain_reduction"] == pytest.approx(0.2)
    assert passed["ablations"]["H2_data (B vs A)"] == pytest.approx(0.05)
    assert evaluate_script.gate(_scores(0.040, 0.04, 0.04, 0.035), sizes)["decision"] == "stop"
    worse_chat = evaluate_script.gate(_scores(0.040, 0.04, 0.04, 0.030, chat_d=0.011), sizes)
    assert worse_chat["decision"] == "stop" and worse_chat["chat_reduction"] < -0.05
    too_big = dict(sizes, **{evaluate_script.FULL: 101})
    assert evaluate_script.gate(_scores(0.040, 0.04, 0.04, 0.030), too_big)["decision"] == "stop"


def test_gate_on_the_4b_scores_autoround_and_glaze2_alone() -> None:
    scores = _scores(0.040, 0.04, 0.04, 0.030)
    pair = {name: scores[name] for name in (evaluate_script.BASELINE, evaluate_script.FULL)}
    result = evaluate_script.gate(pair, {name: 100 for name in pair})
    assert result["decision"] == "go" and result["in_domain_reduction"] == pytest.approx(0.25)
    assert result["ablations"] == {"H4_all (D vs A)": pytest.approx(0.25)}
    assert (result["baseline"], result["candidate"]) == (
        evaluate_script.BASELINE,
        evaluate_script.FULL,
    )


# ---------------------------------------------------------------- end to end, tiny


class _FakeVisionLanguageModel(torch.nn.Module):
    def __init__(self, text: torch.nn.Module) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = text
        self.model.visual = torch.nn.Linear(2, 2)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.model.language_model.embed_tokens


def test_glaze2_quantizes_a_tiny_model_within_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_glaze2 import _snapshot

    import qwen35_compression.models as models
    from qwen35_compression.export import MANIFEST_NAME
    from qwen35_compression.glaze2.evaluate import evaluate_export
    from qwen35_compression.glaze2.pipeline import directory_bytes, quantize

    teacher = tiny_teacher(**WIDE)
    snapshot = _snapshot(teacher, tmp_path / "bf16")
    outputs = tmp_path / "outputs"
    data = outputs / "data"
    data.mkdir(parents=True)
    length = len(random_blocks(1)[0])
    for name, count, seed in (("calibration", 16, 1), ("held_out", 8, 2)):
        ids = random_blocks(count, seed=seed)
        save_blocks(
            data / f"{name}.jsonl",
            PackedBlocks(
                ids,
                [[1] * length] * count,
                [[0] * (length // 2) + [2] * (length - length // 2)] * count,
            ),
        )

    def tiny(variants: dict) -> None:
        for name in (FULL, UNIFORM):
            settings = variants[name]["glaze2"]
            settings.update(
                calibration_blocks=str(data / "calibration.jsonl"),
                held_out_blocks=str(data / "held_out.jsonl"),
                max_iters=6,
                min_iters=3,
                eval_every=3,
                fisher_chunk_tokens=16,
                margin_bytes=0,
            )
        variants[FULL]["glaze2"]["budget_fraction"] = 0.5

    path = _config_with(
        tmp_path, tiny, {"outputs": str(outputs), "results": str(tmp_path / "results")}
    )
    config = load_config(path)
    # AutoRound's export stands in as the byte target: the BF16 checkpoint's size gives room.
    target = outputs / "autoround_w4a16_g128"
    target.mkdir()
    (target / "model.safetensors").write_bytes(b"\0" * directory_bytes(snapshot))
    monkeypatch.setenv("QWEN35_CODE_REVISION", "test")
    monkeypatch.setattr(models, "download_model", lambda config: (snapshot, "rev"))
    monkeypatch.setattr(
        models,
        "load_resolved_model",
        lambda *a, **k: (_FakeVisionLanguageModel(tiny_teacher(**WIDE)), None),
    )

    for name in (UNIFORM, FULL):
        output_dir, manifest = quantize(config, config.variant(name), log=lambda _: None)
        record = manifest["glaze2"]
        assert (output_dir / MANIFEST_NAME).exists()
        assert record["export"]["total_bytes"] <= directory_bytes(target)
        assert len(record["layers"]) == len(teacher.layers)
        assert all(x["held_out_best"] <= x["held_out_rtn"] for x in record["layers"])
        scores = evaluate_export(
            teacher,
            teacher.get_input_embeddings().weight,
            output_dir,
            _blocks(data / "held_out.jsonl"),
        )
        assert scores["math"]["mean_kl"] > 0 and scores["chat"]["tokens"] > 0
    uniform = json.loads((outputs / UNIFORM / MANIFEST_NAME).read_text())["glaze2"]
    full = json.loads((outputs / FULL / MANIFEST_NAME).read_text())["glaze2"]
    assert uniform["allocation"]["options"] == {"w4g128": len(uniform["allocation"]["choice"])}
    assert uniform["export"]["vision_linears"] == 0 and full["export"]["vision_linears"] == 1
    assert full["allocation"]["extra_bytes"] > 0 and len(full["allocation"]["options"]) > 1
    assert full["allocation"]["budget"] <= uniform_language_bytes(teacher) * 0.5 + 1
    with pytest.raises(FileExistsError):
        quantize(config, config.variant(FULL), log=lambda _: None)


def uniform_language_bytes(teacher: torch.nn.Module) -> int:
    from qwen35_compression.glaze2.quant import BASE

    return sum(
        BASE.tensor_bytes(*m.weight.shape) + 16
        for n, m in teacher.named_modules()
        if isinstance(m, torch.nn.Linear) and n.startswith("layers.")
    )


def _blocks(path: Path):
    from qwen35_compression.glaze2.data import load_blocks

    return load_blocks(path)


def test_pilot_answers_cover_every_domain() -> None:
    answers = _load("glaze2_answers")
    prompts = [{"id": str(i), "domain": d} for d in ("math", "mcq", "chat") for i in range(5)]
    first = answers.rotate(prompts)[:3]
    assert [p["domain"] for p in first] == ["math", "mcq", "chat"]
    assert len(answers.rotate(prompts)) == 15


def test_gate_tolerates_a_domain_without_tokens() -> None:
    sizes = {name: 100 for name in evaluate_script.VARIANTS}
    scores = _scores(0.040, 0.04, 0.04, 0.030)
    for entry in scores.values():
        entry["chat"]["mean_kl"] = None
    result = evaluate_script.gate(scores, sizes)
    assert result["decision"] == "go" and result["chat_reduction"] is None


def test_autoround_reads_glaze2_block_files(tmp_path: Path) -> None:
    from qwen35_compression.runner import blocks_dataset

    path = tmp_path / "blocks.jsonl"
    save_blocks(path, PackedBlocks([[1, 2, 3], [4, 5, 6]], [[0, 1, 1]] * 2, [[0, 0, 0]] * 2))
    dataset, collate = blocks_dataset(path)
    assert len(dataset) == 2
    batch = collate([dataset[1]])
    assert batch["input_ids"].tolist() == [[4, 5, 6]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1]]
    with pytest.raises(ValueError, match="batch size 1"):
        collate([dataset[0], dataset[1]])


def test_4b_driver_pilots_then_gates_before_the_suite() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_glaze2_4b.py", "--dry-run", "--code-revision", "abc"],
        check=True,
        capture_output=True,
        text=True,
    )
    steps = json.loads(result.stdout)
    assert list(steps) == [
        "answers",
        *(f"{stage}:{name}" for stage in ("pilot", "full") for name in STAGE_STEPS),
        "drift",
        "suite",
    ]
    pilot, full = steps["pilot:evaluate"], steps["full:evaluate"]
    assert pilot[pilot.index("--variants") + 1] == "autoround_w4a16_g128,glaze2_pilot_w4a16_g128"
    assert full[full.index("--variants") + 1] == "autoround_w4a16_g128,glaze2_w4a16_g128"
    blocks = steps["full:blocks"]
    assert blocks[blocks.index("--output-dir") + 1].endswith("outputs/feature1/glaze2/data")
    assert steps["drift"][steps["drift"].index("--variants") + 1] == "glaze2_w4a16_g128"
    assert "--skip-bf16" in steps["drift"]
    suite = steps["suite"]
    assert suite[suite.index("--variant") + 1] == "glaze2_w4a16_g128"
    assert "--skip-bootstrap" in suite and "--pilot-limit" in suite
    # The 4B spends every byte the 8-bit vision tower frees, and the pilot only cuts iterations.
    config = load_config("configs/feature1.yaml")
    full_settings = config.variant("glaze2_w4a16_g128").glaze2
    pilot_settings = config.variant("glaze2_pilot_w4a16_g128").glaze2
    assert full_settings.budget_fraction is None and full_settings.max_iters == 400
    assert pilot_settings.max_iters < full_settings.max_iters
    assert pilot_settings.byte_target == full_settings.byte_target == "autoround_w4a16_g128"


STAGE_STEPS = ("blocks", "quantize", "evaluate", "vllm_check")
