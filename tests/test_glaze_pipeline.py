"""Glaze's place in the pipeline: config, dispatch, drift comparisons, CLIs and the B1 driver."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from qwen35_compression.config import GlazeConfig, load_config
from qwen35_compression.export import MANIFEST_NAME, verify_export, write_export_manifest
from qwen35_compression.io import inventory

GLAZE = "glaze_v1_w4a16_g128"


def _load(name: str):
    sys.path.insert(0, "scripts")
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


drift = _load("drift_scores")
study = _load("run_glaze_study")


# ---------------------------------------------------------------- configuration


def test_glaze_v1_is_configured_on_autorounds_own_data() -> None:
    config = load_config("configs/feature1.yaml")
    variant = config.variant(GLAZE)
    init = config.variant("autoround_w4a16_g128")
    assert (variant.method, variant.init) == ("glaze", "autoround_w4a16_g128")
    assert (variant.bits, variant.group_size, variant.ignore) == (4, 128, init.ignore)
    glaze = variant.glaze
    assert glaze is not None and glaze.data == "calibration"
    # Same 128 blocks AutoRound calibrated on, and 32 it never saw.
    assert glaze.train_blocks == init.calibration_samples == 128
    assert glaze.dev_blocks == 32
    # B1's diagnosis: 2,805 moves per step overshoot, 281 (1e-5) lower the KL.
    assert glaze.flip_fractions == (1e-5, 2e-5, 4e-5) and glaze.momentum == 0.9
    assert glaze.max_relative_change == 0.01
    assert config.variant("gptq_w4a16_g128").glaze is None


def _config_with(tmp_path: Path, change) -> Path:
    """configs/feature1.yaml with its variants edited by `change`, written to tmp_path."""
    raw = yaml.safe_load(Path("configs/feature1.yaml").read_text(encoding="utf-8"))
    variants = yaml.safe_load(Path("configs/variants/feature1.yaml").read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in variants["variants"]}
    change(by_name)
    (tmp_path / "variants.yaml").write_text(yaml.safe_dump(variants), encoding="utf-8")
    raw["variants_path"] = str(tmp_path / "variants.yaml")
    path = tmp_path / "configs" / "feature1.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _glaze(items: dict) -> dict:
    return items[GLAZE]["glaze"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda v: v[GLAZE].pop("init"), "requires init"),
        (lambda v: v[GLAZE].update(init="nope"), "not a configured variant"),
        (lambda v: v[GLAZE].update(init="awq_w4a16_g128"), "glaze init must be one of"),
        (lambda v: v[GLAZE].update(group_size=64), "match its init"),
        (lambda v: v[GLAZE].update(ignore=["lm_head"]), "same layers unquantized"),
        (lambda v: _glaze(v).update(typo=1), "unknown glaze settings"),
        (lambda v: _glaze(v).update(data="answers"), "unsupported glaze data"),
        (lambda v: _glaze(v).update(epochs=0), "epochs must be positive"),
        (lambda v: _glaze(v).update(flip_fractions=[]), "flip fractions"),
        (lambda v: _glaze(v).update(flip_fractions=[0.0, 0.1]), "flip fractions"),
        (lambda v: _glaze(v).update(momentum=1.0), "momentum"),
        (lambda v: _glaze(v).update(max_relative_change=0), "max_relative_change"),
        (lambda v: _glaze(v).update(max_relative_change=1.5), "max_relative_change"),
        (lambda v: _glaze(v).update(micro_batch_tokens=3000), "multiple of the 2048-token"),
        (lambda v: _glaze(v).update(tokens_per_step=12288), "divide"),
        (lambda v: _glaze(v).update(dev_blocks=30), "whole steps"),
        (lambda v: _glaze(v).update(warmup_steps=64), "warmup"),
        (lambda v: _glaze(v).update(max_memory_fraction=1.5), "memory fraction"),
        (lambda v: _glaze(v).update(train_blocks=64), "init's calibration samples"),
        (lambda v: _glaze(v).update(train_start=-8), "must not be negative"),
        (lambda v: _glaze(v).update(dev_start=100), "overlap"),
        (lambda v: _glaze(v).update(train_start=64), "start after the init's"),
        (
            lambda v: _glaze(v).update(train_start=160, train_blocks=120, dev_start=96),
            "never calibrated on",
        ),
        (lambda v: v["autoround_w4a16_g128"].pop("calibration_samples"), "calibration_samples"),
        (lambda v: v["gptq_w4a16_g128"].update(init=GLAZE), "only glaze variants"),
    ],
)
def test_glaze_settings_are_validated(tmp_path: Path, change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_config_with(tmp_path, change))


def test_glaze_v1b_differs_from_v1_only_in_its_training_blocks() -> None:
    from dataclasses import asdict

    config = load_config("configs/feature1.yaml")
    v1, v1b = config.variant(GLAZE), config.variant("glaze_v1b_w4a16_g128")
    assert (v1b.method, v1b.init, v1b.group_size, v1b.ignore) == (
        v1.method,
        v1.init,
        v1.group_size,
        v1.ignore,
    )
    assert v1b.glaze is not None and v1.glaze is not None
    # Fresh blocks that calibrated nothing, and the same held-out blocks as B1.
    assert (v1b.glaze.train_start, v1b.glaze.train_blocks) == (160, 120)
    assert (v1b.glaze.dev_start, v1b.glaze.dev_blocks) == (128, 32)
    assert (v1.glaze.train_start, v1.glaze.dev_start) == (0, None)
    placement = {"train_start", "train_blocks", "dev_start"}
    first, second = asdict(v1.glaze), asdict(v1b.glaze)
    assert {key for key in first if first[key] != second[key]} == placement


def test_glaze_defaults_describe_stage_b1() -> None:
    defaults = GlazeConfig()
    assert (defaults.train_blocks, defaults.dev_blocks, defaults.epochs) == (128, 32, 4)
    assert (defaults.tokens_per_step, defaults.micro_batch_tokens) == (16384, 8192)


def test_quantize_hands_glaze_variants_to_refine(monkeypatch: pytest.MonkeyPatch) -> None:
    import qwen35_compression.glaze.train as train
    from qwen35_compression.runner import quantize

    calls = []
    monkeypatch.setattr(train, "refine", lambda config, variant: calls.append(variant.name) or 7)
    config = load_config("configs/feature1.yaml")
    assert quantize(config, config.variant(GLAZE)) == 7
    assert calls == [GLAZE]


def test_refine_checks_its_inputs_before_loading_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qwen35_compression.models as models
    from qwen35_compression.glaze.train import refine

    def no_loading(*args, **kwargs):
        raise AssertionError("refine loaded a model before checking its inputs")

    monkeypatch.setattr(models, "load_resolved_model", no_loading)
    config = load_config(_config_with(tmp_path, lambda v: None))
    variant = config.variant(GLAZE)
    with pytest.raises(FileNotFoundError, match="manifest"):
        refine(config, variant, output_dir=tmp_path / "out")  # no init export yet
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "x").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        refine(config, variant, output_dir=occupied)
    with pytest.raises(ValueError, match="not a glaze variant"):
        refine(config, config.variant("gptq_w4a16_g128"), output_dir=tmp_path / "x")


def test_export_manifest_carries_method_records(tmp_path: Path) -> None:
    config = load_config("configs/feature1.yaml")
    variant = config.variant(GLAZE)
    (tmp_path / "config.json").write_text('{"quantization_config": {}}', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    manifest = write_export_manifest(
        tmp_path, config, variant, "rev", 1.0, None, extra={"glaze": {"best_step": 3}}
    )
    assert manifest["glaze"] == {"best_step": 3} and manifest["method"] == "glaze"
    with pytest.raises(ValueError, match="clash"):
        write_export_manifest(tmp_path, config, variant, "rev", 1.0, None, extra={"files": []})


# ---------------------------------------------------------------- drift comparisons


def test_answer_records_count_the_same_positions_as_the_tallies() -> None:
    ranks = [1, 2] * 4100 + [3] * 20
    logprobs = [-0.5] * len(ranks)
    record = drift.answer_record("minerva_math500", 7, ranks, logprobs)
    assert record["tokens"] == drift.BUCKETS[-1] == 8192
    assert record["flips"] == 4096
    assert record["nll"] == pytest.approx(0.5 * 8192)
    tally = drift.Tally()
    tally.add(ranks, logprobs)
    assert sum(tally.tokens) == record["tokens"] and sum(tally.flips) == record["flips"]


def _result(name: str, rows: list[tuple[int, int, float]]) -> dict:
    return {
        "name": name,
        "per_answer": [
            {"task": "minerva_math500", "doc_id": i, "tokens": t, "flips": f, "nll": n}
            for i, (t, f, n) in enumerate(rows)
        ],
    }


def test_paired_difference_is_token_weighted_with_an_interval() -> None:
    base = _result("init", [(100, 5, 20.0), (300, 12, 45.0), (50, 3, 9.0)])
    other = _result("glaze", [(100, 4, 18.0), (300, 10, 41.0), (50, 3, 8.5)])
    diff = drift.paired_difference(base, other, resamples=500)
    assert diff["mean_nll"]["difference"] == pytest.approx((-2.0 - 4.0 - 0.5) / 450)
    assert diff["flip_rate"]["difference"] == pytest.approx(-3 / 450)
    assert diff["mean_nll"]["low"] <= diff["mean_nll"]["difference"] <= diff["mean_nll"]["high"]
    assert diff["answers"] == {"count": 3, "tokens": 450}
    same = drift.paired_difference(base, base, resamples=50)
    assert same["mean_nll"] == {"difference": 0.0, "low": 0.0, "high": 0.0}
    with pytest.raises(ValueError, match="same answers"):
        drift.paired_difference(base, _result("x", [(100, 4, 18.0)]))
    with pytest.raises(ValueError, match="tokenized"):
        drift.paired_difference(base, _result("x", [(99, 4, 18.0), (300, 1, 1.0), (50, 1, 1.0)]))
    with pytest.raises(ValueError, match="rescore"):
        drift.paired_difference({"name": "old"}, base)


def test_report_prints_paired_rows_against_a_base() -> None:
    base = _result("init", [(100, 5, 20.0), (300, 12, 45.0)])
    other = _result("glaze", [(100, 4, 18.0), (300, 10, 41.0)])
    lines = drift.paired_rows(base, [base, other])
    assert lines[0].startswith("change from init") and len(lines) == 4
    assert lines[3].startswith("| glaze | -0.0150")


# ---------------------------------------------------------------- the B1 driver


def test_gate_passes_stops_or_asks() -> None:
    init_nll = study.BF16_MEAN_NLL + 0.0417
    improved = {"difference": -0.004, "low": -0.005, "high": -0.003}
    math_ok = {"delta": -1.0, "low": -4.0, "high": 2.0}
    gate = study.evaluate_gate(improved, init_nll, math_ok)
    assert gate["decision"] == "pass"
    assert gate["excess_reduction"] == pytest.approx(0.004 / 0.0417)
    assert gate["glaze_excess_loss"] == pytest.approx(0.0377)
    assert study.evaluate_gate(improved, init_nll, {"delta": -3.0})["decision"] == "ask"
    noisy = {"difference": -0.004, "low": -0.006, "high": 0.001}
    assert study.evaluate_gate(noisy, init_nll, math_ok)["decision"] == "ask"
    flat = {"difference": -0.0005, "low": -0.001, "high": -0.0001}
    assert study.evaluate_gate(flat, init_nll, math_ok)["decision"] == "stop"
    worse = {"difference": 0.002, "low": 0.001, "high": 0.003}
    assert study.evaluate_gate(worse, init_nll, math_ok)["decision"] == "stop"


def test_scoring_needs_a_dev_kl_reduction() -> None:
    record = {"dev_at_init": {"mean_kl": 0.0100}, "dev_best": {"mean_kl": 0.0097}}
    assert study.dev_kl_reduction(record) == pytest.approx(0.03)
    assert study.dev_kl_reduction(record) >= study.SCORE_MIN_DEV_REDUCTION
    record["dev_best"] = {"mean_kl": 0.0099}
    assert study.dev_kl_reduction(record) < study.SCORE_MIN_DEV_REDUCTION
    # No step beat the init: the export is the init's, and nothing is gained.
    record["dev_best"] = record["dev_at_init"]
    assert study.dev_kl_reduction(record) == 0


class _Scored(Exception):
    """Raised by the fake runner when the study reaches drift scoring."""


def _run_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_best: float):
    """study() with every download and subprocess faked; training lowers dev KL to dev_best."""
    import argparse

    import qwen35_compression.models as models

    path = _config_with(tmp_path, lambda variants: None)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["paths"] = {"outputs": str(tmp_path / "outputs"), "results": str(tmp_path / "results")}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(path)
    variant = config.variant(GLAZE)
    ran: list[str] = []

    def run_logged(command, log, root):
        ran.append(" ".join(command))
        if "drift_scores.py" in " ".join(command) and "pilot" not in " ".join(command):
            raise _Scored(command)
        return 0.0

    record = {
        "dev_at_init": {"mean_kl": 0.0100},
        "dev_best": {"mean_kl": dev_best},
        "flip_fraction": 1e-5,
    }
    monkeypatch.setattr(models, "download_model", lambda config: (tmp_path / "bf16", "rev"))
    monkeypatch.setattr(study, "fetch_published", lambda repo: (tmp_path / "init", "initrev"))
    monkeypatch.setattr(study, "install_init", lambda *args: {})
    monkeypatch.setattr(study, "fetch_traces", lambda traces: "tracesrev")
    monkeypatch.setattr(study, "run_logged", run_logged)
    monkeypatch.setattr(study, "drift_summary", lambda p: {"flip_rate": 0.05, "mean_nll": 0.17})
    monkeypatch.setattr(study, "verify_export", lambda directory, v: {"glaze": record})
    args = argparse.Namespace(
        skip_bootstrap=True,
        pilot_steps=5,
        pilot_limit=10,
        code_revision="abc",
        output=tmp_path / "out",
        diagnose=False,
    )
    return config, variant, args, ran


def test_study_stops_before_scoring_when_dev_kl_barely_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, variant, args, ran = _run_study(tmp_path, monkeypatch, dev_best=0.0099)
    manifest = study.study(args, config, variant)
    assert manifest["status"] == "stopped" and "1.00%" in manifest["stopped"]
    assert manifest["glaze"]["dev_kl_reduction"] == pytest.approx(0.01)
    assert set(manifest["steps"]) == {
        "download_bf16",
        "download_init",
        "install_init",
        "fla",
        "traces",
        "pilot_train",
        "pilot_score",
        "train",
    }
    assert not any("math500" in command or "init.json" in command for command in ran)
    stored = json.loads((tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8"))
    assert stored["status"] == "stopped"


def test_study_scores_when_dev_kl_falls_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, variant, args, ran = _run_study(tmp_path, monkeypatch, dev_best=0.0097)
    with pytest.raises(_Scored):
        study.study(args, config, variant)
    stored = json.loads((tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8"))
    assert stored["status"] == "failed" and stored["steps"]["drift:init"]["status"] == "failed"
    assert stored["glaze"]["dev_kl_reduction"] == pytest.approx(0.03)


def test_published_init_is_installed_exactly_as_its_manifest_lists(tmp_path: Path) -> None:
    config = load_config("configs/feature1.yaml")
    variant = config.variant("autoround_w4a16_g128")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"quantization_config": {}}', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    manifest = {
        "variant": variant.name,
        "method": "autoround",
        "code_revision": "abc",
        "files": inventory(snapshot),
    }
    (snapshot / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (snapshot / "README.md").write_text("card", encoding="utf-8")  # not part of the export
    init_dir = tmp_path / "outputs" / variant.name
    study.install_init(snapshot, init_dir, variant)
    assert sorted(p.name for p in init_dir.iterdir()) == sorted(
        ["config.json", "model.safetensors", MANIFEST_NAME]
    )
    assert verify_export(init_dir, variant)["code_revision"] == "abc"
    # A second call reuses the verified directory.
    study.install_init(snapshot, init_dir, variant)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True)


def test_study_dry_run_lists_every_step() -> None:
    result = _run(
        "scripts/run_glaze_study.py", "--variant", GLAZE, "--dry-run", "--code-revision", "abc"
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["init_repo"] == "lazybrick/Qwen3.5-4B-Kiln-AutoRound-W4A16-g128"
    commands = plan["commands"]
    assert list(commands) == [
        "bootstrap",
        "preflight",
        "fla",
        "pilot",
        "train",
        "math500:autoround_w4a16_g128",
        f"math500:{GLAZE}",
    ]
    pilot = commands["pilot"]
    assert pilot[pilot.index("--pilot-steps") + 1] == "5"
    assert pilot[pilot.index("--output") + 1].endswith(f"{GLAZE}-pilot")
    assert "--output" not in commands["train"] and "--pilot-steps" not in commands["train"]
    for name in ("math500:autoround_w4a16_g128", f"math500:{GLAZE}"):
        command = commands[name]
        assert command[command.index("--suite") + 1] == "configs/evaluation/feature1_math500.yaml"
        assert (
            "--skip-bootstrap" in command and command[command.index("--code-revision") + 1] == "abc"
        )
    assert plan["glaze"]["schedule"]["steps"] == 64


def test_study_diagnose_runs_only_setup_and_the_diagnosis() -> None:
    result = _run(
        "scripts/run_glaze_study.py",
        "--variant",
        GLAZE,
        "--diagnose",
        "--dry-run",
        "--code-revision",
        "abc",
        "--output",
        "results/feature1/glaze-diagnosis",
    )
    assert result.returncode == 0, result.stderr
    commands = json.loads(result.stdout)["commands"]
    assert list(commands) == ["bootstrap", "preflight", "fla", "diagnose"]
    diagnose = commands["diagnose"]
    assert diagnose[1] == "scripts/glaze_diagnose.py"
    assert diagnose[diagnose.index("--variant") + 1] == GLAZE
    assert diagnose[diagnose.index("--output") + 1].endswith("glaze-diagnosis/diagnosis.json")


def test_diagnosis_summary_keeps_the_headline_numbers(tmp_path: Path) -> None:
    report = {
        "kl": {"base": 0.0075, "repeat": 0.0075, "with_gradient": 0.0075},
        "moves": {
            "guarded_top_2805": {
                "moves": 2805,
                "kl_down": 0.0074,
                "kl_up": 0.0077,
                "best_step_fraction": 0.8,
                "by_family": {"layers.N.mlp.up_proj.scales": 2805},
            }
        },
        "pilots": {"guarded_0.0001": {"train_kl": [0.0075, 0.0074], "passed": True}},
    }
    path = tmp_path / "diagnosis.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert study.diagnosis_summary(path) == {
        "kl": report["kl"],
        "moves": {
            "guarded_top_2805": {"moves": 2805, "kl_down": 0.0074, "best_step_fraction": 0.8}
        },
        "pilots": {"guarded_0.0001": True},
    }


def test_diagnose_cli_checks_its_arguments(tmp_path: Path) -> None:
    other = _run("scripts/glaze_diagnose.py", "--variant", "gptq_w4a16_g128", "--output", "x")
    assert other.returncode != 0 and "not a glaze variant" in other.stderr
    existing = tmp_path / "diagnosis.json"
    existing.write_text("{}", encoding="utf-8")
    again = _run("scripts/glaze_diagnose.py", "--variant", GLAZE, "--output", str(existing))
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr
    missing = _run("scripts/glaze_diagnose.py", "--variant", GLAZE)
    assert missing.returncode != 0 and "--output" in missing.stderr


def test_study_rejects_variants_it_cannot_run() -> None:
    result = _run("scripts/run_glaze_study.py", "--variant", "gptq_w4a16_g128", "--dry-run")
    assert result.returncode != 0 and "not a glaze variant" in result.stderr
    result = _run("scripts/run_glaze_study.py", "--variant", GLAZE, "--pilot-steps", "1")
    assert result.returncode != 0 and "--pilot-steps" in result.stderr


def test_study_hides_brotli_before_any_download() -> None:
    result = _run(
        "-c",
        "import runpy, sys; sys.path.insert(0, 'scripts');"
        "runpy.run_path('scripts/run_glaze_study.py', run_name='probe');"
        "print(sys.modules['brotli'], sys.modules['brotlicffi'])",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "None None"


def test_glaze_cli_dry_run_and_argument_checks() -> None:
    result = _run("scripts/glaze.py", "--variant", GLAZE, "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schedule"] == {
        "steps": 64,
        "steps_per_epoch": 16,
        "blocks_per_step": 8,
        "micro_batches_per_step": 2,
        "trained_tokens": 64 * 16384,
        "probe_tokens": 3 * 8 * 16384,
    }
    # The A100 40 GB plan: well under the 85% guard even with a 1.4x margin.
    assert plan["estimate"]["memory_gib"] * 1.4 < 0.85 * 39.5
    assert plan["estimate"]["train_minutes"] == pytest.approx(11.0, abs=0.1)
    pilot = _run("scripts/glaze.py", "--variant", GLAZE, "--pilot-steps", "3", "--dry-run")
    assert pilot.returncode != 0 and "--output" in pilot.stderr
    other = _run("scripts/glaze.py", "--variant", "gptq_w4a16_g128", "--dry-run")
    assert other.returncode != 0 and "not a glaze variant" in other.stderr


# ---------------------------------------------------------------- refine, end to end on CPU


class _FakeVisionLanguageModel(torch.nn.Module):
    """What load_resolved_model returns, around the tiny text model: text, vision, tied head."""

    def __init__(self, text: torch.nn.Module) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = text
        self.model.visual = torch.nn.Linear(2, 2)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.model.language_model.embed_tokens


def _tiny_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A config whose glaze variant refines a tiny export built like the real one."""
    import glaze_tiny

    import qwen35_compression.glaze.data as data
    import qwen35_compression.models as models

    def tiny_settings(variants: dict) -> None:
        # Every glaze variant must still validate at the tiny block length.
        for item in variants.values():
            if item.get("method") == "glaze":
                item["glaze"].update(
                    tokens_per_step=4 * glaze_tiny.BLOCK,
                    micro_batch_tokens=2 * glaze_tiny.BLOCK,
                    epochs=1,
                    probe_steps=2,
                    eval_every_steps=16,
                    logit_chunk_tokens=64,
                )

    path = _config_with(tmp_path, tiny_settings)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["calibration"]["max_sequence_length"] = glaze_tiny.BLOCK
    raw["calibration"]["lock_path"] = None
    raw["paths"] = {"outputs": str(tmp_path / "outputs"), "results": str(tmp_path / "results")}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(path)

    teacher = glaze_tiny.tiny_teacher()
    init_variant = config.variant("autoround_w4a16_g128")
    init_dir = config.paths.outputs / init_variant.name
    glaze_tiny.write_tiny_export(teacher, init_dir)
    monkeypatch.setenv("QWEN35_CODE_REVISION", "test")
    write_export_manifest(init_dir, config, init_variant, "rev", 1.0, None)

    monkeypatch.setattr(
        models, "load_resolved_model", lambda *a, **k: (_FakeVisionLanguageModel(teacher), None)
    )
    monkeypatch.setattr(models, "resolve_revision", lambda config: "rev")
    monkeypatch.setattr(
        data, "calibration_blocks", lambda processor, calibration: glaze_tiny.random_blocks(170)
    )
    return config


def test_refine_trains_and_writes_a_verified_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen35_compression.glaze.train import refine

    config = _tiny_study(tmp_path, monkeypatch)
    variant = config.variant(GLAZE)
    output_dir, manifest = refine(config, variant, log=lambda _: None)
    assert output_dir == config.paths.outputs / GLAZE
    assert verify_export(output_dir, variant)["method"] == "glaze"
    record = manifest["glaze"]
    assert record["init"]["variant"] == "autoround_w4a16_g128"
    assert record["data"]["train_blocks"] == 128 and record["data"]["dev_blocks"] == 32
    assert record["data"]["train_start"] == 0 and record["data"]["dev_start"] == 128
    assert record["flip_fraction"] in variant.glaze.flip_fractions
    assert set(record["probe_dev_kl"]) == {repr(f) for f in variant.glaze.flip_fractions}
    changed = record["changed"]
    assert 0 < changed["scales_total"] and changed["scales"] <= changed["scales_total"]
    assert 0 < changed["norm_values_total"]
    assert record["dev_best"]["mean_kl"] <= record["dev_at_init"]["mean_kl"]
    assert len(record["history"]) == 32
    # The whole manifest is plain JSON, as written.
    stored = json.loads((output_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert stored["glaze"]["best_step"] == record["best_step"]
    with pytest.raises(FileExistsError):
        refine(config, variant, log=lambda _: None)


def test_refine_pilot_writes_its_own_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen35_compression.glaze.train import refine

    config = _tiny_study(tmp_path, monkeypatch)
    variant = config.variant(GLAZE)
    pilot_dir = tmp_path / "pilot"
    output_dir, manifest = refine(
        config, variant, output_dir=pilot_dir, pilot_steps=3, log=lambda _: None
    )
    assert output_dir == pilot_dir and verify_export(pilot_dir, variant)
    assert len(manifest["glaze"]["pilot"]["train_kl"]) == 3
    assert manifest["glaze"]["pilot"]["flip_fraction"] == min(variant.glaze.flip_fractions)
    assert "best_step" not in manifest["glaze"]
    assert not (config.paths.outputs / GLAZE).exists()


def test_prepare_and_diagnose_run_on_a_tiny_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen35_compression.glaze.diagnose import diagnose
    from qwen35_compression.glaze.student import trainable_state
    from qwen35_compression.glaze.train import prepare

    config = _tiny_study(tmp_path, monkeypatch)
    variant = config.variant(GLAZE)
    setup = prepare(config, variant, log=lambda _: None)
    assert setup.init_variant.name == "autoround_w4a16_g128" and setup.revision == "rev"
    assert len(setup.train) == 128 and len(setup.dev) == 32
    before = trainable_state(setup.student)
    report = diagnose(setup.trainer, setup.train, setup.schedule, log=lambda _: None)
    after = trainable_state(setup.student)
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert report["batch"]["blocks"] == setup.schedule.blocks_per_step
    smallest = min(variant.glaze.flip_fractions)
    assert set(report["pilots"]) == {
        f"guarded_{smallest:g}",
        f"guarded_{smallest / 10:g}",
        f"scales_only_{smallest:g}",
    }
    assert all(len(pilot["train_kl"]) == 5 for pilot in report["pilots"].values())
    # Nothing was written: a diagnosis leaves no export behind.
    assert not (config.paths.outputs / GLAZE).exists()
