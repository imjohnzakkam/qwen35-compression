from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


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
panel = _load("panel_scores")


def _sample(doc_id: int, prompt: str, answer: str) -> str:
    row = {
        "doc_id": doc_id,
        "arguments": {"gen_args_0": {"arg_0": prompt, "arg_1": {}}},
        "resps": [[answer]],
    }
    return json.dumps(row) + "\n"


def test_task_names_from_both_sample_layouts() -> None:
    assert (
        drift.task_of(Path("samples_mmlu_pro_law_2026-10-02T20-24-40.960879.jsonl"))
        == "mmlu_pro_law"
    )
    assert (
        drift.task_of(Path("samples_minerva_math500_2026-10-02T20-24-40.960879.jsonl"))
        == "minerva_math500"
    )
    assert drift.task_of(Path("mmlu_pro_law.jsonl")) == "mmlu_pro_law"


def test_traces_keep_the_mmlu_pro_subset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "minerva_math500.jsonl").write_text(
        "".join(_sample(i, f"p{i}", f"a{i}") for i in range(3)), encoding="utf-8"
    )
    (tmp_path / "samples_mmlu_pro_law_2026-10-02T20-24-40.1.jsonl").write_text(
        "".join(_sample(i, f"q{i}", f"b{i}") for i in range(5)), encoding="utf-8"
    )
    (tmp_path / "ifeval.jsonl").write_text(_sample(0, "x", "y"), encoding="utf-8")
    traces = drift.load_traces(tmp_path, {"mmlu_pro_law": [1, 3]})
    assert [(t.task, t.doc_id) for t in traces] == [
        ("minerva_math500", 0),
        ("minerva_math500", 1),
        ("minerva_math500", 2),
        ("mmlu_pro_law", 1),
        ("mmlu_pro_law", 3),
    ]
    assert traces[3].prompt == "q1" and traces[3].answer == "b1"
    assert len(drift.load_traces(tmp_path, {"mmlu_pro_law": [1, 3]}, limit=1)) == 2


def test_tally_buckets_flips_and_nll_by_position() -> None:
    tally = drift.Tally()
    ranks = [1] * 256 + [2] * 256 + [1] * 10
    logprobs = [-0.5] * len(ranks)
    tally.add(ranks, logprobs)
    summary = tally.summary()
    first, second, third = summary["buckets"][:3]
    assert (first["tokens"], first["flip_rate"]) == (256, 0.0)
    assert (second["tokens"], second["flip_rate"]) == (256, 1.0)
    assert (third["tokens"], third["flip_rate"]) == (10, 0.0)
    assert summary["mean_nll"] == 0.5 and summary["answers"] == 1


def test_cohorts_separate_long_finished_and_capped_answers() -> None:
    assert drift.cohorts(100) == ["all", "finished"]
    assert drift.cohorts(3000) == ["all", "finished", "long"]
    assert drift.cohorts(8190) == ["all", "capped"]


def test_report_rows_print_one_line_per_model() -> None:
    tally = drift.Tally()
    tally.add([1, 2], [-0.1, -0.2])
    result = {
        "name": "gptq",
        "buckets": list(drift.BUCKETS),
        "scores": {"both": {"long": tally.summary()}},
    }
    rows = drift.report_rows([result], "both", "long")
    assert rows[-1].startswith("| gptq | 50.00 |")


def test_paired_delta_is_exact_for_constant_differences() -> None:
    reference = {i: 1.0 for i in range(100)}
    candidate = {i: 1.0 if i % 10 else 0.0 for i in range(100)}
    result = panel.paired_delta(candidate, reference, resamples=200)
    assert result["delta"] == -10.0 and result["questions"] == 100
    assert result["low"] <= -10.0 <= result["high"]


def test_drift_study_rejects_unknown_models() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_drift_study.py", "--components", "mlp", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "unknown models" in result.stderr


def test_drift_study_quantizes_configured_variants() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_drift_study.py",
            "--components",
            "",
            "--published",
            "",
            "--variants",
            "autoround_w4a16_g128",
            "--skip-bf16",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["full"]["quantized"] == ["autoround_w4a16_g128"]
    rejected = subprocess.run(
        [sys.executable, "scripts/run_drift_study.py", "--variants", "nope", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0 and "unknown models" in rejected.stderr
