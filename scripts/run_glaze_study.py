#!/usr/bin/env python3
"""Glaze stage B1 on one GPU: refine AutoRound's export against BF16, then score both.

Steps, stopping at the first failure:
1. Bootstrap the evaluation environment; download BF16, the init's published export and BF16's
   drift traces; install flash-linear-attention.
2. Pilot: a few Glaze steps on one fixed batch (the KL must fall), exported and drift-scored on
   the first answers of each task (the export must load in vLLM and score like a 4-bit model).
3. The full Glaze run.
4. Drift for the init and for Glaze, with per-answer records.
5. MATH-500 free generation for both (accuracy and answers that reach the token cap).
6. A report: paired drift intervals, MATH-500 deltas and the stage gate.

With --diagnose, only step 1 and the diagnosis run: measurements of the pilot batch's steps and
short pilots of candidate settings (scripts/glaze_diagnose.py), with no export.

    python scripts/run_glaze_study.py --variant glaze_v1_w4a16_g128 --dry-run
    python scripts/run_glaze_study.py --variant glaze_v1_w4a16_g128 --diagnose --dry-run
"""

from __future__ import annotations

import sys

# The GPU image's base Python ships brotlicffi, and httpx 0.28 fails mid-download decoding Brotli
# with it (see run_drift_study.py). Hidden before httpx loads, the Hub answers with gzip.
for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import _bootstrap  # noqa: E402, F401

from qwen35_compression.config import ExperimentConfig, VariantConfig, load_config  # noqa: E402
from qwen35_compression.drift import (  # noqa: E402
    TEXT_PYTHON,
    TRACES_REPO,
    fetch_published,
    fetch_traces,
    score_command,
)
from qwen35_compression.export import MANIFEST_NAME, verify_export  # noqa: E402
from qwen35_compression.feature1 import FLA_VERSION, fla_install_command, run_logged  # noqa: E402
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_INITS = {
    "autoround_w4a16_g128": "lazybrick/Qwen3.5-4B-Kiln-AutoRound-W4A16-g128",
    "gptq_w4a16_g128": "lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128",
}
MATH500_SUITE = Path("configs/evaluation/feature1_math500.yaml")
# BF16's mean NLL on its own MATH-500 and MMLU-Pro answers (drift study, 2,304,479 tokens).
BF16_MEAN_NLL = 0.12797209507699497
# The pilot export must score like a 4-bit model (AutoRound: 4.96% flips), not like a broken one.
PILOT_MAX_FLIP_RATE = 0.10
# Gate 1: Glaze's excess loss at least 5% below the init's with a paired interval excluding 0,
# and MATH-500 at most 2 points below. Under 2%, scales and norms lack the capacity.
GATE_REDUCTION = 0.05
GATE_FALLBACK_BELOW = 0.02
GATE_MATH500_POINTS = -2.0


def install_init(snapshot: Path, init_dir: Path, variant: VariantConfig) -> dict[str, Any]:
    """Copy a published export into the outputs directory, where every tool expects it."""
    if init_dir.exists() and any(init_dir.iterdir()):
        return verify_export(init_dir, variant)
    init_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((snapshot / MANIFEST_NAME).read_text(encoding="utf-8"))
    for name in [item["path"] for item in manifest["files"]] + [MANIFEST_NAME]:
        shutil.copyfile(snapshot / name, init_dir / name)
    return verify_export(init_dir, variant)


def drift_summary(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    overall = result["scores"]["both"]["all"]
    return {"flip_rate": overall["flip_rate"], "mean_nll": overall["mean_nll"]}


def evaluate_gate(
    nll_difference: dict[str, float],
    init_mean_nll: float,
    math500_delta: dict[str, float] | None,
) -> dict[str, Any]:
    """Stage B1's gate from Glaze minus init: paired mean NLL and the MATH-500 change."""
    excess = init_mean_nll - BF16_MEAN_NLL
    reduction = -nll_difference["difference"] / excess
    significant = nll_difference["high"] < 0
    math_ok = math500_delta is not None and math500_delta["delta"] >= GATE_MATH500_POINTS
    if reduction >= GATE_REDUCTION and significant and math_ok:
        decision = "pass"
    elif reduction < GATE_FALLBACK_BELOW:
        decision = "stop"
    else:
        decision = "ask"
    return {
        "init_excess_loss": excess,
        "glaze_excess_loss": excess + nll_difference["difference"],
        "excess_reduction": reduction,
        "nll_interval_excludes_zero": significant,
        "math500_delta": math500_delta,
        "decision": decision,
    }


def diagnosis_summary(path: Path) -> dict[str, Any]:
    """The headline numbers of a diagnosis report, for the run manifest."""
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "kl": report["kl"],
        "moves": {
            name: {key: result[key] for key in ("moves", "kl_down", "best_step_fraction")}
            for name, result in report["moves"].items()
        },
        "pilots": {name: result["passed"] for name, result in report["pilots"].items()},
    }


def math500_record(run: Path, tokenizer: Any) -> dict[str, Any]:
    """MATH-500 accuracy, answers at the token cap, and per-question correctness of one run."""
    import panel_scores

    rows = panel_scores._samples(run, "minerva_math500")
    correct = {row["doc_id"]: float(row["math_verify"]) for row in rows}
    capped = sum(panel_scores._at_cap(tokenizer, row) for row in rows)
    return {
        "accuracy": 100 * sum(correct.values()) / len(correct),
        "at_cap": capped,
        "questions": len(correct),
        "per_question": correct,
    }


def commands(
    args: argparse.Namespace, config: ExperimentConfig, variant: VariantConfig, output: Path
) -> dict[str, list[str]]:
    """Every subprocess the study runs, so the dry run shows exactly what will execute."""
    config_path = str(config.source_path)
    pilot_dir = config.paths.outputs / f"{variant.name}-pilot"
    glaze = [sys.executable, "scripts/glaze.py", "--config", config_path, "--variant"]
    steps = {
        "bootstrap": [sys.executable, "scripts/bootstrap_gpu.py", "--scope", "text"],
        "preflight": [str(TEXT_PYTHON), "scripts/preflight.py", "--profile", "gpu-text"],
        "fla": fla_install_command(sys.executable),
    }
    if args.diagnose:
        steps["diagnose"] = [
            sys.executable,
            "scripts/glaze_diagnose.py",
            "--config",
            config_path,
            "--variant",
            variant.name,
            "--output",
            str(output / "diagnosis.json"),
        ]
        return steps
    steps |= {
        "pilot": [
            *glaze,
            variant.name,
            "--pilot-steps",
            str(args.pilot_steps),
            "--output",
            str(pilot_dir),
        ],
        "train": [*glaze, variant.name],
    }
    for name in (str(variant.init), variant.name):
        steps[f"math500:{name}"] = [
            sys.executable,
            "scripts/run_feature1_variant.py",
            "--config",
            config_path,
            "--variant",
            name,
            "--suite",
            str(MATH500_SUITE),
            "--skip-bootstrap",
            "--code-revision",
            args.code_revision,
            "--output",
            str(output / "math500" / name),
        ]
    return steps


def study(args: argparse.Namespace, config: ExperimentConfig, variant: VariantConfig) -> dict:
    from qwen35_compression.models import download_model

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "run.log"
    steps = commands(args, config, variant, output)
    init_variant = config.variant(str(variant.init))
    init_dir = config.paths.outputs / init_variant.name
    export_dir = config.paths.outputs / variant.name
    pilot_dir = config.paths.outputs / f"{variant.name}-pilot"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "glaze",
        "variant": variant.name,
        "init": init_variant.name,
        "code_revision": args.code_revision,
        "config_digest": config.digest,
        "research_result": True,
        "status": "running",
        "steps": {},
    }

    def save() -> None:
        write_json(output / "run_manifest.json", manifest)

    def step(name: str, action: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        manifest["steps"][name] = {"status": "running"}
        save()
        try:
            value = action()
        except BaseException as error:
            manifest["steps"][name] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.perf_counter() - started),
            }
            manifest["status"] = "failed"
            save()
            raise
        manifest["steps"][name] = {
            "status": "passed",
            "seconds": round(time.perf_counter() - started),
        }
        save()
        return value

    def logged(name: str) -> Callable[[], float]:
        return lambda: run_logged(steps[name], log, ROOT)

    if not args.skip_bootstrap:
        step("bootstrap", lambda: subprocess.run(steps["bootstrap"], cwd=ROOT, check=True))
        step("preflight", lambda: subprocess.run(steps["preflight"], cwd=ROOT, check=True))
    snapshot, revision = step("download_bf16", lambda: download_model(config))
    manifest["base_model"] = {"id": config.model.id, "revision": revision}
    repo = PUBLISHED_INITS[init_variant.name]
    init_path, init_revision = step("download_init", lambda: fetch_published(repo))
    manifest["init_source"] = {"repo": repo, "revision": init_revision}
    step("install_init", lambda: install_init(init_path, init_dir, init_variant))
    step("fla", logged("fla"))
    manifest["flash_linear_attention"] = FLA_VERSION
    if args.diagnose:
        step("diagnose", logged("diagnose"))
        manifest["diagnosis"] = diagnosis_summary(output / "diagnosis.json")
        manifest["status"] = "passed"
        save()
        return manifest
    traces = output / "traces"
    manifest["traces"] = {
        "repo": TRACES_REPO,
        "revision": step("traces", lambda: fetch_traces(traces)),
    }

    # Pilot: a disposable export from a few steps; it must load in vLLM and score sanely.
    if pilot_dir.exists():
        shutil.rmtree(pilot_dir)
    step("pilot_train", logged("pilot"))
    pilot_score = score_command(
        pilot_dir, "pilot", traces, output / "pilot.json", args.pilot_limit, snapshot
    )
    step("pilot_score", lambda: run_logged(pilot_score, log, ROOT))
    manifest["pilot"] = drift_summary(output / "pilot.json")
    if manifest["pilot"]["flip_rate"] > PILOT_MAX_FLIP_RATE:
        manifest["status"] = "failed"
        save()
        raise SystemExit(f"pilot export scores like a broken model: {manifest['pilot']}")

    if export_dir.exists() and any(export_dir.iterdir()):
        step("train", lambda: verify_export(export_dir, variant))
    else:
        step("train", logged("train"))
    exported = verify_export(export_dir, variant)
    manifest["glaze"] = {
        key: exported["glaze"].get(key)
        for key in (
            "flip_fraction",
            "probe_dev_kl",
            "changed",
            "best_step",
            "dev_at_init",
            "dev_best",
        )
    }

    drift = {}
    for name, path in (("init", init_dir), ("glaze", export_dir)):
        command = score_command(path, name, traces, output / f"{name}.json", None, snapshot)
        step(f"drift:{name}", lambda command=command: run_logged(command, log, ROOT))
        drift[name] = drift_summary(output / f"{name}.json")
    manifest["drift"] = drift

    for name in (init_variant.name, variant.name):
        step(f"math500:{name}", logged(f"math500:{name}"))

    def report() -> None:
        import drift_scores
        import panel_scores
        from transformers import AutoTokenizer

        init_result = json.loads((output / "init.json").read_text(encoding="utf-8"))
        glaze_result = json.loads((output / "glaze.json").read_text(encoding="utf-8"))
        paired = drift_scores.paired_difference(init_result, glaze_result)
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        math = {
            name: math500_record(output / "math500" / name, tokenizer)
            for name in (init_variant.name, variant.name)
        }
        delta = panel_scores.paired_delta(
            math[variant.name]["per_question"], math[init_variant.name]["per_question"]
        )
        manifest["paired_drift"] = paired
        manifest["math500"] = {
            name: {k: v for k, v in record.items() if k != "per_question"}
            for name, record in math.items()
        }
        manifest["math500_delta"] = delta
        manifest["gate"] = evaluate_gate(paired["mean_nll"], drift["init"]["mean_nll"], delta)
        rows = drift_scores.report_rows([init_result, glaze_result], "both", "all")
        rows += [""] + drift_scores.paired_rows(init_result, [init_result, glaze_result])
        (output / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    step("report", report)
    manifest["status"] = "passed"
    save()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--variant", required=True, help="A glaze variant from the config")
    parser.add_argument("--output", type=Path, default=Path("results/feature1/glaze-b1"))
    parser.add_argument("--pilot-steps", type=int, default=5)
    parser.add_argument(
        "--pilot-limit", type=int, default=10, help="Answers per task the pilot export scores"
    )
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument(
        "--diagnose", action="store_true", help="Only diagnose the pilot batch; export nothing"
    )
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    if args.pilot_steps < 2 or args.pilot_limit < 1:
        parser.error("--pilot-steps must be at least 2 and --pilot-limit at least 1")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    config = load_config(args.config)
    variant = config.variant(args.variant)
    if variant.method != "glaze":
        parser.error(f"{variant.name} is not a glaze variant")
    if variant.init not in PUBLISHED_INITS:
        parser.error(f"no published export for the init {variant.init!r}")

    if args.dry_run:
        from qwen35_compression.glaze.train import plan

        print(
            json.dumps(
                {
                    "variant": variant.name,
                    "init": variant.init,
                    "init_repo": PUBLISHED_INITS[str(variant.init)],
                    "output": str(args.output.resolve()),
                    "glaze": plan(config, variant),
                    "commands": commands(args, config, variant, args.output.resolve()),
                },
                indent=2,
            )
        )
        return
    study(args, config, variant)


if __name__ == "__main__":
    main()
