#!/usr/bin/env python3
"""Glaze v2 phase 2 on one GPU: Qwen3.5-4B, against AutoRound's published export.

The proxy study (phase 1) chose the method; here only Glaze v2 in full is built. In order,
stopping at the first failure:
1. Setup: both evaluator environments, flash-linear-attention, the BF16 snapshot and AutoRound's
   published export (the byte target and the KL baseline).
2. BF16's answers to the calibration prompts (vLLM, once).
3. Pilot: every stage below on 16 calibration and 8 held-out blocks, 10 iterations per layer.
4. Glaze v2 on 512 calibration blocks; held-out KL to BF16 against AutoRound; vLLM loads it.
5. Gate (phase 1's): at least 15% lower in-domain KL than AutoRound, chat no more than 5% worse,
   no larger. On "stop" the run ends here.
6. Drift scores on BF16's 1,501 MATH-500 and MMLU-Pro answers (as AutoRound's were scored).
7. The full benchmark suite with the BF16 protocol, after its own two-question pilot.

    python scripts/run_glaze2_4b.py --dry-run
"""

from __future__ import annotations

import sys

# The GPU image's base Python ships brotlicffi, which breaks httpx downloads (run_drift_study.py).
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

from qwen35_compression.config import ExperimentConfig, load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON  # noqa: E402
from qwen35_compression.feature1 import FLA_VERSION, fla_install_command, run_logged  # noqa: E402
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = Path("data/glaze2/prompts.jsonl")
BASELINE = "autoround_w4a16_g128"
BASELINE_REPO = "lazybrick/Qwen3.5-4B-Kiln-AutoRound-W4A16-g128"
FULL = "glaze2_w4a16_g128"
PILOT = "glaze2_pilot_w4a16_g128"
STAGES = {
    "pilot": (PILOT, {"calibration_blocks": 16, "held_out_blocks": 8}),
    "full": (FULL, {"calibration_blocks": 512, "held_out_blocks": 64}),
}
SUITE_PILOT_LIMIT = 2


def stage_commands(
    config: ExperimentConfig,
    variant: str,
    sizes: dict[str, int],
    answers: Path,
    results: Path,
) -> dict[str, list[str]]:
    """Every subprocess of one stage (pilot or full), in order."""
    config_path = str(config.source_path)
    settings = config.variant(variant).glaze2
    assert settings is not None
    return {
        "blocks": [
            sys.executable,
            "scripts/glaze2_blocks.py",
            "--config",
            config_path,
            "--answers",
            str(answers),
            "--output-dir",
            str(settings.calibration_blocks.parent),
            "--calibration-blocks",
            str(sizes["calibration_blocks"]),
            "--held-out-blocks",
            str(sizes["held_out_blocks"]),
        ],
        "quantize": [
            sys.executable,
            "scripts/quantize.py",
            "--config",
            config_path,
            "--variant",
            variant,
        ],
        "evaluate": [
            sys.executable,
            "scripts/glaze2_evaluate.py",
            "--config",
            config_path,
            "--variants",
            f"{BASELINE},{variant}",
            "--output",
            str(results / "evaluation.json"),
        ],
        "vllm_check": [
            str(TEXT_PYTHON),
            "scripts/glaze2_vllm_check.py",
            "--model",
            str(config.paths.outputs / variant),
            "--output",
            str(results / "vllm_check.json"),
        ],
    }


def plan(config: ExperimentConfig, output: Path, snapshot: Path | str, code_revision: str):
    """The commands after setup, by step name."""
    answers = config.paths.outputs / "glaze2" / "data" / "answers.jsonl"
    steps: dict[str, list[str]] = {
        "answers": [
            str(TEXT_PYTHON),
            "scripts/glaze2_answers.py",
            "--model",
            str(snapshot),
            "--prompts",
            str(PROMPTS),
            "--output",
            str(answers),
        ]
    }
    for stage, (variant, sizes) in STAGES.items():
        commands = stage_commands(config, variant, sizes, answers, output / stage)
        steps.update({f"{stage}:{name}": command for name, command in commands.items()})
    steps["drift"] = [
        sys.executable,
        "scripts/run_drift_study.py",
        "--config",
        str(config.source_path),
        "--output",
        str(output / "drift"),
        "--components",
        "",
        "--published",
        "",
        "--variants",
        FULL,
        "--skip-bf16",
        "--skip-fla",
        "--skip-bootstrap",
        "--code-revision",
        code_revision,
    ]
    steps["suite"] = [
        sys.executable,
        "scripts/run_feature1_variant.py",
        "--config",
        str(config.source_path),
        "--variant",
        FULL,
        "--output",
        str(output / "suite"),
        "--pilot-limit",
        str(SUITE_PILOT_LIMIT),
        "--skip-bootstrap",
        "--code-revision",
        code_revision,
    ]
    return answers, steps


def study(args: argparse.Namespace) -> dict[str, Any]:
    from qwen35_compression.drift import copy_published
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "run.log"
    # AutoRound needed this on the 4B; Glaze's layer copies come and go the same way.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "glaze2_4b",
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

    def logged(command: list[str]) -> Callable[[], float]:
        return lambda: run_logged(command, log, ROOT)

    if not args.skip_bootstrap:
        step(
            "bootstrap",
            lambda: subprocess.run(
                [sys.executable, "scripts/bootstrap_gpu.py"], cwd=ROOT, check=True
            ),
        )
        step(
            "preflight",
            lambda: subprocess.run(
                [str(TEXT_PYTHON), "scripts/preflight.py", "--profile", "gpu-text"],
                cwd=ROOT,
                check=True,
            ),
        )
    step("fla", logged(fla_install_command(sys.executable)))
    manifest["flash_linear_attention"] = FLA_VERSION
    snapshot, revision = step("download_bf16", lambda: download_model(config))
    manifest["base_model"] = {"id": config.model.id, "revision": revision}
    baseline = config.paths.outputs / BASELINE
    if not baseline.exists():
        baseline_revision = step("fetch_baseline", lambda: copy_published(BASELINE_REPO, baseline))
        manifest["baseline"] = {"repo": BASELINE_REPO, "revision": baseline_revision}

    answers, steps = plan(config, output, snapshot, args.code_revision)
    if not answers.exists():
        step("answers", logged(steps["answers"]))
    # Kept with the results, which are fetched even when a later step fails.
    shutil.copyfile(answers, output / "answers.jsonl")

    for stage in STAGES:
        for name, command in steps.items():
            if name.startswith(stage + ":"):
                step(name, logged(command))
        evaluation = json.loads((output / stage / "evaluation.json").read_text(encoding="utf-8"))
        manifest[stage] = {"gate": evaluation["gate"], "bytes": evaluation["bytes"]}
        save()

    manifest["gate"] = manifest["full"]["gate"]
    manifest["decision"] = manifest["gate"]["decision"]
    save()
    print("decision=" + manifest["decision"], flush=True)
    if manifest["decision"] == "go":
        try:
            step("drift", logged(steps["drift"]))
        except subprocess.CalledProcessError:
            # The suite does not depend on it; the failure stays in the manifest.
            manifest["status"] = "running"
            save()
        step("suite", logged(steps["suite"]))
    failed = [name for name, entry in manifest["steps"].items() if entry["status"] == "failed"]
    manifest["status"] = "failed" if failed else "passed"
    save()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/feature1/glaze2_4b"))
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    if not (ROOT / PROMPTS).exists():
        parser.error(f"missing {PROMPTS}: run scripts/glaze2_prompts.py first")
    if args.dry_run:
        config = load_config(args.config)
        _, steps = plan(config, args.output.resolve(), "<bf16 snapshot>", args.code_revision)
        print(json.dumps(steps, indent=2))
        return
    study(args)


if __name__ == "__main__":
    main()
