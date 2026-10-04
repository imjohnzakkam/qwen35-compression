#!/usr/bin/env python3
"""Glaze v2 phase 1 on one GPU: the Qwen3.5-0.8B proxy study, pilot first.

BF16 answers all calibration prompts once (vLLM: its first start on a machine spends ~10 minutes
compiling kernels, so it starts once). Then, for the pilot config and the full config in turn,
stopping at the first failure:
1. The answers packed into calibration and held-out blocks (the pilot's are tiny).
2. Four exports at AutoRound's size: A AutoRound as configured, B AutoRound on the new blocks,
   C Glaze v2 rounding at uniform 4-bit g128, D Glaze v2 in full.
3. Held-out KL to BF16 for all four, and phase 1's gate (D at least 15% below A in-domain).
4. D loads and answers in stock vLLM.

    python scripts/run_glaze2_proxy.py --dry-run
"""

from __future__ import annotations

import sys

# The GPU image's base Python ships brotlicffi, which breaks httpx downloads (run_drift_study.py).
for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
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
VARIANTS = (
    "autoround_w4a16_g128",
    "autoround_glaze2_data_w4a16_g128",
    "glaze2_uniform_w4a16_g128",
    "glaze2_w4a16_g128",
)
FULL_VARIANT = "glaze2_w4a16_g128"
# The pilot's sizes: enough to run every stage, nothing more.
PILOT_SIZES = {"calibration_blocks": 16, "held_out_blocks": 8}
FULL_SIZES = {"calibration_blocks": 512, "held_out_blocks": 64}


def answers_command(config: ExperimentConfig, snapshot: Path | str) -> list[str]:
    """BF16's answers to every prompt, written once for both stages."""
    return [
        str(TEXT_PYTHON),
        "scripts/glaze2_answers.py",
        "--model",
        str(snapshot),
        "--prompts",
        str(PROMPTS),
        "--output",
        str(config.paths.outputs / "data" / "answers.jsonl"),
    ]


def stage_commands(
    config: ExperimentConfig, sizes: dict[str, Any], answers: Path, results: Path
) -> dict[str, list[str]]:
    """Every subprocess of one stage (pilot or full), in order, from the shared answers."""
    config_path = str(config.source_path)
    data = config.paths.outputs / "data"
    steps = {
        "blocks": [
            sys.executable,
            "scripts/glaze2_blocks.py",
            "--config",
            config_path,
            "--answers",
            str(answers),
            "--output-dir",
            str(data),
            "--calibration-blocks",
            str(sizes["calibration_blocks"]),
            "--held-out-blocks",
            str(sizes["held_out_blocks"]),
        ],
    }
    for name in VARIANTS:
        steps[f"quantize:{name}"] = [
            sys.executable,
            "scripts/quantize.py",
            "--config",
            config_path,
            "--variant",
            name,
        ]
    steps["evaluate"] = [
        sys.executable,
        "scripts/glaze2_evaluate.py",
        "--config",
        config_path,
        "--output",
        str(results / "evaluation.json"),
    ]
    steps["vllm_check"] = [
        str(TEXT_PYTHON),
        "scripts/glaze2_vllm_check.py",
        "--model",
        str(config.paths.outputs / FULL_VARIANT),
        "--output",
        str(results / "vllm_check.json"),
    ]
    return steps


def study(args: argparse.Namespace) -> dict[str, Any]:
    from qwen35_compression.models import download_model

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "run.log"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "glaze2_proxy",
        "code_revision": args.code_revision,
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

    if not args.skip_bootstrap:
        step(
            "bootstrap",
            lambda: subprocess.run(
                [sys.executable, "scripts/bootstrap_gpu.py", "--scope", "text"],
                cwd=ROOT,
                check=True,
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
    step("fla", lambda: run_logged(fla_install_command(sys.executable), log, ROOT))
    manifest["flash_linear_attention"] = FLA_VERSION
    full_config = load_config(args.config)
    snapshot, revision = step("download_bf16", lambda: download_model(full_config))
    manifest["base_model"] = {"id": full_config.model.id, "revision": revision}
    answers = full_config.paths.outputs / "data" / "answers.jsonl"
    if not answers.exists():
        step("answers", lambda: run_logged(answers_command(full_config, snapshot), log, ROOT))
    for stage, config_path, sizes in (
        ("pilot", args.pilot_config, PILOT_SIZES),
        ("full", args.config, FULL_SIZES),
    ):
        config = load_config(config_path)
        results = output / stage
        for name, command in stage_commands(config, sizes, answers, results).items():
            step(f"{stage}:{name}", lambda c=command: run_logged(c, log, ROOT))
        evaluation = json.loads((results / "evaluation.json").read_text(encoding="utf-8"))
        manifest[stage] = {"gate": evaluation["gate"], "bytes": evaluation["bytes"]}
        save()
    manifest["status"] = "passed"
    manifest["decision"] = manifest["full"]["gate"]["decision"]
    save()
    print("decision=" + manifest["decision"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/glaze2_proxy.yaml"))
    parser.add_argument(
        "--pilot-config", type=Path, default=Path("configs/glaze2_proxy_pilot.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("results/glaze2_proxy"))
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    if not PROMPTS.exists() and not (ROOT / PROMPTS).exists():
        parser.error(f"missing {PROMPTS}: run scripts/glaze2_prompts.py first")
    if args.dry_run:
        full_config = load_config(args.config)
        plan: dict[str, Any] = {
            "answers": answers_command(full_config, "<bf16 snapshot>"),
        }
        shared = full_config.paths.outputs / "data" / "answers.jsonl"
        for stage, config_path, sizes in (
            ("pilot", args.pilot_config, PILOT_SIZES),
            ("full", args.config, FULL_SIZES),
        ):
            config = load_config(config_path)
            plan[stage] = stage_commands(config, sizes, shared, args.output.resolve() / stage)
        print(json.dumps(plan, indent=2))
        return
    study(args)


if __name__ == "__main__":
    main()
