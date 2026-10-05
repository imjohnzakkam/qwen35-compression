#!/usr/bin/env python3
"""Two controls for Glaze v2 on Qwen3.5-4B, on one GPU: is its gain the data or the allocator?

1. Data: AutoRound, one 4-bit g128 scheme, tuned on Glaze v2's calibration blocks instead of
   UltraChat (`autoround_glaze2_data_w4a16_g128`).
2. Allocation: AutoRound's own mixed precision. AutoScheme picks each Linear's option from Glaze's
   menu at Glaze's language-model budget, and AutoRound tunes it on the same blocks
   (`autoround_autoscheme_w4a16`).

In order, after setup (evaluator environment, flash-linear-attention, the BF16 snapshot, AutoRound's
and Glaze v2's published exports, Glaze's blocks rebuilt from its saved answers and checked against
its run): a pilot of both (16 blocks, 10 steps; the allocation on 2 short samples; vLLM loads the
mixed export), then each control in full: quantize, MATH-500 and IFEval after a two-question pilot.
Last, validation KL to BF16 for AutoRound, both controls and Glaze v2.

The two controls are independent: one failing skips only its own remaining steps.

    python scripts/run_glaze_controls.py --dry-run
"""

from __future__ import annotations

import sys

# The GPU image's base Python ships brotlicffi, which breaks httpx downloads.
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
SUITE = Path("configs/evaluation/feature1_controls.yaml")
PUBLISHED = {
    "autoround_w4a16_g128": "lazybrick/Qwen3.5-4B-Kiln-AutoRound-W4A16-g128",
    "glaze2_w4a16_g128": "lazybrick/Qwen3.5-4B-Kiln-Glaze-W4A16",
}
GLAZE = "glaze2_w4a16_g128"
# Block digests logged by Glaze v2's 4B run (code d3a2e2b, 2026-10-05); rebuilt blocks must match.
BLOCKS = {
    "full": {
        "dir": "glaze2",
        "sizes": (512, 64),
        "calibration": "0bbceb412b9a7853c3af8b935e126fd28586a84dbd2cef4365b5691d8933a810",
        "held_out": "85b8dd5caf4b1b5bcf9c20d94a0d310b35d2b1977625902c46fae5fcaf38bfa2",
    },
    "pilot": {
        "dir": "glaze2_pilot",
        "sizes": (16, 8),
        "calibration": "55dd6e1eab1191823071c629247dba8e84e8ebcdaaa6a9f7311d7b6d0eba3445",
        "held_out": "996c40a1f6b9b81df830b26eb3ce58bb80709eb8a5ce499b4bfd1e695dd56b40",
    },
}
# control -> (pilot variant, full variant, uses an AutoScheme allocation)
CONTROLS = {
    "data": ("autoround_glaze2_data_pilot_w4a16_g128", "autoround_glaze2_data_w4a16_g128", False),
    "allocation": ("autoround_autoscheme_pilot_w4a16", "autoround_autoscheme_w4a16", True),
}
# AutoScheme's scoring data: the pilot only checks the plumbing. 16 blocks is what Glaze's own
# allocation needed (its 16-block pilot chose the same allocation as the full 512).
ALLOCATION_SAMPLES = {"pilot": (2, 512), "full": (16, 2048)}
SUITE_PILOT_LIMIT = 2


def plan(config: ExperimentConfig, output: Path, code_revision: str) -> dict[str, list[str]]:
    """Every command after setup, by step name, in run order."""
    config_path = str(config.source_path)
    data = config.paths.outputs / "glaze2" / "data"
    answers = data / "answers.jsonl"
    steps: dict[str, list[str]] = {}
    for stage, spec in BLOCKS.items():
        calibration, held_out = spec["sizes"]
        steps[f"blocks:{stage}"] = [
            sys.executable,
            "scripts/glaze2_blocks.py",
            "--config",
            config_path,
            "--answers",
            str(answers),
            "--output-dir",
            str(config.paths.outputs / spec["dir"] / "data"),
            "--calibration-blocks",
            str(calibration),
            "--held-out-blocks",
            str(held_out),
        ]
    for stage, index in (("pilot", 0), ("full", 1)):
        for control, variants in CONTROLS.items():
            variant, allocated = variants[index], variants[2]
            if allocated:
                nsamples, seqlen = ALLOCATION_SAMPLES[stage]
                steps[f"{stage}:{control}:allocate"] = [
                    sys.executable,
                    "scripts/autoscheme_allocate.py",
                    "--config",
                    config_path,
                    "--variant",
                    variant,
                    "--reference",
                    str(config.paths.outputs / GLAZE),
                    "--nsamples",
                    str(nsamples),
                    "--seqlen",
                    str(seqlen),
                ]
            steps[f"{stage}:{control}:quantize"] = [
                sys.executable,
                "scripts/quantize.py",
                "--config",
                config_path,
                "--variant",
                variant,
            ]
            if stage == "pilot" and allocated:
                # The uniform pilot has AutoRound's published layout; the mixed one must load.
                steps[f"{stage}:{control}:vllm_check"] = [
                    str(TEXT_PYTHON),
                    "scripts/glaze2_vllm_check.py",
                    "--model",
                    str(config.paths.outputs / variant),
                    "--output",
                    str(output / stage / control / "vllm_check.json"),
                ]
            if stage == "full":
                steps[f"{stage}:{control}:suite"] = [
                    sys.executable,
                    "scripts/run_feature1_variant.py",
                    "--config",
                    config_path,
                    "--variant",
                    variant,
                    "--suite",
                    str(SUITE),
                    "--output",
                    str(output / "suite" / control),
                    "--pilot-limit",
                    str(SUITE_PILOT_LIMIT),
                    "--text-only",
                    "--skip-bootstrap",
                    "--code-revision",
                    code_revision,
                ]
    steps["evaluate"] = [
        sys.executable,
        "scripts/glaze2_evaluate.py",
        "--config",
        config_path,
        "--variants",
        ",".join(["autoround_w4a16_g128", *(v[1] for v in CONTROLS.values()), GLAZE]),
        "--output",
        str(output / "evaluation.json"),
    ]
    return steps


def control_of(step: str) -> str | None:
    parts = step.split(":")
    return parts[1] if len(parts) == 3 and parts[1] in CONTROLS else None


def check_blocks(config: ExperimentConfig, stage: str) -> dict[str, Any]:
    spec = BLOCKS[stage]
    record = json.loads(
        (config.paths.outputs / spec["dir"] / "data" / "blocks.json").read_text(encoding="utf-8")
    )
    for split in ("calibration", "held_out"):
        if record[split]["sha256"] != spec[split]:
            raise ValueError(f"{stage} {split} blocks differ from Glaze v2's run")
    return record


def study(args: argparse.Namespace) -> dict[str, Any]:
    from qwen35_compression.drift import copy_published
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "run.log"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "glaze_controls",
        "code_revision": args.code_revision,
        "config_digest": config.digest,
        "research_result": True,
        "status": "running",
        "steps": {},
        "skipped": [],
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

    try:
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
        step("fla", logged(fla_install_command(sys.executable)))
        manifest["flash_linear_attention"] = FLA_VERSION
        _, revision = step("download_bf16", lambda: download_model(config))
        manifest["base_model"] = {"id": config.model.id, "revision": revision}
        manifest["published"] = {}
        for name, repo in PUBLISHED.items():
            target = config.paths.outputs / name
            if not target.exists():
                manifest["published"][name] = {
                    "repo": repo,
                    "revision": step(
                        f"fetch:{name}", lambda r=repo, t=target: copy_published(r, t)
                    ),
                }
        steps = plan(config, output, args.code_revision)
        for stage in BLOCKS:
            step(f"blocks:{stage}", logged(steps[f"blocks:{stage}"]))
            manifest[f"blocks_{stage}"] = check_blocks(config, stage)
            save()
    except BaseException:
        manifest["status"] = "failed"
        save()
        raise

    failed_controls: set[str] = set()
    for name, command in steps.items():
        if name.startswith("blocks:"):
            continue
        control = control_of(name)
        if control in failed_controls:
            manifest["skipped"].append(name)
            save()
            continue
        if name == "evaluate" and failed_controls:
            # Score only the exports that exist.
            dropped = {CONTROLS[c][1] for c in failed_controls}
            index = command.index("--variants") + 1
            command = list(command)
            command[index] = ",".join(v for v in command[index].split(",") if v not in dropped)
        try:
            step(name, logged(command))
        except subprocess.CalledProcessError:
            if control is None:
                break
            failed_controls.add(control)
            continue
        if name.endswith(":allocate"):
            # Kept with the results, which are fetched even when a later step fails.
            variant = command[command.index("--variant") + 1]
            allocation = config.variant(variant).allocation
            if allocation is not None and allocation.exists():
                stage = name.split(":")[0]
                (output / f"allocation_{stage}.json").write_text(
                    allocation.read_text(encoding="utf-8"), encoding="utf-8"
                )
    failed = [name for name, entry in manifest["steps"].items() if entry["status"] == "failed"]
    manifest["status"] = "failed" if failed or manifest["skipped"] else "passed"
    save()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/feature1/glaze_controls"))
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    config = load_config(args.config)
    answers = config.paths.outputs / "glaze2" / "data" / "answers.jsonl"
    if not args.dry_run and not answers.exists():
        parser.error(f"missing {answers}: Glaze v2's saved answers (its run's answers.jsonl)")
    if args.dry_run:
        print(json.dumps(plan(config, args.output.resolve(), args.code_revision), indent=2))
        return
    study(args)


if __name__ == "__main__":
    main()
