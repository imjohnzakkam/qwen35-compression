#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.feature1 import (
    EVALUATOR_ENV,
    build_text_eval_commands,
    build_vlm_eval_command,
    load_benchmark_suite,
    run_logged,
    suite_label,
    suite_record,
    vision_protocol,
)
from qwen35_compression.io import write_json
from qwen35_compression.models import download_model, resolve_revision
from qwen35_compression.provenance import git_revision

ROOT = Path(__file__).resolve().parent.parent


def _run(command: list[str], log_path: Path) -> float:
    return run_logged(command, log_path, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Feature 1 BF16 baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument(
        "--suite",
        type=Path,
        help="Benchmark suite override, e.g. configs/evaluation/feature1_thinking.yaml",
    )
    parser.add_argument("--output", type=Path, help="Defaults to results/feature1/bf16[-<suite>]")
    parser.add_argument(
        "--limit", help="Pilot-only: first N questions of every text task and vision dataset"
    )
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.text_only and args.vision_only:
        raise ValueError("--text-only and --vision-only are mutually exclusive")
    code_revision = args.code_revision or git_revision(ROOT)
    if not code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = code_revision

    config = load_config(args.config)
    if config.feature != "feature1":
        raise ValueError("run_feature1_bf16.py accepts only a feature1 config")
    if config.evaluation.suite_path is None:
        raise ValueError("Feature 1 requires a benchmark suite")
    suite = load_benchmark_suite(args.suite or config.evaluation.suite_path)
    label = suite_label(suite, config.evaluation.suite_path)
    if not suite.vision_tasks:
        if args.vision_only:
            raise ValueError(f"suite {label!r} has no vision tasks")
        args.text_only = True
    default_output = Path("results/feature1") / ("bf16" if label is None else f"bf16-{label}")
    output = (args.output or default_output).resolve()
    text_python = ROOT / ".venv-gpu-text" / "bin" / "python"
    vision_python = ROOT / ".venv-gpu-vision" / "bin" / "python"
    toolkit_dir = ROOT / "external" / "VLMEvalKit"
    revision = config.model.revision or "RESOLVED_AT_RUNTIME"

    text_commands = build_text_eval_commands(
        Path(config.model.id),
        suite,
        output / "text",
        revision,
        limit_override=args.limit,
        python_executable=text_python,
    )
    vision_command = build_vlm_eval_command(
        Path("PINNED_SNAPSHOT_AT_RUNTIME"), suite, output / "vision", toolkit_dir
    )
    vision_command[0] = str(vision_python)
    if args.limit:
        vision_command.extend(("--limit", str(args.limit)))
    smoke_python = text_python if args.text_only else vision_python
    smoke_command = [
        str(smoke_python),
        "scripts/smoke_vllm.py",
        "--model-path",
        "PINNED_SNAPSHOT_AT_RUNTIME",
        "--output",
        str(output / "runtime_smoke.json"),
        "--max-model-len",
        str(suite.max_model_len),
    ]
    if not args.text_only:
        smoke_command.extend(
            ("--image", "data/calibration/feature1_multimodal/images/0000.jpg")
        )
    if args.dry_run:
        bootstrap = [sys.executable, "scripts/bootstrap_gpu.py"]
        if args.text_only:
            bootstrap.extend(("--scope", "text"))
        print(
            json.dumps(
                {
                    "bootstrap": bootstrap,
                    "text_preflight": [
                        str(text_python),
                        "scripts/preflight.py",
                        "--profile",
                        "gpu-text",
                    ],
                    "vision_preflight": [
                        str(vision_python),
                        "scripts/preflight.py",
                        "--profile",
                        "gpu-vision",
                    ],
                    "benchmark_suite": suite_record(suite),
                    "runtime_smoke": smoke_command,
                    "text": dict(text_commands),
                    "vision": None if args.text_only else vision_command,
                },
                indent=2,
            )
        )
        return

    bootstrap = [sys.executable, "scripts/bootstrap_gpu.py"]
    if args.text_only:
        bootstrap.extend(("--scope", "text"))
    subprocess.run(bootstrap, cwd=ROOT, check=True)
    if not args.vision_only:
        subprocess.run(
            [str(text_python), "scripts/preflight.py", "--profile", "gpu-text"],
            cwd=ROOT,
            check=True,
        )
    if not args.text_only:
        subprocess.run(
            [str(vision_python), "scripts/preflight.py", "--profile", "gpu-vision"],
            cwd=ROOT,
            check=True,
        )

    revision = resolve_revision(config)
    snapshot, downloaded_revision = download_model(config)
    if downloaded_revision != revision:
        raise RuntimeError("downloaded model revision changed during the run")
    text_commands = build_text_eval_commands(
        Path(config.model.id),
        suite,
        output / "text",
        revision,
        limit_override=args.limit,
        python_executable=text_python,
    )
    vision_command = build_vlm_eval_command(snapshot, suite, output / "vision", toolkit_dir)
    vision_command[0] = str(vision_python)
    if args.limit:
        vision_command.extend(("--limit", str(args.limit)))
    smoke_command[smoke_command.index("--model-path") + 1] = str(snapshot)

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    if args.text_only:
        scope = "text_only"
    elif args.vision_only:
        scope = "vision_only"
    else:
        scope = "text_and_vision"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "feature": "feature1",
        "variant": "bf16",
        "model_id": config.model.id,
        "model_revision": revision,
        "code_revision": code_revision,
        "config_digest": config.digest,
        "benchmark_suite": suite_record(suite),
        "scope": scope,
        "vision_protocol": None if args.text_only else vision_protocol(suite),
        "limit": args.limit,
        # score_vision.py scores the same first-N subset of a pilot.
        "vision_limit": None if args.limit is None or args.text_only else int(args.limit),
        "enable_thinking": suite.enable_thinking,
        "evaluator_env": EVALUATOR_ENV,
        "research_result": args.limit is None,
        "status": "running",
        "durations_seconds": {},
    }
    write_json(output / "run_manifest.json", manifest)
    durations = manifest["durations_seconds"]
    assert isinstance(durations, dict)
    try:
        durations["runtime_smoke"] = _run(smoke_command, log_path)
        if not args.vision_only:
            for stage, command in text_commands:
                durations[stage] = _run(command, log_path)
        if not args.text_only:
            durations["vision"] = _run(vision_command, log_path)
        manifest["status"] = "passed"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        write_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
