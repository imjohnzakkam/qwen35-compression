#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.feature1 import (
    build_lm_eval_command,
    build_vlm_eval_command,
    load_benchmark_suite,
)
from qwen35_compression.io import write_json
from qwen35_compression.models import download_model, resolve_revision
from qwen35_compression.provenance import git_revision

ROOT = Path(__file__).resolve().parent.parent


def _run(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Feature 1 BF16 baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/feature1/bf16"))
    parser.add_argument("--limit", help="Pilot-only per-task limit")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit and not args.text_only:
        raise ValueError("--limit is text-only pilot mode; also pass --text-only")
    code_revision = args.code_revision or git_revision(ROOT)
    if not code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = code_revision

    config = load_config(args.config)
    if config.feature != "feature1":
        raise ValueError("run_feature1_bf16.py accepts only a feature1 config")
    if config.evaluation.suite_path is None:
        raise ValueError("Feature 1 requires a benchmark suite")
    suite = load_benchmark_suite(config.evaluation.suite_path)
    output = args.output.resolve()
    text_python = ROOT / ".venv-gpu-text" / "bin" / "python"
    vision_python = ROOT / ".venv-gpu-vision" / "bin" / "python"
    toolkit_dir = ROOT / "external" / "VLMEvalKit"
    revision = config.model.revision or "RESOLVED_AT_RUNTIME"

    text_command = build_lm_eval_command(
        Path(config.model.id),
        suite,
        output / "text",
        revision,
        limit_override=args.limit,
        backend_override="vllm",
        python_executable=text_python,
    )
    vision_command = build_vlm_eval_command(
        Path("PINNED_SNAPSHOT_AT_RUNTIME"), suite, output / "vision", toolkit_dir
    )
    vision_command[0] = str(vision_python)
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
                    "runtime_smoke": smoke_command,
                    "text": text_command,
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
    text_command = build_lm_eval_command(
        Path(config.model.id),
        suite,
        output / "text",
        revision,
        limit_override=args.limit,
        backend_override="vllm",
        python_executable=text_python,
    )
    vision_command = build_vlm_eval_command(snapshot, suite, output / "vision", toolkit_dir)
    vision_command[0] = str(vision_python)
    smoke_command[smoke_command.index("--model-path") + 1] = str(snapshot)

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "feature": "feature1",
        "variant": "bf16",
        "model_id": config.model.id,
        "model_revision": revision,
        "code_revision": code_revision,
        "config_digest": config.digest,
        "scope": "text_only" if args.text_only else "text_and_vision",
        "limit": args.limit,
        "research_result": args.limit is None,
        "status": "running",
        "durations_seconds": {},
    }
    write_json(output / "run_manifest.json", manifest)
    durations = manifest["durations_seconds"]
    assert isinstance(durations, dict)
    try:
        durations["runtime_smoke"] = _run(smoke_command, log_path)
        durations["text"] = _run(text_command, log_path)
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
