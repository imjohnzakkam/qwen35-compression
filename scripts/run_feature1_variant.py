#!/usr/bin/env python3
"""Quantize one Feature 1 variant and score it with the BF16 baseline's exact protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.export import verify_export
from qwen35_compression.feature1 import (
    EVALUATOR_ENV,
    build_lm_eval_command,
    build_vlm_eval_command,
    load_benchmark_suite,
    run_logged,
)
from qwen35_compression.io import write_json
from qwen35_compression.models import resolve_revision
from qwen35_compression.provenance import git_revision

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize and evaluate one Feature 1 variant")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, help="Defaults to <paths.results>/<variant>")
    parser.add_argument("--limit", help="Pilot-only per-task limit")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Reuse evaluator environments already built on this machine",
    )
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
        raise ValueError("run_feature1_variant.py accepts only a feature1 config")
    if config.evaluation.suite_path is None:
        raise ValueError("Feature 1 requires a benchmark suite")
    variant = config.variant(args.variant)
    if variant.method == "bf16":
        raise ValueError("the BF16 baseline is produced by scripts/run_feature1_bf16.py")
    suite = load_benchmark_suite(config.evaluation.suite_path)
    export_dir = config.paths.outputs / variant.name
    output = (args.output or config.paths.results / variant.name).resolve()
    text_python = ROOT / ".venv-gpu-text" / "bin" / "python"
    vision_python = ROOT / ".venv-gpu-vision" / "bin" / "python"
    toolkit_dir = ROOT / "external" / "VLMEvalKit"

    bootstrap = [sys.executable, "scripts/bootstrap_gpu.py"]
    if args.text_only:
        bootstrap.extend(("--scope", "text"))
    text_preflight = [str(text_python), "scripts/preflight.py", "--profile", "gpu-text"]
    vision_preflight = [str(vision_python), "scripts/preflight.py", "--profile", "gpu-vision"]
    # Compression runs in this interpreter's environment, which carries llmcompressor.
    quantize_command = [
        sys.executable,
        "scripts/quantize.py",
        "--config",
        str(config.source_path),
        "--variant",
        variant.name,
    ]
    smoke_python = text_python if args.text_only else vision_python
    smoke_command = [
        str(smoke_python),
        "scripts/smoke_vllm.py",
        "--model-path",
        str(export_dir),
        "--output",
        str(output / "runtime_smoke.json"),
        "--max-model-len",
        str(suite.max_model_len),
    ]
    if not args.text_only:
        smoke_command.extend(("--image", "data/calibration/feature1_multimodal/images/0000.jpg"))
    text_command = build_lm_eval_command(
        export_dir,
        suite,
        output / "text",
        None,
        limit_override=args.limit,
        backend_override="vllm",
        python_executable=text_python,
    )
    vision_command = build_vlm_eval_command(export_dir, suite, output / "vision", toolkit_dir)
    vision_command[0] = str(vision_python)
    export_exists = export_dir.is_dir() and any(export_dir.iterdir())

    if args.dry_run:
        print(
            json.dumps(
                {
                    "variant": variant.name,
                    "method": variant.method,
                    "export_dir": str(export_dir),
                    "bootstrap": None if args.skip_bootstrap else bootstrap,
                    "text_preflight": text_preflight,
                    "vision_preflight": None if args.text_only else vision_preflight,
                    "quantize": None if export_exists else quantize_command,
                    "runtime_smoke": smoke_command,
                    "text": text_command,
                    "vision": None if args.text_only else vision_command,
                },
                indent=2,
            )
        )
        return

    if not args.skip_bootstrap:
        subprocess.run(bootstrap, cwd=ROOT, check=True)
    subprocess.run(text_preflight, cwd=ROOT, check=True)
    if not args.text_only:
        subprocess.run(vision_preflight, cwd=ROOT, check=True)

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "feature": "feature1",
        "variant": variant.name,
        "method": variant.method,
        "model_id": config.model.id,
        "model_revision": resolve_revision(config),
        "code_revision": code_revision,
        "config_digest": config.digest,
        "export_dir": str(export_dir),
        "scope": "text_only" if args.text_only else "text_and_vision",
        "limit": args.limit,
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
        if export_exists:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"= reusing existing export {export_dir}\n")
        else:
            durations["quantize"] = run_logged(quantize_command, log_path, ROOT)
        export_manifest = verify_export(export_dir, variant)
        manifest["export"] = {
            "code_revision": export_manifest["code_revision"],
            "total_bytes": export_manifest["total_bytes"],
            "files": len(export_manifest["files"]),
        }
        durations["runtime_smoke"] = run_logged(smoke_command, log_path, ROOT)
        durations["text"] = run_logged(text_command, log_path, ROOT)
        if not args.text_only:
            durations["vision"] = run_logged(vision_command, log_path, ROOT)
        manifest["status"] = "passed"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        write_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
