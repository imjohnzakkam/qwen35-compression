#!/usr/bin/env python3
"""Quantize one Feature 1 variant and score it with the BF16 baseline's exact protocol."""

from __future__ import annotations

import sys

# The GPU image's base Python ships brotlicffi, which breaks httpx downloads from the Hub
# (run_drift_study.py); hidden before httpx loads.
for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402, F401

from qwen35_compression.config import ExperimentConfig, VariantConfig, load_config  # noqa: E402
from qwen35_compression.export import verify_export  # noqa: E402
from qwen35_compression.feature1 import (  # noqa: E402
    EVALUATOR_ENV,
    FLA_METHODS,
    FLA_VERSION,
    BenchmarkSuite,
    build_text_eval_commands,
    build_vlm_eval_command,
    fla_install_command,
    load_benchmark_suite,
    run_logged,
    suite_label,
    suite_record,
    vision_protocol,
)
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.models import resolve_revision  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize and evaluate one Feature 1 variant")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--suite",
        type=Path,
        help="Benchmark suite override, e.g. configs/evaluation/feature1_thinking.yaml",
    )
    parser.add_argument(
        "--output", type=Path, help="Defaults to <paths.results>/<variant>[-<suite>]"
    )
    parser.add_argument(
        "--limit", help="Pilot-only: first N questions of every text task and vision dataset"
    )
    parser.add_argument(
        "--pilot-limit",
        type=int,
        help="Run a --limit N pilot into <output>-pilot first; start the full run only if it "
        "passes. The export it builds is reused, and setup is paid once.",
    )
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Reuse evaluator environments already built on this machine",
    )
    parser.add_argument(
        "--from-hub",
        metavar="REPO",
        help="Score this published Kiln export instead of quantizing (copied into the outputs "
        "directory; its compression manifest must name the variant)",
    )
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit and args.pilot_limit:
        raise ValueError("--limit and --pilot-limit are mutually exclusive")
    if args.text_only and args.vision_only:
        raise ValueError("--text-only and --vision-only are mutually exclusive")
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
    suite = load_benchmark_suite(args.suite or config.evaluation.suite_path)
    label = suite_label(suite, config.evaluation.suite_path)
    if not suite.vision_tasks:
        if args.vision_only:
            raise ValueError(f"suite {label!r} has no vision tasks")
        args.text_only = True
    run_name = variant.name if label is None else f"{variant.name}-{label}"
    output = (args.output or config.paths.results / run_name).resolve()
    if args.pilot_limit and not args.dry_run:
        pilot = output.with_name(output.name + "-pilot")
        run(args, config, variant, suite, pilot, str(args.pilot_limit), code_revision)
        args.skip_bootstrap = True
    run(args, config, variant, suite, output, args.limit, code_revision)


def run(
    args: argparse.Namespace,
    config: ExperimentConfig,
    variant: VariantConfig,
    suite: BenchmarkSuite,
    output: Path,
    limit: str | None,
    code_revision: str,
) -> None:
    """One run of the variant: quantize unless its export exists, then smoke, text and vision."""
    export_dir = config.paths.outputs / variant.name
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
    text_commands = build_text_eval_commands(
        export_dir,
        suite,
        output / "text",
        None,
        limit_override=limit,
        python_executable=text_python,
    )
    vision_command = build_vlm_eval_command(export_dir, suite, output / "vision", toolkit_dir)
    vision_command[0] = str(vision_python)
    if limit:
        vision_command.extend(("--limit", str(limit)))
    if args.from_hub and not args.dry_run and not export_dir.exists():
        from qwen35_compression.drift import copy_published

        copy_published(args.from_hub, export_dir)
    export_exists = export_dir.is_dir() and any(export_dir.iterdir())

    if args.dry_run:
        print(
            json.dumps(
                {
                    "variant": variant.name,
                    "method": variant.method,
                    "pilot_limit": args.pilot_limit,
                    "from_hub": args.from_hub,
                    "export_dir": str(export_dir),
                    "bootstrap": None if args.skip_bootstrap else bootstrap,
                    "text_preflight": text_preflight,
                    "vision_preflight": None if args.text_only else vision_preflight,
                    "quantize": None if export_exists or args.from_hub else quantize_command,
                    "flash_linear_attention": (
                        fla_install_command(sys.executable)
                        if variant.method in FLA_METHODS and not (export_exists or args.from_hub)
                        else None
                    ),
                    "benchmark_suite": suite_record(suite),
                    "runtime_smoke": smoke_command,
                    "text": dict(text_commands),
                    "vision": None if args.text_only else vision_command,
                },
                indent=2,
            )
        )
        return

    if not args.skip_bootstrap:
        subprocess.run(bootstrap, cwd=ROOT, check=True)
    if not args.vision_only:
        subprocess.run(text_preflight, cwd=ROOT, check=True)
    if not args.text_only:
        subprocess.run(vision_preflight, cwd=ROOT, check=True)

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
        "variant": variant.name,
        "method": variant.method,
        "model_id": config.model.id,
        "model_revision": resolve_revision(config),
        "code_revision": code_revision,
        "config_digest": config.digest,
        "benchmark_suite": suite_record(suite),
        "export_dir": str(export_dir),
        "from_hub": args.from_hub,
        "scope": scope,
        "vision_protocol": None if args.text_only else vision_protocol(suite),
        "limit": limit,
        # score_vision.py scores the same first-N subset of a pilot.
        "vision_limit": None if limit is None or args.text_only else int(limit),
        "enable_thinking": suite.enable_thinking,
        "evaluator_env": EVALUATOR_ENV,
        "research_result": limit is None,
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
            if variant.method in FLA_METHODS:
                run_logged(fla_install_command(sys.executable), log_path, ROOT)
                manifest["flash_linear_attention"] = FLA_VERSION
            durations["quantize"] = run_logged(quantize_command, log_path, ROOT)
        export_manifest = verify_export(export_dir, variant)
        manifest["export"] = {
            "code_revision": export_manifest["code_revision"],
            "total_bytes": export_manifest["total_bytes"],
            "files": len(export_manifest["files"]),
        }
        durations["runtime_smoke"] = run_logged(smoke_command, log_path, ROOT)
        if not args.vision_only:
            for stage, command in text_commands:
                durations[stage] = run_logged(command, log_path, ROOT)
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
