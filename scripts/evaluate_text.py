#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.feature1 import (
    build_lm_eval_command,
    load_benchmark_suite,
    run_command,
)
from qwen35_compression.models import resolve_revision


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Feature 1 text benchmarks")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", help="Temporary per-task pilot limit")
    parser.add_argument(
        "--backend",
        choices=("hf-multimodal", "vllm"),
        help="Override the backend pinned in the benchmark suite",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        help="Interpreter containing the selected backend",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.evaluation.suite_path is None:
        raise ValueError("benchmark evaluation requires evaluation.suite_path")
    suite = load_benchmark_suite(config.evaluation.suite_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    revision = resolve_revision(config) if str(args.model_path) == config.model.id else None
    run_command(
        build_lm_eval_command(
            args.model_path,
            suite,
            args.output,
            revision,
            limit_override=args.limit,
            backend_override=args.backend,
            python_executable=args.python_executable,
        )
    )


if __name__ == "__main__":
    main()
