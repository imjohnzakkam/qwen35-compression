#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.feature1 import (
    build_vlm_eval_command,
    load_benchmark_suite,
    run_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Feature 1 vision benchmarks")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--toolkit-dir", type=Path, default=Path("external/VLMEvalKit")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.evaluation.suite_path is None:
        raise ValueError("benchmark evaluation requires evaluation.suite_path")
    suite = load_benchmark_suite(config.evaluation.suite_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        build_vlm_eval_command(
            args.model_path, suite, args.output_dir, args.toolkit_dir
        )
    )


if __name__ == "__main__":
    main()
