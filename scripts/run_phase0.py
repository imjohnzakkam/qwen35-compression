#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from qwen35_compression.config import load_config
from qwen35_compression.io import write_json


def _run(command: list[str], root: Path) -> float:
    started = time.perf_counter()
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument(
        "--variants",
        nargs="*",
        help="Optional subset; defaults to every variant in the config",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config.phase != "phase0":
        raise ValueError("run_phase0.py accepts only a phase0 config")
    root = config.source_path.parent.parent
    variants = args.variants or [variant.name for variant in config.variants]
    for name in variants:
        config.variant(name)

    durations = {}
    durations["download"] = _run(
        [sys.executable, "scripts/download_model.py", "--config", str(config.source_path)], root
    )
    for name in variants:
        variant = config.variant(name)
        if variant.method != "bf16":
            durations[f"quantize:{name}"] = _run(
                [
                    sys.executable,
                    "scripts/quantize.py",
                    "--config",
                    str(config.source_path),
                    "--variant",
                    name,
                ],
                root,
            )
            durations[f"verify:{name}"] = _run(
                [
                    sys.executable,
                    "scripts/verify_export.py",
                    "--config",
                    str(config.source_path),
                    "--variant",
                    name,
                ],
                root,
            )
        durations[f"evaluate:{name}"] = _run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--config",
                str(config.source_path),
                "--variant",
                name,
            ],
            root,
        )
    summary_path = config.paths.results / "phase0_summary.json"
    write_json(
        summary_path,
        {
            "phase": config.phase,
            "model_id": config.model.id,
            "config_digest": config.digest,
            "variants": variants,
            "durations_seconds": durations,
            "status": "passed",
        },
    )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
