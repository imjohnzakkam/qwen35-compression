#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.preflight import run_preflight, write_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Feature 1 runtime before GPU work")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--profile", choices=("local", "gpu-text", "gpu-vision"), default="local")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_preflight(args.config, args.profile)
    if args.output:
        write_preflight(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
