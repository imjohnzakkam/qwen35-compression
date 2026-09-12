#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.feature1 import prepare_calibration_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Feature 1 calibration data")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    lock = prepare_calibration_dataset(config.calibration)
    print(f"Wrote {config.calibration.path}")
    print(f"Pinned dataset revision {lock['revision']} in {config.calibration.lock_path}")


if __name__ == "__main__":
    main()
