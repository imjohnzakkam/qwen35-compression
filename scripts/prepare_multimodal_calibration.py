#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.multimodal import prepare_multimodal_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Feature 1 image-text calibration data")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    if config.multimodal_calibration is None:
        raise ValueError("config does not define multimodal_calibration")
    lock = prepare_multimodal_calibration(config.multimodal_calibration)
    print(f"Wrote {config.multimodal_calibration.path}")
    print(f"Downloaded {len(lock['images'])} pinned images")
    print(f"Pinned dataset revision {lock['revision']}")


if __name__ == "__main__":
    main()
