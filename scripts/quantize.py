#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.runner import quantize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    variant = config.variant(args.variant)
    output_dir, manifest = quantize(config, variant)
    print(f"output={output_dir}")
    print(f"bytes={manifest['total_bytes']}")


if __name__ == "__main__":
    main()
