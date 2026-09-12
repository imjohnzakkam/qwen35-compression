#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config)
    variants = args.variants or [variant.name for variant in config.variants]
    for name in variants:
        config.variant(name)
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--config",
                str(config.source_path),
                "--variant",
                name,
            ],
            cwd=config.source_path.parent.parent,
            check=True,
        )


if __name__ == "__main__":
    main()
