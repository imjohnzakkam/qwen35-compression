#!/usr/bin/env python3
from __future__ import annotations

import argparse

from qwen35_compression.config import load_config
from qwen35_compression.export import verify_export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    variant = config.variant(args.variant)
    manifest = verify_export(config.paths.outputs / variant.name, variant)
    print(f"verified={variant.name}")
    print(f"files={len(manifest['files'])}")
    print(f"bytes={manifest['total_bytes']}")


if __name__ == "__main__":
    main()
