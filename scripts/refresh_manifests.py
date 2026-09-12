#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.export import MANIFEST_NAME, write_export_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    for variant in config.variants:
        if variant.method == "bf16":
            continue
        output_dir = config.paths.outputs / variant.name
        manifest_path = output_dir / MANIFEST_NAME
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = write_export_manifest(
            output_dir=output_dir,
            config=config,
            variant=variant,
            model_revision=existing["model_revision"],
            elapsed_seconds=existing["elapsed_seconds"],
            peak_memory_bytes=existing["peak_memory_bytes"],
        )
        print(f"refreshed={variant.name} code_revision={manifest['code_revision']}")


if __name__ == "__main__":
    main()
