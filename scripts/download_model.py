#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.models import download_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    path, revision = download_model(config)
    print(f"model={config.model.id}")
    print(f"revision={revision}")
    print(f"cache_path={path}")


if __name__ == "__main__":
    main()
