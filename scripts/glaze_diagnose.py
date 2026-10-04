#!/usr/bin/env python3
"""Measure why a Glaze variant's steps raise the KL on its pilot batch; pilot candidate settings.

Loads the teacher and the student as training does, changes nothing on disk except the report.

python scripts/glaze_diagnose.py --variant glaze_v1_w4a16_g128 --output DIR/diagnosis.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must be set before torch initializes CUDA (see scripts/quantize.py).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import _bootstrap  # noqa: E402, F401

from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.glaze.diagnose import diagnose  # noqa: E402
from qwen35_compression.glaze.train import prepare  # noqa: E402
from qwen35_compression.io import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True, help="The JSON report to write")
    args = parser.parse_args()
    config = load_config(args.config)
    variant = config.variant(args.variant)
    if variant.method != "glaze":
        parser.error(f"{variant.name} is not a glaze variant")
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    setup = prepare(config, variant)
    report = diagnose(setup.trainer, setup.train, setup.schedule)
    report.update(
        {
            "variant": variant.name,
            "init": setup.init_variant.name,
            "base_revision": setup.revision,
            "code_revision": os.environ.get("QWEN35_CODE_REVISION"),
        }
    )
    write_json(args.output, report)
    print(f"diagnosis={args.output}")


if __name__ == "__main__":
    main()
