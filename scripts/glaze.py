#!/usr/bin/env python3
"""Train and export one Glaze variant: its init's export, refined end to end against BF16.

python scripts/glaze.py --variant glaze_v1_w4a16_g128 --dry-run
python scripts/glaze.py --variant glaze_v1_w4a16_g128 --pilot-steps 5 --output DIR
python scripts/glaze.py --variant glaze_v1_w4a16_g128
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Must be set before torch initializes CUDA (see scripts/quantize.py).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import _bootstrap  # noqa: E402, F401

from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.glaze.train import plan, refine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, help="Defaults to <paths.outputs>/<variant>")
    parser.add_argument(
        "--pilot-steps",
        type=int,
        help="Only N steps on one fixed batch (the KL must fall), then export to --output",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pilot_steps is not None:
        if args.pilot_steps < 2:
            parser.error("--pilot-steps must be at least 2")
        if args.output is None:
            parser.error("a pilot needs --output, so it never takes the variant's export directory")
    config = load_config(args.config)
    variant = config.variant(args.variant)
    if variant.method != "glaze":
        parser.error(f"{variant.name} is not a glaze variant")
    output = args.output.resolve() if args.output else None

    if args.dry_run:
        print(json.dumps(plan(config, variant, output, args.pilot_steps), indent=2))
        return
    output_dir, manifest = refine(config, variant, output_dir=output, pilot_steps=args.pilot_steps)
    record = manifest["glaze"]
    summary = {
        key: record[key]
        for key in ("scale_learning_rate", "best_step", "dev_at_init", "dev_best", "pilot")
        if key in record
    }
    print(f"output={output_dir}")
    print(f"bytes={manifest['total_bytes']}")
    print("glaze=" + json.dumps(summary))


if __name__ == "__main__":
    main()
