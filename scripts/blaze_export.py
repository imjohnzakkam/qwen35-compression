#!/usr/bin/env python3
"""Write a fake-quantized Blaze export: the vocabulary ladder and/or one precision for the body.

The export is the BF16 checkpoint with the chosen tensors replaced by the values a real Blaze
export stores, so stock vLLM scores it like any BF16 model.

    python scripts/blaze_export.py --model SNAPSHOT --ladder --body-bits 6 --output DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.blaze.census import load_census
from qwen35_compression.blaze.export import write_fake_export
from qwen35_compression.blaze.ladder import DEFAULT_AT_LEAST, DEFAULT_COVER, plan_ladder
from qwen35_compression.glaze2.quant import Option


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="BF16 snapshot directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--census", type=Path, default=Path("data/blaze/census.json"))
    parser.add_argument("--ladder", action="store_true", help="Ladder the vocabulary table")
    parser.add_argument("--cover", type=float, default=DEFAULT_COVER)
    parser.add_argument("--at-least", type=int, default=DEFAULT_AT_LEAST)
    parser.add_argument("--body-bits", type=int, default=0, help="0 keeps the body in BF16")
    parser.add_argument("--body-group", type=int, default=128)
    parser.add_argument("--device", default=None, help="Default: cuda when available")
    args = parser.parse_args()
    if not args.ladder and not args.body_bits:
        parser.error("nothing to do: pass --ladder and/or --body-bits")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ladder = (
        plan_ladder(load_census(args.census), args.cover, args.at_least) if args.ladder else None
    )
    body = Option(args.body_bits, args.body_group) if args.body_bits else None
    record = write_fake_export(args.model, args.output, ladder, body, device)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
