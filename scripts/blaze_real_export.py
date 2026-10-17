#!/usr/bin/env python3
"""Write a real Blaze export: compressed-tensors body and the tiered vocabulary table.

vLLM serves it with the `blaze-vllm` plugin (plugins/blaze_vllm). Its stored values are the
fake-quantized export's (scripts/blaze_export.py with the same flags).

    python scripts/blaze_real_export.py --model SNAPSHOT --body-bits 8 --output DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.blaze.census import load_census
from qwen35_compression.blaze.ladder import DEFAULT_AT_LEAST, DEFAULT_COVER, plan_ladder
from qwen35_compression.blaze.real import write_real_export
from qwen35_compression.glaze2.quant import Option


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="BF16 snapshot directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--census", type=Path, default=Path("data/blaze/census.json"))
    parser.add_argument("--body-bits", type=int, default=8)
    parser.add_argument("--body-group", type=int, default=128)
    parser.add_argument("--device", default=None, help="Default: cuda when available")
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ladder = plan_ladder(load_census(args.census), DEFAULT_COVER, DEFAULT_AT_LEAST)
    body = Option(args.body_bits, args.body_group)
    record = write_real_export(args.model, args.output, ladder, body, device)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
