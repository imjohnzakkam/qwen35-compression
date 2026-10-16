#!/usr/bin/env python3
"""Ladder an existing 4-bit export's vocabulary table (and optionally widen some body units).

    python scripts/blaze_addon.py --source EXPORT --output DIR
    python scripts/blaze_addon.py --source EXPORT --bf16 SNAPSHOT --plan plan.json --output DIR

With --plan (scripts/blaze_market.py --mode fill), every Linear the plan stores above its lower
option is re-quantized from the BF16 checkpoint at that option.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.blaze.addon import ladder_export
from qwen35_compression.blaze.census import load_census
from qwen35_compression.blaze.ladder import DEFAULT_AT_LEAST, DEFAULT_COVER, plan_ladder
from qwen35_compression.glaze2.quant import Option


def upgrades(plan_path: Path) -> dict[str, Option]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    lower = plan["lower"]
    named = {f"w{b}g128": Option(b, 128) for b in (4, 5, 6, 8)}
    return {
        key.removesuffix(".weight"): named[name]
        for key, name in plan["linear_options"].items()
        if name != lower
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="compressed-tensors export")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--census", type=Path, default=Path("data/blaze/census.json"))
    parser.add_argument("--bf16", type=Path, help="BF16 snapshot, for --plan's upgrades")
    parser.add_argument("--plan", type=Path, help="Units to widen, from blaze_market.py fill")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.plan and not args.bf16:
        parser.error("--plan needs --bf16")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ladder = plan_ladder(load_census(args.census), DEFAULT_COVER, DEFAULT_AT_LEAST)
    widen = upgrades(args.plan) if args.plan else None
    record = ladder_export(args.source, args.output, ladder, args.bf16, widen, device)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
