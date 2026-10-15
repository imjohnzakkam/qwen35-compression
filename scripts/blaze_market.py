#!/usr/bin/env python3
"""Blaze's byte market for gate G2: the body's options at a standard W4A16 export's size.

Every unit starts at the upper option; the cheapest units by measured price step down to the
lower option until the export (laddered table, body, BF16 vision tower and norms, as the real
format stores them) is no larger than a standard W4A16 g128 export of the same checkpoint.

    python scripts/blaze_market.py --prices prices.json --output plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.blaze.body import TEXT_PREFIX
from qwen35_compression.blaze.census import load_census
from qwen35_compression.blaze.export import size_plan
from qwen35_compression.blaze.ladder import plan_ladder
from qwen35_compression.blaze.market import cheapest_cover, steps
from qwen35_compression.config import load_config
from qwen35_compression.glaze2.quant import Option
from qwen35_compression.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--census", type=Path, default=Path("data/blaze/census.json"))
    parser.add_argument("--upper", type=int, default=6)
    parser.add_argument("--lower", type=int, default=4)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument(
        "--allow-over",
        action="store_true",
        help="A pilot's few priced units cannot reach the size: step them all down and go on",
    )
    args = parser.parse_args()

    from qwen35_compression.models import download_model

    snapshot, _ = download_model(load_config(args.config))
    upper, lower = Option(args.upper, args.group), Option(args.lower, args.group)
    units = json.loads(args.prices.read_text(encoding="utf-8"))["units"]
    ladder = plan_ladder(load_census(args.census))
    target = size_plan(Path(snapshot), Option(4, 128))["total_bytes"]
    everything_upper = size_plan(Path(snapshot), upper, ladder)["total_bytes"]
    per_weight = (upper.bits - lower.bits) / 8
    market = steps(
        {u: e["prices"][upper.name] for u, e in units.items()},
        {u: e["prices"][lower.name] for u, e in units.items()},
        {u: int(e["weights"] * per_weight) for u, e in units.items()},
    )
    try:
        stepped = cheapest_cover(market, everything_upper - target)
    except ValueError:
        if not args.allow_over:
            raise
        stepped = sorted(units)
    options = {
        f"{TEXT_PREFIX}{member}.weight": (lower if unit in stepped else upper)
        for unit, entry in units.items()
        for member in entry["members"]
    }
    final = size_plan(Path(snapshot), upper, ladder, options)
    if final["total_bytes"] > target and not args.allow_over:
        raise SystemExit(f"plan is {final['total_bytes'] - target:,} bytes over the target")
    write_json(
        args.output,
        {
            "schema_version": 1,
            "upper": upper.name,
            "lower": lower.name,
            "target_bytes": target,
            "everything_upper_bytes": everything_upper,
            "plan_bytes": final["total_bytes"],
            "parts": final["parts"],
            "lower_units": stepped,
            "units": len(units),
            "predicted_cost": sum(s.cost for s in market if s.unit in stepped),
            "linear_options": {key: option.name for key, option in sorted(options.items())},
        },
    )
    print(
        f"market: {len(stepped)} of {len(units)} units at {lower.name}; "
        f"{final['total_bytes']:,} bytes against {target:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
