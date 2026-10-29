#!/usr/bin/env python3
"""Blaze's byte market: the body's options at a standard W4A16 export's size.

Two modes, both bounded by a standard W4A16 g128 export of the same checkpoint, counted as the
real formats store them:

- `cover` (G2): every unit starts at the upper option; the units whose measured step-down price
  is lowest move to the lower option until the export fits.
- `fill` (G2b): every unit starts at the lower option and the vision tower may be stored smaller
  (`--vision-bits`); the units with the most predicted gain move up while the bytes allow. A
  unit's price at any width comes from its price measured at `--rank-bits`, scaled by the
  measured bits rule (excess loss falls `--bits-factor` per two added bits), because prices
  measured directly at 6 bits and above sit near the measurement's noise.

    python scripts/blaze_market.py --mode fill --prices prices.json --lower 6 --upper 8 \\
        --vision-bits 8 --output plan.json
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
from qwen35_compression.blaze.market import best_fill, cheapest_cover, steps
from qwen35_compression.config import load_config
from qwen35_compression.glaze2.quant import Option
from qwen35_compression.io import write_json

# G1: 4-bit to 6-bit round-to-nearest bodies differed 19.5x in excess loss.
MEASURED_FACTOR_PER_TWO_BITS = 19.5


def predicted(price: float, measured_bits: int, bits: int, factor: float) -> float:
    return price / factor ** ((bits - measured_bits) / 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--census", type=Path, default=Path("data/blaze/census.json"))
    parser.add_argument("--mode", choices=("cover", "fill"), default="cover")
    parser.add_argument("--upper", type=int, default=6)
    parser.add_argument("--lower", type=int, default=4)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--vision-bits", type=int, default=0, help="0 keeps the vision tower BF16")
    parser.add_argument("--rank-bits", type=int, default=4, help="fill: the measured price width")
    parser.add_argument("--bits-factor", type=float, default=MEASURED_FACTOR_PER_TWO_BITS)
    parser.add_argument(
        "--allow-over",
        action="store_true",
        help="cover: a pilot's few priced units cannot reach the size; step them all down",
    )
    args = parser.parse_args()

    from qwen35_compression.models import download_model

    snapshot, _ = download_model(load_config(args.config))
    source = Path(snapshot)
    upper, lower = Option(args.upper, args.group), Option(args.lower, args.group)
    vision = Option(args.vision_bits, 128) if args.vision_bits else None
    units = json.loads(args.prices.read_text(encoding="utf-8"))["units"]
    ladder = plan_ladder(load_census(args.census))
    target = size_plan(source, Option(4, 128))["total_bytes"]
    per_weight = (upper.bits - lower.bits) / 8
    extra = {u: int(e["weights"] * per_weight) for u, e in units.items()}

    if args.mode == "cover":
        start = size_plan(source, upper, ladder, vision=vision)["total_bytes"]
        market = steps(
            {u: e["prices"][upper.name] for u, e in units.items()},
            {u: e["prices"][lower.name] for u, e in units.items()},
            extra,
        )
        try:
            moved = cheapest_cover(market, start - target)
        except ValueError:
            if not args.allow_over:
                raise
            moved = sorted(units)
        options = {
            f"{TEXT_PREFIX}{member}.weight": (lower if unit in moved else upper)
            for unit, entry in units.items()
            for member in entry["members"]
        }
        predicted_cost = sum(s.cost for s in market if s.unit in moved)
    else:
        start = size_plan(source, lower, ladder, vision=vision)["total_bytes"]
        rank = Option(args.rank_bits, args.group).name
        gains = {
            u: predicted(e["prices"][rank], args.rank_bits, lower.bits, args.bits_factor)
            - predicted(e["prices"][rank], args.rank_bits, upper.bits, args.bits_factor)
            for u, e in units.items()
        }
        moved = best_fill(gains, extra, target - start)
        options = {
            f"{TEXT_PREFIX}{member}.weight": (upper if unit in moved else lower)
            for unit, entry in units.items()
            for member in entry["members"]
        }
        predicted_cost = sum(
            predicted(
                e["prices"][rank],
                args.rank_bits,
                (upper if u in moved else lower).bits,
                args.bits_factor,
            )
            for u, e in units.items()
        )
    final = size_plan(source, lower if args.mode == "fill" else upper, ladder, options, vision)
    if final["total_bytes"] > target and not args.allow_over:
        raise SystemExit(f"plan is {final['total_bytes'] - target:,} bytes over the target")
    write_json(
        args.output,
        {
            "schema_version": 2,
            "mode": args.mode,
            "upper": upper.name,
            "lower": lower.name,
            "vision": vision.name if vision else None,
            "target_bytes": target,
            "start_bytes": start,
            "plan_bytes": final["total_bytes"],
            "parts": final["parts"],
            "moved_units": moved,
            "units": len(units),
            "predicted_cost": predicted_cost,
            "linear_options": {key: option.name for key, option in sorted(options.items())},
        },
    )
    direction = "down to " + lower.name if args.mode == "cover" else "up to " + upper.name
    print(
        f"market: {len(moved)} of {len(units)} units {direction}; "
        f"{final['total_bytes']:,} bytes against {target:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
