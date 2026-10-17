#!/usr/bin/env python3
"""Blaze prices for gate G2: each layer's token mixer and FFN alone, at each option, as KL to BF16.

Round-to-nearest, on WikiText-2's training text (the same pinned file as G1's prices). The
market (scripts/blaze_market.py) reads the difference between a unit's two prices as the cost
of storing it at the lower option.

    python scripts/blaze_prices.py --bits 4,6 --output results/blaze/g2/prices.json
"""

from __future__ import annotations

import sys

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402, F401
from blaze_sensitivity import (  # noqa: E402
    REPLAY_FILE,
    REPLAY_REPO,
    REPLAY_REVISION,
    replay_documents,
)

from qwen35_compression.blaze.body import layer_part_units  # noqa: E402
from qwen35_compression.blaze.sensitivity import measure_units, pack_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.glaze2.quant import Option  # noqa: E402
from qwen35_compression.io import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", default="4,6", help="Comma-separated bit widths")
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--layers", default="", help="Comma-separated layer indices (a pilot)")
    args = parser.parse_args()

    import torch

    from qwen35_compression.glaze2.pipeline import load_teacher

    started = time.perf_counter()
    teacher, model, processor, _, revision = load_teacher(load_config(args.config))
    head = model.get_output_embeddings().weight
    tokenizer = getattr(processor, "tokenizer", processor)
    documents = [text for text in replay_documents() if text.strip()]
    rows = tokenizer(documents, add_special_tokens=False)["input_ids"]
    blocks = pack_text(rows, args.length, args.blocks)
    units = layer_part_units(teacher)
    if args.layers:
        keep = {int(index) for index in args.layers.split(",")}
        units = {name: m for name, m in units.items() if int(name.split(".")[1]) in keep}
    record_units: dict[str, dict] = {}
    for bits in (int(b) for b in args.bits.split(",")):
        option = Option(bits, args.group)
        prices = measure_units(teacher, head, blocks, units, option, batch=args.batch)
        for name, price in prices.items():
            entry = record_units.setdefault(
                name, {"members": price["members"], "weights": price["weights"], "prices": {}}
            )
            entry["prices"][option.name] = price["mean_kl"]
    write_json(
        args.output,
        {
            "schema_version": 1,
            "model_revision": revision,
            "replay": {"repo": REPLAY_REPO, "revision": REPLAY_REVISION, "file": REPLAY_FILE},
            "blocks": args.blocks,
            "length": args.length,
            "units": record_units,
            "seconds": round(time.perf_counter() - started),
            "peak_memory_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
        },
    )
    print(f"priced {len(record_units)} units at {args.bits} bits", flush=True)


if __name__ == "__main__":
    main()
