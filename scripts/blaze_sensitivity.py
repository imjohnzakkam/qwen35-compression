#!/usr/bin/env python3
"""Blaze prices for gate G1(c): each FFN layer alone at 4-bit round-to-nearest, as KL to BF16.

The replay text is WikiText-2's training split: generic, and disjoint from every evaluation set.
Writes each layer's price and how concentrated the prices are (the share held by the most
expensive half of the FFN weights).

    python scripts/blaze_sensitivity.py --output results/blaze/g1/ffn_prices.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.blaze.body import ffn_layer_units
from qwen35_compression.blaze.sensitivity import concentration, measure_units, pack_text
from qwen35_compression.config import load_config
from qwen35_compression.glaze2.quant import Option
from qwen35_compression.io import write_json

REPLAY_REPO = "EleutherAI/wikitext_document_level"
REPLAY_REVISION = "647234772b9554e208af6c826f23b99e3cac88c8"
REPLAY_FILE = "wikitext-2-raw-v1/wikitext-2-raw-v1-train.parquet"


def replay_documents() -> list[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPLAY_REPO, REPLAY_FILE, repo_type="dataset", revision=REPLAY_REVISION)
    return [row["page"] for row in pq.read_table(path, columns=["page"]).to_pylist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--layers", default="", help="Comma-separated layer indices; default all")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group", type=int, default=128)
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
    units = ffn_layer_units(teacher)
    if args.layers:
        keep = {f"layers.{int(index)}.mlp" for index in args.layers.split(",")}
        units = {name: members for name, members in units.items() if name in keep}
    option = Option(args.bits, args.group)
    prices = measure_units(teacher, head, blocks, units, option, batch=args.batch)
    share = concentration(
        {name: price["mean_kl"] for name, price in prices.items()},
        {name: price["weights"] for name, price in prices.items()},
    )
    record = {
        "schema_version": 1,
        "model_revision": revision,
        "option": option.name,
        "replay": {"repo": REPLAY_REPO, "revision": REPLAY_REVISION, "file": REPLAY_FILE},
        "blocks": args.blocks,
        "length": args.length,
        "units": prices,
        "concentration": share,
        "seconds": round(time.perf_counter() - started),
        "peak_memory_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
    }
    write_json(args.output, record)
    print(f"concentration of FFN prices: {share:.3f} over {len(prices)} layers", flush=True)


if __name__ == "__main__":
    main()
