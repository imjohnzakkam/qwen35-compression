#!/usr/bin/env python3
"""Write a fake-quantized Blaze export whose body is GPTQ-rounded on generic text.

GPTQ runs on WikiText-2's training text (the same pinned file as the prices), layer by layer on
the quantized chain, at the body option given; the vocabulary is laddered as in blaze_export.py.

    python scripts/blaze_gptq.py --ladder --bits 6 --output DIR
"""

from __future__ import annotations

import sys

# The GPU image's base Python ships brotlicffi, which breaks httpx downloads (run_drift_study.py).
for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402, F401
from blaze_sensitivity import (  # noqa: E402
    REPLAY_FILE,
    REPLAY_REPO,
    REPLAY_REVISION,
    replay_documents,
)

from qwen35_compression.blaze.body import TEXT_PREFIX  # noqa: E402
from qwen35_compression.blaze.census import load_census  # noqa: E402
from qwen35_compression.blaze.export import MANIFEST, write_fake_export  # noqa: E402
from qwen35_compression.blaze.gptq import DAMP, gptq_body  # noqa: E402
from qwen35_compression.blaze.ladder import (  # noqa: E402
    DEFAULT_AT_LEAST,
    DEFAULT_COVER,
    plan_ladder,
)
from qwen35_compression.blaze.sensitivity import pack_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.glaze2.quant import Option  # noqa: E402


def _linear_names(text_model) -> list[str]:
    from torch import nn

    return [
        name
        for name, module in text_model.named_modules()
        if isinstance(module, nn.Linear) and name.startswith("layers.")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--census", type=Path, default=Path("data/blaze/census.json"))
    parser.add_argument("--ladder", action="store_true", help="Ladder the vocabulary table")
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--damp", type=float, default=DAMP)
    parser.add_argument("--layers", default="", help="Comma-separated layer indices (a pilot)")
    parser.add_argument("--plan", type=Path, help="Per-Linear options from scripts/blaze_market.py")
    parser.add_argument(
        "--vision-bits", type=int, default=0, help="Round the vision tower's Linears (0: BF16)"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Write the real format (blaze.real), not a fake-quantized export; needs --ladder",
    )
    args = parser.parse_args()
    if args.real and not args.ladder:
        parser.error("--real stores the laddered table; pass --ladder")

    import torch

    from qwen35_compression.glaze2.pipeline import load_teacher

    started = time.perf_counter()
    teacher, model, processor, snapshot, revision = load_teacher(load_config(args.config))
    tokenizer = getattr(processor, "tokenizer", processor)
    documents = [text for text in replay_documents() if text.strip()]
    rows = tokenizer(documents, add_special_tokens=False)["input_ids"]
    blocks = pack_text(rows, args.length, args.blocks)
    option = Option(args.bits, args.group)
    # Checkpoint key -> option; GPTQ takes module paths relative to the text model.
    options: dict[str, Option] = {}
    if args.plan:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        named = {o.name: o for o in (Option(b, args.group) for b in (2, 3, 4, 5, 6, 8))}
        options = {key: named[name] for key, name in plan["linear_options"].items()}
    by_module = {
        key.removeprefix(TEXT_PREFIX).removesuffix(".weight"): value
        for key, value in options.items()
    }
    layers = [int(index) for index in args.layers.split(",")] if args.layers else None
    per_linear = (
        {name: by_module.get(name, option) for name in _linear_names(teacher)}
        if by_module
        else option
    )
    kept: dict = {} if args.real else None
    stored = gptq_body(teacher, blocks, per_linear, args.batch, layers, args.damp, codes=kept)
    replaced = {f"{TEXT_PREFIX}{name}.weight": weight for name, weight in stored.items()}
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    gptq_seconds = round(time.perf_counter() - started)
    del teacher, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ladder = (
        plan_ladder(load_census(args.census), DEFAULT_COVER, DEFAULT_AT_LEAST)
        if args.ladder
        else None
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = Option(args.vision_bits, 128) if args.vision_bits else None
    if args.real:
        from qwen35_compression.blaze.real import write_real_export

        record = write_real_export(
            snapshot,
            args.output,
            ladder,
            options or option,
            device,
            quantized={f"{TEXT_PREFIX}{name}.weight": item for name, item in kept.items()},
            vision=vision,
        )
        record["real"] = True
    else:
        record = write_fake_export(
            snapshot,
            args.output,
            ladder,
            option,
            device,
            replaced=replaced,
            options=options or None,
            vision=vision,
        )
    record["gptq"] = {
        "model_revision": revision,
        "replay": {"repo": REPLAY_REPO, "revision": REPLAY_REVISION, "file": REPLAY_FILE},
        "blocks": args.blocks,
        "length": args.length,
        "damp": args.damp,
        "layers": layers,
        "plan": str(args.plan) if args.plan else None,
        "linears": len(stored),
        "seconds": gptq_seconds,
        "peak_memory_bytes": peak,
    }
    (args.output / MANIFEST).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["gptq"], indent=2), flush=True)


if __name__ == "__main__":
    main()
