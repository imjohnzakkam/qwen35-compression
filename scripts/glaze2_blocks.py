#!/usr/bin/env python3
"""Pack BF16's answers into Glaze v2's calibration and held-out blocks.

Answers cut off at the token limit are left out. Prompts are split into calibration and held-out
by their id (content, not position), the domains are interleaved to the configured token shares,
and each set is packed into fixed-length blocks with answer and domain masks.

python scripts/glaze2_blocks.py --config configs/glaze2_proxy.yaml --answers A --output-dir D
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.glaze2.data import (
    chat_sample,
    interleave,
    is_held_out,
    pack_samples,
    save_blocks,
)

SHARES = {"math": 0.4, "mcq": 0.4, "chat": 0.2}
TEMPLATE_KWARGS = {"enable_thinking": False}


def digest(blocks) -> str:
    return hashlib.sha256(json.dumps(blocks.ids, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-blocks", type=int, default=512)
    parser.add_argument("--held-out-blocks", type=int, default=64)
    parser.add_argument("--held-out-share", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if (output / "calibration.jsonl").exists():
        parser.error(f"refusing to overwrite blocks in {output}")

    from transformers import AutoTokenizer

    from qwen35_compression.models import download_model

    config = load_config(args.config)
    snapshot, _ = download_model(config)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    length = config.calibration.max_sequence_length
    rows = [json.loads(line) for line in Path(args.answers).read_text().splitlines() if line]
    finished = [r for r in rows if r["finish"] == "stop" and r["answer"].strip()]
    calibration, held_out = [], []
    for row in finished:
        sample = chat_sample(
            tokenizer, row["domain"], row["prompt"], row["answer"], TEMPLATE_KWARGS
        )
        (held_out if is_held_out(row["id"], args.held_out_share) else calibration).append(sample)
    cal_blocks = pack_samples(
        interleave(calibration, SHARES, args.seed), length, args.calibration_blocks
    )
    dev_blocks = pack_samples(interleave(held_out, SHARES, args.seed), length, args.held_out_blocks)
    output.mkdir(parents=True, exist_ok=True)
    save_blocks(output / "calibration.jsonl", cal_blocks)
    save_blocks(output / "held_out.jsonl", dev_blocks)
    record = {
        "answers": len(rows),
        "finished": len(finished),
        "calibration": {
            "blocks": len(cal_blocks),
            "tokens": cal_blocks.tokens_by_domain(),
            "sha256": digest(cal_blocks),
        },
        "held_out": {
            "blocks": len(dev_blocks),
            "tokens": dev_blocks.tokens_by_domain(),
            "sha256": digest(dev_blocks),
        },
        "shares": SHARES,
        "block_tokens": length,
    }
    (output / "blocks.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("blocks=" + json.dumps(record))


if __name__ == "__main__":
    main()
