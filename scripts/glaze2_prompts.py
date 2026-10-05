#!/usr/bin/env python3
"""Build Glaze v2's calibration prompts: in-domain, from training sets, decontaminated.

Math from MATH's training split, multiple choice from ARC-Challenge and SciQ, chat from ultrachat
conversations past the ones the other methods calibrate on. Any prompt sharing a 13-gram with
MATH-500 or MMLU-Pro's test questions is dropped. Writes the prompts and a lock recording every
source revision and count.

python scripts/glaze2_prompts.py --output data/glaze2/prompts.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.glaze2.data import (
    LETTERS,
    Decontaminator,
    Prompt,
    math_prompt,
    mcq_prompt,
    prompt_id,
    sciq_options,
)

MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
SOURCES = {
    "math_train": "EleutherAI/hendrycks_math",
    "math500": "HuggingFaceH4/MATH-500",
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "arc": "allenai/ai2_arc",
    "sciq": "allenai/sciq",
    "chat": "HuggingFaceH4/ultrachat_200k",
}
# Well past the start of the split. The other methods' 512 conversations are a random sample of
# all of it, so a few may recur (2 of the 600 chat prompts did).
CHAT_SKIP = 20_000


def revision(repo: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo).sha


def build(args: argparse.Namespace) -> dict:
    from datasets import load_dataset

    revisions = {name: revision(repo) for name, repo in SOURCES.items()}

    def load(name: str, *config: str, split: str, streaming: bool = False):
        return load_dataset(
            SOURCES[name], *config, split=split, revision=revisions[name], streaming=streaming
        )

    math_tests = [row["problem"] for row in load("math500", split="test")]
    mmlu_tests = [
        row["question"] + " " + " ".join(row["options"]) for row in load("mmlu_pro", split="test")
    ]
    math_guard = Decontaminator(math_tests)
    mcq_guard = Decontaminator(mmlu_tests)
    chat_guard = Decontaminator(math_tests + mmlu_tests)
    rng = random.Random(args.seed)
    counts: dict[str, dict[str, int]] = {}
    prompts: list[Prompt] = []

    def keep(name: str, domain: str, texts: list[str], guard: Decontaminator, take: int | None):
        clean = [t for t in texts if guard.clean(t)]
        counts[name] = {"pool": len(texts), "removed": len(texts) - len(clean)}
        if take is not None and take < len(clean):
            clean = rng.sample(clean, take)
        counts[name]["kept"] = len(clean)
        prompts.extend(Prompt(prompt_id(name, t), domain, t, name) for t in clean)

    problems = [
        row["problem"] for c in MATH_CONFIGS for row in load("math_train", c, split="train")
    ]
    keep("math_train", "math", [math_prompt(p) for p in problems], math_guard, args.math)

    arc = []
    for row in load("arc", "ARC-Challenge", split="train"):
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        # Some ARC questions number their options; letters keep one answer format throughout.
        order = sorted(range(len(labels)), key=lambda i: labels[i])
        arc.append(mcq_prompt(row["question"], [texts[i] for i in order][: len(LETTERS)]))
    keep("arc", "mcq", arc, mcq_guard, None)
    sciq = [
        mcq_prompt(row["question"], sciq_options(row, args.seed))
        for row in load("sciq", split="train")
    ]
    keep("sciq", "mcq", sciq, mcq_guard, args.sciq)

    chats = []
    for row in load("chat", split="train_sft", streaming=True).skip(CHAT_SKIP):
        first = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        if 20 <= len(first) <= 4000:
            chats.append(first)
        if len(chats) >= args.chat * 2:
            break
    keep("chat", "chat", chats, chat_guard, args.chat)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(p.record(), sort_keys=True) for p in prompts]
    payload = ("\n".join(lines) + "\n").encode()
    output.write_bytes(payload)
    lock = {
        "path": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "prompts": len(prompts),
        "seed": args.seed,
        "ngram": 13,
        "sources": {
            name: {"repo": SOURCES[name], "revision": rev} for name, rev in revisions.items()
        },
        "counts": counts,
        "chat_skip": CHAT_SKIP,
    }
    output.with_suffix(".lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="data/glaze2/prompts.jsonl")
    parser.add_argument("--math", type=int, default=1600)
    parser.add_argument("--sciq", type=int, default=1100)
    parser.add_argument("--chat", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    lock = build(parser.parse_args())
    print(json.dumps({k: lock[k] for k in ("prompts", "counts", "sha256")}, indent=2), flush=True)
    # The streamed ultrachat reader leaves non-daemon threads that keep the process alive.
    os._exit(0)


if __name__ == "__main__":
    main()
