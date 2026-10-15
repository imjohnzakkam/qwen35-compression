#!/usr/bin/env python3
"""Blaze stage 1 (CPU): count vocabulary use and plan the ladder for every Qwen3.5 size.

Counts come from public training splits (never a test split) and from BF16's answers to Glaze
v2's training prompts, so no benchmark question or answer informs the ladder. Writes the census
with its lock, then prints coverage and the ladder's bytes per model size.

    uv run python scripts/blaze_census.py \
        --answers logs/jarvis/glaze2-4b-a100-20261005/results/answers.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen35_compression.blaze.census import Census, coverage, file_sha256, save_census
from qwen35_compression.blaze.ladder import plan_ladder
from qwen35_compression.config import load_config


@dataclass(frozen=True)
class Corpus:
    name: str
    repo: str
    files: str
    columns: tuple[str, ...]
    limit: int | None = None


# Training splits only.
CORPORA = (
    Corpus("math_train", "EleutherAI/hendrycks_math", "*/train-*.parquet", ("problem", "solution")),
    Corpus("gsm8k_train", "openai/gsm8k", "main/train-*.parquet", ("question", "answer")),
    Corpus(
        "arc_train", "allenai/ai2_arc", "ARC-Challenge/train-*.parquet", ("question", "choices")
    ),
    Corpus(
        "sciq_train",
        "allenai/sciq",
        "data/train-*.parquet",
        ("question", "support", "correct_answer"),
    ),
    Corpus("hellaswag_train", "Rowan/hellaswag", "data/train-*.parquet", ("ctx", "endings")),
    Corpus(
        "wikitext_train",
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1/wikitext-2-raw-v1-train.parquet",
        ("page",),
    ),
    Corpus(
        "ultrachat_train",
        "HuggingFaceH4/ultrachat_200k",
        "data/train_sft-00000-*.parquet",
        ("messages",),
        20_000,
    ),
    Corpus(
        "coco_captions_train", "phiyodr/coco2017", "data/train-*.parquet", ("captions",), 20_000
    ),
)
# Hidden size and whether the vocabulary table is tied, per Qwen3.5 size.
FAMILY = {
    "0.8B": (1024, True),
    "2B": (2048, True),
    "4B": (2560, True),
    "9B": (4096, False),
    "27B": (5120, False),
}
COVER_POINTS = (16_384, 32_768, 65_536)


def flatten(value: Any) -> Iterator[str]:
    """Every string in a cell: plain text, a list of options, or chat messages."""
    if value is None:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        if "content" in value:
            yield from flatten(value["content"])
            return
        for item in value.values():
            yield from flatten(item)
    else:
        for item in value:
            yield from flatten(item)


def corpus_texts(corpus: Corpus) -> tuple[list[str], str]:
    """A corpus's rows as text, and the dataset revision they came from."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(corpus.repo, repo_type="dataset", allow_patterns=[corpus.files])
    )
    texts: list[str] = []
    for path in sorted(snapshot.glob(corpus.files)):
        for row in pq.read_table(path, columns=list(corpus.columns)).to_pylist():
            texts.append(
                "\n".join(part for column in corpus.columns for part in flatten(row[column]))
            )
            if corpus.limit and len(texts) >= corpus.limit:
                return texts, snapshot.name
    if not texts:
        raise FileNotFoundError(f"no rows for {corpus.repo} {corpus.files}")
    return texts, snapshot.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument(
        "--answers", type=Path, required=True, help="BF16's answers to Glaze v2's training prompts"
    )
    parser.add_argument("--output", type=Path, default=Path("data/blaze/census.json"))
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    config = load_config(args.config)
    snapshot = Path(
        snapshot_download(
            config.model.id,
            revision=config.model.revision,
            allow_patterns=["tokenizer.json", "config.json"],
        )
    )
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    vocab = int(model_config.get("text_config", model_config)["vocab_size"])
    census = Census(vocab)

    def count(source: str, texts: list[str]) -> None:
        tokens = 0
        for start in range(0, len(texts), 2_000):
            encodings = tokenizer.encode_batch(
                texts[start : start + 2_000], add_special_tokens=False
            )
            tokens += census.add(source, (encoding.ids for encoding in encodings))
        print(f"{source:22s} {len(texts):7,d} texts {tokens:12,d} tokens", flush=True)

    sources: dict[str, Any] = {}
    for corpus in CORPORA:
        texts, revision = corpus_texts(corpus)
        sources[corpus.name] = {
            "repo": corpus.repo,
            "revision": revision,
            "files": corpus.files,
            "rows": len(texts),
        }
        count(corpus.name, texts)
    rows = [
        json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line
    ]
    answers = [row["answer"] for row in rows if row.get("answer")]
    sources["bf16_answers"] = {
        "path": args.answers.as_posix(),
        "sha256": file_sha256(args.answers),
        "rows": len(answers),
    }
    count("bf16_answers", answers)

    lock = save_census(
        census,
        args.output,
        {"tokenizer": {"model": config.model.id, "revision": snapshot.name}, "sources": sources},
    )
    plan = plan_ladder(census)
    order, own = census.ranking(), census.counts["bf16_answers"]
    print(f"\n{census.used_rows():,} of {vocab:,} rows used; {vocab - census.used_rows():,} never")
    for count_rows in COVER_POINTS:
        share = coverage(order, own, count_rows)
        print(f"busiest {count_rows:6,d} rows hold {share:.4%} of BF16's own answer tokens")
    print(f"ladder: {plan.hot_rows:,} rows at 8 bits, {vocab - plan.hot_rows:,} at 4 bits")
    for size, (hidden, tied) in FAMILY.items():
        bf16 = vocab * hidden * 2 * (1 if tied else 2)
        laddered = plan.table_bytes(hidden)
        if not tied:
            laddered += plan.rest_option.tensor_bytes(vocab, hidden)
        print(
            f"Qwen3.5-{size:4s} vocabulary {bf16 / 1e9:6.3f} GB in BF16,"
            f" {laddered / 1e9:6.3f} GB laddered"
        )
    print(f"\nwrote {lock['path']} (sha256 {lock['sha256'][:12]})")


if __name__ == "__main__":
    main()
