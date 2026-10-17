#!/usr/bin/env python3
"""How much code and non-English text a ladder's 8-bit rows cover, on text no census saw.

Tokenizes HumanEval, MBPP and MultiPL-E (five languages) for code, and MGSM (ten languages) and
Belebele (eight more) for non-English text, and reports, per source and per domain, the share of
tokens whose row is in each census's 8-bit rung (DEFAULT_COVER, DEFAULT_AT_LEAST).

    uv run python scripts/blaze_domain_coverage.py --census data/blaze/census.json \
        data/blaze/census_multilingual.json --output data/blaze/domain_coverage.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from qwen35_compression.blaze.census import load_census
from qwen35_compression.blaze.ladder import plan_ladder

TOKENIZER = "Qwen/Qwen3.5-4B"
HUMANEVAL = ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544")
MBPP = ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c")
MULTIPL_E = ("nuprl/MultiPL-E", "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2")
MGSM = ("juletxara/mgsm", "b2f13d426afe3be8d69a7e739b36724db8b66bbc")
BELEBELE = ("facebook/belebele", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2")
MULTIPL_E_LANGUAGES = {"js": "JavaScript", "cpp": "C++", "java": "Java", "go": "Go", "rs": "Rust"}
MGSM_LANGUAGES = {
    "zh": "Chinese", "ja": "Japanese", "ru": "Russian", "de": "German", "fr": "French",
    "es": "Spanish", "th": "Thai", "sw": "Swahili", "bn": "Bengali", "te": "Telugu",
}  # fmt: skip
BELEBELE_LANGUAGES = {
    "arb_Arab": "Arabic", "hin_Deva": "Hindi", "kor_Hang": "Korean", "vie_Latn": "Vietnamese",
    "tur_Latn": "Turkish", "tam_Taml": "Tamil", "heb_Hebr": "Hebrew", "pes_Arab": "Persian",
}  # fmt: skip


def sources() -> Iterator[tuple[str, str, list[str]]]:
    """(domain, source, texts) for every held-out source."""
    from datasets import load_dataset

    def load(pin: tuple[str, str], *args, **kwargs):
        return load_dataset(pin[0], *args, revision=pin[1], **kwargs)

    yield (
        "code",
        "HumanEval (Python)",
        [r["prompt"] + r["canonical_solution"] for r in load(HUMANEVAL, split="test")],
    )
    mbpp = load(MBPP, "full", split="test")
    yield "code", "MBPP (Python)", [r["text"] + "\n" + r["code"] for r in mbpp]
    for config, name in MULTIPL_E_LANGUAGES.items():
        rows = load(MULTIPL_E, f"humaneval-{config}", split="test")
        yield "code", f"MultiPL-E ({name})", [r["prompt"] for r in rows]
    for config, name in MGSM_LANGUAGES.items():
        rows = load(MGSM, config, split="test")
        yield (
            "multilingual",
            f"MGSM ({name})",
            [r["question"] + " " + (r["answer"] or "") for r in rows],
        )
    for config, name in BELEBELE_LANGUAGES.items():
        rows = load(BELEBELE, config, split="test")
        passages = list(dict.fromkeys(r["flores_passage"] for r in rows))
        yield "multilingual", f"Belebele ({name})", passages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", type=Path, nargs="+", default=[Path("data/blaze/census.json")])
    parser.add_argument("--output", type=Path, default=Path("data/blaze/domain_coverage.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    ladders = {}
    for path in args.census:
        census = load_census(path)
        plan = plan_ladder(census)
        hot = np.zeros(census.vocab_size, dtype=bool)
        hot[list(plan.order[: plan.hot_rows])] = True
        ladders[path.as_posix()] = (plan.hot_rows, hot)

    rows = []
    totals: dict[str, dict[str, list[int]]] = {name: {} for name in ladders}
    for domain, source, texts in sources():
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        ids = np.concatenate([np.asarray(row, dtype=np.int64) for row in encoded if row])
        row = {"domain": domain, "source": source, "tokens": int(ids.size), "hot_share": {}}
        for name, (_, hot) in ladders.items():
            inside = int(hot[ids].sum())
            row["hot_share"][name] = inside / ids.size
            total = totals[name].setdefault(domain, [0, 0])
            total[0] += inside
            total[1] += int(ids.size)
        rows.append(row)
        shares = "  ".join(f"{100 * v:6.2f}%" for v in row["hot_share"].values())
        print(f"{domain:12s} {source:24s} {ids.size:>9,}  {shares}", flush=True)
    result = {
        "tokenizer": TOKENIZER,
        "censuses": {name: {"hot_rows": hot_rows} for name, (hot_rows, _) in ladders.items()},
        "sources": rows,
        "domains": {
            name: {domain: inside / count for domain, (inside, count) in by_domain.items()}
            for name, by_domain in totals.items()
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["domains"], indent=2))


if __name__ == "__main__":
    main()
