#!/usr/bin/env python3
"""A multilingual census: Blaze's census plus FineWeb-2 web text in 30 languages.

The ladder's census (data/blaze/census.json) is English training text and English answers, and
its 8-bit rows cover only about a quarter of non-English tokens (scripts/blaze_domain_coverage.py).
This adds the first N tokens of FineWeb-2's training split in each language as its own source,
so a ladder planned from the result spends its 8-bit rows on more scripts; plan it with
`--cover 1e-9` (EQUAL_SIZE_COVER) to keep the English ladder's 65,536 rows and bytes.
FineWeb-2 is web text and holds no benchmark data.

    uv run python scripts/blaze_census_multilingual.py --tokens 300000
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from qwen35_compression.blaze.census import load_census, save_census
from qwen35_compression.blaze.ladder import plan_ladder

REPO = "HuggingFaceFW/fineweb-2"
REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
TOKENIZER = "Qwen/Qwen3.5-4B"
# Planned with a negligible cover, the ladder keeps DEFAULT_AT_LEAST (65,536) 8-bit rows, as the
# English census's does: the table's bytes stay the same and only which rows are 8-bit changes.
# (At DEFAULT_COVER this census would want 152,794 rows, about 112 MB more.)
EQUAL_SIZE_COVER = 1e-9
LANGUAGES = (
    "cmn_Hani", "jpn_Jpan", "kor_Hang", "rus_Cyrl", "ukr_Cyrl", "deu_Latn", "fra_Latn",
    "spa_Latn", "por_Latn", "ita_Latn", "nld_Latn", "pol_Latn", "tur_Latn", "vie_Latn",
    "ind_Latn", "tha_Thai", "arb_Arab", "fas_Arab", "heb_Hebr", "hin_Deva", "ben_Beng",
    "tam_Taml", "tel_Telu", "mar_Deva", "urd_Arab", "swh_Latn", "ell_Grek", "ces_Latn",
    "ron_Latn", "hun_Latn",
)  # fmt: skip


def count_language(language: str, tokens: int, vocab: int) -> tuple[str, np.ndarray, int]:
    """Token counts of the first `tokens` tokens of a language's training text."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    stream = load_dataset(REPO, language, split="train", streaming=True, revision=REVISION)
    counts = np.zeros(vocab, dtype=np.int64)
    seen = documents = 0
    for row in stream:
        ids = np.asarray(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        ids = ids[: tokens - seen]
        counts += np.bincount(ids, minlength=vocab)
        seen += len(ids)
        documents += 1
        if seen >= tokens:
            break
    print(f"{language}: {documents} documents, {seen:,} tokens", flush=True)
    return language, counts, documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--census", type=Path, default=Path("data/blaze/census.json"))
    parser.add_argument("--output", type=Path, default=Path("data/blaze/census_multilingual.json"))
    parser.add_argument("--tokens", type=int, default=300_000, help="Tokens per language")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    census = load_census(args.census)
    documents: dict[str, int] = {}
    jobs = [(language, args.tokens, census.vocab_size) for language in LANGUAGES]
    # Pool's exit terminates the workers: streaming readers can keep an idle worker from exiting.
    with get_context("spawn").Pool(args.workers) as pool:
        for language, counts, docs in pool.starmap(count_language, jobs):
            census.counts[f"fineweb2_{language}"] = counts
            documents[language] = docs
    lock = save_census(
        census,
        args.output,
        {
            "base_census": {
                "path": args.census.as_posix(),
                "sha256": json.loads(
                    args.census.with_name(args.census.stem + ".lock.json").read_text()
                )["sha256"],
            },
            "fineweb2": {
                "repo": REPO,
                "revision": REVISION,
                "split": "train",
                "tokens_per_language": args.tokens,
                "documents": documents,
                "tokenizer": TOKENIZER,
            },
        },
    )
    plan = plan_ladder(census, cover=EQUAL_SIZE_COVER)
    print(
        json.dumps(
            {"tokens": census.tokens(), "used_rows": lock["used_rows"], "hot_rows": plan.hot_rows},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
