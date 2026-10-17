#!/usr/bin/env python3
"""BF16's own answers on code and non-English prompts, as sample files drift_scores.py can score.

The vocabulary ladder's census is mostly English text (training splits only), so its 8-bit rows
cover code well but leave most non-English tokens at 4 bits. These answers are what a drift
score of a laddered export is measured along, outside the domains the census and the record's
benchmarks cover:

- code: HumanEval's 164 problems and MBPP's first 200 test problems, in Python;
- multilingual: MGSM's first N questions in each of its 11 languages (English as the control).

Answers are greedy, instruct mode (no thinking), at most --max-tokens tokens. Each file is
`domain_<group>_<name>.jsonl` with lm-eval's sample fields (doc_id, arguments, resps).

    .venv-gpu-text/bin/python scripts/blaze_domain_answers.py --model SNAPSHOT --output DIR
"""

from __future__ import annotations

import sys

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HUMANEVAL = ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544")
MBPP = ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c")
MGSM = ("juletxara/mgsm", "b2f13d426afe3be8d69a7e739b36724db8b66bbc")
MGSM_LANGUAGES = ("bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh")
MBPP_PROBLEMS = 200


def _load(repo: str, revision: str, *args, **kwargs):
    from datasets import load_dataset

    return load_dataset(repo, *args, revision=revision, **kwargs)


def prompts(per_language: int) -> dict[str, list[str]]:
    """File name -> the user messages, in doc_id order."""
    out: dict[str, list[str]] = {}
    humaneval = _load(*HUMANEVAL, split="test")
    out["domain_code_humaneval"] = [
        "Complete the following Python function. Reply with the full function in a code block."
        f"\n\n```python\n{row['prompt']}```"
        for row in humaneval
    ]
    mbpp = _load(*MBPP, "full", split="test")
    out["domain_code_mbpp"] = [
        f"{row['text']}\nYour code should pass these tests:\n" + "\n".join(row["test_list"])
        for row in list(mbpp)[:MBPP_PROBLEMS]
    ]
    for language in MGSM_LANGUAGES:
        mgsm = _load(*MGSM, language, split="test")
        out[f"domain_ml_{language}"] = [row["question"] for row in list(mgsm)[:per_language]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-language", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="First N prompts per file (pilot)")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    files = prompts(args.per_language)
    rendered = {
        name: [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in (texts[: args.limit] if args.limit else texts)
        ]
        for name, texts in files.items()
    }
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=42,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    flat = [
        (name, index, prompt)
        for name, rows in rendered.items()
        for index, prompt in enumerate(rows)
    ]
    outputs = llm.generate([prompt for _, _, prompt in flat], params)
    args.output.mkdir(parents=True, exist_ok=True)
    handles = {
        name: (args.output / f"{name}.jsonl").open("w", encoding="utf-8") for name in rendered
    }
    tokens = 0
    for (name, index, prompt), output in zip(flat, outputs, strict=True):
        answer = output.outputs[0]
        tokens += len(answer.token_ids)
        row = {
            "doc_id": index,
            "arguments": {"gen_args_0": {"arg_0": prompt}},
            "resps": [[answer.text]],
            "finish_reason": answer.finish_reason,
        }
        handles[name].write(json.dumps(row, ensure_ascii=False) + "\n")
    for handle in handles.values():
        handle.close()
    summary = {
        "model": args.model,
        "files": {name: len(rows) for name, rows in rendered.items()},
        "answer_tokens": tokens,
        "max_tokens": args.max_tokens,
        "sources": {"humaneval": HUMANEVAL, "mbpp": MBPP, "mgsm": MGSM},
    }
    (args.output / "answers.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
