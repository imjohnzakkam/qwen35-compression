#!/usr/bin/env python3
"""Check that vLLM serves a real Blaze export as its fake-quantized twin: same tokens, same logits.

Both exports are loaded in turn and asked for the top-20 log-probabilities of the next token after
each of a few fixed prompts, greedily for N tokens. The real export must load, its stored table
must equal the fake one's (read back from the files), and its greedy tokens and log-probabilities
must agree with the fake export's to kernel rounding. Writes a JSON record; exits 1 on a failure.

    .venv-gpu-text/bin/python scripts/blaze_real_check.py --real DIR --fake DIR --output FILE
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

PROMPTS = (
    "Solve for x: 3x + 7 = 22. Show your steps.",
    "Explain why the sky is blue in two sentences.",
    "Write a Python function that reverses a linked list.",
    "Which is larger, 2^10 or 10^3? Answer briefly.",
    "Translate 'good morning, how are you?' into French and Hindi.",
    "List three causes of the French Revolution.",
    "What is the derivative of x^3 * sin(x)?",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
)
# Greedy tokens may part where two candidates are within kernel rounding of each other; the
# check asks for most prompts to agree throughout, and for close log-probabilities where they do.
MIN_AGREEING_PROMPTS = 6
MAX_MEAN_LOGPROB_GAP = 0.05


def generate(model: Path, tokens: int) -> list[dict]:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=str(model), max_model_len=2048, gpu_memory_utilization=0.6, seed=0)
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in PROMPTS
    ]
    params = SamplingParams(temperature=0.0, max_tokens=tokens, logprobs=1)
    outputs = llm.generate(prompts, params)
    records = []
    for output in outputs:
        sample = output.outputs[0]
        records.append(
            {
                "text": sample.text,
                "tokens": list(sample.token_ids),
                "logprobs": [
                    step[token].logprob
                    for step, token in zip(sample.logprobs, sample.token_ids, strict=True)
                ],
            }
        )
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return records


def compare(real: list[dict], fake: list[dict]) -> dict:
    agreeing, gaps = 0, []
    for a, b in zip(real, fake, strict=True):
        same = 0
        for x, y in zip(a["tokens"], b["tokens"], strict=False):
            if x != y:
                break
            same += 1
        agreeing += same == len(b["tokens"])
        gaps.extend(abs(p - q) for p, q in zip(a["logprobs"][:same], b["logprobs"][:same]))
    return {
        "prompts": len(fake),
        "agreeing_prompts": agreeing,
        "mean_logprob_gap": sum(gaps) / len(gaps) if gaps else None,
        "max_logprob_gap": max(gaps) if gaps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", type=Path)
    parser.add_argument("--fake", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--generate", type=Path, help="Internal: one model's answers to --output")
    args = parser.parse_args()

    if args.generate:
        # One engine per process: a second engine in the same process may not get its memory.
        records = generate(args.generate, args.tokens)
        args.output.write_text(json.dumps(records) + "\n", encoding="utf-8")
        return

    import subprocess

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import torch
    from safetensors import safe_open

    from qwen35_compression.blaze.export import TIED_TABLE
    from qwen35_compression.blaze.real import read_table

    fake_table = None
    for path in sorted(args.fake.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as shard:
            if TIED_TABLE in shard.keys():
                fake_table = shard.get_tensor(TIED_TABLE)
    table_equal = fake_table is not None and bool(torch.equal(read_table(args.real), fake_table))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    answers = {}
    for name, model in (("real", args.real), ("fake", args.fake)):
        target = args.output.with_name(f"{args.output.stem}-{name}.json")
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--generate",
                str(model),
                "--output",
                str(target),
                "--tokens",
                str(args.tokens),
            ],
            check=True,
        )
        answers[name] = json.loads(target.read_text(encoding="utf-8"))
    summary = compare(answers["real"], answers["fake"])
    summary["table_equal"] = table_equal
    summary["passed"] = (
        table_equal
        and summary["agreeing_prompts"] >= MIN_AGREEING_PROMPTS
        and (summary["mean_logprob_gap"] or 0.0) <= MAX_MEAN_LOGPROB_GAP
    )
    record = {"summary": summary, **answers}
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("real:", answers["real"][0]["text"][:200].replace("\n", " "), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
