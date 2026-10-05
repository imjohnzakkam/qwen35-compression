#!/usr/bin/env python3
"""Load an export in vLLM, vision tower included, and answer two short prompts.

Mixed 4- and 8-bit config groups with exact module targets must load in stock vLLM; this is the
check before any 4B spend. Runs in the text-evaluation environment.

python scripts/glaze2_vllm_check.py --model EXPORT --output OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPTS = ("What is 17 times 23? Answer with the number only.", "Name the capital of France.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        seed=42,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    params = SamplingParams(temperature=0.0, max_tokens=32)
    results = llm.chat(
        [[{"role": "user", "content": p}] for p in PROMPTS],
        params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    answers = [r.outputs[0].text for r in results]
    record = {"model": args.model, "answers": dict(zip(PROMPTS, answers, strict=True))}
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("vllm_check=" + json.dumps(answers))


if __name__ == "__main__":
    main()
