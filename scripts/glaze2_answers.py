#!/usr/bin/env python3
"""BF16's own answers to Glaze v2's calibration prompts, generated with vLLM.

Greedy, in instruct mode (thinking off), the chat template the evaluation suite uses. Runs in the
text-evaluation environment, where vLLM lives.

python scripts/glaze2_answers.py --model MODEL --prompts data/glaze2/prompts.jsonl --output OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rotate(prompts: list[dict]) -> list[dict]:
    """The prompts taken one domain at a time in turn, so a pilot's first N cover every domain."""
    pools: dict[str, list[dict]] = {}
    for prompt in prompts:
        pools.setdefault(prompt["domain"], []).append(prompt)
    order: list[dict] = []
    while any(pools.values()):
        for pool in pools.values():
            if pool:
                order.append(pool.pop(0))
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="BF16 snapshot directory")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, help="Only the first N prompts (pilot)")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite {output}")

    from vllm import LLM, SamplingParams

    prompts = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line]
    if args.limit:
        prompts = rotate(prompts)[: args.limit]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=42,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=42)
    conversations = [[{"role": "user", "content": p["text"]}] for p in prompts]
    results = llm.chat(
        conversations,
        params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    finished = 0
    with output.open("w", encoding="utf-8") as handle:
        for prompt, result in zip(prompts, results, strict=True):
            completion = result.outputs[0]
            finished += completion.finish_reason == "stop"
            handle.write(
                json.dumps(
                    {
                        "id": prompt["id"],
                        "domain": prompt["domain"],
                        "prompt": prompt["text"],
                        "answer": completion.text,
                        "finish": completion.finish_reason,
                        "tokens": len(completion.token_ids),
                    }
                )
                + "\n"
            )
    print(f"answers={len(prompts)} finished={finished} output={output}")


if __name__ == "__main__":
    main()
