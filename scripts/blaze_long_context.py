#!/usr/bin/env python3
"""How many long requests a checkpoint's GPU memory holds at once, and how fast it serves them.

At a fixed vLLM memory budget (0.85 of the GPU), the KV cache gets what the weights leave, so a
checkpoint that holds fewer bytes on the GPU keeps more long requests in flight. This reads the
cache size vLLM reports at start-up, then serves N requests of L prompt tokens and M generated
tokens each (distinct random tokens, prefix caching off) and times the whole batch.

    .venv-gpu-text/bin/python scripts/blaze_long_context.py --model DIR --name NAME --output FILE
"""

from __future__ import annotations

import sys

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

# Keep the engine in this process so its start-up log (the KV cache size) reaches our handler.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_CACHE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_CONCURRENCY = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
# Token ids drawn from ordinary vocabulary, clear of the special tokens at the table's end.
TOKEN_RANGE = (1_000, 150_000)


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def kv_report(lines: list[str]) -> dict:
    sizes = [int(m.group(1).replace(",", "")) for line in lines if (m := _CACHE.search(line))]
    concurrency = [
        {"tokens_per_request": int(m.group(1).replace(",", "")), "requests": float(m.group(2))}
        for line in lines
        if (m := _CONCURRENCY.search(line))
    ]
    return {"kv_cache_tokens": min(sizes) if sizes else None, "max_concurrency": concurrency}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=32_768)
    args = parser.parse_args()

    import numpy as np

    lines = _Lines()
    logging.getLogger("vllm").addHandler(lines)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=str(args.model),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=False,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    kv = kv_report(lines.lines)
    rng = np.random.default_rng(0)
    prompts = [
        TokensPrompt(prompt_token_ids=rng.integers(*TOKEN_RANGE, args.prompt_tokens).tolist())
        for _ in range(args.requests)
    ]
    params = SamplingParams(temperature=0.0, max_tokens=args.new_tokens, ignore_eos=True)
    llm.generate([TokensPrompt(prompt_token_ids=[1_000] * 64)], params, use_tqdm=False)  # warm-up
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    seconds = time.perf_counter() - started
    generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    prompt_tokens = args.requests * args.prompt_tokens
    per_request = args.prompt_tokens + args.new_tokens
    result = {
        "name": args.name,
        "model": str(args.model),
        "settings": {
            "prompt_tokens": args.prompt_tokens,
            "new_tokens": args.new_tokens,
            "requests": args.requests,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": 0.85,
        },
        **kv,
        "requests_that_fit": None
        if kv["kv_cache_tokens"] is None
        else kv["kv_cache_tokens"] // per_request,
        "seconds": round(seconds, 2),
        "prompt_tokens_per_second": round(prompt_tokens / seconds, 1),
        "generated_tokens_per_second": round(generated / seconds, 1),
        "total_tokens_per_second": round((prompt_tokens + generated) / seconds, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
