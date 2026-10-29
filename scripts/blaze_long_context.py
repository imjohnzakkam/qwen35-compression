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
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_CACHE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_CONCURRENCY = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_MEMORY = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
# Token ids drawn from ordinary vocabulary, clear of the special tokens at the table's end.
TOKEN_RANGE = (1_000, 150_000)


def kv_report(lines: list[str]) -> dict:
    sizes = [int(m.group(1).replace(",", "")) for line in lines if (m := _CACHE.search(line))]
    concurrency = [
        {"tokens_per_request": int(m.group(1).replace(",", "")), "requests": float(m.group(2))}
        for line in lines
        if (m := _CONCURRENCY.search(line))
    ]
    memory = [float(m.group(1)) for line in lines if (m := _MEMORY.search(line))]
    return {
        "kv_cache_tokens": min(sizes) if sizes else None,
        "kv_cache_gib": min(memory) if memory else None,
        "max_concurrency": concurrency,
    }


def run_and_read_log(argv: list[str]) -> list[str]:
    """Run the measurement in a child process and keep its output: vLLM's engine logs the KV
    cache size from its own process, where no handler of ours can see it."""
    child = subprocess.Popen(
        [sys.executable, __file__, *argv, "--child"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    for line in child.stdout:
        sys.stdout.write(line)
        lines.append(line)
    if child.wait():
        raise subprocess.CalledProcessError(child.returncode, child.args)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.child:
        kv = kv_report(run_and_read_log(sys.argv[1:]))
        result = json.loads(args.output.read_text(encoding="utf-8"))
        per_request = args.prompt_tokens + args.new_tokens
        tokens = kv["kv_cache_tokens"]
        result.update(kv, requests_that_fit=None if tokens is None else tokens // per_request)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return

    import numpy as np
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
        "seconds": round(seconds, 2),
        "prompt_tokens_per_second": round(prompt_tokens / seconds, 1),
        "generated_tokens_per_second": round(generated / seconds, 1),
        "total_tokens_per_second": round((prompt_tokens + generated) / seconds, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
