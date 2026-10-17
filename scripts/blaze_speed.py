#!/usr/bin/env python3
"""Decode speed and weight memory of one checkpoint in vLLM, on fixed synthetic work.

Two measures after a warm-up: batched throughput (256 requests of 512 generated tokens each,
end of sequence ignored) and single-stream speed (4 requests of 256 tokens, one at a time). The
weight memory is the bytes of the loaded model's parameters and buffers on the GPU.

    .venv-gpu-text/bin/python scripts/blaze_speed.py --model DIR --name NAME --output FILE
"""

from __future__ import annotations

import os
import sys

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)
# One process, so the model's memory can be read directly.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

TOPICS = (
    "the history of mathematics",
    "how vaccines work",
    "a trip across the Himalayas",
    "the design of a compiler",
    "climate change and oceans",
    "the economics of cities",
    "learning to play the violin",
    "the life cycle of stars",
)


def model_bytes(worker) -> int:
    model = worker.model_runner.model
    tensors = list(model.parameters()) + list(model.buffers())
    seen, total = set(), 0
    for tensor in tensors:
        if tensor.data_ptr() in seen:
            continue
        seen.add(tensor.data_ptr())
        total += tensor.numel() * tensor.element_size()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument(
        "--single-copy",
        action="store_true",
        help="Serve a real Blaze export's vocabulary table once (BLAZE_SINGLE_COPY=1)",
    )
    args = parser.parse_args()
    if args.single_copy:
        os.environ["BLAZE_SINGLE_COPY"] = "1"

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=str(args.model), max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    try:
        weights = llm.collective_rpc(model_bytes)[0]
    except Exception as error:  # noqa: BLE001 - recorded, not fatal
        weights = None
        print(f"weight memory not read: {error}", flush=True)
    tokenizer = llm.get_tokenizer()

    def prompt(index: int) -> str:
        text = f"Write a long, detailed essay about {TOPICS[index % len(TOPICS)]} (part {index})."
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def run(count: int, tokens: int) -> tuple[int, float]:
        params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = llm.generate([prompt(i) for i in range(count)], params, use_tqdm=False)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        return sum(len(o.outputs[0].token_ids) for o in outputs), seconds

    run(8, 64)  # warm-up
    batch_tokens, batch_seconds = run(args.requests, args.tokens)
    single_tokens, single_seconds = 0, 0.0
    for _ in range(4):
        tokens, seconds = run(1, 256)
        single_tokens += tokens
        single_seconds += seconds
    record = {
        "name": args.name,
        "single_copy": args.single_copy,
        "model": str(args.model),
        "gpu": torch.cuda.get_device_name(),
        "weight_memory_bytes": weights,
        "checkpoint_bytes": sum(p.stat().st_size for p in args.model.glob("*.safetensors")),
        "batched": {
            "requests": args.requests,
            "tokens": batch_tokens,
            "seconds": round(batch_seconds, 2),
            "tokens_per_second": round(batch_tokens / batch_seconds, 1),
        },
        "single_stream": {
            "tokens": single_tokens,
            "seconds": round(single_seconds, 2),
            "tokens_per_second": round(single_tokens / single_seconds, 1),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
