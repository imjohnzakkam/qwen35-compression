#!/usr/bin/env python3
"""Benchmark Blaze's 6-bit matmul against BF16 (cuBLAS) and vLLM's Marlin at 4 and 8 bits.

Shapes are Qwen3.5-4B's body Linears; batch sizes run from one decode token to a large batch.
For each, the 6-bit kernel's output is checked against its torch reference, and every kernel is
timed with CUDA events. The summary adds one decode token's body time across all 32 layers at
batch 1, the regime where weight bytes set the speed. Writes a JSON record.

    .venv-gpu-text/bin/python scripts/blaze_kernel_bench.py --output results/blaze/kernel.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "plugins" / "blaze_vllm"))

# Qwen3.5-4B body Linears as vLLM fuses them: (name, in, out, per decode step and layer type).
SHAPES = {
    "gate_up": (2560, 18432),
    "down": (9216, 2560),
    "deltanet_in": (2560, 12288),
    "deltanet_out": (4096, 2560),
}
# How many of each a decode token runs: 32 FFNs, 24 DeltaNet layers (in and out).
PER_TOKEN = {"gate_up": 32, "down": 32, "deltanet_in": 24, "deltanet_out": 24}
BATCHES = (1, 4, 16, 64, 256)


def timed(fn, iters: int = 100, warmup: int = 20) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # microseconds


def marlin(weight, bits: int):
    """A Marlin GEMM closure for `weight` [n, k] at `bits`, or None when unavailable."""
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            GPTQ_MARLIN_MAX_PARALLEL,
            GPTQ_MARLIN_MIN_THREAD_N,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
            MarlinWorkspace,
            marlin_quantize,
        )
        from vllm.scalar_type import scalar_types

        qtype = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
        n, k = weight.shape
        _, q_w, s, g_idx, sort, _ = marlin_quantize(weight.T.contiguous(), qtype, 128, False)
        ws = MarlinWorkspace(n, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL)

        def run(x):
            return ops.marlin_gemm(
                a=x,
                c=None,
                b_q_weight=q_w,
                b_bias=None,
                b_scales=s,
                a_scales=None,
                global_scale=None,
                b_zeros=None,
                g_idx=g_idx,
                perm=sort,
                workspace=ws.scratch,
                b_q_type=qtype,
                size_m=x.shape[0],
                size_n=n,
                size_k=k,
                is_k_full=True,
            )

        return run
    except Exception as error:  # noqa: BLE001 - recorded, the 6-bit results still stand
        print(f"marlin {bits}-bit unavailable: {type(error).__name__}: {error}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from blaze_vllm import w6

    torch.manual_seed(0)
    device = "cuda"
    results: dict = {"gpu": torch.cuda.get_device_name(), "shapes": {}}
    for name, (k, n) in SHAPES.items():
        weight = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
        grouped = weight.float().reshape(n, k // 128, 128)
        scale = (grouped.abs().amax(-1) / 31).to(torch.bfloat16).clamp_min(1e-8)
        codes = torch.clamp(torch.round(grouped / scale.float().unsqueeze(-1)), -32, 31)
        lo, hi = w6.pack(codes.reshape(n, k).to(torch.int8))
        kernels = {
            "bf16": lambda x, w=weight: F.linear(x, w),
            "w6_blaze": lambda x: w6.matmul(x, lo, hi, scale),
            "w4_marlin": marlin(weight, 4),
            "w8_marlin": marlin(weight, 8),
        }
        record: dict = {"k": k, "n": n, "batches": {}}
        for m in BATCHES:
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            got, want = w6.matmul(x, lo, hi, scale), w6.reference(x, lo, hi, scale)
            err = ((got.float() - want.float()).norm() / want.float().norm()).item()
            row = {"w6_rel_error": err}
            for kernel, fn in kernels.items():
                row[kernel + "_us"] = round(timed(lambda: fn(x)), 1) if fn else None
            record["batches"][str(m)] = row
            print(name, m, {key: value for key, value in row.items()}, flush=True)
        results["shapes"][name] = record
    per_token = {}
    for kernel in ("bf16", "w6_blaze", "w4_marlin", "w8_marlin"):
        times = [results["shapes"][s]["batches"]["1"][kernel + "_us"] for s in SHAPES]
        if all(t is not None for t in times):
            per_token[kernel] = round(
                sum(t * PER_TOKEN[s] for s, t in zip(SHAPES, times, strict=True)) / 1000, 3
            )
    results["decode_body_ms_per_token_batch1"] = per_token
    results["passed"] = all(
        row["w6_rel_error"] < 1e-2
        for shape in results["shapes"].values()
        for row in shape["batches"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"per_token_ms": per_token, "passed": results["passed"]}), flush=True)
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
