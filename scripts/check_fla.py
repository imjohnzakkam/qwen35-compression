#!/usr/bin/env python3
"""Compare transformers' PyTorch DeltaNet fallback with flash-linear-attention's kernel.

Runs one Gated DeltaNet layer of the model forward and backward on an AutoRound-sized batch
(8 x 2,048 tokens), once with each implementation of the chunked delta rule, and records peak
GPU memory, time, and how closely the two outputs agree. Run after installing
flash-linear-attention; a side that runs out of memory is recorded as such.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def run_layer(layer, kernel, inputs):
    import torch

    layer.chunk_gated_delta_rule = kernel
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        x = inputs.detach().clone().requires_grad_(True)
        out = layer(x)
        out = out[0] if isinstance(out, tuple) else out
        out.float().pow(2).mean().backward()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "out_of_memory"}, None
    return {
        "status": "ok",
        "seconds": round(time.perf_counter() - started, 3),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }, out.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=2048)
    args = parser.parse_args()

    import fla
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    layer = next(m for m in model.modules() if isinstance(m, qwen.Qwen3_5GatedDeltaNet))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(0)
    hidden = model.config.get_text_config().hidden_size
    inputs = 0.1 * torch.randn(args.batch, args.tokens, hidden, dtype=torch.bfloat16, device="cuda")

    fallback, reference = run_layer(layer, qwen.torch_chunk_gated_delta_rule, inputs)
    kernel, candidate = run_layer(layer, qwen.chunk_gated_delta_rule, inputs)
    result = {
        "fla_version": fla.__version__,
        "transformers_uses_fla_kernel": qwen.chunk_gated_delta_rule is not None,
        "shape": [args.batch, args.tokens, hidden],
        "torch_fallback": fallback,
        "fla": kernel,
    }
    if reference is not None and candidate is not None:
        diff = (candidate.float() - reference.float()).abs().max().item()
        scale = reference.float().abs().max().item()
        result["max_abs_diff"] = diff
        result["max_rel_diff"] = diff / scale if scale else None
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
