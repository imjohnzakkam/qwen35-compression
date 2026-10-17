#!/usr/bin/env python3
"""Check, on a GPU, that Blaze's one-copy lookup reads exactly the rows vLLM would.

For each vocabulary block shape a Qwen3.5 real export uses (65,536 hot rows at 8 bits and 182,784
cold rows at 4 bits, hidden 2,048 for the 2B and 2,560 for the 4B), random codes and scales are
packed as compressed-tensors stores them, repacked by vLLM's own `gptq_marlin_repack` and
`marlin_permute_scales` (exactly as its Marlin kernel does at load), and every row is gathered
back with `blaze_vllm.marlin_rows.gather`. Each row must equal, bit for bit, both codes x scales
and vLLM's quantized-embedding gather on the stored layout (the `twice` layout's lookup). It also
times both gathers on a decode-sized and a prefill-sized batch.

    .venv-gpu-text/bin/python scripts/blaze_vocab_check.py --output vocab_check.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SHAPES = [(65_536, 2_560, 8), (182_784, 2_560, 4), (65_536, 2_048, 8), (182_784, 2_048, 4)]
GROUP = 128
CHUNK = 8_192


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

    return pack_to_int32(codes.to(torch.int8), bits)


def check(rows: int, hidden: int, bits: int, generator: torch.Generator) -> dict:
    from blaze_vllm import marlin_rows
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_embedding import (  # noqa: E501
        _dequant_gather_triton,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales

    device = "cuda"
    half = 1 << (bits - 1)
    codes = torch.randint(-half, half, (rows, hidden), generator=generator).to(device)
    scales = (torch.rand(rows, hidden // GROUP, generator=generator) * 0.02 + 1e-4).to(
        device, torch.bfloat16
    )
    stored = pack(codes, bits)  # [rows, hidden / pack], compressed-tensors layout
    # vLLM's Marlin kernel: GPTQ layout (input dim first), then repack and permute scales.
    marlin = ops.gptq_marlin_repack(
        stored.t().contiguous(),
        perm=torch.empty(0, dtype=torch.int, device=device),
        size_k=hidden,
        size_n=rows,
        num_bits=bits,
        is_a_8bit=False,
    )
    permuted = marlin_permute_scales(scales.t().contiguous(), hidden, rows, GROUP)
    table = marlin_rows.position_table(bits).to(device)

    mismatched_truth = mismatched_vllm = 0
    for start in range(0, rows, CHUNK):
        ids = torch.arange(start, min(start + CHUNK, rows), device=device)
        got = marlin_rows.gather(ids, marlin, permuted, table, bits, hidden)
        truth = (
            (codes[ids].float().reshape(len(ids), -1, GROUP) * scales[ids].float().unsqueeze(-1))
            .reshape(len(ids), hidden)
            .to(torch.bfloat16)
        )
        vllm_rows = _dequant_gather_triton(ids, stored, scales, hidden, bits)
        mismatched_truth += int((got != truth).any(dim=1).sum())
        mismatched_vllm += int((got != vllm_rows).any(dim=1).sum())

    timing = {}
    for name, batch in (("decode_256", 256), ("prefill_8192", 8192)):
        ids = torch.randint(0, rows, (batch,), generator=generator).to(device)
        for label, fn in (
            ("marlin_rows", lambda: marlin_rows.gather(ids, marlin, permuted, table, bits, hidden)),
            ("vllm_stored", lambda: _dequant_gather_triton(ids, stored, scales, hidden, bits)),
        ):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            timing[f"{name}_{label}_us"] = round((time.perf_counter() - started) / 20 * 1e6, 1)
    return {
        "rows": rows,
        "hidden": hidden,
        "bits": bits,
        "marlin_shape": list(marlin.shape),
        "rows_differing_from_codes_x_scales": mismatched_truth,
        "rows_differing_from_vllm_lookup": mismatched_vllm,
        "exact": mismatched_truth == 0 and mismatched_vllm == 0,
        **timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    results = []
    for rows, hidden, bits in SHAPES:
        result = check(rows, hidden, bits, generator)
        print(json.dumps(result), flush=True)
        results.append(result)
    record = {
        "gpu": torch.cuda.get_device_name(),
        "shapes": results,
        "exact": all(r["exact"] for r in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"exact={record['exact']}", flush=True)
    if not record["exact"]:
        raise SystemExit("one-copy lookup differs from vLLM's rows")


if __name__ == "__main__":
    main()
