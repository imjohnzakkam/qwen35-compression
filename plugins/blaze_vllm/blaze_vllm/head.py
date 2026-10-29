"""Logits straight from a compressed-tensors packed block, so one copy serves lookups and logits.

`packed` holds each row's signed INT codes, offset to unsigned and packed densely into int32
(compressed-tensors' pack-quantized layout, `32 // bits` codes per word); `scale` holds one
scale per row and group. The kernel dequantizes a tile of rows exactly as the quantized
embedding does (code times scale, rounded to the activations' dtype) and multiplies it with the
hidden states, accumulating in fp32. `reference` is the same computation in plain torch.
"""

from __future__ import annotations

import torch


def reference(
    x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, bits: int
) -> torch.Tensor:
    """x [m, k] @ dequantized(packed)ᵀ -> [m, rows], in torch."""
    rows, words = packed.shape
    per_word = 32 // bits
    hidden = words * per_word
    shifts = torch.arange(per_word, device=packed.device, dtype=torch.int32) * bits
    codes = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    codes = codes.reshape(rows, hidden) - (1 << (bits - 1))
    group = hidden // scale.shape[1]
    # Dequantized in the scales' dtype, as the quantized embedding gives the same rows.
    weight = (codes.float().reshape(rows, -1, group) * scale.float().unsqueeze(-1)).to(scale.dtype)
    weight = weight.reshape(rows, hidden)
    return (x.float() @ weight.float().T).to(x.dtype)


_kernel = None


def _build():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def packed_logits_kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        out_ptr,
        m,
        rows,
        hidden,
        words,
        groups,
        stride_xm,
        stride_om,
        BITS: tl.constexpr,
        PER_WORD: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, GROUP)
        mask_m = offs_m < m
        mask_n = offs_n < rows
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, groups):
            k = g * GROUP + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            word = tl.load(
                packed_ptr + offs_n[:, None] * words + (k[None, :] // PER_WORD),
                mask=mask_n[:, None],
                other=0,
            )
            code = ((word >> ((k[None, :] % PER_WORD) * BITS)) & ((1 << BITS) - 1)) - (
                1 << (BITS - 1)
            )
            s = tl.load(scale_ptr + offs_n * groups + g, mask=mask_n, other=0.0)
            w = (code.to(tl.float32) * s.to(tl.float32)[:, None]).to(s.dtype).to(x.dtype)
            acc += tl.dot(x, tl.trans(w))
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    return packed_logits_kernel


def packed_logits(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """x [m, k] @ dequantized(packed)ᵀ -> [m, rows], written into `out` when given."""
    global _kernel
    if not x.is_cuda:
        result = reference(x, packed, scale, bits)
        if out is not None:
            out.copy_(result)
            return out
        return result
    if _kernel is None:
        _kernel = _build()
    from vllm.triton_utils import triton

    x = x.contiguous()
    m, hidden = x.shape
    rows, words = packed.shape
    groups = scale.shape[1]
    if out is None:
        out = torch.empty(m, rows, dtype=x.dtype, device=x.device)
    block_m = 16 if m <= 16 else 64
    block_n = 64
    grid = (triton.cdiv(rows, block_n), triton.cdiv(m, block_m))
    _kernel[grid](
        x,
        packed,
        scale,
        out,
        m,
        rows,
        hidden,
        words,
        groups,
        x.stride(0),
        out.stride(0),
        BITS=bits,
        PER_WORD=32 // bits,
        GROUP=hidden // groups,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out
