"""A 6-bit weight-only matmul (W6A16, symmetric, group 128) for Blaze's 6-bit body units.

Codes q in [-32, 31] are stored offset to u = q + 32 in two bit planes, so a kernel unpacks
them with shifts alone: `lo` holds u's low 4 bits, 8 codes per int32 along the input dimension,
and `hi` its high 2 bits, 16 codes per int32. One BF16 scale per row and group of 128 inputs; the
weight is code times scale, rounded to BF16, as for every other Blaze option. The Triton kernel
tiles rows and inputs, can split the input dimension across programs for small batches, and
accumulates in fp32. `reference` is the same computation in plain torch.
"""

from __future__ import annotations

import torch

GROUP = 128


def pack(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int codes [n, k] in [-32, 31] -> (lo [n, k / 8], hi [n, k / 16]) int32 planes."""
    rows, cols = codes.shape
    if cols % 16:
        raise ValueError(f"{cols} inputs are not a multiple of 16")
    u = (codes.to(torch.int32) + 32) & 63
    lo = (u & 15).reshape(rows, cols // 8, 8)
    hi = (u >> 4).reshape(rows, cols // 16, 16)
    lo_shift = torch.arange(8, dtype=torch.int32, device=codes.device) * 4
    hi_shift = torch.arange(16, dtype=torch.int32, device=codes.device) * 2
    # Sum of disjoint bit fields: an OR without int32 overflow trouble in torch.
    lo_words = (lo << lo_shift).sum(dim=-1, dtype=torch.int64)
    hi_words = (hi << hi_shift).sum(dim=-1, dtype=torch.int64)
    as_int32 = lambda w: torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)  # noqa: E731
    return as_int32(lo_words), as_int32(hi_words)


def unpack(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """The int32 codes [n, k] the planes hold."""
    rows = lo.shape[0]
    lo_shift = torch.arange(8, dtype=torch.int32, device=lo.device) * 4
    hi_shift = torch.arange(16, dtype=torch.int32, device=lo.device) * 2
    low = ((lo.unsqueeze(-1) >> lo_shift) & 15).reshape(rows, -1)
    high = ((hi.unsqueeze(-1) >> hi_shift) & 3).reshape(rows, -1)
    return (low | (high << 4)) - 32


def dequantize(lo: torch.Tensor, hi: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    codes = unpack(lo, hi).float()
    rows, cols = codes.shape
    weight = codes.reshape(rows, cols // GROUP, GROUP) * scale.float().unsqueeze(-1)
    return weight.to(scale.dtype).reshape(rows, cols)


def reference(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, scale: torch.Tensor):
    """x [m, k] @ dequantized weightᵀ -> [m, n], accumulated in fp32."""
    return (x.float() @ dequantize(lo, hi, scale).float().T).to(x.dtype)


_kernel = None


def _build():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def w6_kernel(
        x_ptr,
        lo_ptr,
        hi_ptr,
        scale_ptr,
        out_ptr,
        m,
        n,
        k,
        groups,
        stride_xm,
        stride_om,
        GROUPS_PER_SPLIT: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < m
        mask_n = offs_n < n
        lo_cols = k // 8
        hi_cols = k // 16
        lo_shift = tl.arange(0, 8) * 4
        hi_shift = tl.arange(0, 16) * 2
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        first = pid_k * GROUPS_PER_SPLIT
        for gi in range(0, GROUPS_PER_SPLIT):
            g = first + gi
            if g < groups:
                kk = g * 128 + tl.arange(0, 128)
                x = tl.load(
                    x_ptr + offs_m[:, None] * stride_xm + kk[None, :],
                    mask=mask_m[:, None],
                    other=0.0,
                )
                lw = tl.load(
                    lo_ptr + offs_n[:, None] * lo_cols + g * 16 + tl.arange(0, 16)[None, :],
                    mask=mask_n[:, None],
                    other=0,
                )
                hw = tl.load(
                    hi_ptr + offs_n[:, None] * hi_cols + g * 8 + tl.arange(0, 8)[None, :],
                    mask=mask_n[:, None],
                    other=0,
                )
                low = tl.reshape((lw[:, :, None] >> lo_shift[None, None, :]) & 15, (BLOCK_N, 128))
                high = tl.reshape((hw[:, :, None] >> hi_shift[None, None, :]) & 3, (BLOCK_N, 128))
                code = (low | (high << 4)) - 32
                s = tl.load(scale_ptr + offs_n * groups + g, mask=mask_n, other=0.0)
                w = (code.to(tl.float32) * s.to(tl.float32)[:, None]).to(s.dtype).to(x.dtype)
                acc += tl.dot(x, tl.trans(w))
        out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]
        if SPLIT == 1:
            tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            tl.atomic_add(out, acc, mask=mask)

    return w6_kernel


def choose(m: int, n: int, k: int, sms: int) -> tuple[int, int, int]:
    """(BLOCK_M, BLOCK_N, SPLIT): enough programs to fill the GPU at small batches."""
    block_m = 16 if m <= 16 else 32 if m <= 32 else 64
    block_n = 64
    tiles = -(-n // block_n) * -(-m // block_m)
    groups = k // GROUP
    split = 1
    while tiles * split < 2 * sms and split * 2 <= groups and split < 16:
        split *= 2
    return block_m, block_n, split


def matmul(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, scale: torch.Tensor):
    """x [m, k] @ dequantized weightᵀ -> [m, n] in x's dtype."""
    global _kernel
    if not x.is_cuda:
        return reference(x, lo, hi, scale)
    if _kernel is None:
        _kernel = _build()
    from vllm.triton_utils import triton

    x = x.contiguous()
    m, k = x.shape
    n = lo.shape[0]
    groups = k // GROUP
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    block_m, block_n, split = choose(m, n, k, sms)
    per_split = -(-groups // split)
    if split == 1:
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    else:
        out = torch.zeros(m, n, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m), split)
    _kernel[grid](
        x,
        lo,
        hi,
        scale,
        out,
        m,
        n,
        k,
        groups,
        x.stride(0),
        out.stride(0),
        GROUPS_PER_SPLIT=per_split,
        SPLIT=split,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out if split == 1 else out.to(x.dtype)
