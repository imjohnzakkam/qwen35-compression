"""Embedding rows read straight from a Marlin-repacked weight, so the table is held once.

vLLM's LM head runs a quantized table through Marlin, which repacks it at load into its own tile
layout; the lookup then needed a second copy in the stored layout. Here the lookup gathers and
dequantizes rows from the Marlin layout itself (symmetric INT4/INT8, group 128, no act-order).

Marlin's layout (vLLM 0.29, `marlin_utils_test.marlin_weights` and `gptq_marlin_repack`): the
unsigned codes u[k, n] (input k, output n) are cut into 16x16 tiles, each k-tile row is laid out
tile by tile (n // 16), and every 1024 consecutive codes (four tiles, 64 outputs) are permuted by
`weight_perm`; then 32 / bits codes are packed per int32, code i of a word at bit bits * i.
So code (k, n) sits in packed row k // 16, at flat position (n // 64) * 1024 + T[n % 64, k % 16],
where T inverts the permutation within one 16 x 64 block. Scales s[g, n] are permuted by an 8x8
transpose within each run of 64 outputs: s[g, n] is stored in column
(n // 64) * 64 + (n % 8) * 8 + (n % 64) // 8.
"""

from __future__ import annotations

import torch

TILE = 16
BLOCK_N = 64
CHUNK = TILE * BLOCK_N  # 1024 codes: four 16x16 tiles


def weight_perm(bits: int) -> torch.Tensor:
    """Marlin's permutation of 1024 codes (vLLM's `get_weight_perm`, 16-bit activations)."""
    perm: list[int] = []
    for i in range(32):
        block: list[int] = []
        col = i // 4
        for half in (0, 1):
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1):
                block.append(16 * row + col + 8 * half)
        for j in range(4):
            perm.extend(p + 256 * j for p in block)
    interleave = {4: [0, 2, 4, 6, 1, 3, 5, 7], 8: [0, 2, 1, 3]}[bits]
    table = torch.tensor(perm).reshape(-1, len(interleave))[:, interleave]
    return table.reshape(-1)


def position_table(bits: int) -> torch.Tensor:
    """T[n % 64, k % 16]: a code's flat position within its 1024-code block (int32, 64 x 16)."""
    inverse = torch.empty(CHUNK, dtype=torch.long)
    inverse[weight_perm(bits)] = torch.arange(CHUNK)
    n = torch.arange(BLOCK_N).unsqueeze(1)
    k = torch.arange(TILE).unsqueeze(0)
    before = (n // TILE) * (TILE * TILE) + k * TILE + n % TILE
    return inverse[before].to(torch.int32)


def scale_column(n: torch.Tensor) -> torch.Tensor:
    """Where output n's scales sit in Marlin's permuted scale rows."""
    return (n // BLOCK_N) * BLOCK_N + (n % 8) * 8 + (n % BLOCK_N) // 8


def pack_reference(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Signed codes [n, k] -> Marlin's packed int32 [k / 16, n * 16 * bits / 32] (for tests)."""
    rows, cols = codes.shape
    u = (codes.to(torch.int64) + (1 << (bits - 1))).T  # [k, n] unsigned
    u = u.reshape(cols // TILE, TILE, rows // TILE, TILE).permute(0, 2, 1, 3)
    u = u.reshape(cols // TILE, rows * TILE)
    u = u.reshape(-1, CHUNK)[:, weight_perm(bits)].reshape(cols // TILE, rows * TILE)
    per_word = 32 // bits
    shifts = torch.arange(per_word) * bits
    words = (u.reshape(u.shape[0], -1, per_word) << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def scales_reference(scales: torch.Tensor) -> torch.Tensor:
    """Scales [n, groups] -> Marlin's permuted [groups, n] (for tests)."""
    s = scales.T.contiguous()
    groups, rows = s.shape
    perm = torch.tensor([i + 8 * j for i in range(8) for j in range(8)])
    return s.reshape(-1, BLOCK_N)[:, perm].reshape(groups, rows)


def gather_reference(
    ids: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, bits: int, hidden: int
) -> torch.Tensor:
    """Rows `ids` of the dequantized table, read from Marlin's layout in plain torch."""
    table = position_table(bits).to(packed.device).long()
    n = ids.reshape(-1, 1).long()
    k = torch.arange(hidden, device=packed.device).unsqueeze(0)
    flat = (n // BLOCK_N) * CHUNK + table[n % BLOCK_N, k % TILE]
    per_word = 32 // bits
    words = packed.long()[k // TILE, flat // per_word] & 0xFFFFFFFF
    codes = ((words >> ((flat % per_word) * bits)) & ((1 << bits) - 1)) - (1 << (bits - 1))
    group = hidden // scales.shape[0]
    scale = scales[k // group, scale_column(n)]
    return (codes.float() * scale.float()).to(scales.dtype)


try:  # vLLM's Triton; absent in CPU-only test environments, which use gather_reference
    from vllm.triton_utils import tl, triton
except ImportError:
    tl = triton = None

# Defined at import, as vLLM's own quantized-embedding kernel is: vLLM compiles the model's
# forward with torch.compile, which can launch a Triton kernel but not define one mid-trace.
if triton is not None:

    @triton.jit
    def _marlin_rows_kernel(
        ids_ptr,
        packed_ptr,
        scale_ptr,
        table_ptr,
        out_ptr,
        hidden,
        packed_cols,
        scale_cols,
        BITS: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = k < hidden
        n = tl.load(ids_ptr + row).to(tl.int64)
        position = tl.load(table_ptr + (n % 64) * 16 + k % 16, mask=mask, other=0).to(tl.int64)
        flat = (n // 64) * 1024 + position
        per_word: tl.constexpr = 32 // BITS
        word = tl.load(
            packed_ptr + (k // 16).to(tl.int64) * packed_cols + flat // per_word,
            mask=mask,
            other=0,
        )
        shift = ((flat % per_word) * BITS).to(tl.int32)
        code = ((word >> shift) & ((1 << BITS) - 1)) - (1 << (BITS - 1))
        column = (n // 64) * 64 + (n % 8) * 8 + (n % 64) // 8
        scale = tl.load(scale_ptr + (k // GROUP).to(tl.int64) * scale_cols + column, mask=mask)
        out = code.to(tl.float32) * scale.to(tl.float32)
        tl.store(out_ptr + row * hidden + k, out.to(out_ptr.dtype.element_ty), mask=mask)


def gather(
    ids: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    table: torch.Tensor,
    bits: int,
    hidden: int,
) -> torch.Tensor:
    """Rows `ids` [T] of a Marlin-repacked table, dequantized: [T, hidden] in the scales' dtype.
    `table` is position_table(bits) on the weight's device."""
    if packed.shape[0] * TILE != hidden or scales.shape[1] * bits != packed.shape[1] * 32 // TILE:
        raise ValueError(
            f"not a Marlin-repacked weight for {hidden} inputs: {tuple(packed.shape)}, "
            f"scales {tuple(scales.shape)}"
        )
    if not packed.is_cuda:
        return gather_reference(ids, packed, scales, bits, hidden)
    flat = ids.reshape(-1).contiguous()
    out = torch.empty(flat.numel(), hidden, dtype=scales.dtype, device=packed.device)
    block = min(triton.next_power_of_2(hidden), 1024)
    grid = (flat.numel(), triton.cdiv(hidden, block))
    _marlin_rows_kernel[grid](
        flat,
        packed,
        scales,
        table,
        out,
        hidden,
        packed.shape[1],
        scales.shape[1],
        BITS=bits,
        GROUP=hidden // scales.shape[0],
        BLOCK=block,
    )
    return out
