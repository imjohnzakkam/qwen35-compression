"""Serve a checkpoint's 6-bit groups on vLLM's 8-bit kernels, with the same codes and scales.

A 6-bit code q in [-32, 31] is also an 8-bit code, so a 6-bit group (symmetric, group 128) runs
exactly on the 8-bit Marlin kernel once its words are repacked: the file keeps 6 bits per weight,
and the GPU holds 8. compressed-tensors packs B-bit codes densely along the input dimension,
offset to u = q + 2^(B-1): element j of each run of 32 starts at bit B * j of the run's B words,
and may cross into the next word. With `BLAZE_W6_AS_W8=1`, the plugin builds 8-bit layers for
the 6-bit groups (`widen_schemes`) and repacks their words as they load (`widened`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

ENV = "BLAZE_W6_AS_W8"
FROM_BITS, TO_BITS = 6, 8
PACKED = ".weight_packed"


def six_bit_modules(quantization_config: Mapping[str, Any]) -> set[str]:
    """Checkpoint module names in the config's 6-bit groups (named targets only)."""
    modules: set[str] = set()
    for group in quantization_config.get("config_groups", {}).values():
        if (group.get("weights") or {}).get("num_bits") != FROM_BITS:
            continue
        for target in group["targets"]:
            if target.startswith("re:") or target == "Linear":
                raise ValueError(f"6-bit groups must name their modules, not {target!r}")
            modules.add(target)
    return modules


def widen_schemes(quant_config: Any) -> int:
    """Make vLLM's compressed-tensors config build 8-bit layers where the checkpoint has 6-bit
    groups; return how many targets changed. Call before the model's layers are built."""
    changed = 0
    for scheme in getattr(quant_config, "target_scheme_map", {}).values():
        weights = scheme.get("weights")
        if weights is not None and weights.num_bits == FROM_BITS:
            scheme["weights"] = weights.model_copy(update={"num_bits": TO_BITS})
            changed += 1
    return changed


def _codes(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Dense B-bit words [n, k * B / 32] -> unsigned codes u [n, k] (int64)."""
    rows, words = packed.shape
    if words % bits:
        raise ValueError(f"{words} words do not hold whole runs of 32 {bits}-bit codes")
    runs = words // bits
    unsigned = packed.to(torch.int64) & 0xFFFFFFFF
    unsigned = unsigned.reshape(rows, runs, bits)
    start = torch.arange(32, device=packed.device) * bits
    word, offset = start // 32, start % 32
    low = unsigned[..., word] >> offset
    # The next word's low bits, for codes that cross a word boundary (zero where none do).
    spill = torch.where(offset + bits > 32, 32 - offset, 64)
    nxt = (word + 1).clamp(max=bits - 1)
    high = torch.where(spill < 64, unsigned[..., nxt] << spill.clamp(max=63), 0)
    return ((low | high) & ((1 << bits) - 1)).reshape(rows, runs * 32)


def _words(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Unsigned codes u [n, k] -> dense B-bit int32 words, for B dividing 32."""
    rows, columns = codes.shape
    per_word = 32 // bits
    shifts = torch.arange(per_word, device=codes.device) * bits
    words = (codes.reshape(rows, columns // per_word, per_word) << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def widen_packed(packed: torch.Tensor) -> torch.Tensor:
    """A 6-bit group's words -> the 8-bit words of the same signed codes."""
    unsigned = _codes(packed, FROM_BITS)
    shift = (1 << (TO_BITS - 1)) - (1 << (FROM_BITS - 1))
    return _words(unsigned + shift, TO_BITS)


def widened(
    weights: Iterable[tuple[str, torch.Tensor]], modules: set[str]
) -> Iterator[tuple[str, torch.Tensor]]:
    """The checkpoint's tensors, with the words of `modules` (6-bit) repacked as 8-bit."""
    for name, tensor in weights:
        if name.endswith(PACKED) and name.removesuffix(PACKED) in modules:
            tensor = widen_packed(tensor)
        yield name, tensor
