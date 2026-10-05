"""Symmetric INT group quantization as compressed-tensors stores it, and its learned form.

A weight row is cut into groups of `group` inputs. Each group has one BF16 scale s and integer
codes q in [-2^(k-1), 2^(k-1) - 1]; the weight is s * q, computed in BF16. Glaze v2 learns, per
weight, a rounding offset V in [-0.5, 0.5] and, per group, a clip factor alpha in [0.5, 1]
(AutoRound's two variables), with the scale cast to BF16 inside the forward pass so the optimized
model is exactly the one the export stores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

SCALE_DTYPE = torch.bfloat16
# AutoRound's clip range: a group's scale may shrink to half its round-to-nearest value.
ALPHA_MIN = 0.5


@dataclass(frozen=True, order=True)
class Option:
    """One storage choice for a tensor: bits per code and inputs per scale."""

    bits: int
    group: int

    def bytes_per_weight(self) -> float:
        """Packed codes plus one BF16 scale per group."""
        return self.bits / 8 + 2 / self.group

    def tensor_bytes(self, rows: int, columns: int) -> int:
        return rows * columns * self.bits // 8 + rows * (columns // self.group) * 2

    @property
    def name(self) -> str:
        return f"w{self.bits}g{self.group}"


BASE = Option(4, 128)
# vLLM 0.29's Marlin kernels take 4 or 8 bits and groups of 32, 64 or 128.
OPTIONS = (Option(4, 128), Option(4, 64), Option(4, 32), Option(8, 128))


def code_range(bits: int) -> tuple[int, int]:
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _grouped(weight: torch.Tensor, group: int) -> torch.Tensor:
    rows, columns = weight.shape
    if columns % group:
        raise ValueError(f"{columns} inputs are not a multiple of group {group}")
    return weight.reshape(rows, columns // group, group)


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest, with the gradient of the identity."""
    return x + (torch.round(x) - x).detach()


def ste_bf16(x: torch.Tensor) -> torch.Tensor:
    """Cast to the BF16 grid and back, with the gradient of the identity."""
    return x + (x.to(SCALE_DTYPE).to(x.dtype) - x).detach()


def group_scales(
    weight: torch.Tensor, option: Option, alpha: torch.Tensor | None = None
) -> torch.Tensor:
    """BF16-grid scales (kept in the weight's dtype): max |w| of each group over 2^(k-1) - 1,
    shrunk by alpha. A group of zeros gets the smallest BF16 normal so codes stay finite."""
    _, top = code_range(option.bits)
    peak = _grouped(weight, option.group).abs().amax(dim=-1)
    if alpha is not None:
        peak = peak * alpha
    scale = ste_bf16(peak / top)
    tiny = torch.finfo(SCALE_DTYPE).tiny
    return torch.where(scale > 0, scale, torch.full_like(scale, tiny))


def quantize(
    weight: torch.Tensor,
    option: Option,
    offsets: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer codes (as a float tensor, differentiable through the STE) and their scales."""
    low, high = code_range(option.bits)
    scale = group_scales(weight, option, alpha)
    ratio = _grouped(weight, option.group) / scale.unsqueeze(-1)
    if offsets is not None:
        ratio = ratio + _grouped(offsets, option.group)
    codes = torch.clamp(ste_round(ratio), low, high)
    return codes.reshape(weight.shape), scale


def dequantize(codes: torch.Tensor, scale: torch.Tensor, option: Option) -> torch.Tensor:
    """BF16 weights as compressed-tensors computes them: code times BF16 scale, in BF16."""
    grouped = _grouped(codes, option.group).to(SCALE_DTYPE)
    weight = grouped * scale.to(SCALE_DTYPE).unsqueeze(-1)
    return weight.reshape(codes.shape)


def fake_quantize(weight: torch.Tensor, option: Option) -> torch.Tensor:
    """Round-to-nearest quantization, back in the weight's dtype."""
    with torch.no_grad():
        codes, scale = quantize(weight.float(), option)
        return dequantize(codes, scale, option).to(weight.dtype)


def rtn_error(weight: torch.Tensor, option: Option) -> torch.Tensor:
    """W - Q(W) under round-to-nearest, in fp32."""
    return weight.float() - fake_quantize(weight, option).float()


class LearnedQuantLinear(nn.Module):
    """A bias-free Linear whose weight is quantized with learned rounding and clipping.

    `offsets` (one per weight, in [-0.5, 0.5]) and `alpha` (one per group, in [0.5, 1]) are the
    trained variables; the BF16 weight is frozen. `export()` gives the codes and scales the
    forward pass computed with.
    """

    def __init__(self, linear: nn.Linear, option: Option) -> None:
        super().__init__()
        if linear.bias is not None:
            raise ValueError("Glaze v2 quantizes bias-free Linear layers only")
        rows, columns = linear.weight.shape
        if columns % option.group:
            raise ValueError(f"{columns} inputs are not a multiple of group {option.group}")
        self.option = option
        self.out_features, self.in_features = rows, columns
        self.register_buffer("weight", linear.weight.detach().clone(), persistent=False)
        device = linear.weight.device
        self.offsets = nn.Parameter(torch.zeros(rows, columns, device=device))
        self.alpha = nn.Parameter(torch.ones(rows, columns // option.group, device=device))

    def clamp_(self) -> None:
        with torch.no_grad():
            self.offsets.clamp_(-0.5, 0.5)
            self.alpha.clamp_(ALPHA_MIN, 1.0)

    def quantized_weight(self) -> torch.Tensor:
        codes, scale = quantize(self.weight.float(), self.option, self.offsets, self.alpha)
        grouped = _grouped(codes, self.option.group) * scale.unsqueeze(-1)
        # Rounded to BF16 as the kernel's product is, with the identity's gradient.
        return ste_bf16(grouped.reshape(codes.shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.quantized_weight().to(inputs.dtype))

    @torch.no_grad()
    def export(self) -> tuple[torch.Tensor, torch.Tensor]:
        """int8 codes and BF16 scales, exactly what the forward pass used."""
        codes, scale = quantize(self.weight.float(), self.option, self.offsets, self.alpha)
        return codes.to(torch.int8), scale.to(SCALE_DTYPE)
