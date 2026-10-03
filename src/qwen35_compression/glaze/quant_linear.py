"""A Linear layer made of frozen INT4 codes and trainable per-group scales."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

INT4_MIN, INT4_MAX = -8, 7


def unpack_int4(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """compressed-tensors' packed int32 words to int8 codes in [-8, 7]."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    return unpack_from_int32(packed, 4, torch.Size(shape))


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """int8 codes in [-8, 7] to compressed-tensors' packed int32 words."""
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

    return pack_to_int32(codes, 4)


def dequantize_groups(codes: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """Weights as compressed-tensors dequantizes them: codes cast to the scale dtype, times scale.

    For BF16 scales both operands are exact, so the product is rounded once to BF16, bit for bit
    what `compressed_tensors.quantization.lifecycle.forward.dequantize` returns.
    """
    rows, columns = codes.shape
    grouped = codes.view(rows, columns // group_size, group_size).to(scales.dtype)
    return (grouped * scales.unsqueeze(-1)).view(rows, columns)


class GroupQuantLinear(nn.Module):
    """W4A16 Linear: frozen INT4 codes times trainable group scales, without bias.

    `scales` holds the scales in fp32 at values of their storage dtype's grid, so the forward
    pass computes exactly the weights the exported scales give. Glaze moves them one grid step
    at a time (see glaze.grid); `initial_scales` keeps the export's values for comparison.
    """

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor, group_size: int) -> None:
        super().__init__()
        if codes.dtype is not torch.int8 or codes.ndim != 2:
            raise ValueError("codes must be a 2-D int8 tensor")
        rows, columns = codes.shape
        if group_size <= 0 or columns % group_size:
            raise ValueError(f"{columns} input features are not a multiple of group {group_size}")
        if not scales.is_floating_point() or tuple(scales.shape) != (rows, columns // group_size):
            raise ValueError(
                f"scales must be floating point with shape {(rows, columns // group_size)}, "
                f"got {scales.dtype} {tuple(scales.shape)}"
            )
        if int(codes.min()) < INT4_MIN or int(codes.max()) > INT4_MAX:
            raise ValueError("codes are outside the INT4 range [-8, 7]")
        # Scales may be negative: AutoRound's symmetric INT4 uses the full [-8, 7] range, giving
        # each group the sign of its largest-magnitude weight. Grid steps keep the sign.
        if not bool(torch.isfinite(scales).all()):
            raise ValueError("group scales must be finite")
        self.in_features = columns
        self.out_features = rows
        self.group_size = group_size
        self.storage_dtype = scales.dtype
        self.register_buffer("codes", codes.contiguous())
        self.register_buffer("initial_scales", scales.detach().clone().contiguous())
        working = torch.float64 if scales.dtype is torch.float64 else torch.float32
        self.scales = nn.Parameter(scales.detach().to(working).contiguous())

    def stored_scales(self) -> torch.Tensor:
        """The scales as the export stores them; exact, and differentiable in `scales`."""
        return self.scales.to(self.storage_dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = dequantize_groups(self.codes, self.stored_scales(), self.group_size)
        return F.linear(inputs, weight.to(inputs.dtype))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, storage_dtype={self.storage_dtype}"
        )
