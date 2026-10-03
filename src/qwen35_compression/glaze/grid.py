"""Descent on a number grid: Glaze's optimizer for values stored in BF16.

Every trained value (group scales, norm weights) is stored in BF16 and starts exactly on a BF16
grid point. A continuous optimizer moves all of them by about the same amount per step; none
changes until it crosses half a grid step, and then most change at once. With 27.9M AutoRound
scales that is a full-ulp jump (0.4-0.8%) of most scales in one step, which raised the KL 12x
in B1's first pilot. Here each step instead moves only the values with the largest predicted
loss decrease, each by exactly one grid step, so the change per step is bounded and every
intermediate model is one the export can store.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

SIXTEEN_BIT = (torch.bfloat16, torch.float16)


def grid_neighbor(
    values: torch.Tensor, direction: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """The next point of `dtype`'s grid from `values` in `direction` (+1 up, -1 down).

    `values` must already lie on the grid. A step that would reach zero, change sign or
    overflow, or a zero direction, gives NaN: such a value never moves.
    """
    stored = values.to(dtype)
    if dtype in SIXTEEN_BIT:
        # IEEE sign-magnitude: in the raw 16-bit pattern, +1 is one step away from zero and -1
        # one step towards it, for either sign.
        away = torch.sign(stored.float()) == direction
        bits = stored.view(torch.int16).to(torch.int32) + torch.where(away, 1, -1)
        result = bits.to(torch.int16).view(dtype).to(values.dtype)
    else:
        limit = torch.where(direction > 0, float("inf"), float("-inf")).to(dtype)
        result = torch.nextafter(stored, limit).to(values.dtype)
    invalid = (
        ~torch.isfinite(result)
        | (result == 0)
        | (values == 0)
        | (direction == 0)
        | (torch.sign(result) != torch.sign(values))
    )
    return torch.where(invalid, torch.full_like(result, float("nan")), result)


class GridDescent:
    """Each step, move the `flips` values with the largest predicted decrease one grid step.

    The prediction is first order: -m * (neighbor - value), with m an exponential moving
    average of the gradient. A value that moves has its average reset, so it moves again only
    on fresh evidence. Values never leave the grid, change sign or reach zero.
    """

    def __init__(
        self, parameters: Sequence[torch.Tensor], dtype: torch.dtype, momentum: float = 0.9
    ) -> None:
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("nothing to optimize")
        self.dtype = dtype
        self.momentum = momentum
        self.averages = [torch.zeros_like(p) for p in self.parameters]
        with torch.no_grad():
            for parameter in self.parameters:
                if not torch.equal(parameter.to(dtype).to(parameter.dtype), parameter):
                    raise ValueError(f"parameters must start on the {dtype} grid")

    @property
    def size(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self, flips: int) -> int:
        """Apply up to `flips` one-step moves; return how many were made."""
        gains, targets = [], []
        for parameter, average in zip(self.parameters, self.averages, strict=True):
            if parameter.grad is not None:
                average.mul_(self.momentum).add_(parameter.grad, alpha=1 - self.momentum)
            direction = -torch.sign(average)
            target = grid_neighbor(parameter, direction, self.dtype)
            gain = -average * (target - parameter)
            gains.append(torch.nan_to_num(gain, nan=float("-inf")).flatten())
            targets.append(target)
        everything = torch.cat(gains)
        count = min(max(int(flips), 0), everything.numel())
        if count == 0:
            return 0
        best = torch.topk(everything, count)
        chosen = best.indices[best.values > 0]
        start = 0
        for parameter, average, target in zip(self.parameters, self.averages, targets, strict=True):
            end = start + parameter.numel()
            local = chosen[(chosen >= start) & (chosen < end)] - start
            if local.numel():
                parameter.view(-1)[local] = target.view(-1)[local]
                average.view(-1)[local] = 0
            start = end
        return int(chosen.numel())
