"""Byte-budgeted precision allocation: which storage option each language-model unit gets.

A unit is the set of Linear layers vLLM fuses into one kernel (q/k/v, gate/up, the DeltaNet
in_proj_qkv/in_proj_z and in_proj_b/in_proj_a); they must share one option. Each unit's predicted
KL saving for an option comes from a Kronecker-factored diagonal Fisher (output-gradient and
input second moments) and the round-to-nearest error of the option against 4-bit g128. A
multiple-choice knapsack then picks one option per unit within the byte budget.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from qwen35_compression.glaze2.quant import BASE, OPTIONS, Option, rtn_error

# Members vLLM 0.29 fuses for Qwen3.5 (its packed_modules_mapping), by their last name.
FUSED = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z"),
    ("in_proj_b", "in_proj_a"),
)


def unit_of(name: str) -> str:
    """The unit a Linear belongs to: its parent path and the fused members it shares a kernel
    with, e.g. `layers.3.self_attn.q_proj` -> `layers.3.self_attn.{q_proj,k_proj,v_proj}`."""
    parent, _, leaf = name.rpartition(".")
    for members in FUSED:
        if leaf in members:
            return f"{parent}.{{{','.join(members)}}}"
    return name


def group_units(names: Iterable[str]) -> dict[str, list[str]]:
    units: dict[str, list[str]] = {}
    for name in names:
        units.setdefault(unit_of(name), []).append(name)
    return units


@dataclass
class FisherFactors:
    """Per Linear: mean squared output gradient per output channel, and mean squared input per
    input channel, over the calibration tokens."""

    output: torch.Tensor
    input: torch.Tensor


@dataclass
class Unit:
    name: str
    members: list[str]
    shapes: list[tuple[int, int]]
    # Option -> predicted KL saving against 4-bit g128 (0 for the base itself).
    saving: dict[Option, float] = field(default_factory=dict)

    def bytes(self, option: Option) -> int:
        return sum(option.tensor_bytes(rows, columns) for rows, columns in self.shapes)

    def options(self) -> list[Option]:
        """Options every member's input width allows (groups must divide it)."""
        return [o for o in OPTIONS if all(columns % o.group == 0 for _, columns in self.shapes)]


def predicted_saving(weight: torch.Tensor, factors: FisherFactors, option: Option) -> float:
    """1/2 * sum_rj a_r b_j (e_base^2 - e_option^2): the drop in a second-order KL estimate."""
    if option == BASE:
        return 0.0
    a = factors.output.float().to(weight.device)
    b = factors.input.float().to(weight.device)
    difference = rtn_error(weight, BASE).square() - rtn_error(weight, option).square()
    return float(0.5 * (a @ difference @ b))


def build_units(weights: dict[str, torch.Tensor], fisher: dict[str, FisherFactors]) -> list[Unit]:
    units = []
    for unit_name, members in sorted(group_units(weights).items()):
        unit = Unit(unit_name, members, [tuple(weights[m].shape) for m in members])
        for option in unit.options():
            unit.saving[option] = sum(
                predicted_saving(weights[m], fisher[m], option) for m in members
            )
        units.append(unit)
    return units


@dataclass
class Allocation:
    choice: dict[str, Option]
    extra_bytes: int
    predicted_saving: float

    def option_of(self, linear: str) -> Option:
        return self.choice[unit_of(linear)]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for option in self.choice.values():
            counts[option.name] = counts.get(option.name, 0) + 1
        return dict(sorted(counts.items()))


def allocate(units: Sequence[Unit], budget: int, resolution: int = 1 << 16) -> Allocation:
    """One option per unit maximizing the predicted saving, with extra bytes over 4-bit g128 at
    most `budget`. Exact dynamic programming over bins of `resolution` bytes (costs rounded up,
    so the budget always holds)."""
    if budget < 0:
        raise ValueError(f"negative byte budget: {budget}")
    bins = budget // resolution
    best = np.full(bins + 1, -np.inf)
    best[0] = 0.0
    picks: list[np.ndarray] = []
    for unit in units:
        base = unit.bytes(BASE)
        choices = [o for o in unit.options() if o in unit.saving]
        if BASE not in choices:
            raise ValueError(f"{unit.name} cannot take 4-bit g128")
        new = np.full(bins + 1, -np.inf)
        pick = np.zeros(bins + 1, dtype=np.int64)
        for index, option in enumerate(choices):
            cost = -(-(unit.bytes(option) - base) // resolution)  # ceiling
            if cost > bins:
                continue
            shifted = np.full(bins + 1, -np.inf)
            shifted[cost:] = best[: bins + 1 - cost] + unit.saving[option]
            better = shifted > new
            new[better] = shifted[better]
            pick[better] = index
        best = new
        picks.append(pick)
    end = int(np.argmax(best))
    choice: dict[str, Option] = {}
    extra = 0
    for unit, pick in zip(reversed(units), reversed(picks), strict=True):
        choices = [o for o in unit.options() if o in unit.saving]
        option = choices[int(pick[end])]
        choice[unit.name] = option
        extra += unit.bytes(option) - unit.bytes(BASE)
        end -= -(-(unit.bytes(option) - unit.bytes(BASE)) // resolution)
    total = sum(units_by_name(units)[name].saving[o] for name, o in choice.items())
    return Allocation(dict(sorted(choice.items())), extra, total)


def units_by_name(units: Sequence[Unit]) -> dict[str, Unit]:
    return {unit.name: unit for unit in units}


def uniform(units: Sequence[Unit], option: Option = BASE) -> Allocation:
    return Allocation({unit.name: option for unit in units}, 0, 0.0)


def layer_index(name: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None
