"""Blaze's byte market at a fixed size: which units step down from the upper option to the lower.

Every body unit starts at the upper option (6-bit g128 on the 4B). When that is over the byte
budget, some units must take the lower option (4-bit g128). Each unit's cost of stepping down is
its measured price at the lower option minus its price at the upper one (KL to BF16 with that
unit alone quantized; the drift study found such prices add up), and stepping down saves the
difference in stored bytes. The cheapest set of steps that saves enough is an exact min-cost
covering knapsack over the saved bytes in fixed-size bins, rounded down so the result always fits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

BIN_BYTES = 64 * 1024


@dataclass(frozen=True)
class Step:
    unit: str
    cost: float  # price at the lower option minus price at the upper, >= 0 after clipping
    saved: int  # bytes saved by the lower option


def steps(
    upper: Mapping[str, float], lower: Mapping[str, float], saved: Mapping[str, int]
) -> list[Step]:
    if set(upper) != set(lower) or set(upper) != set(saved):
        raise ValueError("prices and savings must name the same units")
    return [Step(u, max(0.0, lower[u] - upper[u]), int(saved[u])) for u in sorted(upper)]


def cheapest_cover(options: list[Step], deficit: int, bin_bytes: int = BIN_BYTES) -> list[str]:
    """The units to step down: least total cost whose savings reach `deficit` bytes."""
    if deficit <= 0:
        return []
    need = -(-deficit // bin_bytes)
    if sum(step.saved // bin_bytes for step in options) < need:
        raise ValueError(f"stepping every unit down saves less than {deficit:,} bytes")
    inf = float("inf")
    # best[b]: least cost reaching at least b bins; choice[i][b]: whether item i was taken.
    best = [0.0] + [inf] * need
    choice: list[list[bool]] = []
    for step in options:
        size = step.saved // bin_bytes
        taken = [False] * (need + 1)
        for bins in range(need, 0, -1):
            source = max(0, bins - size)
            if size and best[source] + step.cost < best[bins]:
                best[bins] = best[source] + step.cost
                taken[bins] = True
        choice.append(taken)
    chosen: list[str] = []
    bins = need
    for index in range(len(options) - 1, -1, -1):
        if bins > 0 and choice[index][bins]:
            chosen.append(options[index].unit)
            bins = max(0, bins - options[index].saved // bin_bytes)
    if bins > 0:
        raise AssertionError("knapsack backtrack did not reach the deficit")
    return sorted(chosen)


def best_fill(
    gains: Mapping[str, float], costs: Mapping[str, int], budget: int, bin_bytes: int = BIN_BYTES
) -> list[str]:
    """The units to step up: most total gain whose extra bytes stay within `budget`.

    An exact 0/1 knapsack over costs rounded up to whole bins, so the result always fits.
    """
    if set(gains) != set(costs):
        raise ValueError("gains and costs must name the same units")
    capacity = max(0, budget) // bin_bytes
    names = sorted(gains)
    sizes = [-(-int(costs[name]) // bin_bytes) for name in names]
    best = [0.0] * (capacity + 1)
    taken = [[False] * (capacity + 1) for _ in names]
    for index, name in enumerate(names):
        size, gain = sizes[index], max(0.0, gains[name])
        for room in range(capacity, size - 1, -1):
            if best[room - size] + gain > best[room]:
                best[room] = best[room - size] + gain
                taken[index][room] = True
    chosen: list[str] = []
    room = capacity
    for index in range(len(names) - 1, -1, -1):
        if taken[index][room]:
            chosen.append(names[index])
            room -= sizes[index]
    return sorted(chosen)
