"""Stage 4's measurement: what one unit's precision costs, as KL to BF16 on generic text.

The BF16 model's final hidden states on the replay blocks are computed once. Then one unit at a
time is fake-quantized in place, the blocks are run again, the KL from BF16's next-token
distribution is measured at every position, and the unit is restored, so each price is that
unit's alone. The drift study found per-component excess losses add up (101% of the whole),
which is what lets a knapsack sum these prices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from qwen35_compression.blaze.body import swapped, unit_size
from qwen35_compression.glaze.losses import DistillStats, distill_step
from qwen35_compression.glaze2.quant import Option


def pack_text(token_rows: Sequence[Sequence[int]], length: int, count: int) -> list[list[int]]:
    """Join tokenized documents in order and cut `count` blocks of `length` tokens."""
    need = length * count
    stream: list[int] = []
    for row in token_rows:
        stream.extend(int(token) for token in row)
        if len(stream) >= need:
            break
    if len(stream) < need:
        raise ValueError(f"{len(stream)} tokens, {need} needed for {count} blocks of {length}")
    return [stream[start : start + length] for start in range(0, need, length)]


@torch.no_grad()
def final_states(
    model: nn.Module, blocks: Sequence[Sequence[int]], batch: int
) -> list[torch.Tensor]:
    device = next(model.parameters()).device
    states = []
    for start in range(0, len(blocks), batch):
        ids = torch.tensor(blocks[start : start + batch], device=device)
        states.append(model(input_ids=ids, use_cache=False).last_hidden_state)
    return states


@torch.no_grad()
def kl_to_reference(
    model: nn.Module,
    head_weight: torch.Tensor,
    blocks: Sequence[Sequence[int]],
    reference: Sequence[torch.Tensor],
    batch: int,
    chunk_tokens: int,
) -> DistillStats:
    """KL(reference || model) over every position of the blocks, through one shared LM head."""
    stats = DistillStats()
    for index, start in enumerate(range(0, len(blocks), batch)):
        ids = torch.tensor(blocks[start : start + batch], device=head_weight.device)
        hidden = model(input_ids=ids, use_cache=False).last_hidden_state
        mask = torch.ones(hidden.shape[:-1], dtype=torch.bool, device=hidden.device)
        stats.merge(
            distill_step(
                hidden,
                reference[index],
                head_weight,
                mask,
                chunk_tokens,
                float(mask.sum()),
                backward=False,
            )
        )
    return stats


def measure_units(
    model: nn.Module,
    head_weight: torch.Tensor,
    blocks: Sequence[Sequence[int]],
    units: dict[str, list[str]],
    option: Option,
    batch: int = 4,
    chunk_tokens: int = 512,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    """Each unit's price: mean KL to BF16 with that unit alone at `option`."""
    reference = final_states(model, blocks, batch)
    prices: dict[str, dict[str, Any]] = {}
    for unit, names in units.items():
        with swapped(model, names, option):
            stats = kl_to_reference(model, head_weight, blocks, reference, batch, chunk_tokens)
        prices[unit] = {
            "members": list(names),
            "weights": unit_size(model, names),
            **stats.summary(),
        }
        log(f"blaze price: {unit} at {option.name}: mean KL {stats.mean_kl:.6f}")
    return prices


def concentration(prices: dict[str, float], sizes: dict[str, int], share: float = 0.5) -> float:
    """Share of the summed price held by the units with the highest price per weight that make up
    `share` of all the weights; the unit straddling that line counts in proportion.

    Near 1, a few units hold the error and a 4/8-bit menu can buy it back cheaply. Near `share`,
    the error is spread evenly and only a finer menu (6 bits) helps.
    """
    if set(prices) != set(sizes) or not prices:
        raise ValueError("prices and sizes must name the same, non-empty set of units")
    total_price = sum(prices.values())
    total_size = sum(sizes.values())
    if total_price <= 0 or total_size <= 0:
        raise ValueError("prices and sizes must have positive totals")
    budget = share * total_size
    held = 0.0
    for unit in sorted(prices, key=lambda name: prices[name] / sizes[name], reverse=True):
        take = min(sizes[unit], budget)
        held += prices[unit] * take / sizes[unit]
        budget -= take
        if budget <= 0:
            break
    return held / total_price
