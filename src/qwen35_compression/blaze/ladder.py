"""Stage 2: the vocabulary ladder. Busy rows keep 8 bits; every other row drops to 4.

Rows are renumbered by use, so each rung is one contiguous block and the LM head runs one
GEMM per rung; an index map (one int32 per row) sends token ids to stored rows and back.
Numerically the ladder is per-row symmetric round-to-nearest at two precisions, so the
fake-quantized table (the stored values, written back in BF16) gives exactly the outputs a
tiered kernel would. That is what the quality gates score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from qwen35_compression.blaze.census import Census
from qwen35_compression.glaze2.quant import Option, fake_quantize

HOT_OPTION = Option(8, 128)
REST_OPTION = Option(4, 128)
# The 8-bit rung covers this share of the census's tokens, and never fewer rows than AT_LEAST.
DEFAULT_COVER = 0.999
DEFAULT_AT_LEAST = 65_536
INDEX_BYTES = 4
CHUNK_ROWS = 16_384


@dataclass(frozen=True)
class LadderPlan:
    """Row ids in stored order, busiest first, and how many of them form the 8-bit rung."""

    order: tuple[int, ...]
    hot_rows: int
    hot_option: Option = HOT_OPTION
    rest_option: Option = REST_OPTION

    def __post_init__(self) -> None:
        if sorted(self.order) != list(range(len(self.order))):
            raise ValueError("a ladder's order must be a permutation of the vocabulary")
        if not 0 <= self.hot_rows <= len(self.order):
            raise ValueError(f"hot_rows {self.hot_rows} is outside 0..{len(self.order)}")

    @property
    def vocab_size(self) -> int:
        return len(self.order)

    def hot_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        mask[list(self.order[: self.hot_rows])] = True
        return mask

    def table_bytes(self, hidden: int) -> int:
        """Bytes of the laddered table as a tiered export stores it, index map included."""
        rest = self.vocab_size - self.hot_rows
        return (
            self.hot_option.tensor_bytes(self.hot_rows, hidden)
            + self.rest_option.tensor_bytes(rest, hidden)
            + INDEX_BYTES * self.vocab_size
        )

    def record(self, hidden: int) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hot_rows": self.hot_rows,
            "hot_option": self.hot_option.name,
            "rest_option": self.rest_option.name,
            "table_bytes": self.table_bytes(hidden),
            "bf16_bytes": self.vocab_size * hidden * 2,
        }


def plan_ladder(
    census: Census,
    cover: float = DEFAULT_COVER,
    at_least: int = DEFAULT_AT_LEAST,
    sources: list[str] | None = None,
) -> LadderPlan:
    """The smallest prefix of the usage ranking holding `cover` of the census's tokens, raised to
    `at_least` rows (and capped at the vocabulary)."""
    if not 0 < cover <= 1:
        raise ValueError(f"cover must be in (0, 1], got {cover}")
    order = census.ranking(sources)
    counts = census.total(sources)[order]
    total = int(counts.sum())
    if total == 0:
        raise ValueError("the census has no tokens")
    reached = int(np.searchsorted(np.cumsum(counts), cover * total, side="left")) + 1
    hot = min(max(reached, at_least), census.vocab_size)
    return LadderPlan(tuple(int(row) for row in order), hot)


def ladder_table(
    weight: torch.Tensor, plan: LadderPlan, chunk_rows: int = CHUNK_ROWS
) -> torch.Tensor:
    """The table with every row fake-quantized at its rung's precision, in token order."""
    if weight.dim() != 2 or weight.shape[0] != plan.vocab_size:
        raise ValueError(f"table shape {tuple(weight.shape)} does not match {plan.vocab_size} rows")
    mask = plan.hot_mask().to(weight.device)
    out = torch.empty_like(weight)
    for option, rows in (
        (plan.hot_option, mask.nonzero().squeeze(-1)),
        (plan.rest_option, (~mask).nonzero().squeeze(-1)),
    ):
        for start in range(0, rows.numel(), chunk_rows):
            index = rows[start : start + chunk_rows]
            out[index] = fake_quantize(weight[index], option)
    return out


def renumber(weight: torch.Tensor, plan: LadderPlan) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows in stored order (the 8-bit rung first) and the index map: token id -> stored row."""
    order = torch.tensor(plan.order, dtype=torch.long, device=weight.device)
    index = torch.empty_like(order)
    index[order] = torch.arange(order.numel(), device=weight.device)
    return weight[order], index.to(torch.int32)


def restore(stored: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Token order back from stored order."""
    return stored[index.long()]
