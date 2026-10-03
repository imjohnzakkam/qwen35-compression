"""Glaze training data: the init's packed calibration blocks, split into train and dev."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from qwen35_compression.config import CalibrationConfig, GlazeConfig


def calibration_blocks(processor: Any, calibration: CalibrationConfig) -> list[list[int]]:
    """Every packed calibration block, in the order AutoRound sees them."""
    from qwen35_compression.calibration import build_calibration_dataset

    dataset, _ = build_calibration_dataset(processor, calibration, pack=True)
    # Packed rows hold one sequence each: input_ids is [[block]].
    return [list(row[0]) for row in dataset["input_ids"]]


def split_blocks(
    blocks: Sequence[Sequence[int]], glaze: GlazeConfig
) -> tuple[list[list[int]], list[list[int]]]:
    """The first `train_blocks` (the init's own calibration) and the next `dev_blocks`."""
    needed = glaze.train_blocks + glaze.dev_blocks
    if len(blocks) < needed:
        raise ValueError(f"{len(blocks)} calibration blocks, {needed} needed for train and dev")
    lengths = {len(block) for block in blocks[:needed]}
    if len(lengths) != 1:
        raise ValueError(f"calibration blocks differ in length: {sorted(lengths)}")
    train = [list(block) for block in blocks[: glaze.train_blocks]]
    dev = [list(block) for block in blocks[glaze.train_blocks : needed]]
    return train, dev


def epoch_order(count: int, epoch: int, seed: int) -> list[int]:
    """A fixed shuffle of block indices for one epoch."""
    import torch

    generator = torch.Generator().manual_seed(seed * 1009 + epoch)
    return torch.randperm(count, generator=generator).tolist()


def groups(indices: Sequence[int], size: int) -> list[list[int]]:
    """Consecutive groups of `size`; the caller guarantees an exact fit."""
    if size <= 0 or len(indices) % size:
        raise ValueError(f"{len(indices)} items do not split into groups of {size}")
    return [list(indices[start : start + size]) for start in range(0, len(indices), size)]


def blocks_digest(blocks: Sequence[Sequence[int]]) -> str:
    """SHA-256 of the token ids, recorded so a run's training data can be checked later."""
    payload = json.dumps([list(block) for block in blocks], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
