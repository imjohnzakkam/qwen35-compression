"""Why a Glaze step raises the KL: measurements on one fixed batch, before any full run.

B1's second pilot moved the 2,805 most promising values one BF16 grid step each and the KL on
its fixed batch rose 32%. For a set of one-step moves, this measures the KL after them (downhill,
against the gradient) and after the opposite moves (uphill). Their difference gives the slope
actually seen along the moves, to compare with the gradient's prediction; their sum gives the
curvature, and so how large a fraction of one grid step would have been best. Every evaluated
value is one the export can store. It also records which tensors the moves come from, and runs
short pilots of candidate settings, each from the init.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from qwen35_compression.glaze.grid import guarded_neighbor
from qwen35_compression.glaze.student import family, load_trainable_state, trainable_state
from qwen35_compression.glaze.train import GlazeTrainer, Schedule

PILOT_STEPS = 5


def is_scale(name: str) -> bool:
    return name.endswith(".scales")


def is_norm(name: str) -> bool:
    return not is_scale(name)


@dataclass
class MoveSet:
    """One grid step for chosen values: per trainable tensor, flat indices and both targets."""

    name: str
    indices: list[torch.Tensor]
    originals: list[torch.Tensor]
    downhill: list[torch.Tensor]
    uphill: list[torch.Tensor]
    predicted_down: float
    predicted_up: float

    @property
    def size(self) -> int:
        return sum(int(index.numel()) for index in self.indices)

    @torch.no_grad()
    def apply(self, parameters: Sequence[torch.Tensor], targets: list[torch.Tensor]) -> None:
        for parameter, index, target in zip(parameters, self.indices, targets, strict=True):
            if index.numel():
                parameter.view(-1)[index] = target


class Neighbors:
    """Both grid neighbors of every trainable value and the predicted gain of the downhill one.

    Unguarded, a value may move wherever it keeps its own sign (B1's first two pilots); guarded,
    no move may change its multiplier by more than `max_relative_change`. Values with no valid
    move either way get no gain, so every selected move can be measured in both directions.
    """

    def __init__(self, trainer: GlazeTrainer, gradients: list[torch.Tensor], guard: bool) -> None:
        cap = trainer.glaze.max_relative_change if guard else math.inf
        self.down: list[torch.Tensor] = []
        self.up: list[torch.Tensor] = []
        self.gains: list[torch.Tensor] = []
        for (_, parameter, offset), gradient in zip(trainer.trainable, gradients, strict=True):
            value = parameter.detach()
            shift = offset if guard else 0.0
            direction = torch.sign(gradient)
            down = guarded_neighbor(value, -direction, trainer.storage_dtype, shift, cap)
            up = guarded_neighbor(value, direction, trainer.storage_dtype, shift, cap)
            gain = -gradient * (down - value)
            invalid = torch.isnan(down) | torch.isnan(up)
            self.down.append(down)
            self.up.append(up)
            self.gains.append(torch.where(invalid, float("-inf"), gain).flatten())


def _split(chosen: torch.Tensor, sizes: Sequence[int]) -> list[torch.Tensor]:
    """Global flat indices into per-tensor flat indices."""
    pieces, start = [], 0
    for size in sizes:
        local = chosen[(chosen >= start) & (chosen < start + size)] - start
        pieces.append(torch.sort(local).values)
        start += size
    return pieces


def top_indices(
    trainer: GlazeTrainer,
    neighbors: Neighbors,
    count: int,
    select: Callable[[str], bool] | None = None,
) -> list[torch.Tensor]:
    """The `count` values with the largest predicted gain among the selected tensors."""
    gains = [
        gain if select is None or select(name) else torch.full_like(gain, float("-inf"))
        for (name, _, _), gain in zip(trainer.trainable, neighbors.gains, strict=True)
    ]
    everything = torch.cat(gains)
    best = torch.topk(everything, min(count, everything.numel()))
    chosen = best.indices[best.values > 0]
    return _split(chosen, [gain.numel() for gain in gains])


def random_indices(
    trainer: GlazeTrainer,
    neighbors: Neighbors,
    count: int,
    select: Callable[[str], bool],
    seed: int,
) -> list[torch.Tensor]:
    """`count` values drawn uniformly from the selected tensors' movable values."""
    movable = torch.cat(
        [
            (gain > 0) & torch.full_like(gain, select(name), dtype=torch.bool)
            for (name, _, _), gain in zip(trainer.trainable, neighbors.gains, strict=True)
        ]
    )
    candidates = movable.nonzero().squeeze(-1)
    generator = torch.Generator().manual_seed(seed)
    pick = torch.randperm(candidates.numel(), generator=generator)[:count]
    chosen = candidates[pick.to(candidates.device)]
    return _split(chosen, [gain.numel() for gain in neighbors.gains])


def move_set(
    name: str,
    trainer: GlazeTrainer,
    gradients: list[torch.Tensor],
    neighbors: Neighbors,
    indices: list[torch.Tensor],
) -> MoveSet:
    originals, downhill, uphill = [], [], []
    predicted_down = predicted_up = 0.0
    for parameter, gradient, down, up, index in zip(
        trainer.parameters, gradients, neighbors.down, neighbors.up, indices, strict=True
    ):
        value = parameter.detach().view(-1)[index].clone()
        slope = gradient.view(-1)[index]
        originals.append(value)
        downhill.append(down.view(-1)[index])
        uphill.append(up.view(-1)[index])
        predicted_down += float((slope.double() * (downhill[-1] - value).double()).sum())
        predicted_up += float((slope.double() * (uphill[-1] - value).double()).sum())
    return MoveSet(name, indices, originals, downhill, uphill, predicted_down, predicted_up)


def refused(
    trainer: GlazeTrainer, neighbors: Neighbors, indices: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Of the given (unguarded) moves, those the multiplier guard refuses."""
    kept = []
    for (_, parameter, offset), down, index in zip(
        trainer.trainable, neighbors.down, indices, strict=True
    ):
        value = parameter.detach().view(-1)[index]
        target = down.view(-1)[index]
        guarded = guarded_neighbor(
            value,
            torch.sign(target - value),
            trainer.storage_dtype,
            offset,
            trainer.glaze.max_relative_change,
        )
        kept.append(index[torch.isnan(guarded)])
    return kept


def breakdown(trainer: GlazeTrainer, indices: list[torch.Tensor]) -> dict[str, int]:
    """Moves per tensor family, largest first."""
    counts: Counter[str] = Counter()
    for (name, _, _), index in zip(trainer.trainable, indices, strict=True):
        if index.numel():
            counts[family(name)] += int(index.numel())
    return dict(counts.most_common())


def measure(
    trainer: GlazeTrainer,
    batch: Sequence[Sequence[int]],
    schedule: Schedule,
    moves: MoveSet,
    base_kl: float,
) -> dict[str, Any]:
    """KL after the moves and after the opposite moves, and what they say about the step."""
    kls = {}
    for direction, targets in (("down", moves.downhill), ("up", moves.uphill)):
        moves.apply(trainer.parameters, targets)
        try:
            kls[direction] = trainer.evaluate(batch, schedule).mean_kl
        finally:
            moves.apply(trainer.parameters, moves.originals)
    change_down, change_up = kls["down"] - base_kl, kls["up"] - base_kl
    # Quadratic model along the moves: change = first order + curvature (the same both ways).
    curvature = (change_down + change_up - moves.predicted_down - moves.predicted_up) / 2
    seen = (change_down - change_up) / 2
    predicted = (moves.predicted_down - moves.predicted_up) / 2
    # Without positive curvature the model has no best step; None keeps the JSON strict.
    best = -moves.predicted_down / (2 * curvature) if curvature > 0 else None
    return {
        "moves": moves.size,
        "kl_down": kls["down"],
        "kl_up": kls["up"],
        "change_down": change_down,
        "change_up": change_up,
        "predicted_down": moves.predicted_down,
        "slope_seen_over_predicted": seen / predicted if predicted else None,
        "curvature": curvature,
        # The fraction of one grid step that the quadratic model says is best: under 0.5, the
        # downhill step raises the KL.
        "best_step_fraction": best,
    }


def run_pilot(
    trainer: GlazeTrainer,
    train: Sequence[Sequence[int]],
    schedule: Schedule,
    fraction: float,
    select: Callable[[str], bool] | None,
    guard: bool,
) -> dict[str, Any]:
    """A fixed-batch pilot from the init; the student is restored afterwards."""
    initial = trainable_state(trainer.student)
    try:
        optimizer = trainer.make_optimizer(select=select, guard=guard)
        losses = trainer.pilot(
            train, schedule, PILOT_STEPS, fraction, optimizer=optimizer, check=False
        )
    finally:
        load_trainable_state(trainer.student, initial)
    return {
        "flip_fraction": fraction,
        "flips": trainer.flips(fraction),
        "train_kl": losses,
        "passed": min(losses[1:]) < losses[0],
    }


def diagnose(
    trainer: GlazeTrainer,
    train: Sequence[Sequence[int]],
    schedule: Schedule,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Every measurement, on the pilot's fixed batch: the first `blocks_per_step` blocks."""
    glaze = trainer.glaze
    indices = list(range(schedule.blocks_per_step))
    batch = [train[i] for i in indices]
    base = trainer.evaluate(batch, schedule).mean_kl
    repeat = trainer.evaluate(batch, schedule).mean_kl
    log(f"diag base KL {base:.6f}, repeated {repeat:.6f}")
    with_gradient = trainer.gradient(train, indices, schedule).mean_kl
    gradients = [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in trainer.parameters
    ]
    for parameter in trainer.parameters:
        parameter.grad = None
    report: dict[str, Any] = {
        "batch": {"blocks": len(batch), "tokens": len(batch) * schedule.block_tokens},
        "kl": {"base": base, "repeat": repeat, "with_gradient": with_gradient},
        "trainable": {
            "scales": sum(p.numel() for n, p, _ in trainer.trainable if is_scale(n)),
            "norm_values": sum(p.numel() for n, p, _ in trainer.trainable if is_norm(n)),
        },
        "max_relative_change": glaze.max_relative_change,
        "moves": measure_moves(trainer, batch, schedule, gradients, base, log),
        "pilots": {},
    }
    # The neighbor tables are released by now, so the pilots peak as a training step does.
    del gradients
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    smallest = min(glaze.flip_fractions)
    pilots = {
        f"guarded_{smallest:g}": (smallest, None, True),
        f"guarded_{smallest / 10:g}": (smallest / 10, None, True),
        f"scales_only_{smallest:g}": (smallest, is_scale, True),
    }
    for name, (fraction, select, guard) in pilots.items():
        result = run_pilot(trainer, train, schedule, fraction, select, guard)
        report["pilots"][name] = result
        log(f"diag pilot {name}: KL {result['train_kl']}, passed {result['passed']}")
    return report


def measure_moves(
    trainer: GlazeTrainer,
    batch: Sequence[Sequence[int]],
    schedule: Schedule,
    gradients: list[torch.Tensor],
    base: float,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Each candidate set of one-step moves, measured both ways from the init."""
    glaze = trainer.glaze
    count = trainer.flips(min(glaze.flip_fractions))
    unguarded = Neighbors(trainer, gradients, guard=False)
    guarded = Neighbors(trainer, gradients, guard=True)
    top = top_indices(trainer, unguarded, count)
    sets = {}
    for share in (100, 10):
        smaller = top_indices(trainer, unguarded, max(1, count // share))
        sets[f"unguarded_top_{max(1, count // share)}"] = (unguarded, smaller)
    sets[f"unguarded_top_{count}"] = (unguarded, top)
    sets[f"refused_of_unguarded_top_{count}"] = (unguarded, refused(trainer, unguarded, top))
    sets[f"guarded_top_{count}"] = (guarded, top_indices(trainer, guarded, count))
    sets[f"scales_top_{count}"] = (guarded, top_indices(trainer, guarded, count, is_scale))
    sets[f"norms_guarded_top_{count}"] = (guarded, top_indices(trainer, guarded, count, is_norm))
    sets[f"random_scales_{count}"] = (
        guarded,
        random_indices(trainer, guarded, count, is_scale, glaze.seed),
    )

    results: dict[str, Any] = {}
    for name, (neighbors, chosen) in sets.items():
        moves = move_set(name, trainer, gradients, neighbors, chosen)
        result = measure(trainer, batch, schedule, moves, base)
        result["by_family"] = breakdown(trainer, chosen)
        results[name] = result
        families = ", ".join(f"{k} {v}" for k, v in list(result["by_family"].items())[:3])
        log(
            f"diag {name}: {result['moves']} moves, KL down {result['kl_down']:.6f} "
            f"up {result['kl_up']:.6f}, best step fraction {result['best_step_fraction']}, "
            f"slope seen/predicted {result['slope_seen_over_predicted']}; {families}"
        )
    return results
