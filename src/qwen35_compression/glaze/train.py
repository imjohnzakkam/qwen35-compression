"""Glaze training: distil the student's group scales and norm weights from the BF16 teacher."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from qwen35_compression.config import (
    ExperimentConfig,
    GlazeConfig,
    VariantConfig,
    glaze_dev_start,
)
from qwen35_compression.glaze.budget import check_memory, check_time, memory_estimate_gib
from qwen35_compression.glaze.data import blocks_digest, epoch_order, groups
from qwen35_compression.glaze.grid import GridDescent
from qwen35_compression.glaze.losses import DistillStats, distill_step
from qwen35_compression.glaze.quant_linear import GroupQuantLinear
from qwen35_compression.glaze.student import (
    load_trainable_state,
    trainable_parameters,
    trainable_state,
)

# Measured training throughput on one A100 40 GB, for the dry-run estimate only: AutoRound's
# 26 s per layer for 200 steps of 16,384 tokens scales to 7.6 minutes per million tokens.
MINUTES_PER_MILLION_TOKENS = 7.6
# Qwen3.5-4B, from the AutoRound export: used for the dry-run memory estimate.
QWEN35_4B_SHAPES = {
    "quantized_params": 3.569e9,
    "scales": 27.88e6,
    "embedding_params": 635.7e6,
    "layers": 32,
    "hidden": 2560,
    "intermediate": 9216,
    "vocab": 248320,
}


class PilotFailed(RuntimeError):
    """The short pilot showed the full run would not train."""


@dataclass(frozen=True)
class Schedule:
    block_tokens: int
    blocks_per_micro: int
    micro_per_step: int
    steps_per_epoch: int
    epochs: int

    @property
    def blocks_per_step(self) -> int:
        return self.blocks_per_micro * self.micro_per_step

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.epochs


def make_schedule(glaze: GlazeConfig, block_tokens: int) -> Schedule:
    blocks_per_micro = glaze.micro_batch_tokens // block_tokens
    micro_per_step = glaze.tokens_per_step // glaze.micro_batch_tokens
    return Schedule(
        block_tokens=block_tokens,
        blocks_per_micro=blocks_per_micro,
        micro_per_step=micro_per_step,
        steps_per_epoch=glaze.train_blocks // (blocks_per_micro * micro_per_step),
        epochs=glaze.epochs,
    )


def learning_rate_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup over `warmup` steps, then cosine decay towards 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class FitResult:
    best_state: dict[str, torch.Tensor]
    best_step: int
    initial_dev: dict[str, Any]
    best_dev: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0


class GlazeTrainer:
    """Moves a student's scales and norm weights along their BF16 grid to match a teacher."""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        head_weight: torch.Tensor,
        glaze: GlazeConfig,
        log: Callable[[str], None] = print,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.head_weight = head_weight
        self.glaze = glaze
        self.log = log
        self.device = head_weight.device
        layers = [module for module in student.modules() if isinstance(module, GroupQuantLinear)]
        if not layers:
            raise ValueError("the student has no quantized layers to train")
        dtypes = {layer.storage_dtype for layer in layers}
        if len(dtypes) != 1:
            raise ValueError(f"quantized layers store scales in several dtypes: {dtypes}")
        self.storage_dtype = dtypes.pop()
        # (name, parameter, offset): each value sets the multiplier offset + value.
        self.trainable = trainable_parameters(student)
        self.parameters = [parameter for _, parameter, _ in self.trainable]
        self._names = {id(parameter): name for name, parameter, _ in self.trainable}

    @property
    def size(self) -> int:
        """Number of trainable values (scales and norm weights)."""
        return sum(parameter.numel() for parameter in self.parameters)

    def flips(self, fraction: float) -> int:
        return max(1, round(fraction * self.size))

    def make_optimizer(
        self, select: Callable[[str], bool] | None = None, guard: bool = True
    ) -> GridDescent:
        """GridDescent over the trainable tensors whose names pass `select` (default: all).

        With `guard`, no move changes a multiplier by more than `max_relative_change`; without
        it, only the stored value's own sign and zero are protected (B1's first two pilots).
        """
        chosen = [item for item in self.trainable if select is None or select(item[0])]
        if not chosen:
            raise ValueError("no trainable tensor passes the selection")
        return GridDescent(
            [parameter for _, parameter, _ in chosen],
            self.storage_dtype,
            self.glaze.momentum,
            offsets=[offset for _, _, offset in chosen] if guard else None,
            max_relative_change=self.glaze.max_relative_change if guard else math.inf,
        )

    def norm_moves(self, optimizer: GridDescent) -> int:
        """How many of the optimizer's last moves were norm weights (the rest were scales)."""
        return sum(
            moved
            for parameter, moved in zip(optimizer.parameters, optimizer.moved, strict=True)
            if not self._names[id(parameter)].endswith(".scales")
        )

    def _ids(self, blocks: Sequence[Sequence[int]], indices: Sequence[int]) -> torch.Tensor:
        return torch.tensor(
            [list(blocks[i]) for i in indices], dtype=torch.long, device=self.device
        )

    def _micro_batch(self, ids: torch.Tensor, normalizer: float, backward: bool) -> DistillStats:
        with torch.no_grad():
            teacher_hidden = self.teacher(input_ids=ids, use_cache=False).last_hidden_state
        with torch.set_grad_enabled(backward):
            student_hidden = self.student(input_ids=ids, use_cache=False).last_hidden_state
        # Packed calibration blocks: every position is a training position.
        mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        return distill_step(
            student_hidden,
            teacher_hidden,
            self.head_weight,
            mask,
            self.glaze.logit_chunk_tokens,
            normalizer,
            backward,
        )

    def evaluate(self, blocks: Sequence[Sequence[int]], schedule: Schedule) -> DistillStats:
        stats = DistillStats()
        for indices in groups(range(len(blocks)), schedule.blocks_per_micro):
            ids = self._ids(blocks, indices)
            stats.merge(self._micro_batch(ids, float(ids.numel()), backward=False))
        return stats

    def gradient(
        self, blocks: Sequence[Sequence[int]], indices: Sequence[int], schedule: Schedule
    ) -> DistillStats:
        """The mean KL over the indexed blocks, with its gradient left in every `.grad`."""
        for parameter in self.parameters:
            parameter.grad = None
        normalizer = float(len(indices) * schedule.block_tokens)
        stats = DistillStats()
        for micro in groups(indices, schedule.blocks_per_micro):
            stats.merge(self._micro_batch(self._ids(blocks, micro), normalizer, backward=True))
        if not math.isfinite(stats.kl):
            raise FloatingPointError(f"non-finite training KL: {stats.kl}")
        return stats

    def _step(
        self,
        optimizer: GridDescent,
        blocks: Sequence[Sequence[int]],
        indices: Sequence[int],
        schedule: Schedule,
        flips: int,
    ) -> tuple[DistillStats, int]:
        """Gradients from one step's blocks, then up to `flips` grid moves."""
        stats = self.gradient(blocks, indices, schedule)
        return stats, optimizer.step(flips)

    def _check_memory(self) -> None:
        if self.device.type == "cuda":
            check_memory(
                torch.cuda.max_memory_allocated(self.device),
                torch.cuda.get_device_properties(self.device).total_memory,
                self.glaze.max_memory_fraction,
            )

    def pilot(
        self,
        blocks: Sequence[Sequence[int]],
        schedule: Schedule,
        steps: int,
        fraction: float,
        optimizer: GridDescent | None = None,
        check: bool = True,
    ) -> list[float]:
        """A few steps on one fixed batch: the KL must stay finite and fall below its start.

        Without `check`, a KL that does not fall is reported rather than raised.
        """
        if steps < 2:
            raise ValueError("a pilot needs at least 2 steps")
        optimizer = optimizer or self.make_optimizer()
        indices = list(range(schedule.blocks_per_step))
        losses = []
        for step in range(steps):
            stats, moved = self._step(optimizer, blocks, indices, schedule, self.flips(fraction))
            losses.append(stats.mean_kl)
            self._check_memory()
            self.log(
                f"pilot step {step + 1}/{steps}: KL {losses[-1]:.6f}, moved {moved} "
                f"(norms {self.norm_moves(optimizer)})"
            )
        if check and not min(losses[1:]) < losses[0]:
            raise PilotFailed(f"pilot KL did not fall on a fixed batch: {losses}")
        return losses

    def probe(
        self,
        train: Sequence[Sequence[int]],
        dev: Sequence[Sequence[int]],
        schedule: Schedule,
    ) -> tuple[float, dict[str, float]]:
        """Pick the flip fraction whose short run reaches the lowest dev KL."""
        fractions = self.glaze.flip_fractions
        if len(fractions) == 1 or self.glaze.probe_steps == 0:
            return fractions[0], {}
        initial = trainable_state(self.student)
        order = epoch_order(len(train), 0, self.glaze.seed)
        steps = groups(order, schedule.blocks_per_step)[: self.glaze.probe_steps]
        results: dict[str, float] = {}
        for fraction in fractions:
            load_trainable_state(self.student, initial)
            optimizer = self.make_optimizer()
            for indices in steps:
                self._step(optimizer, train, indices, schedule, self.flips(fraction))
                self._check_memory()
            results[repr(fraction)] = self.evaluate(dev, schedule).mean_kl
            self.log(f"probe flip fraction {fraction:g}: dev KL {results[repr(fraction)]:.6f}")
        load_trainable_state(self.student, initial)
        finite = {f: results[repr(f)] for f in fractions if math.isfinite(results[repr(f)])}
        if not finite:
            raise PilotFailed("every probe setting diverged")
        return min(finite, key=finite.__getitem__), results

    def fit(
        self,
        train: Sequence[Sequence[int]],
        dev: Sequence[Sequence[int]],
        schedule: Schedule,
        fraction: float,
    ) -> FitResult:
        """Train for the full schedule; keep the state with the lowest dev KL (the init counts).

        The number of grid moves per step follows a warmup and cosine decay, like a learning
        rate.
        """
        optimizer = self.make_optimizer()
        initial = self.evaluate(dev, schedule)
        self.log(f"dev KL at init: {initial.mean_kl:.6f}")
        best_kl, best_step = initial.mean_kl, 0
        best_state, best_dev = trainable_state(self.student), initial.summary()
        history: list[dict[str, Any]] = []
        started = time.perf_counter()
        paced_from: float | None = None
        step = 0
        for epoch in range(schedule.epochs):
            order = epoch_order(len(train), epoch, self.glaze.seed)
            for indices in groups(order, schedule.blocks_per_step):
                factor = learning_rate_factor(step, self.glaze.warmup_steps, schedule.total_steps)
                flips = max(1, round(self.flips(fraction) * factor))
                stats, moved = self._step(optimizer, train, indices, schedule, flips)
                step += 1
                self._check_memory()
                # Pace from the second step on: the first one compiles the fla kernels.
                if step == 1:
                    paced_from = time.perf_counter()
                elif paced_from is not None:
                    check_time(
                        time.perf_counter() - paced_from,
                        step - 1,
                        schedule.total_steps,
                        self.glaze.max_train_minutes,
                    )
                entry: dict[str, Any] = {
                    "step": step,
                    "train_kl": stats.mean_kl,
                    "moved": moved,
                    "moved_norms": self.norm_moves(optimizer),
                }
                if step % self.glaze.eval_every_steps == 0 or step == schedule.total_steps:
                    dev_stats = self.evaluate(dev, schedule)
                    entry["dev"] = dev_stats.summary()
                    if dev_stats.mean_kl < best_kl:
                        best_kl, best_step = dev_stats.mean_kl, step
                        best_state, best_dev = trainable_state(self.student), dev_stats.summary()
                history.append(entry)
                self.log(
                    f"step {step}/{schedule.total_steps}: train KL {stats.mean_kl:.6f}, "
                    f"moved {moved} (norms {entry['moved_norms']})"
                    + (f", dev KL {entry['dev']['mean_kl']:.6f}" if "dev" in entry else "")
                )
        return FitResult(
            best_state=best_state,
            best_step=best_step,
            initial_dev=initial.summary(),
            best_dev=best_dev,
            history=history,
            seconds=time.perf_counter() - started,
        )


def changed_values(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, int]:
    """How many scales and norm values the run moved, of how many."""
    counts = {"scales": 0, "scales_total": 0, "norm_values": 0, "norm_values_total": 0}
    for name, value in before.items():
        kind = "scales" if name.endswith(".scales") else "norm_values"
        counts[kind] += int((after[name] != value).sum())
        counts[f"{kind}_total"] += value.numel()
    return counts


def plan(
    config: ExperimentConfig,
    variant: VariantConfig,
    output_dir: Path | None = None,
    pilot_steps: int | None = None,
) -> dict[str, Any]:
    """What a run would do, with its memory and time estimates; loads nothing."""
    glaze = _require_glaze(variant)
    schedule = make_schedule(glaze, config.calibration.max_sequence_length)
    trained_tokens = schedule.total_steps * glaze.tokens_per_step
    probe_tokens = (
        len(glaze.flip_fractions) * glaze.probe_steps * glaze.tokens_per_step
        if len(glaze.flip_fractions) > 1
        else 0
    )
    memory = memory_estimate_gib(
        **QWEN35_4B_SHAPES,
        micro_batch_tokens=glaze.micro_batch_tokens,
        chunk_tokens=glaze.logit_chunk_tokens,
    )
    return {
        "variant": variant.name,
        "init": variant.init,
        "init_dir": str(config.paths.outputs / str(variant.init)),
        "output_dir": str(output_dir or config.paths.outputs / variant.name),
        "pilot_steps": pilot_steps,
        "data": glaze.data,
        "schedule": {
            "steps": schedule.total_steps,
            "steps_per_epoch": schedule.steps_per_epoch,
            "blocks_per_step": schedule.blocks_per_step,
            "micro_batches_per_step": schedule.micro_per_step,
            "trained_tokens": trained_tokens,
            "probe_tokens": probe_tokens,
        },
        "estimate": {
            "memory_gib": round(memory["total"], 2),
            "train_minutes": round(
                (trained_tokens + probe_tokens) / 1e6 * MINUTES_PER_MILLION_TOKENS, 1
            ),
        },
    }


def _require_glaze(variant: VariantConfig) -> GlazeConfig:
    if variant.method != "glaze" or variant.glaze is None or variant.init is None:
        raise ValueError(f"{variant.name} is not a glaze variant")
    return variant.glaze


@dataclass
class Setup:
    """The teacher and the student of one Glaze variant, loaded and ready to train."""

    trainer: GlazeTrainer
    student: nn.Module
    train: list[list[int]]
    dev: list[list[int]]
    schedule: Schedule
    init_variant: VariantConfig
    init_dir: Path
    init_manifest: dict[str, Any]
    revision: str


def prepare(
    config: ExperimentConfig, variant: VariantConfig, log: Callable[[str], None] = print
) -> Setup:
    """Verify the init export, load BF16 as the teacher and the export as the student."""
    from qwen35_compression.export import verify_export
    from qwen35_compression.feature1 import require_calibration_lock
    from qwen35_compression.glaze.data import calibration_blocks, split_blocks
    from qwen35_compression.glaze.student import build_student, read_init_export
    from qwen35_compression.models import load_resolved_model, resolve_revision

    glaze = _require_glaze(variant)
    init_variant = config.variant(str(variant.init))
    init_dir = config.paths.outputs / init_variant.name
    init_manifest = verify_export(init_dir, init_variant)
    if config.calibration.lock_path is not None:
        require_calibration_lock(config.calibration)

    torch.manual_seed(glaze.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    revision = resolve_revision(config)
    model, processor = load_resolved_model(f"{config.model.id}@{revision}", config)
    model.requires_grad_(False)
    model.eval()
    # The student is text-only, so the vision tower is not needed on the GPU.
    if getattr(model.model, "visual", None) is not None:
        del model.model.visual
    teacher = model.model.language_model
    head_weight = model.get_output_embeddings().weight
    try:
        from accelerate.hooks import remove_hook_from_module

        # Device-placement hooks would be copied into the student; everything sits on one GPU.
        remove_hook_from_module(teacher, recurse=True)
    except ImportError:
        pass

    init = read_init_export(init_dir)
    student = build_student(teacher, init, train_norms=glaze.train_norms)
    del init
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()

    blocks = calibration_blocks(processor, config.calibration)
    train, dev = split_blocks(blocks, glaze)
    schedule = make_schedule(glaze, config.calibration.max_sequence_length)
    trainer = GlazeTrainer(student, teacher, head_weight, glaze, log=log)
    return Setup(
        trainer, student, train, dev, schedule, init_variant, init_dir, init_manifest, revision
    )


def refine(
    config: ExperimentConfig,
    variant: VariantConfig,
    *,
    output_dir: Path | None = None,
    pilot_steps: int | None = None,
    log: Callable[[str], None] = print,
) -> tuple[Path, dict[str, Any]]:
    """Train a Glaze variant from its init export and write the refined export.

    With `pilot_steps`, only a few steps on one fixed batch run (they must lower the KL), and the
    resulting export exists to check that the whole path, vLLM loading included, works.
    """
    from qwen35_compression.export import write_export_manifest
    from qwen35_compression.glaze.export import write_refined_export
    from qwen35_compression.glaze.student import LANGUAGE_MODEL_PREFIX, exported_tensors

    glaze = _require_glaze(variant)
    output_dir = output_dir or config.paths.outputs / variant.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output_dir}")
    started = time.perf_counter()
    setup = prepare(config, variant, log)
    trainer, student, schedule = setup.trainer, setup.student, setup.schedule
    train, dev, init_variant = setup.train, setup.dev, setup.init_variant
    record: dict[str, Any] = {
        "init": {
            "variant": init_variant.name,
            "method": init_variant.method,
            "code_revision": setup.init_manifest.get("code_revision"),
            "files": {item["path"]: item["sha256"] for item in setup.init_manifest["files"]},
        },
        "data": {
            "source": glaze.data,
            "train_start": glaze.train_start,
            "train_blocks": len(train),
            "dev_start": glaze_dev_start(glaze),
            "dev_blocks": len(dev),
            "block_tokens": schedule.block_tokens,
            "train_sha256": blocks_digest(train),
            "dev_sha256": blocks_digest(dev),
        },
        "settings": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in vars(glaze).items()
        },
        "schedule": {"steps": schedule.total_steps, "steps_per_epoch": schedule.steps_per_epoch},
    }
    if pilot_steps:
        # The smallest setting: the pilot checks the machinery, not the step size.
        fraction = min(glaze.flip_fractions)
        losses = trainer.pilot(train, schedule, pilot_steps, fraction)
        record["pilot"] = {"steps": pilot_steps, "flip_fraction": fraction, "train_kl": losses}
    else:
        initial = trainable_state(student)
        fraction, probe = trainer.probe(train, dev, schedule)
        result = trainer.fit(train, dev, schedule, fraction)
        load_trainable_state(student, result.best_state)
        record.update(
            {
                "flip_fraction": fraction,
                "changed": changed_values(initial, result.best_state),
                "probe_dev_kl": probe,
                "best_step": result.best_step,
                "dev_at_init": result.initial_dev,
                "dev_best": result.best_dev,
                "history": result.history,
                "train_seconds": round(result.seconds, 1),
            }
        )

    write_refined_export(
        setup.init_dir, output_dir, exported_tensors(student, LANGUAGE_MODEL_PREFIX)
    )
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    manifest = write_export_manifest(
        output_dir,
        config,
        variant,
        setup.revision,
        time.perf_counter() - started,
        peak,
        extra={"glaze": record},
    )
    return output_dir, manifest
