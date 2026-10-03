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

from qwen35_compression.config import ExperimentConfig, GlazeConfig, VariantConfig
from qwen35_compression.glaze.budget import check_memory, check_time, memory_estimate_gib
from qwen35_compression.glaze.data import blocks_digest, epoch_order, groups
from qwen35_compression.glaze.losses import DistillStats, distill_step
from qwen35_compression.glaze.quant_linear import GroupQuantLinear
from qwen35_compression.glaze.student import load_trainable_state, trainable_state

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
    """Trains a student's log-scales (and norm masters) against a frozen teacher."""

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
        self.scale_params = [
            module.log_scale for module in student.modules() if isinstance(module, GroupQuantLinear)
        ]
        scale_ids = {id(parameter) for parameter in self.scale_params}
        self.norm_params = [
            parameter
            for parameter in student.parameters()
            if parameter.requires_grad and id(parameter) not in scale_ids
        ]
        if not self.scale_params:
            raise ValueError("the student has no quantized layers to train")

    def make_optimizer(self, scale_lr: float) -> torch.optim.Optimizer:
        parameter_groups: list[dict[str, Any]] = [{"params": self.scale_params, "lr": scale_lr}]
        if self.norm_params:
            parameter_groups.append(
                {"params": self.norm_params, "lr": self.glaze.norm_learning_rate}
            )
        return torch.optim.AdamW(parameter_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

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

    def _step(
        self,
        optimizer: torch.optim.Optimizer,
        blocks: Sequence[Sequence[int]],
        indices: Sequence[int],
        schedule: Schedule,
    ) -> DistillStats:
        optimizer.zero_grad(set_to_none=True)
        normalizer = float(len(indices) * schedule.block_tokens)
        stats = DistillStats()
        for micro in groups(indices, schedule.blocks_per_micro):
            stats.merge(self._micro_batch(self._ids(blocks, micro), normalizer, backward=True))
        if not math.isfinite(stats.kl):
            raise FloatingPointError(f"non-finite training KL: {stats.kl}")
        optimizer.step()
        return stats

    def _check_memory(self) -> None:
        if self.device.type == "cuda":
            check_memory(
                torch.cuda.max_memory_allocated(self.device),
                torch.cuda.get_device_properties(self.device).total_memory,
                self.glaze.max_memory_fraction,
            )

    def pilot(
        self, blocks: Sequence[Sequence[int]], schedule: Schedule, steps: int, scale_lr: float
    ) -> list[float]:
        """A few steps on one fixed batch: the KL must stay finite and fall below its start."""
        if steps < 2:
            raise ValueError("a pilot needs at least 2 steps")
        optimizer = self.make_optimizer(scale_lr)
        indices = list(range(schedule.blocks_per_step))
        losses = []
        for step in range(steps):
            losses.append(self._step(optimizer, blocks, indices, schedule).mean_kl)
            self._check_memory()
            self.log(f"pilot step {step + 1}/{steps}: KL {losses[-1]:.6f}")
        if not min(losses[1:]) < losses[0]:
            raise PilotFailed(f"pilot KL did not fall on a fixed batch: {losses}")
        return losses

    def probe(
        self,
        train: Sequence[Sequence[int]],
        dev: Sequence[Sequence[int]],
        schedule: Schedule,
    ) -> tuple[float, dict[str, float]]:
        """Pick the scale learning rate whose short run reaches the lowest dev KL."""
        rates = self.glaze.scale_learning_rates
        if len(rates) == 1 or self.glaze.probe_steps == 0:
            return rates[0], {}
        initial = trainable_state(self.student)
        order = epoch_order(len(train), 0, self.glaze.seed)
        steps = groups(order, schedule.blocks_per_step)[: self.glaze.probe_steps]
        results: dict[str, float] = {}
        for rate in rates:
            load_trainable_state(self.student, initial)
            optimizer = self.make_optimizer(rate)
            for indices in steps:
                self._step(optimizer, train, indices, schedule)
                self._check_memory()
            results[repr(rate)] = self.evaluate(dev, schedule).mean_kl
            self.log(f"probe scale lr {rate:g}: dev KL {results[repr(rate)]:.6f}")
        load_trainable_state(self.student, initial)
        finite = {rate: results[repr(rate)] for rate in rates if math.isfinite(results[repr(rate)])}
        if not finite:
            raise PilotFailed("every probe learning rate diverged")
        return min(finite, key=finite.__getitem__), results

    def fit(
        self,
        train: Sequence[Sequence[int]],
        dev: Sequence[Sequence[int]],
        schedule: Schedule,
        scale_lr: float,
    ) -> FitResult:
        """Train for the full schedule; keep the state with the lowest dev KL (the init counts)."""
        optimizer = self.make_optimizer(scale_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: learning_rate_factor(step, self.glaze.warmup_steps, schedule.total_steps),
        )
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
                rate = optimizer.param_groups[0]["lr"]
                stats = self._step(optimizer, train, indices, schedule)
                scheduler.step()
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
                entry: dict[str, Any] = {"step": step, "train_kl": stats.mean_kl, "scale_lr": rate}
                if step % self.glaze.eval_every_steps == 0 or step == schedule.total_steps:
                    dev_stats = self.evaluate(dev, schedule)
                    entry["dev"] = dev_stats.summary()
                    if dev_stats.mean_kl < best_kl:
                        best_kl, best_step = dev_stats.mean_kl, step
                        best_state, best_dev = trainable_state(self.student), dev_stats.summary()
                history.append(entry)
                self.log(
                    f"step {step}/{schedule.total_steps}: train KL {stats.mean_kl:.6f}"
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
        len(glaze.scale_learning_rates) * glaze.probe_steps * glaze.tokens_per_step
        if len(glaze.scale_learning_rates) > 1
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
    from qwen35_compression.export import verify_export, write_export_manifest
    from qwen35_compression.feature1 import require_calibration_lock
    from qwen35_compression.glaze.data import calibration_blocks, split_blocks
    from qwen35_compression.glaze.export import write_refined_export
    from qwen35_compression.glaze.student import (
        LANGUAGE_MODEL_PREFIX,
        build_student,
        exported_tensors,
        read_init_export,
    )
    from qwen35_compression.models import load_resolved_model, resolve_revision

    glaze = _require_glaze(variant)
    init_variant = config.variant(str(variant.init))
    init_dir = config.paths.outputs / init_variant.name
    output_dir = output_dir or config.paths.outputs / variant.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output_dir}")
    init_manifest = verify_export(init_dir, init_variant)
    if config.calibration.lock_path is not None:
        require_calibration_lock(config.calibration)

    torch.manual_seed(glaze.seed)
    started = time.perf_counter()
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
    record: dict[str, Any] = {
        "init": {
            "variant": init_variant.name,
            "method": init_variant.method,
            "code_revision": init_manifest.get("code_revision"),
            "files": {item["path"]: item["sha256"] for item in init_manifest["files"]},
        },
        "data": {
            "source": glaze.data,
            "train_blocks": len(train),
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
        losses = trainer.pilot(train, schedule, pilot_steps, max(glaze.scale_learning_rates))
        record["pilot"] = {"steps": pilot_steps, "train_kl": losses}
    else:
        rate, probe = trainer.probe(train, dev, schedule)
        result = trainer.fit(train, dev, schedule, rate)
        load_trainable_state(student, result.best_state)
        record.update(
            {
                "scale_learning_rate": rate,
                "probe_dev_kl": probe,
                "best_step": result.best_step,
                "dev_at_init": result.initial_dev,
                "dev_best": result.best_dev,
                "history": result.history,
                "train_seconds": round(result.seconds, 1),
            }
        )

    write_refined_export(init_dir, output_dir, exported_tensors(student, LANGUAGE_MODEL_PREFIX))
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    manifest = write_export_manifest(
        output_dir,
        config,
        variant,
        revision,
        time.perf_counter() - started,
        peak,
        extra={"glaze": record},
    )
    return output_dir, manifest
