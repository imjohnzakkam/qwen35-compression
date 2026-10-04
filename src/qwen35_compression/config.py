from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_METHODS = {"bf16", "int8", "gptq", "awq", "autoround", "mixed", "glaze"}
# Exports Glaze can start from: symmetric INT4 group-quantized weights in compressed-tensors.
GLAZE_INIT_METHODS = {"gptq", "autoround"}
GLAZE_DATA = {"calibration"}


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class CalibrationSourceConfig:
    dataset_id: str
    split: str
    messages_column: str = "messages"
    revision: str | None = None


@dataclass(frozen=True)
class CalibrationConfig:
    path: Path
    num_samples: int
    max_sequence_length: int
    seed: int
    lock_path: Path | None = None
    source: CalibrationSourceConfig | None = None


@dataclass(frozen=True)
class ImageTextSourceConfig:
    dataset_id: str
    split: str
    image_url_column: str
    captions_column: str
    revision: str | None = None


@dataclass(frozen=True)
class MultimodalCalibrationConfig:
    path: Path
    assets_dir: Path
    lock_path: Path
    num_samples: int
    seed: int
    source: ImageTextSourceConfig


@dataclass(frozen=True)
class EvaluationConfig:
    path: Path | None
    max_samples: int | None
    max_new_tokens: int
    seed: int
    mode: str = "smoke"
    suite_path: Path | None = None


@dataclass(frozen=True)
class PathsConfig:
    outputs: Path
    results: Path


@dataclass(frozen=True)
class QuantizationGroupConfig:
    name: str
    targets: tuple[str, ...]
    scheme: str
    group_size: int | None = None


@dataclass(frozen=True)
class GlazeConfig:
    """End-to-end distillation of a W4A16 export's group scales and norm weights against BF16.

    The INT4 codes of the init export stay frozen. `data: calibration` trains on packed
    calibration blocks: `train_blocks` from block `train_start` (by default the init's own, from
    0), with `dev_blocks` held out from block `dev_start` (by default right after training).
    """

    data: str = "calibration"
    train_start: int = 0
    train_blocks: int = 128
    # None: the block right after the training blocks.
    dev_start: int | None = None
    dev_blocks: int = 32
    epochs: int = 4
    tokens_per_step: int = 16384
    micro_batch_tokens: int = 8192
    # Trained values move one BF16 grid step at a time (qwen35_compression.glaze.grid). Each
    # step moves this fraction of them, the most promising first; a short probe on the dev set
    # picks one of the candidates. The pilot uses the smallest. B1's diagnosis: 2,805 moves per
    # step (1e-4) overshoot, since the best move is 0.2-0.4 of a grid step; 281 (1e-5) lower the KL.
    flip_fractions: tuple[float, ...] = (1e-5, 2e-5, 4e-5)
    momentum: float = 0.9
    # No move may change the multiplier a value sets (a scale; 1 + w for Qwen3.5's RMSNorm) by
    # more than this fraction of itself. Scale steps are at most 2^-7 (0.78%), so it binds on
    # norm weights near w = -1, where one step can switch a channel on or off.
    max_relative_change: float = 0.01
    probe_steps: int = 8
    warmup_steps: int = 3
    train_norms: bool = True
    logit_chunk_tokens: int = 512
    eval_every_steps: int = 16
    seed: int = 42
    # Stop instead of running out of memory or time mid-run.
    max_memory_fraction: float = 0.85
    max_train_minutes: float = 30.0


@dataclass(frozen=True)
class VariantConfig:
    name: str
    method: str
    scheme: str | None = None
    bits: int | None = None
    group_size: int | None = None
    ignore: tuple[str, ...] = field(default_factory=tuple)
    targets: tuple[str, ...] = ("Linear",)
    groups: tuple[QuantizationGroupConfig, ...] = field(default_factory=tuple)
    requires_multimodal_calibration: bool = False
    # Use only the first N calibration samples (after packing, for methods that pack).
    calibration_samples: int | None = None
    # Glaze: the variant whose export it refines, and its training settings.
    init: str | None = None
    glaze: GlazeConfig | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    feature: str
    model: ModelConfig
    calibration: CalibrationConfig
    multimodal_calibration: MultimodalCalibrationConfig | None
    evaluation: EvaluationConfig
    paths: PathsConfig
    variants: tuple[VariantConfig, ...]
    source_path: Path
    digest: str

    def variant(self, name: str) -> VariantConfig:
        for variant in self.variants:
            if variant.name == name:
                return variant
        choices = ", ".join(item.name for item in self.variants)
        raise ValueError(f"unknown variant {name!r}; expected one of: {choices}")


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _glaze_config(raw: Any, variant: str) -> GlazeConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"glaze settings must be a mapping: {variant}")
    known = set(GlazeConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown glaze settings for {variant}: {unknown}")
    values = dict(raw)
    if "flip_fractions" in values:
        values["flip_fractions"] = tuple(float(value) for value in values["flip_fractions"])
    return GlazeConfig(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    source_path = Path(path).resolve()
    raw_bytes = source_path.read_bytes()
    raw: dict[str, Any] = yaml.safe_load(raw_bytes)
    root = source_path.parent.parent

    model = ModelConfig(**raw["model"])
    calibration_raw = raw["calibration"]
    source_raw = calibration_raw.get("source")
    calibration = CalibrationConfig(
        path=_resolve_path(calibration_raw["path"], root),
        num_samples=int(calibration_raw["num_samples"]),
        max_sequence_length=int(calibration_raw["max_sequence_length"]),
        seed=int(calibration_raw["seed"]),
        lock_path=(
            _resolve_path(calibration_raw["lock_path"], root)
            if calibration_raw.get("lock_path")
            else None
        ),
        source=(CalibrationSourceConfig(**source_raw) if source_raw else None),
    )
    multimodal_raw = raw.get("multimodal_calibration")
    multimodal_calibration = None
    if multimodal_raw:
        multimodal_calibration = MultimodalCalibrationConfig(
            path=_resolve_path(multimodal_raw["path"], root),
            assets_dir=_resolve_path(multimodal_raw["assets_dir"], root),
            lock_path=_resolve_path(multimodal_raw["lock_path"], root),
            num_samples=int(multimodal_raw["num_samples"]),
            seed=int(multimodal_raw["seed"]),
            source=ImageTextSourceConfig(**multimodal_raw["source"]),
        )
    evaluation_raw = raw["evaluation"]
    evaluation = EvaluationConfig(
        path=(_resolve_path(evaluation_raw["path"], root) if evaluation_raw.get("path") else None),
        max_samples=evaluation_raw.get("max_samples"),
        max_new_tokens=int(evaluation_raw["max_new_tokens"]),
        seed=int(evaluation_raw["seed"]),
        mode=str(evaluation_raw.get("mode", "smoke")),
        suite_path=(
            _resolve_path(evaluation_raw["suite_path"], root)
            if evaluation_raw.get("suite_path")
            else None
        ),
    )
    paths_raw = raw["paths"]
    paths = PathsConfig(
        outputs=_resolve_path(paths_raw["outputs"], root),
        results=_resolve_path(paths_raw["results"], root),
    )
    variants_raw = raw.get("variants", ())
    if raw.get("variants_path"):
        variants_document = yaml.safe_load(
            _resolve_path(raw["variants_path"], root).read_text(encoding="utf-8")
        )
        variants_raw = variants_document["variants"]
    variants = tuple(
        VariantConfig(
            name=item["name"],
            method=item["method"],
            scheme=item.get("scheme"),
            bits=item.get("bits"),
            group_size=item.get("group_size"),
            ignore=tuple(item.get("ignore", ())),
            targets=tuple(item.get("targets", ("Linear",))),
            groups=tuple(
                QuantizationGroupConfig(
                    name=group["name"],
                    targets=tuple(group["targets"]),
                    scheme=group["scheme"],
                    group_size=group.get("group_size"),
                )
                for group in item.get("groups", ())
            ),
            requires_multimodal_calibration=bool(
                item.get("requires_multimodal_calibration", False)
            ),
            calibration_samples=item.get("calibration_samples"),
            init=item.get("init"),
            glaze=_glaze_config(item.get("glaze"), item["name"]),
        )
        for item in variants_raw
    )
    _validate(
        raw["feature"],
        model,
        calibration,
        multimodal_calibration,
        evaluation,
        variants,
    )
    canonical_raw = {**raw, "variants": variants_raw}
    canonical = json.dumps(canonical_raw, sort_keys=True, separators=(",", ":")).encode()
    return ExperimentConfig(
        feature=raw["feature"],
        model=model,
        calibration=calibration,
        multimodal_calibration=multimodal_calibration,
        evaluation=evaluation,
        paths=paths,
        variants=variants,
        source_path=source_path,
        digest=hashlib.sha256(canonical).hexdigest(),
    )


def _validate(
    feature: str,
    model: ModelConfig,
    calibration: CalibrationConfig,
    multimodal_calibration: MultimodalCalibrationConfig | None,
    evaluation: EvaluationConfig,
    variants: tuple[VariantConfig, ...],
) -> None:
    if not feature:
        raise ValueError("feature must be non-empty")
    if feature == "feature0" and model.id != "Qwen/Qwen3.5-0.8B":
        raise ValueError("Feature 0 is locked to Qwen/Qwen3.5-0.8B")
    if feature == "feature1" and model.id != "Qwen/Qwen3.5-4B":
        raise ValueError("Feature 1 is locked to Qwen/Qwen3.5-4B")
    if calibration.num_samples <= 0 or calibration.max_sequence_length <= 0:
        raise ValueError("calibration counts must be positive")
    if multimodal_calibration is not None and multimodal_calibration.num_samples <= 0:
        raise ValueError("multimodal calibration sample count must be positive")
    if evaluation.mode not in {"smoke", "benchmark"}:
        raise ValueError("evaluation.mode must be smoke or benchmark")
    if evaluation.mode == "smoke" and evaluation.path is None:
        raise ValueError("smoke evaluation requires evaluation.path")
    if evaluation.mode == "benchmark" and evaluation.suite_path is None:
        raise ValueError("benchmark evaluation requires evaluation.suite_path")
    if evaluation.max_samples is not None and evaluation.max_samples <= 0:
        raise ValueError("evaluation.max_samples must be positive or null")
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError("variant names must be unique")
    for variant in variants:
        if variant.method not in ALLOWED_METHODS:
            raise ValueError(f"unsupported method: {variant.method}")
        if variant.method in {"gptq", "awq", "autoround"}:
            if variant.bits != 4 or variant.group_size not in {32, 64, 128}:
                raise ValueError(f"invalid W4A16 settings for {variant.name}")
        if variant.calibration_samples is not None and variant.calibration_samples <= 0:
            raise ValueError(f"calibration_samples must be positive: {variant.name}")
        if variant.method == "mixed" and not variant.groups:
            raise ValueError(f"mixed variant requires groups: {variant.name}")
        if variant.method != "mixed" and variant.groups:
            raise ValueError(f"only mixed variants may define groups: {variant.name}")
        if variant.method == "glaze":
            _validate_glaze(variant, {item.name: item for item in variants}, calibration)
        elif variant.init is not None or variant.glaze is not None:
            raise ValueError(f"only glaze variants may set init or glaze: {variant.name}")


def _validate_glaze(
    variant: VariantConfig,
    by_name: dict[str, VariantConfig],
    calibration: CalibrationConfig,
) -> None:
    name = variant.name
    if variant.init is None or variant.glaze is None:
        raise ValueError(f"glaze variant requires init and glaze settings: {name}")
    init = by_name.get(variant.init)
    if init is None:
        raise ValueError(f"glaze init {variant.init!r} is not a configured variant: {name}")
    if init.method not in GLAZE_INIT_METHODS:
        raise ValueError(
            f"glaze init must be one of {sorted(GLAZE_INIT_METHODS)}, got {init.method}: {name}"
        )
    # The refined export reuses the init's codes, so it must describe the same quantization.
    if (variant.scheme, variant.bits, variant.group_size) != ("W4A16", 4, init.group_size):
        raise ValueError(f"glaze variant must match its init's W4A16 settings: {name}")
    if variant.ignore != init.ignore:
        raise ValueError(
            f"glaze variant must leave the same layers unquantized as its init: {name}"
        )
    glaze = variant.glaze
    if glaze.data not in GLAZE_DATA:
        raise ValueError(f"unsupported glaze data {glaze.data!r}: {name}")
    counts = {
        "train_blocks": glaze.train_blocks,
        "dev_blocks": glaze.dev_blocks,
        "epochs": glaze.epochs,
        "tokens_per_step": glaze.tokens_per_step,
        "micro_batch_tokens": glaze.micro_batch_tokens,
        "logit_chunk_tokens": glaze.logit_chunk_tokens,
        "eval_every_steps": glaze.eval_every_steps,
    }
    for field_name, value in counts.items():
        if value <= 0:
            raise ValueError(f"glaze {field_name} must be positive: {name}")
    if glaze.probe_steps < 0 or glaze.warmup_steps < 0:
        raise ValueError(f"glaze probe_steps and warmup_steps must not be negative: {name}")
    fractions = glaze.flip_fractions
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError(f"glaze flip fractions must be in (0, 1]: {name}")
    if not 0 <= glaze.momentum < 1:
        raise ValueError(f"glaze momentum must be in [0, 1): {name}")
    if not 0 < glaze.max_relative_change <= 1:
        raise ValueError(f"glaze max_relative_change must be in (0, 1]: {name}")
    if not 0 < glaze.max_memory_fraction <= 1 or glaze.max_train_minutes <= 0:
        raise ValueError(f"glaze memory fraction and time budget must be positive: {name}")
    block = calibration.max_sequence_length
    if glaze.micro_batch_tokens % block or glaze.tokens_per_step % glaze.micro_batch_tokens:
        raise ValueError(
            f"glaze micro_batch_tokens must be a multiple of the {block}-token block and divide "
            f"tokens_per_step: {name}"
        )
    blocks_per_step = glaze.tokens_per_step // block
    blocks_per_micro = glaze.micro_batch_tokens // block
    if glaze.train_blocks % blocks_per_step or glaze.dev_blocks % blocks_per_micro:
        raise ValueError(f"glaze block counts must fill whole steps and micro-batches: {name}")
    if glaze.warmup_steps >= glaze.train_blocks // blocks_per_step * glaze.epochs:
        raise ValueError(f"glaze warmup must be shorter than training: {name}")
    if glaze.train_start < 0 or (glaze.dev_start is not None and glaze.dev_start < 0):
        raise ValueError(f"glaze train_start and dev_start must not be negative: {name}")
    train = range(glaze.train_start, glaze.train_start + glaze.train_blocks)
    dev_start = glaze_dev_start(glaze)
    dev = range(dev_start, dev_start + glaze.dev_blocks)
    if set(train) & set(dev):
        raise ValueError(f"glaze training and dev blocks overlap: {name}")
    if init.method == "autoround":
        seen = init.calibration_samples
        if seen is None:
            # Without it there is no telling which blocks calibrated the init.
            raise ValueError(f"glaze needs its autoround init's calibration_samples: {name}")
        if glaze.train_start == 0 and glaze.train_blocks != seen:
            # B1's premise: the same calibration data as the init, so only the objective differs.
            raise ValueError(
                f"glaze train_blocks must equal the init's calibration samples: {name}"
            )
        if 0 < glaze.train_start < seen:
            # Otherwise the training blocks are fresh: none of them calibrated the init.
            raise ValueError(f"glaze fresh training blocks must start after the init's: {name}")
        if dev_start < seen:
            raise ValueError(f"glaze dev blocks must be ones the init never calibrated on: {name}")


def glaze_dev_start(glaze: GlazeConfig) -> int:
    """The first held-out block: `dev_start`, or the block right after the training blocks."""
    if glaze.dev_start is not None:
        return glaze.dev_start
    return glaze.train_start + glaze.train_blocks
