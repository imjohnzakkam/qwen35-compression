from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_METHODS = {"bf16", "int8", "gptq", "awq", "autoround", "mixed"}


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
        path=(
            _resolve_path(evaluation_raw["path"], root)
            if evaluation_raw.get("path")
            else None
        ),
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
        if variant.method == "mixed" and not variant.groups:
            raise ValueError(f"mixed variant requires groups: {variant.name}")
        if variant.method != "mixed" and variant.groups:
            raise ValueError(f"only mixed variants may define groups: {variant.name}")
