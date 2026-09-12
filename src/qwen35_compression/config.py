from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_METHODS = {"bf16", "int8", "gptq", "awq"}


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class CalibrationConfig:
    path: Path
    num_samples: int
    max_sequence_length: int
    seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    path: Path
    max_samples: int | None
    max_new_tokens: int
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    outputs: Path
    results: Path


@dataclass(frozen=True)
class VariantConfig:
    name: str
    method: str
    scheme: str | None = None
    bits: int | None = None
    group_size: int | None = None
    ignore: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExperimentConfig:
    phase: str
    model: ModelConfig
    calibration: CalibrationConfig
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
    calibration = CalibrationConfig(
        path=_resolve_path(calibration_raw["path"], root),
        num_samples=int(calibration_raw["num_samples"]),
        max_sequence_length=int(calibration_raw["max_sequence_length"]),
        seed=int(calibration_raw["seed"]),
    )
    evaluation_raw = raw["evaluation"]
    evaluation = EvaluationConfig(
        path=_resolve_path(evaluation_raw["path"], root),
        max_samples=evaluation_raw.get("max_samples"),
        max_new_tokens=int(evaluation_raw["max_new_tokens"]),
        seed=int(evaluation_raw["seed"]),
    )
    paths_raw = raw["paths"]
    paths = PathsConfig(
        outputs=_resolve_path(paths_raw["outputs"], root),
        results=_resolve_path(paths_raw["results"], root),
    )
    variants = tuple(
        VariantConfig(
            name=item["name"],
            method=item["method"],
            scheme=item.get("scheme"),
            bits=item.get("bits"),
            group_size=item.get("group_size"),
            ignore=tuple(item.get("ignore", ())),
        )
        for item in raw["variants"]
    )
    _validate(raw["phase"], model, calibration, evaluation, variants)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return ExperimentConfig(
        phase=raw["phase"],
        model=model,
        calibration=calibration,
        evaluation=evaluation,
        paths=paths,
        variants=variants,
        source_path=source_path,
        digest=hashlib.sha256(canonical).hexdigest(),
    )


def _validate(
    phase: str,
    model: ModelConfig,
    calibration: CalibrationConfig,
    evaluation: EvaluationConfig,
    variants: tuple[VariantConfig, ...],
) -> None:
    if not phase:
        raise ValueError("phase must be non-empty")
    if phase == "phase0" and model.id != "Qwen/Qwen3.5-0.8B":
        raise ValueError("Phase 0 is locked to Qwen/Qwen3.5-0.8B")
    if calibration.num_samples <= 0 or calibration.max_sequence_length <= 0:
        raise ValueError("calibration counts must be positive")
    if evaluation.max_samples is not None and evaluation.max_samples <= 0:
        raise ValueError("evaluation.max_samples must be positive or null")
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise ValueError("variant names must be unique")
    for variant in variants:
        if variant.method not in ALLOWED_METHODS:
            raise ValueError(f"unsupported method: {variant.method}")
        if variant.method in {"gptq", "awq"}:
            if variant.bits != 4 or variant.group_size not in {32, 64, 128}:
                raise ValueError(f"invalid W4A16 settings for {variant.name}")
