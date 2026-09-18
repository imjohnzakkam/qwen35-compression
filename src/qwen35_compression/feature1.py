"""Feature 1 calibration and benchmark helpers."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from qwen35_compression.config import CalibrationConfig


@dataclass(frozen=True)
class BenchmarkSuite:
    text_tasks: tuple[str, ...]
    vision_tasks: tuple[str, ...]
    fewshot: int
    text_backend: str
    batch_size: str
    max_model_len: int
    gpu_memory_utilization: float
    limit: int | None
    seed: int
    generation: Mapping[str, Any]
    enable_thinking: bool = False


def load_benchmark_suite(path: Path) -> BenchmarkSuite:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected a YAML mapping in {path}")
    return BenchmarkSuite(
        text_tasks=tuple(str(task) for task in raw["text_tasks"]),
        vision_tasks=tuple(str(task) for task in raw["vision_tasks"]),
        fewshot=int(raw.get("fewshot", 0)),
        text_backend=str(raw.get("text_backend", "hf-multimodal")),
        batch_size=str(raw.get("batch_size", "auto")),
        max_model_len=int(raw.get("max_model_len", 4096)),
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.85)),
        limit=(int(raw["limit"]) if raw.get("limit") is not None else None),
        seed=int(raw.get("seed", 42)),
        generation=dict(raw.get("generation", {})),
        enable_thinking=bool(raw.get("enable_thinking", False)),
    )


def prepare_calibration_dataset(config: CalibrationConfig) -> Mapping[str, Any]:
    """Materialize a deterministic calibration subset and provenance lock."""
    if config.source is None or config.lock_path is None:
        raise ValueError("calibration source and lock_path are required")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    source = config.source
    info = HfApi().dataset_info(source.dataset_id, revision=source.revision)
    resolved_revision = info.sha
    if not resolved_revision:
        raise ValueError("Hugging Face did not return a dataset commit SHA")
    dataset = load_dataset(
        source.dataset_id,
        split=source.split,
        revision=resolved_revision,
    )
    if len(dataset) < config.num_samples:
        raise ValueError(
            f"dataset has {len(dataset)} rows; {config.num_samples} were requested"
        )

    indices = random.Random(config.seed).sample(range(len(dataset)), config.num_samples)
    records = [
        _normalise_messages(dataset[index][source.messages_column]) for index in indices
    ]
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    contract = {
        "dataset_id": source.dataset_id,
        "revision": resolved_revision,
        "split": source.split,
        "messages_column": source.messages_column,
        "seed": config.seed,
        "num_samples": config.num_samples,
        "max_sequence_length": config.max_sequence_length,
    }
    lock = {
        "schema_version": 1,
        **contract,
        "source_indices": indices,
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }

    config.path.parent.mkdir(parents=True, exist_ok=True)
    config.lock_path.parent.mkdir(parents=True, exist_ok=True)
    config.path.write_bytes(encoded)
    config.lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock


def _normalise_messages(messages: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    normalised = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    if not normalised:
        raise ValueError("calibration conversations must not be empty")
    return {"messages": normalised}


def build_lm_eval_command(
    model_path: Path,
    suite: BenchmarkSuite,
    output_path: Path,
    revision: str | None = None,
    limit_override: str | None = None,
    backend_override: str | None = None,
    python_executable: str | Path | None = None,
) -> list[str]:
    backend = backend_override or suite.text_backend
    if backend not in {"hf-multimodal", "vllm"}:
        raise ValueError(f"unsupported text backend: {backend}")
    model_args = [f"pretrained={model_path}"]
    if backend == "hf-multimodal":
        model_args.append("trust_remote_code=True")
    else:
        model_args.extend(
            (
                "dtype=bfloat16",
                f"max_model_len={suite.max_model_len}",
                f"gpu_memory_utilization={suite.gpu_memory_utilization}",
            )
        )
    if revision:
        model_args.append(f"revision={revision}")
    # Passed to apply_chat_template by both the hf-multimodal and vllm backends.
    model_args.append(f"enable_thinking={suite.enable_thinking}")
    command = [
        str(python_executable or sys.executable),
        "-m",
        "lm_eval",
        "--model",
        backend,
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(suite.text_tasks),
        "--num_fewshot",
        str(suite.fewshot),
        "--batch_size",
        suite.batch_size,
        "--apply_chat_template",
        "--seed",
        str(suite.seed),
        "--gen_kwargs",
        *(f"{key}={value!r}" for key, value in sorted(suite.generation.items())),
        "--log_samples",
        "--output_path",
        str(output_path),
    ]
    limit = limit_override if limit_override is not None else suite.limit
    if limit is not None:
        command.extend(("--limit", str(limit)))
    return command


# Datasets VLMEvalKit will not score without an LLM judge; the wrapper uses the served
# checkpoint itself for answer extraction there, so record that in every manifest.
VISION_JUDGED_DATASETS = ("MathVista_MINI",)


def vision_protocol(suite: BenchmarkSuite) -> dict[str, Any]:
    return {
        "inference": "vllm openai server via VLMEvalKit LMDeployAPI",
        "temperature": 0.0,
        "max_tokens": int(suite.generation.get("max_gen_toks", 256)),
        "enable_thinking": suite.enable_thinking,
        "judge": {
            "default": "exact_matching",
            **{name: "served checkpoint (extraction only)" for name in VISION_JUDGED_DATASETS},
        },
    }


def build_vlm_eval_command(
    model_path: Path,
    suite: BenchmarkSuite,
    output_dir: Path,
    toolkit_dir: Path = Path("external/VLMEvalKit"),
) -> list[str]:
    wrapper = Path(__file__).resolve().parents[2] / "scripts" / "vlmeval_qwen35.py"
    thinking = [] if suite.enable_thinking else ["--disable-thinking"]
    return [
        sys.executable,
        str(wrapper),
        "--toolkit-dir",
        str(toolkit_dir),
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--data",
        *suite.vision_tasks,
        "--max-new-tokens",
        str(suite.generation.get("max_gen_toks", 256)),
        "--max-model-len",
        str(suite.max_model_len),
        "--gpu-memory-utilization",
        str(suite.gpu_memory_utilization),
        "--seed",
        str(suite.seed),
        *thinking,
    ]


def require_calibration_lock(config: CalibrationConfig) -> Mapping[str, Any]:
    if config.lock_path is None:
        raise ValueError("calibration.lock_path is required")
    lock = json.loads(config.lock_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(config.path.read_bytes()).hexdigest()
    if digest != lock["content_sha256"]:
        raise ValueError("calibration data does not match its provenance lock")
    return lock


def run_command(command: Sequence[str]) -> None:
    import subprocess

    subprocess.run(list(command), check=True)


# lm-eval never shuts the vLLM engine down, and the detached EngineCore child kept the
# harness alive indefinitely after results were written. An in-process engine exits.
EVALUATOR_ENV = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}


def run_logged(command: Sequence[str], log_path: Path, cwd: Path) -> float:
    """Run a step, tee its output to the run log, and return its wall-clock seconds."""
    import os
    import subprocess
    import time

    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write("+ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env={**os.environ, **EVALUATOR_ENV},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))
    return time.perf_counter() - started
