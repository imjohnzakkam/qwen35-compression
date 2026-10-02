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
    # Image prompts run far longer than text ones; a prompt that does not fit beside
    # max_gen_toks is rejected by vLLM and VLMEvalKit scores the failed request as wrong.
    vision_max_model_len: int = 4096
    # MCQ-style answers are free-form explanations the rules cannot parse; these datasets are
    # scored after download by one fixed extractor, identical for every variant.
    vision_answer_extractor: str | None = None
    vision_extracted_tasks: tuple[str, ...] = ()
    # Thinking-mode suites: lm-eval strips everything up to this token before extracting answers.
    think_end_token: str | None = None
    # Sampled suites run once per seed; an empty tuple means one run with `seed`.
    seeds: tuple[int, ...] = ()
    # Fixed per-task document indices (lm-eval --samples), for reproducible subsets.
    samples_path: Path | None = None
    source_path: Path | None = None
    digest: str | None = None


# lm-eval rejects likelihood-scored tasks when enable_thinking=True.
LOGLIKELIHOOD_TASKS = frozenset({"hellaswag", "arc_challenge", "arc_easy", "wikitext"})
# Room a prompt needs beside max_gen_toks. lm-eval silently truncates the start of any prompt
# that does not fit, so a suite that leaves less than this is rejected.
MIN_PROMPT_TOKENS = 2048
# Image prompts (MMMU carries up to 8 images) need far more room than text ones.
MIN_VISION_PROMPT_TOKENS = 16384


def load_benchmark_suite(path: Path) -> BenchmarkSuite:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected a YAML mapping in {path}")
    vision_tasks = tuple(str(task) for task in raw["vision_tasks"])
    extracted = tuple(str(task) for task in raw.get("vision_extracted_tasks", ()))
    extractor = raw.get("vision_answer_extractor")
    if extracted and not extractor:
        raise ValueError("vision_extracted_tasks requires vision_answer_extractor")
    unknown = sorted(set(extracted) - set(vision_tasks))
    if unknown:
        raise ValueError(f"vision_extracted_tasks not in vision_tasks: {unknown}")
    text_tasks = tuple(str(task) for task in raw["text_tasks"])
    enable_thinking = bool(raw.get("enable_thinking", False))
    think_end_token = raw.get("think_end_token")
    if enable_thinking and not think_end_token:
        raise ValueError("enable_thinking requires think_end_token")
    likelihood = sorted(set(text_tasks) & LOGLIKELIHOOD_TASKS)
    if enable_thinking and likelihood:
        raise ValueError(f"thinking suites cannot score likelihood tasks: {likelihood}")
    if enable_thinking and vision_tasks:
        raise ValueError("thinking suites are text-only")
    max_model_len = int(raw.get("max_model_len", 4096))
    vision_max_model_len = int(raw.get("vision_max_model_len", max_model_len))
    max_gen_toks = int(dict(raw.get("generation", {})).get("max_gen_toks", 256))
    if text_tasks and max_model_len - max_gen_toks < MIN_PROMPT_TOKENS:
        raise ValueError(
            f"max_model_len {max_model_len} leaves under {MIN_PROMPT_TOKENS} prompt tokens "
            f"beside max_gen_toks {max_gen_toks}"
        )
    if vision_tasks and vision_max_model_len - max_gen_toks < MIN_VISION_PROMPT_TOKENS:
        raise ValueError(
            f"vision_max_model_len {vision_max_model_len} leaves under "
            f"{MIN_VISION_PROMPT_TOKENS} prompt tokens beside max_gen_toks {max_gen_toks}"
        )
    samples = raw.get("samples_path")
    samples_path = (path.parent / str(samples)).resolve() if samples else None
    if samples_path is not None and not samples_path.is_file():
        raise FileNotFoundError(f"samples_path not found: {samples_path}")
    if samples_path is not None and raw.get("limit") is not None:
        raise ValueError("samples_path and limit are mutually exclusive")
    return BenchmarkSuite(
        text_tasks=text_tasks,
        vision_tasks=vision_tasks,
        fewshot=int(raw.get("fewshot", 0)),
        text_backend=str(raw.get("text_backend", "hf-multimodal")),
        batch_size=str(raw.get("batch_size", "auto")),
        max_model_len=max_model_len,
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.85)),
        limit=(int(raw["limit"]) if raw.get("limit") is not None else None),
        seed=int(raw.get("seed", 42)),
        generation=dict(raw.get("generation", {})),
        enable_thinking=enable_thinking,
        vision_max_model_len=vision_max_model_len,
        vision_answer_extractor=str(extractor) if extractor else None,
        vision_extracted_tasks=extracted,
        think_end_token=str(think_end_token) if think_end_token else None,
        seeds=tuple(int(seed) for seed in raw.get("seeds", ())),
        samples_path=samples_path,
        source_path=path.resolve(),
        digest=hashlib.sha256(raw_bytes).hexdigest(),
    )


def suite_record(suite: BenchmarkSuite) -> dict[str, Any]:
    """What a run manifest records about the benchmark suite that produced it."""
    return {
        "path": str(suite.source_path) if suite.source_path else None,
        "sha256": suite.digest,
        "enable_thinking": suite.enable_thinking,
        "max_gen_toks": int(suite.generation.get("max_gen_toks", 256)),
        "max_model_len": suite.max_model_len,
        "seeds": list(suite.seeds) or [suite.seed],
        "samples": (
            {"path": str(suite.samples_path), "sha256": _file_sha256(suite.samples_path)}
            if suite.samples_path
            else None
        ),
    }


def suite_label(suite: BenchmarkSuite, default_suite: Path | None) -> str | None:
    """None for the config's own suite, else a short name for output paths ("thinking")."""
    if suite.source_path is None or (
        default_suite is not None and suite.source_path == default_suite.resolve()
    ):
        return None
    return suite.source_path.stem.removeprefix("feature1_")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    seed_override: int | None = None,
    device: str | None = None,
    batch_size_override: str | None = None,
) -> list[str]:
    backend = backend_override or suite.text_backend
    seed = suite.seed if seed_override is None else seed_override
    if backend not in {"hf", "hf-multimodal", "vllm"}:
        raise ValueError(f"unsupported text backend: {backend}")
    model_args = [f"pretrained={model_path}"]
    if backend == "hf-multimodal":
        model_args.append("trust_remote_code=True")
    elif backend == "hf":
        # Local validation: same prompt + answer budget as the vLLM runs.
        model_args.extend(("dtype=bfloat16", f"max_length={suite.max_model_len}"))
    else:
        model_args.extend(
            (
                "dtype=bfloat16",
                f"max_model_len={suite.max_model_len}",
                f"gpu_memory_utilization={suite.gpu_memory_utilization}",
                # vLLM's sampling seed; greedy suites are unaffected by it.
                f"seed={seed}",
            )
        )
    if revision:
        model_args.append(f"revision={revision}")
    # Passed to apply_chat_template by both the hf-multimodal and vllm backends.
    model_args.append(f"enable_thinking={suite.enable_thinking}")
    if suite.think_end_token:
        model_args.append(f"think_end_token={suite.think_end_token}")
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
        batch_size_override or suite.batch_size,
        "--apply_chat_template",
        "--seed",
        str(seed),
        "--gen_kwargs",
        *(f"{key}={value!r}" for key, value in sorted(suite.generation.items())),
        "--log_samples",
        "--output_path",
        str(output_path),
    ]
    if device is not None:
        command.extend(("--device", device))
    limit = limit_override if limit_override is not None else suite.limit
    if limit is not None:
        command.extend(("--limit", str(limit)))
    elif suite.samples_path is not None:
        command.extend(("--samples", str(suite.samples_path)))
    return command


def build_text_eval_commands(
    model_path: Path,
    suite: BenchmarkSuite,
    output_path: Path,
    revision: str | None = None,
    limit_override: str | None = None,
    python_executable: str | Path | None = None,
) -> list[tuple[str, list[str]]]:
    """One lm-eval command per seed for sampled suites, else a single command."""
    if not suite.seeds:
        return [
            (
                "text",
                build_lm_eval_command(
                    model_path,
                    suite,
                    output_path,
                    revision,
                    limit_override=limit_override,
                    backend_override="vllm",
                    python_executable=python_executable,
                ),
            )
        ]
    return [
        (
            f"text_seed{seed}",
            build_lm_eval_command(
                model_path,
                suite,
                output_path / f"seed-{seed}",
                revision,
                limit_override=limit_override,
                backend_override="vllm",
                python_executable=python_executable,
                seed_override=seed,
            ),
        )
        for seed in suite.seeds
    ]


# Datasets VLMEvalKit will not score without an LLM judge; the wrapper uses the served
# checkpoint itself for answer extraction there, so record that in every manifest.
def vision_protocol(suite: BenchmarkSuite) -> dict[str, Any]:
    return {
        "inference": "vllm openai server via VLMEvalKit LMDeployAPI",
        "temperature": 0.0,
        "max_tokens": int(suite.generation.get("max_gen_toks", 256)),
        "max_model_len": suite.vision_max_model_len,
        "enable_thinking": suite.enable_thinking,
        "judge": {
            "default": "exact_matching",
            **{
                name: f"{suite.vision_answer_extractor} answer extraction after download"
                for name in suite.vision_extracted_tasks
            },
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
    extracted = (
        ["--extracted-data", *suite.vision_extracted_tasks] if suite.vision_extracted_tasks else []
    )
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
        str(suite.vision_max_model_len),
        "--gpu-memory-utilization",
        str(suite.gpu_memory_utilization),
        "--seed",
        str(suite.seed),
        *extracted,
        *thinking,
    ]


def vlmeval_run_prefix(toolkit_dir: Path, limit: int | None = None) -> list[str]:
    """VLMEvalKit's run.py through scripts/vlmeval_run.py, which applies the fixed final-answer
    rule before any random multiple-choice fill and, for local validation only, a row limit."""
    wrapper = Path(__file__).resolve().parents[2] / "scripts" / "vlmeval_run.py"
    prefix = [str(wrapper), "--toolkit-dir", str(toolkit_dir)]
    if limit is not None:
        prefix.extend(("--limit", str(limit)))
    return [*prefix, "--"]


def build_vision_score_command(
    suite: BenchmarkSuite,
    work_dir: Path,
    model_alias: str,
    toolkit_dir: Path,
    python_executable: Path,
    api_nproc: int = 4,
    limit: int | None = None,
) -> list[str]:
    """VLMEvalKit eval-only command that scores saved predictions with the fixed extractor."""
    if not suite.vision_extracted_tasks or suite.vision_answer_extractor is None:
        raise ValueError("the benchmark suite defines no extractor-scored vision tasks")
    return [
        str(python_executable),
        *vlmeval_run_prefix(toolkit_dir, limit),
        "--model",
        model_alias,
        "--data",
        *suite.vision_extracted_tasks,
        "--work-dir",
        str(work_dir),
        "--mode",
        "eval",
        "--reuse",
        "--judge",
        suite.vision_answer_extractor,
        # VLMEvalKit's default of 32 concurrent requests trips OpenAI rate limits, and a
        # failed extraction becomes a random option; stay well under the account limit.
        "--judge-api-nproc",
        str(api_nproc),
        # run.py builds the inference model even in eval mode; nothing is sent to it.
        "--base-url",
        "http://127.0.0.1:9/v1",
        "--model-class",
        "LMDeployAPI",
        "--key",
        "unused",
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
