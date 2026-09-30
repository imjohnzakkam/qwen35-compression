from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from qwen35_compression.components import targets_for
from qwen35_compression.config import load_config
from qwen35_compression.feature1 import (
    BenchmarkSuite,
    build_lm_eval_command,
    build_vlm_eval_command,
    load_benchmark_suite,
    require_calibration_lock,
    vision_protocol,
)
from qwen35_compression.multimodal import require_multimodal_lock


def test_component_selectors_are_explicit_and_deduplicated() -> None:
    assert targets_for(("deltanet", "vision", "deltanet")) == (
        "re:.*linear_attn.*",
        "re:.*visual.*",
    )


def test_feature1_plan_covers_baselines_sensitivity_and_mixed_precision() -> None:
    plan = load_config(Path("configs/feature1.yaml"))

    assert plan.feature == "feature1"
    assert plan.model.id == "Qwen/Qwen3.5-4B"
    assert plan.calibration.source.dataset_id == "HuggingFaceH4/ultrachat_200k"
    assert plan.calibration.lock_path == Path("data/calibration/feature1.lock.json").resolve()

    names = {variant.name for variant in plan.variants}
    assert {
        "bf16",
        "int8_w8a8",
        "gptq_w4a16_g128",
        "gptq_w4a16_g64",
        "gptq_w4a16_g32",
        "awq_w4a16_g128",
        "sensitivity_deltanet_w4a16_g128",
        "sensitivity_attention_w4a16_g128",
        "sensitivity_ffn_w4a16_g128",
        "sensitivity_vision_w4a16_g128",
        "mixed_w8_sensitive_w4_standard",
    } <= names

    mixed = plan.variant("mixed_w8_sensitive_w4_standard")
    assert mixed.method == "mixed"
    assert {group.scheme for group in mixed.groups} == {"W4A16", "W8A16"}
    vision = plan.variant("sensitivity_vision_w4a16_g128")
    assert vision.requires_multimodal_calibration is True
    assert plan.multimodal_calibration is not None
    assert plan.multimodal_calibration.num_samples == 128
    assert plan.multimodal_calibration.source.dataset_id == "phiyodr/coco2017"


def test_feature1_benchmark_suite_is_explicit() -> None:
    suite = load_benchmark_suite(Path("configs/evaluation/feature1.yaml"))

    assert suite.text_tasks == (
        "mmlu_pro",
        "gsm8k",
        "minerva_math500",
        "ifeval",
        "hellaswag",
        "arc_challenge",
        "wikitext",
    )
    assert suite.vision_tasks == (
        "MMBench_DEV_EN_V11",
        "MMMU_DEV_VAL",
        "MathVista_MINI",
        "TextVQA_VAL",
        "OCRBench",
        "DocVQA_VAL",
    )


def test_lm_eval_command_is_deterministic() -> None:
    suite = BenchmarkSuite(
        text_tasks=("mmlu_pro", "gsm8k"),
        vision_tasks=(),
        fewshot=0,
        text_backend="hf-multimodal",
        batch_size="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        limit=3,
        seed=42,
        generation={},
    )

    command = build_lm_eval_command(
        Path("outputs/model"), suite, Path("results/text"), "abc123", "2"
    )

    assert command[1:4] == ["-m", "lm_eval", "--model"]
    assert command[4] == "hf-multimodal"
    assert "--apply_chat_template" in command
    assert "--log_samples" in command
    assert command[command.index("--tasks") + 1] == "mmlu_pro,gsm8k"
    assert "revision=abc123" in command[command.index("--model_args") + 1]
    assert command[-2:] == ["--limit", "2"]


def test_vlm_eval_command_uses_pinned_task_list() -> None:
    suite = BenchmarkSuite(
        text_tasks=(),
        vision_tasks=("MMMU_DEV_VAL", "OCRBench"),
        fewshot=0,
        text_backend="vllm",
        batch_size="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        limit=None,
        seed=42,
        generation={},
        vision_max_model_len=32768,
    )

    command = build_vlm_eval_command(
        Path("outputs/model"), suite, Path("results/vision"), Path("vendor/vlmeval")
    )

    assert command[1].endswith("scripts/vlmeval_qwen35.py")
    assert command[command.index("--data") + 1 : command.index("--max-new-tokens")] == [
        "MMMU_DEV_VAL",
        "OCRBench",
    ]
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.85"
    assert command[-1] == "--disable-thinking"


def test_pinned_suite_leaves_room_for_image_prompts() -> None:
    suite = load_benchmark_suite(Path("configs/evaluation/feature1.yaml"))
    max_gen_toks = int(suite.generation["max_gen_toks"])

    # The text protocol that produced the BF16 text baseline is unchanged.
    assert suite.max_model_len == 4096
    assert suite.vision_max_model_len - max_gen_toks >= 16384
    assert vision_protocol(suite)["max_model_len"] == suite.vision_max_model_len


def test_vllm_command_uses_single_gpu_optimized_backend() -> None:
    suite = load_benchmark_suite(Path("configs/evaluation/feature1.yaml"))
    command = build_lm_eval_command(
        Path("Qwen/Qwen3.5-4B"),
        suite,
        Path("results/text"),
        "abc123",
        python_executable=Path(".venv-gpu-text/bin/python"),
    )

    assert command[0] == ".venv-gpu-text/bin/python"
    assert command[command.index("--model") + 1] == "vllm"
    model_args = command[command.index("--model_args") + 1]
    assert "dtype=bfloat16" in model_args
    assert "max_model_len=4096" in model_args
    assert "gpu_memory_utilization=0.85" in model_args
    # The 4B chat template thinks by default; the suite pins instruct mode explicitly.
    assert suite.enable_thinking is False
    assert "enable_thinking=False" in model_args
    gen_kwargs = command[command.index("--gen_kwargs") + 1 :]
    assert "max_gen_toks=2048" in gen_kwargs


def test_calibration_lock_rejects_changed_data(tmp_path: Path) -> None:
    plan = load_config(Path("configs/feature1.yaml")).calibration
    data_path = tmp_path / "calibration.jsonl"
    lock_path = tmp_path / "calibration.lock.json"
    data_path.write_text('{"messages": []}\n', encoding="utf-8")
    lock_path.write_text(
        json.dumps({"content_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    local_plan = type(plan)(
        path=data_path,
        lock_path=lock_path,
        num_samples=plan.num_samples,
        max_sequence_length=plan.max_sequence_length,
        seed=plan.seed,
        source=plan.source,
    )
    assert require_calibration_lock(local_plan)["content_sha256"]

    data_path.write_text('{"messages": [{"role": "user"}]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="provenance lock"):
        require_calibration_lock(local_plan)


def test_multimodal_lock_verifies_jsonl_and_images(tmp_path: Path) -> None:
    configured = load_config(Path("configs/feature1.yaml")).multimodal_calibration
    assert configured is not None
    data_path = tmp_path / "data" / "calibration" / "multimodal.jsonl"
    assets_dir = data_path.parent / "images"
    lock_path = data_path.parent / "multimodal.lock.json"
    assets_dir.mkdir(parents=True)
    data_path.write_text('{"messages": []}\n', encoding="utf-8")
    image_path = assets_dir / "0000.jpg"
    image_path.write_bytes(b"test-image")
    lock_path.write_text(
        json.dumps(
            {
                "content_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "images": [
                    {
                        "path": "data/calibration/images/0000.jpg",
                        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    local_config = type(configured)(
        path=data_path,
        assets_dir=assets_dir,
        lock_path=lock_path,
        num_samples=1,
        seed=configured.seed,
        source=configured.source,
    )
    assert require_multimodal_lock(local_config)["images"]

    image_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image does not match"):
        require_multimodal_lock(local_config)


def test_variant_runner_plans_quantize_then_shared_protocol() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_feature1_variant.py",
            "--variant",
            "gptq_w4a16_g128",
            "--code-revision",
            "abc123",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    export_dir = plan["export_dir"]
    assert export_dir.endswith("outputs/feature1/gptq_w4a16_g128")
    assert plan["quantize"][1:] == [
        "scripts/quantize.py",
        "--config",
        str(Path("configs/feature1.yaml").resolve()),
        "--variant",
        "gptq_w4a16_g128",
    ]
    text = plan["text"]
    assert text[0].endswith(".venv-gpu-text/bin/python")
    assert f"pretrained={export_dir}" in text[text.index("--model_args") + 1]
    assert "enable_thinking=False" in text[text.index("--model_args") + 1]
    assert "revision=" not in text[text.index("--model_args") + 1]
    vision = plan["vision"]
    assert vision[0].endswith(".venv-gpu-vision/bin/python")
    assert vision[vision.index("--model-path") + 1] == export_dir
    assert vision[-1] == "--disable-thinking"

    rejected = subprocess.run(
        [
            sys.executable,
            "scripts/run_feature1_variant.py",
            "--variant",
            "bf16",
            "--code-revision",
            "abc123",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "run_feature1_bf16.py" in rejected.stderr
