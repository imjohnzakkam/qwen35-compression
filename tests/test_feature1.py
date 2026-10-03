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
    build_text_eval_commands,
    build_vision_score_command,
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
        "autoround_w4a16_g128",
    } <= names
    autoround = plan.variant("autoround_w4a16_g128")
    assert (autoround.method, autoround.bits, autoround.group_size) == ("autoround", 4, 128)
    assert "lm_head" in autoround.ignore
    # AutoRound's own default sample count; the other methods use the whole calibration set.
    assert autoround.calibration_samples == 128
    assert plan.variant("gptq_w4a16_g128").calibration_samples is None

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


def test_pinned_suite_leaves_room_for_long_answers_and_prompts() -> None:
    suite = load_benchmark_suite(Path("configs/evaluation/feature1.yaml"))
    max_gen_toks = int(suite.generation["max_gen_toks"])

    # At 2048, 8-15% of MMLU-Pro, MATH-500 and MMMU answers were cut off and scored wrong.
    assert max_gen_toks == 8192
    # lm-eval silently truncates prompts that do not fit beside the answer budget.
    assert suite.max_model_len - max_gen_toks >= 4096
    assert suite.vision_max_model_len - max_gen_toks >= 16384
    assert vision_protocol(suite)["max_tokens"] == max_gen_toks
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
    assert "max_model_len=12288" in model_args
    assert "gpu_memory_utilization=0.85" in model_args
    assert "max_num_batched_tokens=4096" in model_args
    assert "seed=42" in model_args
    assert "think_end_token" not in model_args
    assert "--samples" not in command
    # The 4B chat template thinks by default; the suite pins instruct mode explicitly.
    assert suite.enable_thinking is False
    assert "enable_thinking=False" in model_args
    gen_kwargs = command[command.index("--gen_kwargs") + 1 :]
    assert "max_gen_toks=8192" in gen_kwargs


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
    assert list(plan["text"]) == ["text"]
    text = plan["text"]["text"]
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


def _extractor_suite(**overrides: object) -> BenchmarkSuite:
    fields: dict[str, object] = {
        "text_tasks": (),
        "vision_tasks": ("MMMU_DEV_VAL", "OCRBench"),
        "fewshot": 0,
        "text_backend": "vllm",
        "batch_size": "auto",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.85,
        "limit": None,
        "seed": 42,
        "generation": {},
        "vision_answer_extractor": "gpt-4o-mini",
        "vision_extracted_tasks": ("MMMU_DEV_VAL",),
    }
    fields.update(overrides)
    return BenchmarkSuite(**fields)  # type: ignore[arg-type]


def test_pinned_suite_scores_free_form_mcq_with_one_fixed_extractor() -> None:
    suite = load_benchmark_suite(Path("configs/evaluation/feature1.yaml"))

    assert suite.vision_answer_extractor == "gpt-4o-mini"
    assert suite.vision_extracted_tasks == ("MMBench_DEV_EN_V11", "MMMU_DEV_VAL", "MathVista_MINI")
    judge = vision_protocol(suite)["judge"]
    assert judge["default"] == "exact_matching"
    assert judge["MMMU_DEV_VAL"] == "gpt-4o-mini answer extraction after download"
    assert "TextVQA_VAL" not in judge


def test_suite_rejects_extracted_tasks_outside_the_vision_suite(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(
        "text_tasks: []\nvision_tasks: [OCRBench]\n"
        "vision_answer_extractor: gpt-4o-mini\nvision_extracted_tasks: [MMMU_DEV_VAL]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not in vision_tasks"):
        load_benchmark_suite(path)

    path.write_text(
        "text_tasks: []\nvision_tasks: [MMMU_DEV_VAL]\nvision_extracted_tasks: [MMMU_DEV_VAL]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires vision_answer_extractor"):
        load_benchmark_suite(path)


def test_vision_commands_split_rule_and_extractor_scoring() -> None:
    suite = _extractor_suite()

    gpu = build_vlm_eval_command(Path("model"), suite, Path("results/vision"))
    assert gpu[gpu.index("--extracted-data") + 1 : -1] == ["MMMU_DEV_VAL"]
    assert gpu[-1] == "--disable-thinking"

    local = build_vision_score_command(
        suite, Path("scored"), "Qwen3.5-4B-pinned", Path("vendor/vlmeval"), Path("py")
    )
    assert local[0] == "py"
    assert local[1].endswith("scripts/vlmeval_run.py")
    assert local[2:5] == ["--toolkit-dir", "vendor/vlmeval", "--"]
    assert local[local.index("--data") + 1 : local.index("--work-dir")] == ["MMMU_DEV_VAL"]
    assert local[local.index("--mode") + 1] == "eval"
    assert local[local.index("--judge") + 1] == "gpt-4o-mini"
    assert "--reuse" in local
    assert local[local.index("--judge-api-nproc") + 1] == "4"

    limited = build_vision_score_command(
        suite, Path("scored"), "m", Path("vendor/vlmeval"), Path("py"), limit=10
    )
    assert limited[2:7] == ["--toolkit-dir", "vendor/vlmeval", "--limit", "10", "--"]

    with pytest.raises(ValueError, match="no extractor-scored"):
        build_vision_score_command(
            _extractor_suite(vision_extracted_tasks=()), Path("s"), "m", Path("v"), Path("p")
        )


def test_score_vision_dry_run_finds_saved_predictions(tmp_path: Path) -> None:
    run = tmp_path / "feature1-bf16-20260930T000000Z"
    model_dir = run / "vision" / "Qwen3.5-4B-pinned" / "T20260930-000000"
    model_dir.mkdir(parents=True)
    for task in ("MMBench_DEV_EN_V11", "MMMU_DEV_VAL", "MathVista_MINI"):
        (model_dir / f"Qwen3.5-4B-pinned_{task}.xlsx").write_bytes(b"x")
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "passed", "research_result": True}), encoding="utf-8"
    )

    result = subprocess.run(
        [sys.executable, "scripts/score_vision.py", "--run-dir", str(run), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )

    plan = json.loads(result.stdout)
    assert plan["output"] == str(run.with_name(run.name + "-scored").resolve())
    assert sorted(plan["predictions"]) == ["MMBench_DEV_EN_V11", "MMMU_DEV_VAL", "MathVista_MINI"]
    command = plan["command"]
    assert command[command.index("--model") + 1] == "Qwen3.5-4B-pinned"
    assert command[command.index("--work-dir") + 1] == plan["output"]


def test_score_vision_scores_the_suite_the_run_used(tmp_path: Path) -> None:
    from qwen35_compression.feature1 import suite_record

    run = tmp_path / "feature1-autoround-panel"
    model_dir = run / "vision" / "Qwen3.5-4B-pinned" / "T20261003-000000"
    model_dir.mkdir(parents=True)
    (model_dir / "Qwen3.5-4B-pinned_MMMU_DEV_VAL.xlsx").write_bytes(b"x")
    panel = load_benchmark_suite(Path("configs/evaluation/feature1_panel.yaml"))
    remote = "/remote/checkout/configs/evaluation/feature1_panel.yaml"
    record = {**suite_record(panel), "path": remote}
    manifest = {"status": "passed", "research_result": True, "benchmark_suite": record}
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/score_vision.py", "--run-dir", str(run), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert sorted(plan["predictions"]) == ["MMMU_DEV_VAL"]
    command = plan["command"]
    assert command[command.index("--data") + 1 : command.index("--work-dir")] == ["MMMU_DEV_VAL"]

    record["sha256"] = "0" * 64
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    mismatch = subprocess.run(
        [sys.executable, "scripts/score_vision.py", "--run-dir", str(run), "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode != 0 and "sha256 mismatch" in mismatch.stderr


def test_score_vision_redacts_the_api_key_from_written_files(tmp_path: Path) -> None:
    import importlib.util

    sys.path.insert(0, "scripts")
    spec = importlib.util.spec_from_file_location("score_vision", "scripts/score_vision.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    secret = "sk-proj-" + "x" * 40
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.log").write_text(f"API Key: {secret}\n", encoding="utf-8")
    (tmp_path / "clean.txt").write_text("nothing here\n", encoding="utf-8")

    redacted = module.redact_tree(tmp_path, secret)

    assert redacted == ["logs/run.log"]
    log = (tmp_path / "logs" / "run.log").read_text(encoding="utf-8")
    assert secret not in log
    assert "[REDACTED_OPENAI_API_KEY]" in log


THINKING_SUITE = Path("configs/evaluation/feature1_thinking.yaml")


def test_thinking_suite_runs_each_seed_with_qwen_sampling_and_a_fixed_subset() -> None:
    suite = load_benchmark_suite(THINKING_SUITE)

    assert suite.enable_thinking and suite.think_end_token == "</think>"
    assert suite.vision_tasks == ()
    assert suite.seeds == (42, 43)
    assert suite.generation["temperature"] == 1.0 and suite.generation["top_k"] == 20
    assert suite.max_model_len - int(suite.generation["max_gen_toks"]) >= 2048
    commands = build_text_eval_commands(
        Path("model"), suite, Path("out/text"), "rev", python_executable="py"
    )

    assert [label for label, _ in commands] == ["text_seed42", "text_seed43"]
    for (_, command), seed in zip(commands, (42, 43), strict=True):
        model_args = command[command.index("--model_args") + 1].split(",")
        assert f"seed={seed}" in model_args
        assert "enable_thinking=True" in model_args
        assert "think_end_token=</think>" in model_args
        assert command[command.index("--seed") + 1] == str(seed)
        assert command[command.index("--samples") + 1] == str(suite.samples_path)
        assert command[command.index("--output_path") + 1] == f"out/text/seed-{seed}"
        assert "--limit" not in command


def test_committed_thinking_subset_matches_its_generator() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("subset", "scripts/make_thinking_subset.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = json.loads(
        Path("configs/evaluation/feature1_thinking_samples.json").read_text(encoding="utf-8")
    )

    assert committed == module.stratified_subset(module.MMLU_PRO_SIZES, 1000, 42)
    assert sum(len(indices) for indices in committed.values()) == 1001
    for task, indices in committed.items():
        assert indices == sorted(set(indices))
        assert 0 <= indices[0] and indices[-1] < module.MMLU_PRO_SIZES[task]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("enable_thinking: true\ntext_tasks: [ifeval]\n", "requires think_end_token"),
        (
            "enable_thinking: true\nthink_end_token: x\ntext_tasks: [hellaswag]\n",
            "cannot score likelihood tasks",
        ),
        (
            "text_tasks: [ifeval]\nmax_model_len: 4096\ngeneration: {max_gen_toks: 4000}\n",
            "prompt tokens",
        ),
        (
            "text_tasks: [ifeval]\nsamples_path: subset.json\nlimit: 3\n",
            "mutually exclusive",
        ),
    ],
)
def test_suite_rejects_unsafe_settings(tmp_path: Path, body: str, message: str) -> None:
    (tmp_path / "subset.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "suite.yaml"
    path.write_text("vision_tasks: []\n" + body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_benchmark_suite(path)


def test_bf16_driver_runs_the_thinking_track_text_only_into_its_own_directory() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_feature1_bf16.py",
            "--suite",
            str(THINKING_SUITE),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    plan = json.loads(result.stdout)
    assert plan["vision"] is None
    assert plan["bootstrap"][-2:] == ["--scope", "text"]
    assert list(plan["text"]) == ["text_seed42", "text_seed43"]
    output = plan["text"]["text_seed42"][plan["text"]["text_seed42"].index("--output_path") + 1]
    assert output.endswith("results/feature1/bf16-thinking/text/seed-42")
    record = plan["benchmark_suite"]
    assert record["enable_thinking"] is True and record["seeds"] == [42, 43]
    assert record["samples"]["path"].endswith("feature1_thinking_samples.json")


def test_sensitivity_targets_hit_only_their_linear_layers() -> None:
    import re

    plan = load_config(Path("configs/feature1.yaml"))
    modules = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.linear_attn.conv1d",
        "model.language_model.layers.0.linear_attn.norm",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.self_attn.q_norm",
        "model.language_model.layers.3.mlp.down_proj",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.blocks.0.attn.qkv",
        "lm_head",
    ]

    def hits(variant: str) -> set[str]:
        patterns = [t.removeprefix("re:") for t in plan.variant(variant).targets]
        return {m for m in modules if any(re.match(p, m) for p in patterns)}

    assert hits("sensitivity_deltanet_w4a16_g128") == {
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.out_proj",
    }
    assert hits("sensitivity_attention_w4a16_g128") == {
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.o_proj",
    }
    assert hits("sensitivity_ffn_w4a16_g128") == {"model.language_model.layers.3.mlp.down_proj"}


def test_component_targets_match_vllm_fused_layer_names() -> None:
    """vLLM 0.29 names Qwen3.5's text layers language_model.model.layers.N... and fuses
    projections; this replays its compressed-tensors matching (find_matched_target: the layer
    name, else every part of a fused layer) on those names."""
    import re

    fused = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def matches(name: str, targets: tuple[str, ...]) -> bool:
        def hit(value: str) -> bool:
            return any(re.match(t.removeprefix("re:"), value) for t in targets)

        if hit(name):
            return True
        key = next((k for k in fused if name.endswith(k)), None)
        return key is not None and all(hit(name.replace(key, part)) for part in fused[key])

    vllm_layers = {
        "language_model.model.layers.0.linear_attn.in_proj_qkvz": "deltanet",
        "language_model.model.layers.0.linear_attn.in_proj_ba": "deltanet",
        "language_model.model.layers.0.linear_attn.out_proj": "deltanet",
        "language_model.model.layers.3.self_attn.qkv_proj": "attention",
        "language_model.model.layers.3.self_attn.o_proj": "attention",
        "language_model.model.layers.3.mlp.gate_up_proj": "ffn",
        "language_model.model.layers.3.mlp.down_proj": "ffn",
        "visual.blocks.0.mlp.linear_fc1": None,
        "visual.blocks.0.attn.qkv": None,
    }
    plan = load_config(Path("configs/feature1.yaml"))
    for component in ("deltanet", "attention", "ffn"):
        targets = plan.variant(f"sensitivity_{component}_w4a16_g128").targets
        expected = {name for name, owner in vllm_layers.items() if owner == component}
        assert {name for name in vllm_layers if matches(name, targets)} == expected, component


def test_panel_suite_matches_the_instruct_protocol() -> None:
    panel = load_benchmark_suite(Path("configs/evaluation/feature1_panel.yaml"))
    main = load_benchmark_suite(Path("configs/evaluation/feature1.yaml"))
    assert panel.text_tasks == ("mmlu_pro", "minerva_math500", "ifeval")
    assert panel.vision_tasks == panel.vision_extracted_tasks == ("MMMU_DEV_VAL",)
    shared = (
        "max_model_len",
        "vision_max_model_len",
        "generation",
        "enable_thinking",
        "fewshot",
        "seed",
        "vision_answer_extractor",
    )
    for field in shared:
        assert getattr(panel, field) == getattr(main, field), field
    assert panel.samples_path is not None
    assert panel.samples_path.name == "feature1_thinking_samples.json"


def test_fla_installs_pinned_without_touching_torch() -> None:
    from qwen35_compression.feature1 import FLA_METHODS, FLA_VERSION, fla_install_command

    command = fla_install_command("py")
    assert command[:5] == ["uv", "pip", "install", "--no-deps", "--python"]
    assert (
        f"fla-core=={FLA_VERSION}" in command
        and f"flash-linear-attention=={FLA_VERSION}" in command
    )
    assert FLA_METHODS == {"autoround", "glaze"}


def test_calibration_samples_must_be_positive(tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load(Path("configs/feature1.yaml").read_text(encoding="utf-8"))
    variants = yaml.safe_load(Path("configs/variants/feature1.yaml").read_text(encoding="utf-8"))
    variants["variants"][1]["calibration_samples"] = 0
    (tmp_path / "variants.yaml").write_text(yaml.safe_dump(variants), encoding="utf-8")
    raw["variants_path"] = str(tmp_path / "variants.yaml")
    for key in ("calibration", "multimodal_calibration", "evaluation", "paths"):
        section = raw.get(key)
        if isinstance(section, dict):
            for field in ("path", "lock_path", "suite_path", "outputs", "results"):
                if field in section and not Path(section[field]).is_absolute():
                    section[field] = str(Path("configs").resolve() / section[field])
    config = tmp_path / "feature1.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration_samples must be positive"):
        load_config(config)
