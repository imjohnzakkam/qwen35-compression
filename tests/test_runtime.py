from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path


def test_gpu_bootstrap_uses_separate_pinned_environments() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/bootstrap_gpu.py", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    commands = json.loads(result.stdout)
    flattened = [" ".join(command) for command in commands]

    assert any("requirements/gpu-text.lock" in command for command in flattened)
    assert any("requirements/gpu-vision.lock" in command for command in flattened)
    assert any("--torch-backend cu130" in command for command in flattened)
    assert all(
        "--index-strategy unsafe-best-match" in command
        for command in flattened
        if "uv pip sync" in command
    )
    assert not any("torchrun" in command for command in flattened)

    text_result = subprocess.run(
        [sys.executable, "scripts/bootstrap_gpu.py", "--dry-run", "--scope", "text"],
        check=True,
        capture_output=True,
        text=True,
    )
    text_commands = [" ".join(command) for command in json.loads(text_result.stdout)]
    assert any("requirements/gpu-text.lock" in command for command in text_commands)
    assert not any("gpu-vision" in command for command in text_commands)


def test_vlmeval_wrapper_registers_alias_and_forwards_tasks(tmp_path: Path) -> None:
    toolkit = tmp_path / "VLMEvalKit"
    toolkit.mkdir()
    (toolkit / "run.py").write_text("", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/vlmeval_qwen35.py",
            "--toolkit-dir",
            str(toolkit),
            "--model-path",
            "model",
            "--output-dir",
            "results/vision",
            "--data",
            "MMMU_DEV_VAL",
            "OCRBench",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    server = plan["server"]
    assert server[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert server[server.index("--served-model-name") + 1] == "Qwen3.5-4B-pinned"
    infer = plan["infer"]
    # run.py is reached through scripts/vlmeval_run.py, which applies the final-answer rule.
    assert infer[1].endswith("scripts/vlmeval_run.py")
    assert infer[infer.index("--toolkit-dir") + 1] == str(toolkit.resolve())
    assert "--limit" not in infer[: infer.index("--")]
    assert infer[infer.index("--model") + 1] == "Qwen3.5-4B-pinned"
    assert infer[infer.index("--data") + 1 : infer.index("--work-dir")] == [
        "MMMU_DEV_VAL",
        "OCRBench",
    ]
    assert infer[infer.index("--mode") + 1] == "infer"
    assert infer[infer.index("--model-class") + 1] == "LMDeployAPI"
    assert infer[infer.index("--base-url") + 1].startswith("http://127.0.0.1:")
    # Neither dataset is left for the answer extractor, so both are scored by rules on the GPU.
    assert plan["eval"][plan["eval"].index("--judge") + 1] == "exact_matching"
    assert plan["extracted_datasets"] == []
    assert "eval_judged" not in plan


def test_result_download_targets_local_logs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_jarvis_results.py",
            "--instance-id",
            "123",
            "--local-root",
            "logs/jarvis",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    command = json.loads(result.stdout)["command"]
    assert command[:4] == [
        "jl",
        "download",
        "123",
        "/home/qwen35-compression/results/feature1/bf16",
    ]
    assert command[-1] == "-r"
    assert "/logs/jarvis/feature1-bf16-" in command[-2]


def test_download_verification_rejects_pilot(tmp_path: Path) -> None:
    verify_download = runpy.run_path("scripts/fetch_jarvis_results.py")[
        "verify_download"
    ]
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"status": "passed", "research_result": False}), encoding="utf-8"
    )

    try:
        verify_download(tmp_path)
    except ValueError as error:
        assert "limited pilot" in str(error)
    else:
        raise AssertionError("pilot result was accepted as a full baseline")

    receipt = verify_download(tmp_path, allow_pilot=True)
    assert receipt.name == "download_receipt.json"


def test_vllm_runtime_smoke_has_mac_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_vllm.py",
            "--model-path",
            "model",
            "--image",
            "image.jpg",
            "--output",
            str(tmp_path / "smoke.json"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    contract = json.loads(result.stdout)
    assert contract["max_model_len"] == 4096
    assert contract["image"] == "image.jpg"


def test_complete_bf16_pipeline_has_mac_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_feature1_bf16.py",
            "--dry-run",
            "--text-only",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    text = plan["text"]["text"]
    assert text[text.index("--model") + 1] == "vllm"
    assert plan["benchmark_suite"]["max_gen_toks"] == 8192
    assert plan["vision"] is None
    assert plan["runtime_smoke"][0].endswith(".venv-gpu-text/bin/python")


def test_bf16_pilot_limits_text_and_vision() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_feature1_bf16.py", "--dry-run", "--limit", "10"],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    text = plan["text"]["text"]
    assert text[text.index("--limit") + 1] == "10"
    assert plan["vision"][-2:] == ["--limit", "10"]


def test_toolkit_commands_reclone_when_directory_is_not_a_git_checkout(tmp_path: Path) -> None:
    module = runpy.run_path("scripts/bootstrap_gpu.py", run_name="bootstrap_gpu")
    toolkit = {"repository": "https://example.invalid/VLMEvalKit.git", "revision": "abc123"}

    missing = tmp_path / "missing"
    flattened = [" ".join(c) for c in module["toolkit_commands"](toolkit, missing)]
    assert flattened[0].startswith("git clone")
    assert flattened[-1] == f"git -C {missing} checkout abc123"
    assert not any(c.startswith("rm ") for c in flattened)

    uploaded = tmp_path / "uploaded"
    uploaded.mkdir()
    (uploaded / "run.py").write_text("", encoding="utf-8")
    flattened = [" ".join(c) for c in module["toolkit_commands"](toolkit, uploaded)]
    assert flattened[0] == f"rm -rf {uploaded}"
    assert flattened[1].startswith("git clone")
    assert flattened[-1] == f"git -C {uploaded} checkout abc123"

    cloned = tmp_path / "cloned"
    subprocess.run(["git", "init", "-q", str(cloned)], check=True)
    flattened = [" ".join(c) for c in module["toolkit_commands"](toolkit, cloned)]
    assert flattened == [f"git -C {cloned} checkout abc123"]


def test_vlmeval_wrapper_reports_thinking_mode(tmp_path: Path) -> None:
    toolkit = tmp_path / "VLMEvalKit"
    toolkit.mkdir()
    (toolkit / "run.py").write_text("", encoding="utf-8")
    base = [
        sys.executable,
        "scripts/vlmeval_qwen35.py",
        "--toolkit-dir",
        str(toolkit),
        "--model-path",
        "model",
        "--output-dir",
        "results/vision",
        "--data",
        "OCRBench",
        "--dry-run",
    ]
    default = json.loads(subprocess.run(base, check=True, capture_output=True, text=True).stdout)
    assert default["enable_thinking"] is True
    assert "--extra-body" not in default["infer"]
    disabled = json.loads(
        subprocess.run(
            [*base, "--disable-thinking"], check=True, capture_output=True, text=True
        ).stdout
    )
    assert disabled["enable_thinking"] is False
    extra = json.loads(disabled["infer"][disabled["infer"].index("--extra-body") + 1])
    assert extra == {"chat_template_kwargs": {"enable_thinking": False}}


def test_vlmeval_wrapper_leaves_extracted_datasets_for_local_scoring(tmp_path: Path) -> None:
    toolkit = tmp_path / "VLMEvalKit"
    toolkit.mkdir()
    (toolkit / "run.py").write_text("", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/vlmeval_qwen35.py",
            "--toolkit-dir",
            str(toolkit),
            "--model-path",
            "model",
            "--output-dir",
            "results/vision",
            "--data",
            "MMBench_DEV_EN_V11",
            "MathVista_MINI",
            "TextVQA_VAL",
            "--extracted-data",
            "MMBench_DEV_EN_V11",
            "MathVista_MINI",
            "--disable-thinking",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    infer = plan["infer"]
    # Every dataset is inferred on the GPU ...
    assert infer[infer.index("--data") + 1 : infer.index("--work-dir")] == [
        "MMBench_DEV_EN_V11",
        "MathVista_MINI",
        "TextVQA_VAL",
    ]
    # ... but only rule-scored ones are evaluated there; the rest wait for score_vision.py.
    assert plan["extracted_datasets"] == ["MMBench_DEV_EN_V11", "MathVista_MINI"]
    evaluate = plan["eval"]
    assert evaluate[evaluate.index("--data") + 1 : evaluate.index("--work-dir")] == ["TextVQA_VAL"]
    assert evaluate[evaluate.index("--judge") + 1] == "exact_matching"
    assert "eval_judged" not in plan


def test_vlmeval_wrapper_local_server_for_validation(tmp_path: Path) -> None:
    toolkit = tmp_path / "VLMEvalKit"
    toolkit.mkdir()
    (toolkit / "run.py").write_text("", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/vlmeval_qwen35.py",
            "--toolkit-dir",
            str(toolkit),
            "--model-path",
            "model",
            "--output-dir",
            "results/vision",
            "--data",
            "MMMU_DEV_VAL",
            "--server",
            "local",
            "--server-python",
            "local-py",
            "--limit",
            "10",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["server"][0] == "local-py"
    assert plan["server"][1].endswith("scripts/local_vlm_server.py")
    assert plan["server"][plan["server"].index("--device") + 1] == "mps"
    infer = plan["infer"]
    assert infer[infer.index("--limit") + 1] == "10"
    assert infer.index("--limit") < infer.index("--")
