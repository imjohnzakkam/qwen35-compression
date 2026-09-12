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
    command = json.loads(result.stdout)["forwarded_arguments"]

    assert command[:2] == ["--model", "Qwen3.5-4B-pinned"]
    assert command[command.index("--data") + 1 : command.index("--work-dir")] == [
        "MMMU_DEV_VAL",
        "OCRBench",
    ]


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
    assert plan["text"][plan["text"].index("--model") + 1] == "vllm"
    assert plan["vision"] is None
    assert plan["runtime_smoke"][0].endswith(".venv-gpu-text/bin/python")
