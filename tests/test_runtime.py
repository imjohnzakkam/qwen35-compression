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
    assert infer[1].endswith("run.py")
    assert infer[infer.index("--model") + 1] == "Qwen3.5-4B-pinned"
    assert infer[infer.index("--data") + 1 : infer.index("--work-dir")] == [
        "MMMU_DEV_VAL",
        "OCRBench",
    ]
    assert infer[infer.index("--mode") + 1] == "infer"
    assert infer[infer.index("--model-class") + 1] == "LMDeployAPI"
    assert infer[infer.index("--base-url") + 1].startswith("http://127.0.0.1:")
    # Neither dataset needs an LLM judge, so scoring is rule-based and nothing is judged.
    assert plan["eval"][plan["eval"].index("--judge") + 1] == "exact_matching"
    assert plan["eval_judged"] is None and plan["judged_datasets"] == []


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


def test_vlmeval_wrapper_judges_only_datasets_that_require_it(tmp_path: Path) -> None:
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
            "--disable-thinking",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["judged_datasets"] == ["MathVista_MINI"]
    assert plan["eval"][plan["eval"].index("--data") + 1] == "MMBench_DEV_EN_V11"
    judged = plan["eval_judged"]
    assert judged[judged.index("--data") + 1] == "MathVista_MINI"
    assert judged[judged.index("--judge") + 1] == "Qwen3.5-4B-pinned"
    assert judged[judged.index("--judge-base-url") + 1] == judged[judged.index("--base-url") + 1]
    assert json.loads(judged[judged.index("--judge-args") + 1]) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
