from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from qwen35_compression.config import load_config
from qwen35_compression.feature1 import load_benchmark_suite, require_calibration_lock
from qwen35_compression.multimodal import require_multimodal_lock

BASE_IMPORTS = {
    "accelerate": "accelerate",
    "datasets": "datasets",
    "huggingface-hub": "huggingface_hub",
    "lm-eval": "lm_eval",
    "numpy": "numpy",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "safetensors": "safetensors",
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "transformers": "transformers",
}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _package_checks(
    packages: dict[str, str], expected_versions: dict[str, str] | None = None
) -> list[Check]:
    checks = []
    for distribution, module in packages.items():
        version = importlib.metadata.version(distribution)
        if expected_versions and version != expected_versions[distribution]:
            raise ValueError(
                f"{distribution} is {version}; expected {expected_versions[distribution]}"
            )
        importlib.import_module(module)
        checks.append(Check(f"package:{distribution}", "passed", version))
    return checks


def _toolchains(root: Path) -> dict[str, Any]:
    path = root / "configs" / "toolchains.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported toolchain schema: {path}")
    return document


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _locked_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.startswith((" ", "#")):
            continue
        name, version = line.split("==", 1)
        versions[name.lower()] = version
    return versions


def run_preflight(config_path: Path, profile: str = "local") -> dict[str, Any]:
    if profile not in {"local", "gpu-text", "gpu-vision"}:
        raise ValueError("profile must be local, gpu-text, or gpu-vision")
    config = load_config(config_path)
    root = config.source_path.parent.parent
    suite = load_benchmark_suite(config.evaluation.suite_path)  # type: ignore[arg-type]
    checks = _package_checks(BASE_IMPORTS) if profile == "local" else []

    text_lock = require_calibration_lock(config.calibration)
    checks.append(Check("calibration:text", "passed", text_lock["content_sha256"]))
    if config.multimodal_calibration is None:
        raise ValueError("Feature 1 requires multimodal calibration")
    image_lock = require_multimodal_lock(config.multimodal_calibration)
    checks.append(
        Check("calibration:multimodal", "passed", image_lock["content_sha256"])
    )

    if profile == "gpu-vision":
        # Only the text environment carries lm-eval; the vision profile checks its own suite.
        checks.append(Check("benchmarks:text", "skipped", "not used by gpu-vision"))
    else:
        from lm_eval.tasks import TaskManager

        registered = TaskManager().all_tasks
        missing = sorted(set(suite.text_tasks) - set(registered))
        if missing:
            raise ValueError(f"lm-eval tasks are not registered: {missing}")
        checks.append(Check("benchmarks:text", "passed", ",".join(suite.text_tasks)))

    toolchains = _toolchains(root)
    toolkit = toolchains["vlmevalkit"]
    toolkit_dir = (root / toolkit["directory"]).resolve()
    if profile == "gpu-text":
        # The text evaluator never imports VLMEvalKit, and a text-only bootstrap
        # does not provision it; only the vision environment must hold the pin.
        checks.append(Check("toolkit:vlmevalkit", "skipped", "not used by gpu-text"))
    else:
        if not (toolkit_dir / "run.py").is_file():
            raise FileNotFoundError(f"pinned VLMEvalKit checkout is missing: {toolkit_dir}")
        actual_revision = _git_revision(toolkit_dir)
        if actual_revision != toolkit["revision"]:
            raise ValueError(
                f"VLMEvalKit revision is {actual_revision}; expected {toolkit['revision']}"
            )
        checks.append(Check("toolkit:vlmevalkit", "passed", actual_revision))

    for environment in ("gpu_text", "gpu_vision"):
        requirements = root / toolchains[environment]["requirements"]
        if not requirements.is_file():
            raise FileNotFoundError(f"compiled GPU requirements are missing: {requirements}")
        locked = _locked_versions(requirements)
        expected = toolchains[environment]["packages"]
        for name, version in expected.items():
            if name == "vlmeval":
                continue
            if locked.get(name) != version:
                raise ValueError(
                    f"{requirements} pins {name}={locked.get(name)!r}; expected {version}"
                )
        digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
        checks.append(Check(f"requirements:{environment}", "passed", digest))

    import torch

    if profile != "local":
        if platform.system() != "Linux":
            raise RuntimeError("GPU preflight requires Linux")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU preflight requires CUDA")
        if profile == "gpu-text":
            imports = {
                "datasets": "datasets",
                "lm-eval": "lm_eval",
                "torch": "torch",
                "transformers": "transformers",
                "vllm": "vllm",
            }
            checks.extend(_package_checks(imports, toolchains["gpu_text"]["packages"]))
        else:
            if str(toolkit_dir) not in sys.path:
                sys.path.insert(0, str(toolkit_dir))
            imports = {
                "datasets": "datasets",
                "torch": "torch",
                "transformers": "transformers",
                "vlmeval": "vlmeval",
                "vllm": "vllm",
            }
            checks.extend(_package_checks(imports, toolchains["gpu_vision"]["packages"]))
            from vlmeval.dataset import SUPPORTED_DATASETS

            missing_vision = sorted(set(suite.vision_tasks) - set(SUPPORTED_DATASETS))
            if missing_vision:
                raise ValueError(
                    f"VLMEvalKit datasets are not registered: {missing_vision}"
                )
            checks.append(
                Check("benchmarks:vision", "passed", ",".join(suite.vision_tasks))
            )
        checks.append(Check("cuda", "passed", torch.cuda.get_device_name(0)))
    else:
        checks.append(
            Check(
                "cuda",
                "skipped",
                "CUDA-only imports and execution are verified by the GPU preflight",
            )
        )

    return {
        "schema_version": 1,
        "profile": profile,
        "status": "passed",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "feature": config.feature,
        "model_id": config.model.id,
        "config_digest": config.digest,
        "checks": [asdict(check) for check in checks],
    }


def write_preflight(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
