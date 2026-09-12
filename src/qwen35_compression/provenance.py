from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(root: Path) -> str | None:
    injected = os.environ.get("QWEN35_CODE_REVISION")
    if injected:
        return injected
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_record(root: Path) -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "git_revision": git_revision(root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: _version(name)
            for name in (
                "torch",
                "transformers",
                "llmcompressor",
                "compressed-tensors",
                "datasets",
                "lm-eval",
            )
        },
        "cuda": {
            "available": cuda,
            "runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count() if cuda else 0,
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
            if cuda
            else [],
        },
    }
