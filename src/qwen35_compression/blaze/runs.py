"""The bookkeeping every Blaze gate driver shares: a run manifest with one entry per step."""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen35_compression.drift import TEXT_PYTHON
from qwen35_compression.io import write_json


class GateRun:
    """A gate's run directory: `run_manifest.json` records each step's status and seconds."""

    def __init__(self, output: Path, study: str, code_revision: str, **fields: Any) -> None:
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "study": study,
            "code_revision": code_revision,
            **fields,
            "status": "running",
            "steps": {},
        }
        self.save()

    def save(self) -> None:
        write_json(self.output / "run_manifest.json", self.manifest)

    def step(self, name: str, action: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        self.manifest["steps"][name] = {"status": "running"}
        self.save()
        try:
            value = action()
        except BaseException as error:
            self.manifest["steps"][name] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.perf_counter() - started),
            }
            self.manifest["status"] = "failed"
            self.save()
            raise
        self.manifest["steps"][name] = {
            "status": "passed",
            "seconds": round(time.perf_counter() - started),
        }
        self.save()
        return value

    def reuse(self, name: str) -> None:
        self.manifest["steps"][name] = {"status": "reused"}
        self.save()

    def finish(self, **fields: Any) -> None:
        self.manifest.update(fields)
        self.manifest["status"] = "passed"
        self.save()


def bootstrap_text(root: Path) -> None:
    """The text-evaluation environment (vLLM, lm-eval) and its preflight."""
    subprocess.run(
        [sys.executable, "scripts/bootstrap_gpu.py", "--scope", "text"], cwd=root, check=True
    )
    subprocess.run(
        [str(TEXT_PYTHON), "scripts/preflight.py", "--profile", "gpu-text"], cwd=root, check=True
    )


def vision_run_manifest(
    variant: str,
    model_id: str,
    model_revision: str,
    code_revision: str,
    config_digest: str,
    suite_path: Path,
    export: Path,
    limit: int | None,
) -> dict[str, Any]:
    """The record scripts/score_vision.py reads to score a run's vision answers after download."""
    from qwen35_compression.feature1 import load_benchmark_suite, suite_record

    return {
        "schema_version": 1,
        "feature": "feature1",
        "variant": variant,
        "method": "blaze",
        "model_id": model_id,
        "model_revision": model_revision,
        "code_revision": code_revision,
        "config_digest": config_digest,
        "benchmark_suite": suite_record(load_benchmark_suite(suite_path)),
        "export_dir": str(export),
        "scope": "vision_only",
        "limit": limit,
        "vision_limit": limit,
        "enable_thinking": False,
        "research_result": limit is None,
        "status": "passed",
    }
