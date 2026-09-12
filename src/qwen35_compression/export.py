from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen35_compression.config import ExperimentConfig, VariantConfig
from qwen35_compression.io import inventory, write_json
from qwen35_compression.provenance import git_revision

MANIFEST_NAME = "compression_manifest.json"


def write_export_manifest(
    output_dir: Path,
    config: ExperimentConfig,
    variant: VariantConfig,
    model_revision: str,
    elapsed_seconds: float,
    peak_memory_bytes: int | None,
) -> dict[str, Any]:
    files = inventory(output_dir, excluded_names=(MANIFEST_NAME,))
    manifest = {
        "schema_version": 1,
        "phase": config.phase,
        "variant": variant.name,
        "method": variant.method,
        "model_id": config.model.id,
        "model_revision": model_revision,
        "code_revision": git_revision(config.source_path.parent.parent),
        "config_digest": config.digest,
        "quantization": {
            "scheme": variant.scheme,
            "bits": variant.bits,
            "group_size": variant.group_size,
            "ignore": list(variant.ignore),
        },
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    write_json(output_dir / MANIFEST_NAME, manifest)
    return manifest


def verify_export(output_dir: Path, expected: VariantConfig | None = None) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing export manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected is not None and manifest["variant"] != expected.name:
        raise ValueError(
            f"manifest variant {manifest['variant']!r} does not match {expected.name!r}"
        )
    if not manifest.get("code_revision"):
        raise ValueError("export manifest has no code_revision")
    required = {"config.json"}
    names = {item["path"] for item in manifest["files"]}
    missing = required - names
    if missing:
        raise ValueError(f"export is missing required files: {sorted(missing)}")
    if not any(name.endswith(".safetensors") for name in names):
        raise ValueError("export contains no safetensors weights")

    actual = inventory(output_dir, excluded_names=(MANIFEST_NAME,))
    if actual != manifest["files"]:
        raise ValueError("export inventory or digest mismatch")

    config_json = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if manifest["method"] != "bf16" and "quantization_config" not in config_json:
        raise ValueError("compressed export config has no quantization_config")
    return manifest
