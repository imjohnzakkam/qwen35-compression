from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen35_compression.config import ExperimentConfig
from qwen35_compression.io import write_json


def resolve_revision(config: ExperimentConfig) -> str:
    if config.model.revision:
        return config.model.revision
    record_path = config.paths.results / "model.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("model_id") == config.model.id and record.get("revision"):
            return str(record["revision"])

    from huggingface_hub import HfApi

    revision = HfApi().model_info(config.model.id).sha
    write_json(
        record_path,
        {
            "model_id": config.model.id,
            "revision": revision,
            "config_digest": config.digest,
        },
    )
    return revision


def download_model(config: ExperimentConfig) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    revision = resolve_revision(config)
    path = Path(snapshot_download(repo_id=config.model.id, revision=revision))
    return path, revision


def load_model_and_processor(source: str | Path, config: ExperimentConfig) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.model.dtype]
    source_text = str(source)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": config.model.trust_remote_code,
    }
    model = Qwen3_5ForConditionalGeneration.from_pretrained(source_text, **kwargs)
    processor = AutoProcessor.from_pretrained(
        source_text,
        trust_remote_code=config.model.trust_remote_code,
    )
    model.eval()
    return model, processor


def model_source(config: ExperimentConfig, variant_name: str) -> str | Path:
    if variant_name == "bf16":
        revision = resolve_revision(config)
        return f"{config.model.id}@{revision}"
    return config.paths.outputs / variant_name


def split_hub_reference(source: str | Path) -> tuple[str, str | None]:
    source_text = str(source)
    if "@" in source_text and not Path(source_text).exists():
        model_id, revision = source_text.rsplit("@", 1)
        return model_id, revision
    return source_text, None


def load_resolved_model(source: str | Path, config: ExperimentConfig) -> tuple[Any, Any]:
    model_id, revision = split_hub_reference(source)
    if revision is None:
        return load_model_and_processor(model_id, config)

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.model.dtype]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=config.model.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    model.eval()
    return model, processor
