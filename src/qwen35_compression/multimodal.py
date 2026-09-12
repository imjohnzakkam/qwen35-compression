"""Deterministic image-text calibration preparation."""

from __future__ import annotations

import hashlib
import io
import json
import random
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from qwen35_compression.config import MultimodalCalibrationConfig

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _download_image(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported image URL scheme: {parsed.scheme}")
    request = urllib.request.Request(url, headers={"User-Agent": "qwen35-compression/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes: {url}")

    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    return data


def _caption(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return str(value[0])
    raise ValueError("caption column must contain a string or non-empty list")


def prepare_multimodal_calibration(
    config: MultimodalCalibrationConfig,
) -> Mapping[str, Any]:
    """Download only selected images and write a portable JSONL plus lock."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    source = config.source
    info = HfApi().dataset_info(source.dataset_id, revision=source.revision)
    resolved_revision = info.sha
    if not resolved_revision:
        raise ValueError("Hugging Face did not return a dataset commit SHA")
    dataset = load_dataset(
        source.dataset_id,
        split=source.split,
        revision=resolved_revision,
    )
    if len(dataset) < config.num_samples:
        raise ValueError(
            f"dataset has {len(dataset)} rows; {config.num_samples} were requested"
        )

    indices = random.Random(config.seed).sample(range(len(dataset)), config.num_samples)
    config.assets_dir.mkdir(parents=True, exist_ok=True)
    repository_root = config.path.parents[2]
    records: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for position, index in enumerate(indices):
        row = dataset[index]
        url = str(row[source.image_url_column])
        data = _download_image(url)
        image_path = config.assets_dir / f"{position:04d}.jpg"
        image_path.write_bytes(data)
        portable_path = image_path.relative_to(repository_root).as_posix()
        digest = hashlib.sha256(data).hexdigest()
        images.append(
            {
                "source_index": index,
                "source_url": url,
                "path": portable_path,
                "sha256": digest,
                "bytes": len(data),
            }
        )
        records.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": portable_path},
                            {
                                "type": "text",
                                "text": "Describe this image in detail.",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": _caption(row[source.captions_column]),
                    },
                ]
            }
        )

    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    contract = {
        "dataset_id": source.dataset_id,
        "revision": resolved_revision,
        "split": source.split,
        "image_url_column": source.image_url_column,
        "captions_column": source.captions_column,
        "seed": config.seed,
        "num_samples": config.num_samples,
        "prompt": "Describe this image in detail.",
    }
    lock = {
        "schema_version": 1,
        **contract,
        "source_indices": indices,
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "images": images,
    }
    config.path.parent.mkdir(parents=True, exist_ok=True)
    config.path.write_bytes(encoded)
    config.lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock


def require_multimodal_lock(
    config: MultimodalCalibrationConfig,
) -> Mapping[str, Any]:
    lock = json.loads(config.lock_path.read_text(encoding="utf-8"))
    if hashlib.sha256(config.path.read_bytes()).hexdigest() != lock["content_sha256"]:
        raise ValueError("multimodal calibration JSONL does not match its lock")
    repository_root = config.path.parents[2]
    for image in lock["images"]:
        image_path = repository_root / image["path"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != image["sha256"]:
            raise ValueError(f"multimodal image does not match its lock: {image_path}")
    return lock
