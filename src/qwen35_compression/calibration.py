from __future__ import annotations

from typing import Any

from qwen35_compression.config import CalibrationConfig
from qwen35_compression.io import read_jsonl


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({"role": message["role"], "content": content})
    return normalized


def build_calibration_dataset(processor: Any, config: CalibrationConfig) -> tuple[Any, Any]:
    import torch
    from datasets import Dataset

    rows = read_jsonl(config.path, limit=config.num_samples)
    if len(rows) < config.num_samples:
        raise ValueError(
            f"calibration fixture has {len(rows)} rows, "
            f"expected {config.num_samples}: {config.path}"
        )

    dataset = Dataset.from_list(rows).shuffle(seed=config.seed)

    def encode(example: dict[str, Any]) -> dict[str, Any]:
        messages = _normalize_messages(example["messages"])
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            processor_kwargs={
                "return_tensors": "pt",
                "padding": False,
                "truncation": True,
                "max_length": config.max_sequence_length,
                "add_special_tokens": False,
            },
        )

    dataset = dataset.map(encode, batched=False, remove_columns=dataset.column_names)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("compression calibration requires batch size 1")
        return {key: torch.as_tensor(value) for key, value in batch[0].items()}

    return dataset, collate
