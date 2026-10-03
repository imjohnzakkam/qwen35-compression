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


def pack_token_ids(sequences: list[list[int]], length: int) -> list[list[int]]:
    """Join token sequences in order and cut them into equal blocks of `length` tokens.

    The final partial block is dropped. Used for methods that stack every calibration sample into
    one tensor (AutoRound), which requires a single sequence length.
    """
    if length <= 0:
        raise ValueError("pack length must be positive")
    stream = [token for sequence in sequences for token in sequence]
    return [stream[start : start + length] for start in range(0, len(stream) - length + 1, length)]


def build_calibration_dataset(
    processor: Any, config: CalibrationConfig, pack: bool = False
) -> tuple[Any, Any]:
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
    if pack:
        sequences = [torch.as_tensor(ids).reshape(-1).tolist() for ids in dataset["input_ids"]]
        blocks = pack_token_ids(sequences, config.max_sequence_length)
        if not blocks:
            raise ValueError("calibration set is shorter than one packed block")
        dataset = Dataset.from_list(
            [{"input_ids": [block], "attention_mask": [[1] * len(block)]} for block in blocks]
        )

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("compression calibration requires batch size 1")
        return {key: torch.as_tensor(value) for key, value in batch[0].items()}

    return dataset, collate
