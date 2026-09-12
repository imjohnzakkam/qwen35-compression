#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.io import write_json
from qwen35_compression.models import load_resolved_model, model_source, resolve_revision
from qwen35_compression.provenance import environment_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one local image-text generation smoke")
    parser.add_argument("--config", type=Path, default=Path("configs/feature0.yaml"))
    parser.add_argument("--variant", default="bf16")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/feature0/bf16_multimodal.json")
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    import torch

    config = load_config(args.config)
    variant = config.variant(args.variant)
    if variant.method != "bf16":
        raise ValueError("the local multimodal smoke currently accepts only BF16")
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    model, processor = load_resolved_model(model_source(config, variant.name), config)
    device = next(model.parameters()).device
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )
    completion_ids = generated[0, inputs["input_ids"].shape[1] :]
    completion = processor.decode(completion_ids, skip_special_tokens=True).strip()
    if not completion:
        raise RuntimeError("multimodal smoke returned an empty completion")

    root = config.source_path.parent.parent
    write_json(
        args.output,
        {
            "schema_version": 1,
            "feature": config.feature,
            "variant": variant.name,
            "model_id": config.model.id,
            "model_revision": resolve_revision(config),
            "config_digest": config.digest,
            "research_result": False,
            "image": {
                "path": str(image_path),
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            },
            "completion": completion,
            "environment": environment_record(root),
        },
    )
    print(f"result={args.output.resolve()}")
    print(f"completion={completion}")


if __name__ == "__main__":
    main()
