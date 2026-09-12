#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Qwen3.5 text and image generation in vLLM")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    contract = {
        "model_path": str(args.model_path),
        "image": str(args.image) if args.image else None,
        "max_model_len": args.max_model_len,
        "output": str(args.output),
    }
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return
    if platform.system() != "Linux":
        raise RuntimeError("vLLM smoke requires Linux and CUDA")

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams, TextPrompt

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM smoke requires CUDA")
    model_path = args.model_path.resolve()
    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 1},
        seed=42,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=16)
    text_messages = [{"role": "user", "content": "Reply with exactly: READY"}]
    text_prompt = processor.apply_chat_template(
        text_messages, tokenize=False, add_generation_prompt=True
    )
    requests: list[str | TextPrompt] = [text_prompt]
    labels = ["text"]
    if args.image:
        image_path = args.image.resolve()
        image = Image.open(image_path).convert("RGB")
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ]
        image_prompt = processor.apply_chat_template(
            image_messages, tokenize=False, add_generation_prompt=True
        )
        requests.append(TextPrompt(prompt=image_prompt, multi_modal_data={"image": image}))
        labels.append("image")

    generated = llm.generate(requests, sampling)
    completions = {
        label: response.outputs[0].text.strip()
        for label, response in zip(labels, generated, strict=True)
    }
    if any(not value for value in completions.values()):
        raise RuntimeError(f"vLLM smoke produced an empty completion: {completions}")
    report = {
        "schema_version": 1,
        "status": "passed",
        "research_result": False,
        "contract": contract,
        "cuda_device": torch.cuda.get_device_name(0),
        "completions": completions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
