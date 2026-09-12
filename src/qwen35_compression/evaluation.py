from __future__ import annotations

import math
import time
from typing import Any

from qwen35_compression.config import EvaluationConfig
from qwen35_compression.io import read_jsonl


def _device(model: Any) -> Any:
    return next(model.parameters()).device


def evaluate_smoke(
    model: Any,
    processor: Any,
    config: EvaluationConfig,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(config.seed)
    rows = read_jsonl(config.path, limit=config.max_samples)
    if not rows:
        raise ValueError(f"evaluation fixture is empty: {config.path}")

    total_nll = 0.0
    total_tokens = 0
    exact_matches = 0
    generations = []
    started = time.perf_counter()
    device = _device(model)

    for row in rows:
        messages = [{"role": "user", "content": [{"type": "text", "text": row["prompt"]}]}]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :].float()
            labels = inputs["input_ids"][:, 1:]
            nll = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
            )

        token_count = labels.numel()
        total_nll += float(nll.item())
        total_tokens += token_count
        completion_ids = generated[0, inputs["input_ids"].shape[1] :]
        completion = processor.decode(completion_ids, skip_special_tokens=True).strip()
        reference = row["reference"].strip().lower()
        matched = completion.lower().startswith(reference)
        exact_matches += int(matched)
        generations.append(
            {
                "prompt": row["prompt"],
                "reference": row["reference"],
                "completion": completion,
                "prefix_match": matched,
            }
        )

    mean_nll = total_nll / total_tokens
    return {
        "suite": "phase0_fixed_text_smoke_v1",
        "research_result": False,
        "samples": len(rows),
        "tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 80.0)),
        "prefix_accuracy": exact_matches / len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "generations": generations,
    }
