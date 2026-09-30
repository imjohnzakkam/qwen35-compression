#!/usr/bin/env python3
"""Write the fixed MMLU-Pro subset the thinking-mode suite evaluates (lm-eval --samples).

Thinking mode generates thousands of tokens per answer, so the full 12,032-question MMLU-Pro is
out of budget. The subset keeps every subject in proportion to its size, is drawn with a fixed
seed, and is committed, so every model in the thinking track sees exactly the same questions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# MMLU-Pro test split sizes per lm-eval subtask (TIGER-Lab/MMLU-Pro, 12,032 questions), as
# counted in the Feature 1 BF16 text run's per-question logs.
MMLU_PRO_SIZES = {
    "mmlu_pro_biology": 717,
    "mmlu_pro_business": 789,
    "mmlu_pro_chemistry": 1132,
    "mmlu_pro_computer_science": 410,
    "mmlu_pro_economics": 844,
    "mmlu_pro_engineering": 969,
    "mmlu_pro_health": 818,
    "mmlu_pro_history": 381,
    "mmlu_pro_law": 1101,
    "mmlu_pro_math": 1351,
    "mmlu_pro_other": 924,
    "mmlu_pro_philosophy": 499,
    "mmlu_pro_physics": 1299,
    "mmlu_pro_psychology": 798,
}


def stratified_subset(sizes: dict[str, int], total: int, seed: int) -> dict[str, list[int]]:
    population = sum(sizes.values())
    rng = random.Random(seed)
    return {
        task: sorted(rng.sample(range(size), round(total * size / population)))
        for task, size in sorted(sizes.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("configs/evaluation/feature1_thinking_samples.json")
    )
    args = parser.parse_args()
    subset = stratified_subset(MMLU_PRO_SIZES, args.total, args.seed)
    args.output.write_text(json.dumps(subset, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {sum(len(v) for v in subset.values())} MMLU-Pro questions to {args.output}")


if __name__ == "__main__":
    main()
