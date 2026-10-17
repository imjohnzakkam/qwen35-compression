#!/usr/bin/env python3
"""Gate G4's report: Blaze-Q (G2b's export) on the full suite, against BF16 and Glaze v2.

Blaze-Q's scores come from two runs: G4 (MMLU-Pro, GSM8K, HellaSwag, ARC-Challenge, WikiText-2,
MMBench, MathVista, TextVQA, OCRBench, DocVQA) and G3 (MATH-500, IFEval, MMMU). BF16's and
Glaze v2's are their recorded full-suite results (docs/experiments/06-glaze-v2.md). G4 passes
when every task is within a point of BF16, MMMU within two (WikiText-2's perplexity is reported).

    uv run python scripts/blaze_g4_report.py --g4 <G4 run>/g4 \\
        --g4-scored <G4 run>/g4/vision-scored --g2b <G2b run>/g2b --output g4_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# docs/experiments/06-glaze-v2.md, full suite, instruct track (WikiText-2: perplexity).
RECORDED = {
    "MMLU-Pro": (74.6, 73.7),
    "GSM8K": (83.2, 82.8),
    "MATH-500": (83.4, 83.2),
    "IFEval": (82.3, 82.8),
    "HellaSwag": (65.4, 64.8),
    "ARC-Challenge": (51.1, 49.6),
    "WikiText-2": (10.95, 11.30),
    "MMBench": (85.4, 85.7),
    "MMMU": (69.6, 66.7),
    "MathVista": (81.0, 80.6),
    "OCRBench": (86.3, 86.3),
    "DocVQA": (95.3, 95.4),
    "TextVQA": (82.8, 82.2),
}
TEXT_METRICS = {
    "MMLU-Pro": ("mmlu_pro", "exact_match,custom-extract"),
    "GSM8K": ("gsm8k", "exact_match,flexible-extract"),
    "HellaSwag": ("hellaswag", "acc_norm,none"),
    "ARC-Challenge": ("arc_challenge", "acc_norm,none"),
    "WikiText-2": ("wikitext", "word_perplexity,none"),
}
MAX_DROP = 1.0
MAX_MMMU_DROP = 2.0


def _one(root: Path, pattern: str) -> Path:
    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return files[-1]


def _csv_value(path: Path, column: str = "Overall", row: dict[str, str] | None = None) -> float:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    chosen = rows[0]
    if row:
        chosen = next(r for r in rows if all(r.get(k) == v for k, v in row.items()))
    value = float(chosen[column])
    return value * 100 if value <= 1 else value


def blaze_scores(g4: Path, g4_scored: Path, g2b: Path) -> dict[str, float]:
    results = json.loads(_one(g4 / "text", "results_*.json").read_text(encoding="utf-8"))["results"]
    scores: dict[str, float] = {}
    for task, (key, metric) in TEXT_METRICS.items():
        value = float(results[key][metric])
        scores[task] = value if task == "WikiText-2" else 100 * value
    vision = g4 / "vision"
    ocr = json.loads(_one(vision, "*_OCRBench_score.json").read_text(encoding="utf-8"))
    scores["OCRBench"] = float(ocr["Final Score Norm"])
    scores["DocVQA"] = _csv_value(_one(vision, "*_DocVQA_VAL_acc.csv"))
    scores["TextVQA"] = _csv_value(_one(vision, "*_TextVQA_VAL_acc.csv"))
    scores["MMBench"] = _csv_value(
        _one(g4_scored, "*_MMBench_DEV_EN_V11_acc.csv"), row={"split": "dev"}
    )
    scores["MathVista"] = _csv_value(
        _one(g4_scored, "*_MathVista_MINI_*score.csv"), column="acc", row={"Task&Skill": "Overall"}
    )
    panel = json.loads((g2b / "g3_report.json").read_text(encoding="utf-8"))["panel"]
    scores["MATH-500"] = float(panel["MATH-500"]["blaze_q"])
    scores["IFEval"] = float(panel["IFEval"]["blaze_q"])
    scores["MMMU"] = _csv_value(
        _one(g2b / "mmmu-scored", "*_MMMU_DEV_VAL_acc.csv"), row={"split": "validation"}
    )
    return scores


def g4_report(scores: dict[str, float]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    passed = True
    for task, (bf16, glaze) in RECORDED.items():
        value = scores[task]
        row = {"bf16": bf16, "glaze_v2": glaze, "blaze_q": round(value, 2)}
        if task == "WikiText-2":
            row["vs_bf16"] = round(value - bf16, 2)
        else:
            drop = bf16 - value
            row["vs_bf16"] = round(-drop, 2)
            row["within"] = drop <= (MAX_MMMU_DROP if task == "MMMU" else MAX_DROP)
            passed &= row["within"]
        rows[task] = row
    return {
        "tasks": rows,
        "thresholds": {"max_drop": MAX_DROP, "max_mmmu_drop": MAX_MMMU_DROP},
        "checks": {"suite": passed},
        "decision": "go: Blaze-Q holds BF16's full suite"
        if passed
        else "stop: report the drops as measured",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--g4", type=Path, required=True)
    parser.add_argument("--g4-scored", type=Path, required=True)
    parser.add_argument("--g2b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = g4_report(blaze_scores(args.g4, args.g4_scored, args.g2b))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for task, row in report["tasks"].items():
        print(
            f"{task:14s} BF16 {row['bf16']:6.2f}  Glaze v2 {row['glaze_v2']:6.2f}  "
            f"Blaze-Q {row['blaze_q']:6.2f}  ({row['vs_bf16']:+.2f})"
        )
    print("decision=" + report["decision"])


if __name__ == "__main__":
    main()
