#!/usr/bin/env python3
"""Score runs on the reasoning panel (configs/evaluation/feature1_panel.yaml).

A panel run is scored as is. A full instruct-track run is scored on the same questions: MMLU-Pro
restricted to the panel's 1001-question subset, MATH-500, IFEval and MMMU in full. Both use one
protocol, so BF16 and finished variants need no panel rerun.

Also reports how often MMLU-Pro and MATH-500 answers reach the 8,192-token cap (`--loops`, needs
the tokenizer), the main way 4-bit weights lose points on these tasks.

    uv run python scripts/panel_scores.py --run bf16=logs/jarvis/<run> \\
        --run autoround=logs/jarvis/<run>/results --loops
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.feature1 import load_benchmark_suite

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "configs" / "evaluation" / "feature1_panel.yaml"


def _samples(run: Path, task: str) -> list[dict]:
    files = sorted(run.rglob(f"samples_{task}_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no samples for {task} under {run}")
    # One JSON record per "\n"-terminated line; str.splitlines would also split on the Unicode
    # line separators some answers contain.
    with files[-1].open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mmlu_pro_tasks(run: Path) -> list[str]:
    files = run.rglob("samples_mmlu_pro_*.jsonl")
    return sorted({p.name.removeprefix("samples_").rsplit("_", 1)[0] for p in files})


def _mmmu_val(scored: Path) -> float | None:
    files = sorted(scored.rglob("*MMMU_DEV_VAL_acc.csv"))
    if not files:
        return None
    rows = list(csv.DictReader(files[-1].open(encoding="utf-8")))
    row = next(r for r in rows if r.get("split") == "validation")
    value = float(row["Overall"])
    return value * 100 if value <= 1 else value


def panel_scores(
    run: Path, scored: Path | None, subset: dict[str, list[int]], tokenizer=None
) -> dict:
    """Panel scores of one run directory, and its scored-vision directory if any."""
    mmlu, capped_mmlu = [], 0
    for task in _mmlu_pro_tasks(run):
        keep = set(subset.get(task, []))
        for row in _samples(run, task):
            if row["doc_id"] in keep:
                mmlu.append(float(row["exact_match"]))
                if tokenizer is not None:
                    capped_mmlu += _at_cap(tokenizer, row)
    math = _samples(run, "minerva_math500")
    ifeval = _samples(run, "ifeval")
    scores = {
        "MMLU-Pro (1001)": 100 * sum(mmlu) / len(mmlu),
        "MATH-500": 100 * sum(float(r["math_verify"]) for r in math) / len(math),
        "IFEval": 100 * sum(float(r["prompt_level_strict_acc"]) for r in ifeval) / len(ifeval),
        "MMMU (val)": _mmmu_val(scored) if scored else None,
        "questions": {"mmlu_pro": len(mmlu), "math500": len(math), "ifeval": len(ifeval)},
    }
    if tokenizer is not None:
        scores["at cap: MATH-500 %"] = 100 * sum(_at_cap(tokenizer, r) for r in math) / len(math)
        scores["at cap: MMLU-Pro %"] = 100 * capped_mmlu / len(mmlu)
    return scores


def per_question(run: Path, subset: dict[str, list[int]]) -> dict[str, dict]:
    """Per-question correctness (0/1) for the panel's text tasks, keyed by (task, doc_id)."""
    rows: dict[str, dict] = {"MMLU-Pro (1001)": {}, "MATH-500": {}, "IFEval": {}}
    for task in _mmlu_pro_tasks(run):
        keep = set(subset.get(task, []))
        for row in _samples(run, task):
            if row["doc_id"] in keep:
                rows["MMLU-Pro (1001)"][(task, row["doc_id"])] = float(row["exact_match"])
    for row in _samples(run, "minerva_math500"):
        rows["MATH-500"][row["doc_id"]] = float(row["math_verify"])
    for row in _samples(run, "ifeval"):
        rows["IFEval"][row["doc_id"]] = float(row["prompt_level_strict_acc"])
    return rows


def paired_delta(
    candidate: dict, reference: dict, resamples: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Mean difference (points) over shared questions, with a paired bootstrap 95% interval."""
    import random

    keys = sorted(set(candidate) & set(reference), key=str)
    if not keys:
        raise ValueError("no shared questions")
    diffs = [candidate[key] - reference[key] for key in keys]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {
        "delta": 100 * sum(diffs) / n,
        "low": 100 * means[int(0.025 * resamples)],
        "high": 100 * means[int(0.975 * resamples) - 1],
        "questions": n,
    }


def _at_cap(tokenizer, row: dict, cap: int = 8192) -> bool:
    # Responses are decoded text; allow a few tokens for re-tokenisation differences.
    return len(tokenizer(row["resps"][0][0])["input_ids"]) >= cap - 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=RUN_DIR; the scored-vision directory <RUN_DIR>-scored is used when present",
    )
    parser.add_argument("--loops", action="store_true", help="Also count answers at the cap")
    parser.add_argument(
        "--vs",
        help="Reference run NAME: also report each model's change from it, with paired "
        "bootstrap 95%% intervals over the same questions",
    )
    parser.add_argument("--json", type=Path, help="Write the scores here as well")
    args = parser.parse_args()

    suite = load_benchmark_suite(PANEL)
    assert suite.samples_path is not None
    subset = json.loads(suite.samples_path.read_text(encoding="utf-8"))
    tokenizer = None
    if args.loops:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

    results = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        run = Path(path).resolve()
        scored = run.with_name(run.name + "-scored")
        results[name] = panel_scores(run, scored if scored.is_dir() else None, subset, tokenizer)

    metrics = [k for k in next(iter(results.values())) if k != "questions"]
    print("| " + " | ".join(["model", *metrics]) + " |")
    print("| " + " | ".join(["---", *["---:"] * len(metrics)]) + " |")
    for name, scores in results.items():
        cells = ["–" if scores[m] is None else f"{scores[m]:.1f}" for m in metrics]
        print("| " + " | ".join([name, *cells]) + " |")
    if args.vs:
        runs = {spec.partition("=")[0]: Path(spec.partition("=")[2]).resolve() for spec in args.run}
        if args.vs not in runs:
            raise ValueError(f"--vs {args.vs} is not one of the --run names")
        reference = per_question(runs[args.vs], subset)
        tasks = list(reference)
        print(f"\nChange from {args.vs} (points, paired bootstrap 95% interval)")
        print("| " + " | ".join(["model", *tasks]) + " |")
        print("| " + " | ".join(["---", *["---:"] * len(tasks)]) + " |")
        for name, run in runs.items():
            if name == args.vs:
                continue
            candidate = per_question(run, subset)
            deltas = {task: paired_delta(candidate[task], reference[task]) for task in tasks}
            results[name]["vs"] = {args.vs: deltas}
            cells = [
                f"{d['delta']:+.1f} [{d['low']:+.1f}, {d['high']:+.1f}]" for d in deltas.values()
            ]
            print("| " + " | ".join([name, *cells]) + " |")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
