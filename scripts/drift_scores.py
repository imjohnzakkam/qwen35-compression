#!/usr/bin/env python3
"""Teacher-forced divergence from BF16 along BF16's own answers, by token position.

Each BF16 answer from the instruct track (MATH-500 and the 1001-question MMLU-Pro subset) is fed,
prompt and all, through a model in one prefill. At every answer token this records whether the
model's top choice is BF16's token (rank 1) and the model's negative log-likelihood of it. No
text is generated, so a model is scored in minutes instead of an hour of decoding.

Scored on BF16 itself, the flip rate is the noise floor (near-ties under different batching).
For a compressed model, a flip rate that grows with position along the same answers means its
error compounds over the sequence, as a recurrent (DeltaNet) state would; a flat curve means
the error is local.

    drift_scores.py score --model PATH --name gptq --traces DIR --output gptq.json
    drift_scores.py report bf16.json gptq.json ...
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBSET = ROOT / "configs" / "evaluation" / "feature1_thinking_samples.json"
# Answer positions are grouped in these token ranges: [0, 256), [256, 512), ... [4096, 8192).
BUCKETS = (0, 256, 512, 1024, 2048, 4096, 8192)
ANSWER_CAP = 8192
# Answers within this many tokens of the cap are counted as reaching it (re-tokenisation slack).
CAP_SLACK = 12
# The "long" cohort: finished answers of at least this many tokens, so early and late positions
# are compared within the same answers rather than across easy and hard questions.
LONG_ANSWER = 2048


@dataclass
class Trace:
    task: str
    doc_id: int
    prompt: str
    answer: str


def task_of(path: Path) -> str:
    """Task name of a sample file: lm-eval's samples_<task>_<timestamp>.jsonl or <task>.jsonl."""
    stem = path.stem.removeprefix("samples_")
    return re.sub(r"_\d{4}-\d{2}-\d{2}T[\d.-]+$", "", stem)


def load_traces(
    directory: Path, subset: dict[str, list[int]], limit: int | None = None
) -> list[Trace]:
    """BF16 answers for MATH-500 and the MMLU-Pro subset, from lm-eval sample files."""
    traces: list[Trace] = []
    files = [
        path
        for path in sorted(directory.rglob("*.jsonl"))
        if task_of(path) == "minerva_math500" or task_of(path) in subset
    ]
    if not files:
        raise FileNotFoundError(f"no MATH-500 or MMLU-Pro samples under {directory}")
    per_task: dict[str, list[Trace]] = {}
    for path in files:
        task = task_of(path)
        keep = set(subset[task]) if task in subset else None
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if keep is not None and row["doc_id"] not in keep:
                    continue
                arguments = row["arguments"]
                first = arguments["gen_args_0"] if isinstance(arguments, dict) else arguments[0]
                prompt = first["arg_0"] if isinstance(first, dict) else first[0]
                per_task.setdefault(task, []).append(
                    Trace(task, int(row["doc_id"]), prompt, row["resps"][0][0])
                )
    for task in sorted(per_task):
        rows = sorted(per_task[task], key=lambda trace: trace.doc_id)
        traces.extend(rows[:limit] if limit else rows)
    return traces


@dataclass
class Tally:
    """Answer tokens, top-1 flips and summed NLL per position bucket."""

    tokens: list[int] = field(default_factory=lambda: [0] * (len(BUCKETS) - 1))
    flips: list[int] = field(default_factory=lambda: [0] * (len(BUCKETS) - 1))
    nll: list[float] = field(default_factory=lambda: [0.0] * (len(BUCKETS) - 1))
    answers: int = 0

    def add(self, ranks: list[int], logprobs: list[float]) -> None:
        self.answers += 1
        for position, (rank, logprob) in enumerate(zip(ranks, logprobs, strict=True)):
            index = bucket(position)
            if index is None:
                continue
            self.tokens[index] += 1
            self.flips[index] += rank != 1
            self.nll[index] -= logprob

    def summary(self) -> dict:
        rows = []
        for index in range(len(BUCKETS) - 1):
            count = self.tokens[index]
            rows.append(
                {
                    "from": BUCKETS[index],
                    "to": BUCKETS[index + 1],
                    "tokens": count,
                    "flip_rate": self.flips[index] / count if count else None,
                    "mean_nll": self.nll[index] / count if count else None,
                }
            )
        total = sum(self.tokens)
        return {
            "answers": self.answers,
            "tokens": total,
            "flip_rate": sum(self.flips) / total if total else None,
            "mean_nll": sum(self.nll) / total if total else None,
            "buckets": rows,
        }


def bucket(position: int) -> int | None:
    for index in range(len(BUCKETS) - 1):
        if BUCKETS[index] <= position < BUCKETS[index + 1]:
            return index
    return None


def cohorts(answer_tokens: int) -> list[str]:
    """Which summaries an answer of this many tokens counts towards."""
    names = ["all"]
    capped = answer_tokens >= ANSWER_CAP - CAP_SLACK
    names.append("capped" if capped else "finished")
    if not capped and answer_tokens >= LONG_ANSWER:
        names.append("long")
    return names


def score(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    subset = json.loads(Path(args.subset).read_text(encoding="utf-8"))
    traces = load_traces(Path(args.traces), subset, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    encoded = []
    for trace in traces:
        prompt = tokenizer(trace.prompt, add_special_tokens=False)["input_ids"]
        answer = tokenizer(trace.answer, add_special_tokens=False)["input_ids"]
        if not answer:
            continue
        encoded.append((trace, prompt, answer))

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Prompt logprobs are computed per prefill chunk over the full vocabulary (see
        # build_lm_eval_command); 4,096 keeps the peak within a 24 GB GPU.
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        seed=42,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    # rank 1 means the model's top choice is BF16's token; flat logprobs keep ~2M positions compact.
    params = SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False, flat_logprobs=True
    )

    tallies: dict[str, dict[str, Tally]] = {}
    for start in range(0, len(encoded), args.batch):
        chunk = encoded[start : start + args.batch]
        prompts = [TokensPrompt(prompt_token_ids=prompt + answer) for _, prompt, answer in chunk]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        for (trace, prompt, answer), output in zip(chunk, outputs, strict=True):
            ranks, logprobs = [], []
            ids = prompt + answer
            for position in range(len(prompt), len(ids)):
                entry = output.prompt_logprobs[position][ids[position]]
                ranks.append(int(entry.rank))
                logprobs.append(float(entry.logprob))
            task = "math500" if trace.task.startswith("minerva") else "mmlu_pro"
            for group in (task, "both"):
                for cohort in cohorts(len(answer)):
                    tallies.setdefault(group, {}).setdefault(cohort, Tally()).add(ranks, logprobs)
        print(
            f"{args.name}: scored {min(start + args.batch, len(encoded))}/{len(encoded)}",
            flush=True,
        )

    result = {
        "name": args.name,
        "model": args.model,
        "answers": len(encoded),
        "buckets": list(BUCKETS),
        "long_answer_tokens": LONG_ANSWER,
        "scores": {
            group: {cohort: tally.summary() for cohort, tally in by_cohort.items()}
            for group, by_cohort in tallies.items()
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{args.name}: wrote {args.output}")


def report_rows(results: list[dict], group: str = "both", cohort: str = "long") -> list[str]:
    """A Markdown table of flip rate (%) per position bucket, one row per model."""
    edges = results[0]["buckets"]
    labels = [f"{edges[i]}–{edges[i + 1]}" for i in range(len(edges) - 1)]
    lines = [
        f"flip rate (%), {group}, {cohort} answers",
        "| model | " + " | ".join(labels) + " | all positions |",
        "| --- | " + " | ".join("---:" for _ in labels) + " | ---: |",
    ]
    for result in results:
        summary = result["scores"].get(group, {}).get(cohort)
        if summary is None:
            continue
        cells = [
            "–" if row["flip_rate"] is None else f"{100 * row['flip_rate']:.2f}"
            for row in summary["buckets"]
        ]
        overall = "–" if summary["flip_rate"] is None else f"{100 * summary['flip_rate']:.2f}"
        lines.append(f"| {result['name']} | " + " | ".join(cells) + f" | {overall} |")
    return lines


def report(args: argparse.Namespace) -> None:
    results = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.results]
    for cohort in ("long", "finished", "all"):
        print("\n".join(report_rows(results, args.group, cohort)))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("score", help="Score one model (needs vLLM and a GPU)")
    run.add_argument("--model", required=True, help="Model directory or Hub id")
    run.add_argument("--name", required=True)
    run.add_argument("--traces", required=True, help="Directory with BF16 lm-eval sample files")
    run.add_argument("--subset", default=str(SUBSET), help="MMLU-Pro subset (doc ids per task)")
    run.add_argument("--tokenizer", help="Defaults to --model")
    run.add_argument("--output", required=True)
    run.add_argument("--limit", type=int, help="First N answers per task (pilot)")
    run.add_argument("--batch", type=int, default=128, help="Answers per vLLM call")
    run.add_argument("--max-model-len", type=int, default=12288)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    show = commands.add_parser("report", help="Compare scored models")
    show.add_argument("results", nargs="+")
    show.add_argument("--group", default="both", choices=("both", "math500", "mmlu_pro"))
    args = parser.parse_args()
    score(args) if args.command == "score" else report(args)


if __name__ == "__main__":
    main()
