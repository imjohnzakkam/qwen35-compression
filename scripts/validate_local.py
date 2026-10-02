#!/usr/bin/env python3
"""Validate a Feature 1 run end to end on this machine before paying for a GPU.

Runs the same suite as the GPU drivers — tasks, prompts, chat template, enable_thinking, answer
cap, VLMEvalKit's API path and the fixed answer extractor — on the first N questions of every
task, with transformers on the Apple GPU (mps) or CPU in place of vLLM. Then it checks the
outputs and prints a pass/fail report. Scores from it are not research results.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.feature1 import (
    build_lm_eval_command,
    build_vlm_eval_command,
    load_benchmark_suite,
    run_logged,
    suite_label,
    suite_record,
)
from qwen35_compression.io import write_json
from qwen35_compression.models import download_model
from qwen35_compression.provenance import git_revision

ROOT = Path(__file__).resolve().parent.parent
LOCAL_PYTHON = ROOT / ".venv" / "bin" / "python"
VISION_PYTHON = ROOT / ".venv-vision-score" / "bin" / "python"
API_FAILURE = "Failed to obtain answer via API"


def text_commands(args, suite, snapshot: Path, output: Path) -> list[tuple[str, list[str]]]:
    seeds = list(suite.seeds) or [None]
    commands = []
    for seed in seeds:
        stage = "text" if seed is None else f"text_seed{seed}"
        path = output / "text" if seed is None else output / "text" / f"seed-{seed}"
        commands.append(
            (
                stage,
                build_lm_eval_command(
                    snapshot,
                    suite,
                    path,
                    limit_override=str(args.limit),
                    backend_override="hf",
                    python_executable=LOCAL_PYTHON,
                    seed_override=seed,
                    device=args.device,
                    batch_size_override=str(args.batch_size),
                ),
            )
        )
    return commands


def vision_command(args, suite, snapshot: Path, output: Path) -> list[str]:
    command = build_vlm_eval_command(
        snapshot, suite, output / "vision", ROOT / "external" / "VLMEvalKit"
    )
    command[0] = str(VISION_PYTHON)
    return [
        *command,
        "--server",
        "local",
        "--server-python",
        str(LOCAL_PYTHON),
        "--device",
        args.device,
        "--limit",
        str(args.limit),
        # The local server answers one request at a time; a long answer on the Apple GPU can
        # take many minutes, and a timeout would be retried and counted as a failed answer.
        "--api-nproc",
        "1",
        "--timeout",
        "14400",
        "--server-timeout",
        "1800",
    ]


def check_text(output: Path, suite, limit: int, cap: int) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(suite_model(output)))
    report: dict = {"tasks": {}, "problems": []}
    samples = sorted((output / "text").rglob("samples_*.jsonl"))
    results = sorted((output / "text").rglob("results_*.json"))
    if not results:
        report["problems"].append("lm-eval wrote no results file")
    for path in samples:
        task = path.name.removeprefix("samples_").rsplit("_", 1)[0]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        responses = [row["resps"][0][0] for row in rows if row.get("resps")]
        generated = [r for r in responses if isinstance(r, str)]
        lengths = [len(tokenizer(r)["input_ids"]) for r in generated]
        entry = {
            "samples": len(rows),
            "generated": len(generated),
            "mean_answer_tokens": round(sum(lengths) / len(lengths)) if lengths else None,
            "at_cap": sum(n >= cap - 1 for n in lengths),
            "think_tags": sum("<think>" in r or "</think>" in r for r in generated),
        }
        report["tasks"][task] = entry
        if len(rows) == 0:
            report["problems"].append(f"{task}: no samples")
        if entry["think_tags"] and not suite.enable_thinking:
            report["problems"].append(f"{task}: thinking tags in instruct-mode answers")
    expected = set(suite.text_tasks)
    seen = " ".join(report["tasks"])
    for task in expected:
        if task not in seen:
            report["problems"].append(f"{task}: no samples written")
    if results:
        metrics = json.loads(results[-1].read_text(encoding="utf-8")).get("results", {})
        report["metrics"] = {
            name: {k: v for k, v in values.items() if isinstance(v, int | float)}
            for name, values in metrics.items()
        }
    return report


def check_vision(output: Path, suite, limit: int) -> dict:
    report: dict = {"problems": []}
    log = output / "vision" / "local_server_requests.jsonl"
    requests = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    report["requests"] = len(requests)
    report["at_cap"] = sum(r["finish_reason"] == "length" for r in requests)
    report["thinking_open"] = sum(bool(r.get("thinking_in_prompt")) for r in requests)
    report["seconds"] = round(sum(r["seconds"] for r in requests))
    if not requests:
        report["problems"].append("the local server received no requests")
    if report["thinking_open"] and not suite.enable_thinking:
        report["problems"].append("instruct-mode prompts ended with an open <think> block")
    counts = subprocess.run(
        [
            str(VISION_PYTHON),
            "-c",
            "import json,sys,glob,pandas as pd\n"
            "out={}\n"
            "for name in sys.argv[2:]:\n"
            "    files=sorted(glob.glob(f'{sys.argv[1]}/**/*_{name}.xlsx', recursive=True))\n"
            "    if not files: out[name]=None; continue\n"
            "    d=pd.read_excel(files[-1]); p=d['prediction'].astype(str)\n"
            f"    fails=int(p.str.contains('{API_FAILURE}').sum())\n"
            "    out[name]={'rows':len(d),'api_failures':fails}\n"
            "print(json.dumps(out))",
            str(output / "vision"),
            *suite.vision_tasks,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    report["datasets"] = json.loads(counts)
    for name, entry in report["datasets"].items():
        if entry is None:
            report["problems"].append(f"{name}: no predictions file")
        elif entry["api_failures"]:
            report["problems"].append(f"{name}: {entry['api_failures']} failed requests")
    rule_scored = [name for name in suite.vision_tasks if name not in suite.vision_extracted_tasks]
    for name in rule_scored:
        if not list((output / "vision").rglob(f"*_{name}_*acc*.csv")) and not list(
            (output / "vision").rglob(f"*_{name}_*score*")
        ):
            report["problems"].append(f"{name}: no rule-based score file")
    return report


def suite_model(output: Path) -> Path:
    return Path(json.loads((output / "run_manifest.json").read_text())["model_path"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--suite", type=Path, help="Benchmark suite override")
    parser.add_argument("--model-path", type=Path, help="Local checkpoint, e.g. a quantized export")
    parser.add_argument("--variant", default="bf16")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--answer-cap",
        type=int,
        help="Lower answer cap for this local run only. On the Apple GPU one answer that runs to "
        "8,192 tokens holds its whole batch for ~30 min; the GPU run keeps the suite's cap.",
    )
    parser.add_argument("--stages", default="text,vision,score")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:  # 1 only for debugging this script
        raise ValueError("--limit is for a small validation subset (1-50)")
    stages = set(args.stages.split(","))

    config = load_config(args.config)
    suite = load_benchmark_suite(args.suite or config.evaluation.suite_path)
    label = suite_label(suite, config.evaluation.suite_path)
    suite_cap = int(suite.generation.get("max_gen_toks", 256))
    if args.answer_cap is not None:
        suite = dataclasses.replace(
            suite, generation={**suite.generation, "max_gen_toks": args.answer_cap}
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = args.variant if label is None else f"{args.variant}-{label}"
    output = (args.output or ROOT / "results" / "local-validate" / f"{name}-{stamp}").resolve()
    if args.model_path:
        snapshot, revision = args.model_path.resolve(), None
    else:
        snapshot, revision = download_model(config)
    if not suite.vision_tasks:
        stages.discard("vision")
        stages.discard("score")
    if not suite.vision_extracted_tasks:
        stages.discard("score")

    texts = text_commands(args, suite, snapshot, output) if "text" in stages else []
    vision = vision_command(args, suite, snapshot, output) if "vision" in stages else None
    score = [
        sys.executable,
        str(ROOT / "scripts" / "score_vision.py"),
        "--run-dir",
        str(output),
        "--config",
        str(args.config),
        "--allow-pilot",
    ]
    if args.dry_run:
        print(json.dumps({"text": dict(texts), "vision": vision, "score": score}, indent=2))
        return

    output.mkdir(parents=True, exist_ok=False)
    cap = int(suite.generation.get("max_gen_toks", 256))
    manifest: dict = {
        "schema_version": 1,
        "feature": config.feature,
        "variant": args.variant,
        "purpose": "local validation before a GPU run; not a research result",
        "model_id": config.model.id,
        "model_revision": revision,
        "model_path": str(snapshot),
        "code_revision": git_revision(ROOT),
        "config_digest": config.digest,
        "benchmark_suite": suite_record(suite),
        "device": args.device,
        "text_backend": "lm-eval hf (transformers)",
        "vision_server": "scripts/local_vlm_server.py (transformers)",
        # Larger images are downscaled locally; the Apple GPU cannot hold the vision encoder's
        # attention for a full-page scan without flash attention.
        "local_image_max_pixels": 1024 * 1024,
        "limit": args.limit,
        "vision_limit": args.limit,
        "answer_cap": {"suite": suite_cap, "local": cap},
        "enable_thinking": suite.enable_thinking,
        "research_result": False,
        "status": "running",
        "durations_seconds": {},
    }
    write_json(output / "run_manifest.json", manifest)
    log_path = output / "run.log"
    report: dict = {}
    try:
        for stage, command in texts:
            manifest["durations_seconds"][stage] = run_logged(command, log_path, ROOT)
        if vision:
            manifest["durations_seconds"]["vision"] = run_logged(vision, log_path, ROOT)
        # score_vision.py requires a passed run manifest.
        manifest["status"] = "passed"
        write_json(output / "run_manifest.json", manifest)
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        write_json(output / "run_manifest.json", manifest)

    problems: list[str] = []
    if "score" in stages:
        # A failed scoring pass is a validation finding, not a reason to skip the report.
        try:
            manifest["durations_seconds"]["score"] = run_logged(score, log_path, ROOT)
        except subprocess.CalledProcessError:
            problems.append("score_vision.py reported invalid scoring (see run.log)")
        write_json(output / "run_manifest.json", manifest)
    if texts:
        report["text"] = check_text(output, suite, args.limit, cap)
        problems += report["text"]["problems"]
    if vision:
        report["vision"] = check_vision(output, suite, args.limit)
        problems += report["vision"]["problems"]
    scoring_manifest = output.with_name(output.name + "-scored") / "scoring_manifest.json"
    if "score" in stages and scoring_manifest.exists():
        scored = json.loads(scoring_manifest.read_text())
        report["score"] = {
            "extraction_failures": scored["extraction_failures"],
            "final_answer_rule": scored["final_answer_rule"]["answers"],
        }
    report["answer_cap"] = manifest["answer_cap"]
    report["durations_seconds"] = manifest["durations_seconds"]
    report["problems"] = problems
    report["verdict"] = "PASS" if not problems else "FAIL"
    write_json(output / "validation_report.json", report)
    print(json.dumps(report, indent=2))
    print(f"validation {report['verdict']}: {output}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
