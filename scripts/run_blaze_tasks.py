#!/usr/bin/env python3
"""Blaze's open measurements on one GPU: Blaze-S's reasoning panel, task accuracy off English,
and long requests at a fixed memory budget.

1. Blaze-S (AutoRound's published 4-bit body, the multilingual ladder, an 8-bit vision tower):
   MMLU-Pro (1,001), MATH-500 and IFEval, paired against BF16's, GPTQ's and AutoRound's published
   answers to the same questions. Only its drift and MMMU were measured before.
2. Task accuracy where drift was measured: HumanEval and MBPP pass@1 and MGSM in 11 languages,
   on the same prompts as the domain drift (scripts/blaze_domain_answers.py). BF16's answers come
   from the Q6 run; GPTQ W4A16, the multilingual Blaze-Q and Blaze-S answer here.
3. Long context: the KV cache each checkpoint leaves at 0.85 of an A30, and the time to serve
   64 requests of 16,384 prompt tokens (scripts/blaze_long_context.py).

Each measurement is independent: a failed step is recorded and the run moves on.

    python scripts/run_blaze_tasks.py --plan inputs/g2b_plan.json \
        --bf16-answers inputs/domain_traces --pilot-limit 10 --dry-run
"""

from __future__ import annotations

import sys

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import _bootstrap  # noqa: E402, F401
import run_blaze_q6 as q6  # noqa: E402
import run_blaze_s as s  # noqa: E402
import run_blaze_s2 as s2  # noqa: E402

from qwen35_compression.blaze.export import MANIFEST  # noqa: E402
from qwen35_compression.blaze.gates import cost  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON  # noqa: E402
from qwen35_compression.feature1 import (  # noqa: E402
    FLA_VERSION,
    build_text_eval_commands,
    fla_install_command,
    load_benchmark_suite,
    run_logged,
)
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLAZE_Q = "blaze_q6_ml"
BLAZE_S = s2.EXPORT
GPTQ = q6.GPTQ
# The models that answer the domain prompts here; BF16's answers are an input.
ANSWERING = (GPTQ, BLAZE_Q, BLAZE_S)
LONG = ("bf16", GPTQ, BLAZE_Q, BLAZE_S)
PILOT_LONG = ["--prompt-tokens", "2048", "--requests", "4", "--max-model-len", "4096"]
# On an A30: setup and downloads 0.3 h, pilot 0.2 h, the two exports 0.25 h, three answer runs
# 0.3 h, scoring 0.05 h, Blaze-S's panel at 4-bit speed 0.4 h, four long-context runs 0.35 h.
ESTIMATED_HOURS = 1.85


def _env(name: str, command: list[str]) -> list[str]:
    # Blaze-Q's 6-bit groups run on vLLM's Humming kernel, which needs NVRTC on the loader path.
    return q6.with_env("native", command) if name == BLAZE_Q else command


def plan(
    output: Path, outputs: Path, snapshot: Path | str, plan_file: Path, limit: int | None
) -> dict[str, list[str]]:
    exports = outputs / "blaze"
    suffix = "-pilot" if limit else ""
    models = {
        "bf16": Path(snapshot),
        GPTQ: outputs / GPTQ,
        BLAZE_Q: exports / (BLAZE_Q + suffix),
        BLAZE_S: exports / BLAZE_S,
    }
    steps: dict[str, list[str]] = {
        f"export:{BLAZE_S}": s2.addon_command(outputs / s2.SOURCE, models[BLAZE_S]),
        f"export:{BLAZE_Q}": q6.export_command(
            models[BLAZE_Q], plan_file, limit, q6.CENSUS_MULTILINGUAL, q6.EQUAL_SIZE_COVER
        ),
    }
    for name in ANSWERING:
        steps[f"answers:{name}"] = _env(
            name,
            [
                str(TEXT_PYTHON),
                "scripts/blaze_domain_answers.py",
                "--model",
                str(models[name]),
                "--output",
                str(output / "answers" / name),
                *(["--limit", str(q6.DOMAIN_PILOT)] if limit else []),
            ],
        )
    for name in ("bf16", *ANSWERING):
        steps[f"tasks:{name}"] = [
            str(TEXT_PYTHON),
            "scripts/blaze_task_scores.py",
            "--answers",
            str(output / "answers" / name),
            "--name",
            name,
            "--output",
            str(output / f"tasks_{name}.json"),
        ]
    [(_, panel)] = build_text_eval_commands(
        models[BLAZE_S],
        load_benchmark_suite(s.PANEL_SUITE),
        output / "panel" / BLAZE_S,
        None,
        limit_override=str(limit) if limit else None,
        python_executable=TEXT_PYTHON,
    )
    steps[f"panel:{BLAZE_S}"] = panel
    for name in LONG if not limit else (BLAZE_S,):
        steps[f"long:{name}"] = _env(
            name,
            [
                str(TEXT_PYTHON),
                "scripts/blaze_long_context.py",
                "--model",
                str(models[name]),
                "--name",
                name,
                "--output",
                str(output / f"long_{name}.json"),
                *(PILOT_LONG if limit else []),
            ],
        )
    return steps


def tasks_table(output: Path) -> dict[str, Any]:
    from panel_scores import paired_delta

    def load(name: str) -> dict | None:
        path = output / f"tasks_{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    base = load("bf16")
    table: dict[str, Any] = {}
    for name in ("bf16", *ANSWERING):
        result = load(name)
        if result is None:
            continue
        entry: dict[str, Any] = {"summary": result["summary"]}
        if base is not None and name != "bf16":
            entry["vs_bf16"] = {}
            for group, files in (
                ("humaneval", ["domain_code_humaneval"]),
                ("mbpp", ["domain_code_mbpp"]),
                (
                    "mgsm_non_english",
                    [
                        f
                        for f in result["per_file"]
                        if f.startswith("domain_ml_") and f != "domain_ml_en"
                    ],
                ),
            ):
                mine = {(f, k): v for f in files for k, v in result["per_file"].get(f, {}).items()}
                ref = {(f, k): v for f in files for k, v in base["per_file"].get(f, {}).items()}
                if mine and ref:
                    entry["vs_bf16"][group] = paired_delta(mine, ref)
        table[name] = entry
    return table


def panel_table(output: Path) -> dict[str, Any] | None:
    from panel_scores import paired_delta, per_question

    if not any((output / "panel" / BLAZE_S).rglob("samples_*.jsonl")):
        return None
    subset = json.loads(s.SUBSET.read_text(encoding="utf-8"))
    runs = {name: output / "published" / name for name in s.PUBLISHED}
    runs["blaze_s"] = output / "panel" / BLAZE_S
    rows = {name: per_question(path, subset) for name, path in runs.items()}
    table: dict[str, Any] = {}
    for task, reference in rows["bf16"].items():
        entry: dict[str, Any] = {
            name: 100 * sum(rows[name][task].values()) / len(rows[name][task]) for name in runs
        }
        entry["blaze_s_vs_bf16"] = paired_delta(rows["blaze_s"][task], reference)
        entry["blaze_s_vs_autoround"] = paired_delta(rows["blaze_s"][task], rows["autoround"][task])
        entry["blaze_s_vs_gptq"] = paired_delta(rows["blaze_s"][task], rows["gptq"][task])
        table[task] = entry
    return table


def long_table(output: Path) -> dict[str, Any]:
    out = {}
    for name in LONG:
        path = output / f"long_{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import copy_published
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    outputs = config.paths.outputs
    run = GateRun(
        output,
        "blaze_tasks",
        args.code_revision,
        config_digest=config.digest,
        limit=limit,
        research_result=limit is None,
    )
    log = output / "run.log"
    snapshot, revision = run.step("download_bf16", lambda: download_model(config))
    run.manifest["base_model"] = {"id": config.model.id, "revision": revision}
    for name, repo in ((s2.SOURCE, s2.SOURCE_REPO), (GPTQ, q6.GPTQ_REPO)):
        if not (outputs / name).exists():
            run.manifest[f"{name}_revision"] = run.step(
                f"fetch:{name}", lambda repo=repo, name=name: copy_published(repo, outputs / name)
            )
    run.manifest["published_answers"] = run.step(
        "fetch:published", lambda: s.fetch_published(output / "published")
    )
    bf16_answers = output / "answers" / "bf16"
    shutil.copytree(args.bf16_answers, bf16_answers, dirs_exist_ok=True)
    if limit:
        # The pilot scores only the first few BF16 answers per file, as the others answer.
        for path in bf16_answers.glob("domain_*.jsonl"):
            kept = path.read_text(encoding="utf-8").splitlines()[: q6.DOMAIN_PILOT]
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    failed: list[str] = []
    for name, command in plan(output, outputs, snapshot, args.plan, limit).items():
        target = command[command.index("--output") + 1] if "--output" in command else None
        if name.startswith("export:") and target and (Path(target) / MANIFEST).exists():
            run.reuse(name)
            continue
        if target and target.endswith(".json") and Path(target).exists():
            run.reuse(name)
            continue
        try:
            run.step(name, lambda command=command: run_logged(command, log, ROOT))
        except subprocess.CalledProcessError:
            if name.startswith("export:") or limit:
                raise  # every later step needs the exports; the pilot must pass whole
            failed.append(name)
            run.manifest["status"] = "running"
            run.save()
    if limit:
        run.finish()
        return run.manifest
    report = {
        "tasks": tasks_table(output),
        "panel": panel_table(output),
        "long_context": long_table(output),
        "failed_steps": failed,
    }
    write_json(output / "tasks_report.json", report)
    run.finish(failed_steps=failed)
    print("failed=" + ",".join(failed) if failed else "all steps passed", flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/tasks"))
    parser.add_argument("--plan", type=Path, required=True, help="G2b's market plan")
    parser.add_argument(
        "--bf16-answers", type=Path, required=True, help="BF16's domain answers (Q6 run)"
    )
    parser.add_argument("--pilot-limit", type=int, help="Run small into <output>-pilot first")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    output = args.output.resolve()
    pilot_output = output.with_name(output.name + "-pilot")

    if args.dry_run:
        outputs = load_config(args.config).paths.outputs
        print(
            json.dumps(
                {
                    "pilot": plan(pilot_output, outputs, "<4b>", args.plan, args.pilot_limit)
                    if args.pilot_limit
                    else None,
                    "full": plan(output, outputs, "<4b>", args.plan, None),
                    "estimate_on_a30": cost(ESTIMATED_HOURS),
                },
                indent=2,
            )
        )
        return
    for path in (args.plan, q6.CENSUS_MULTILINGUAL, s2.CENSUS):
        if not path.is_file():
            parser.error(f"missing {path}")
    if not any(args.bf16_answers.glob("domain_*.jsonl")):
        parser.error(f"no domain answers under {args.bf16_answers}")
    if not args.skip_bootstrap:
        bootstrap_text(ROOT)
    uv = shutil.which("uv") or "uv"
    subprocess.run(
        [uv, "pip", "install", "--python", str(TEXT_PYTHON), "--no-deps", str(q6.PLUGIN)],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(fla_install_command(sys.executable), cwd=ROOT, check=True)
    print(f"flash-linear-attention {FLA_VERSION} installed; blaze-vllm installed", flush=True)
    if args.pilot_limit:
        run_gate(args, pilot_output, args.pilot_limit)
    run_gate(args, output, None)


if __name__ == "__main__":
    main()
