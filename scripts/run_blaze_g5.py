#!/usr/bin/env python3
"""Blaze gate G5 on one GPU: does Qwen3.5-2B agree with the 4B?

At a standard W4A16 export's size, the 2B's ladder frees enough bytes for an all-8-bit body
(Blaze-Q). G5 compares the two, both against BF16:

1. Setup: the text-evaluation environment and the 2B's BF16 snapshot.
2. BF16 on the text panel (MMLU-Pro's 1001-question subset, MATH-500, IFEval): its scores, and its
   MATH-500 and MMLU-Pro answers as the 2B's drift traces.
3. Exports: Feature 1's GPTQ W4A16 g128 recipe (scripts/quantize.py), and Blaze-Q, fake-quantized
   (the ladder and every body Linear at 8-bit g128 round-to-nearest; no calibration).
4. Drift scores of BF16, W4A16 and Blaze-Q on BF16's answers; the panel for W4A16 and Blaze-Q.
5. The gate report (blaze.gates.g5_report): Blaze-Q's excess loss at most a third of W4A16's, no
   larger as the real formats store them, and no panel task more than a point below BF16.

With --pilot-limit, a pilot runs first into <output>-pilot (N questions per task); the exports
are shared, so the full run reuses them. The full gate runs only if the pilot passes.

    python scripts/run_blaze_g5.py --pilot-limit 10 --dry-run
"""

from __future__ import annotations

import sys

# The GPU image's base Python ships brotlicffi, which breaks httpx downloads (run_drift_study.py).
for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import _bootstrap  # noqa: E402, F401

from qwen35_compression.blaze.export import MANIFEST, size_plan  # noqa: E402
from qwen35_compression.blaze.gates import cost, g5_report  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON, score_command  # noqa: E402
from qwen35_compression.feature1 import (  # noqa: E402
    build_text_eval_commands,
    load_benchmark_suite,
    run_logged,
)
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CENSUS = Path("data/blaze/census.json")
W4A16 = "gptq_w4a16_g128"
BLAZE_Q = "blaze_q_w8"
MODELS = ("bf16", W4A16, BLAZE_Q)
# On an A30: setup and the 2B download about 0.3 h, GPTQ about 0.3 h, three panel runs about
# 1.2 h (the 4B wrote ~1,500 tokens per MATH-500/MMLU-Pro answer at ~1,600 tokens/s), three
# drift scores about 0.3 h; the pilot about 0.3 h.
ESTIMATED_HOURS = {"pilot": 0.3, "full": 2.1}


def exports_of(config_outputs: Path) -> dict[str, Path]:
    return {W4A16: config_outputs / W4A16, BLAZE_Q: config_outputs / "blaze" / BLAZE_Q}


def plan(
    config_path: Path,
    output: Path,
    exports: dict[str, Path],
    snapshot: Path | str,
    limit: int | None,
) -> dict[str, list[str]]:
    """Every command after setup, by step name, in run order."""
    suite = load_benchmark_suite(load_config(config_path).evaluation.suite_path)
    limit_text = str(limit) if limit else None
    models = {"bf16": Path(snapshot), **exports}

    def panel(name: str) -> list[str]:
        [(_, command)] = build_text_eval_commands(
            models[name],
            suite,
            output / "panel" / name,
            None,
            limit_override=limit_text,
            python_executable=TEXT_PYTHON,
        )
        return command

    traces = output / "panel" / "bf16"
    steps: dict[str, list[str]] = {
        "panel:bf16": panel("bf16"),
        f"export:{W4A16}": [
            sys.executable,
            "scripts/quantize.py",
            "--config",
            str(config_path),
            "--variant",
            W4A16,
        ],
        f"export:{BLAZE_Q}": [
            sys.executable,
            "scripts/blaze_export.py",
            "--model",
            str(snapshot),
            "--census",
            str(CENSUS),
            "--ladder",
            "--body-bits",
            "8",
            "--output",
            str(exports[BLAZE_Q]),
        ],
    }
    for name in MODELS:
        steps[f"score:{name}"] = score_command(
            models[name], name, traces, output / f"{name}.json", limit, Path(snapshot)
        )
    for name in (W4A16, BLAZE_Q):
        steps[f"panel:{name}"] = panel(name)
    return steps


def panel_drops(output: Path) -> dict[str, Any]:
    """Per task: BF16's score, Blaze-Q's and W4A16's, and Blaze-Q's paired change from BF16."""
    from panel_scores import paired_delta, per_question

    subset = json.loads(
        (ROOT / "configs" / "evaluation" / "feature1_thinking_samples.json").read_text(
            encoding="utf-8"
        )
    )
    rows = {name: per_question(output / "panel" / name, subset) for name in MODELS}
    table: dict[str, Any] = {}
    for task, reference in rows["bf16"].items():
        table[task] = {
            name: 100 * sum(rows[name][task].values()) / len(rows[name][task]) for name in MODELS
        }
        table[task]["blaze_q_vs_bf16"] = paired_delta(rows[BLAZE_Q][task], reference)
        table[task]["w4a16_vs_bf16"] = paired_delta(rows[W4A16][task], reference)
    return table


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    exports = exports_of(config.paths.outputs)
    run = GateRun(
        output,
        "blaze_g5",
        args.code_revision,
        config_digest=config.digest,
        limit=limit,
        research_result=limit is None,
    )
    log = output / "run.log"
    snapshot, revision = run.step("download_bf16", lambda: download_model(config))
    run.manifest["base_model"] = {"id": config.model.id, "revision": revision}
    for name, command in plan(args.config, output, exports, snapshot, limit).items():
        if name.startswith("export:"):
            target = exports[name.split(":", 1)[1]]
            done = target / ("config.json" if name.endswith(W4A16) else MANIFEST)
            if done.exists():
                run.reuse(name)
                continue
        run.step(name, lambda command=command: run_logged(command, log, ROOT))

    from qwen35_compression.blaze.census import load_census
    from qwen35_compression.blaze.ladder import plan_ladder
    from qwen35_compression.glaze2.quant import Option

    ladder = plan_ladder(load_census(ROOT / CENSUS))
    standard = size_plan(Path(snapshot), Option(4, 128))
    blaze_q = size_plan(Path(snapshot), Option(8, 128), ladder)
    w4a16_files = sum(p.stat().st_size for p in exports[W4A16].glob("*.safetensors"))
    sizes = {"w4a16": standard["total_bytes"], "blaze_q": blaze_q["total_bytes"]}

    def load(name: str) -> dict[str, Any]:
        return json.loads((output / f"{name}.json").read_text(encoding="utf-8"))

    table = panel_drops(output) if not limit else None
    drops = (
        {task: -entry["blaze_q_vs_bf16"]["delta"] for task, entry in table.items()}
        if table
        else None
    )
    report = g5_report(load("bf16"), load(W4A16), load(BLAZE_Q), sizes, drops)
    report["size_plans"] = {
        "w4a16": standard,
        "blaze_q": blaze_q,
        "w4a16_safetensors_bytes": w4a16_files,
    }
    report["panel"] = table
    write_json(output / "g5_report.json", report)
    run.finish(checks=report["checks"], decision=report["decision"])
    print("decision=" + report["decision"], flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/blaze_2b.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/g5"))
    parser.add_argument("--pilot-limit", type=int, help="Run small into <output>-pilot first")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    if not (ROOT / CENSUS).exists():
        parser.error(f"missing {CENSUS}: run scripts/blaze_census.py first")
    output = args.output.resolve()
    pilot_output = output.with_name(output.name + "-pilot")

    if args.dry_run:
        exports = exports_of(load_config(args.config).paths.outputs)
        hours = ESTIMATED_HOURS["full"] + (ESTIMATED_HOURS["pilot"] if args.pilot_limit else 0)
        print(
            json.dumps(
                {
                    "pilot": plan(
                        args.config, pilot_output, exports, "<bf16 snapshot>", args.pilot_limit
                    )
                    if args.pilot_limit
                    else None,
                    "full": plan(args.config, output, exports, "<bf16 snapshot>", None),
                    "estimate_on_a30": cost(hours),
                },
                indent=2,
            )
        )
        return

    if not args.skip_bootstrap:
        bootstrap_text(ROOT)
    if args.pilot_limit:
        run_gate(args, pilot_output, args.pilot_limit)
        if args.pilot_only:
            return
    run_gate(args, output, None)


if __name__ == "__main__":
    main()
