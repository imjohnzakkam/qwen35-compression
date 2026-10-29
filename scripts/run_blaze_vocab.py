#!/usr/bin/env python3
"""Blaze's one-copy vocabulary on one GPU: exact rows, the same drift, less memory, same speed.

The `blaze-vllm` plugin used to hold each vocabulary block twice on the GPU: once for lookups
(the stored layout) and once in Marlin's layout for the LM head. Its default layout now keeps
only the Marlin copy and gathers lookup rows from it (`blaze_vllm.marlin_rows`). This run checks
that the change is free:

1. `blaze_vocab_check.py`: on vLLM's own Marlin repack, every row of every block shape a real
   export uses equals codes x scales and vLLM's own lookup, bit for bit.
2. The corrected Blaze-S (and, with --blaze-q, the real 4B Blaze-Q) rebuilt as before and scored
   on BF16's 1,501 answers with one copy: drift must match the two-copy runs (`--reuse`), answer by
   answer; speed and weight memory are measured against theirs.

    python scripts/run_blaze_vocab.py --bf16 inputs/g2_bf16.json --reuse inputs/two_copy \
        --plan inputs/g2b_plan.json --blaze-q --dry-run
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
import run_blaze_s2 as s2  # noqa: E402

from qwen35_compression.blaze.export import MANIFEST  # noqa: E402
from qwen35_compression.blaze.gates import cost, drift_point, excess_loss  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON, score_command  # noqa: E402
from qwen35_compression.feature1 import FLA_VERSION, fla_install_command, run_logged  # noqa: E402
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = Path("plugins/blaze_vllm")
# The two-copy runs these are compared with: drift and speed result files (--reuse holds them).
TWO_COPY = {
    "blaze_s2": ("blaze_s2.json", "speed_blaze_s2.json"),
    "blaze_q6": ("blaze_q6_native.json", "speed_blaze_q6_native.json"),
}
# Drift may differ from the two-copy run by no more than this per token (kernel reduction order).
DRIFT_TOLERANCE = 1e-4
# On an A30: setup and downloads 0.25 h, Blaze-S export 0.05 h, the row check 0.05 h, pilot
# 0.05 h, drift 0.12 h and speed 0.08 h per model; Blaze-Q's GPTQ export 0.25 h.
ESTIMATED_HOURS = {"s": 0.6, "s+q": 1.05}


def with_layout(layout: str, command: list[str], extra: dict[str, str] | None = None) -> list[str]:
    env = {"BLAZE_VOCAB": layout, **(extra or {})}
    return ["env", *(f"{k}={v}" for k, v in env.items()), *command]


def plan(
    output: Path,
    outputs: Path,
    snapshot: Path | str,
    plan_file: Path | None,
    limit: int | None,
) -> dict[str, list[str]]:
    snapshot = Path(snapshot)
    exports = outputs / "blaze"
    models = {"blaze_s2": exports / s2.EXPORT}
    rows = [str(TEXT_PYTHON), "scripts/blaze_vocab_check.py", "--output", str(output / "rows.json")]
    # The pilot checks rows first, so a layout error stops the run before any long step.
    steps: dict[str, list[str]] = {"rows": rows} if limit else {}
    steps["export:blaze_s2"] = s2.addon_command(outputs / s2.SOURCE, models["blaze_s2"])
    if plan_file is not None and not limit:
        models["blaze_q6"] = exports / q6.EXPORT
        steps["export:blaze_q6"] = q6.export_command(models["blaze_q6"], plan_file, None)
    if not limit:
        steps["rows"] = rows
    native = q6.MODES["native"]  # Humming's NVRTC library path, for the 6-bit body
    for name, model in models.items():
        steps[f"score:{name}"] = with_layout(
            "marlin",
            score_command(model, name, output / "traces", output / f"{name}.json", limit, snapshot),
            native,
        )
        if limit:
            break  # one model's drift on a few answers shows the layout loads and runs
        steps[f"speed:{name}"] = with_layout(
            "marlin",
            [
                str(TEXT_PYTHON),
                "scripts/blaze_speed.py",
                "--model",
                str(model),
                "--name",
                name,
                "--output",
                str(output / f"speed_{name}.json"),
            ],
            native,
        )
    return steps


def per_answer_gap(one: dict, two: dict) -> float:
    """Largest per-token NLL difference between two scores of the same answers."""
    left = {(r["task"], r["doc_id"]): r for r in one["per_answer"]}
    right = {(r["task"], r["doc_id"]): r for r in two["per_answer"]}
    if set(left) != set(right):
        raise ValueError("the two scores cover different answers")
    return max(abs(left[k]["nll"] - right[k]["nll"]) / max(left[k]["tokens"], 1) for k in left)


def vocab_report(output: Path, reuse: Path, bf16: dict, models: list[str]) -> dict[str, Any]:
    def load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    rows = load(output / "rows.json")
    report: dict[str, Any] = {"rows": rows, "models": {}}
    checks = {"rows_exact": rows["exact"]}
    for name in models:
        drift_file, speed_file = TWO_COPY[name]
        one, two = load(output / f"{name}.json"), load(reuse / drift_file)
        speed_one, speed_two = load(output / f"speed_{name}.json"), load(reuse / speed_file)
        gap = per_answer_gap(one, two)
        entry = {
            "one_copy": {**drift_point(one), "excess_loss": excess_loss(one, bf16)},
            "two_copy": {**drift_point(two), "excess_loss": excess_loss(two, bf16)},
            "largest_per_token_gap": gap,
            "weight_memory_bytes": {
                "one_copy": speed_one["weight_memory_bytes"],
                "two_copy": speed_two["weight_memory_bytes"],
            },
            "single_stream_tok_s": {
                "one_copy": speed_one["single_stream"]["tokens_per_second"],
                "two_copy": speed_two["single_stream"]["tokens_per_second"],
            },
            "batched_tok_s": {
                "one_copy": speed_one["batched"]["tokens_per_second"],
                "two_copy": speed_two["batched"]["tokens_per_second"],
            },
        }
        report["models"][name] = entry
        checks[f"{name}_same_drift"] = gap <= DRIFT_TOLERANCE
        checks[f"{name}_less_memory"] = (
            entry["weight_memory_bytes"]["one_copy"] < entry["weight_memory_bytes"]["two_copy"]
        )
        checks[f"{name}_speed_within_3pct"] = (
            entry["single_stream_tok_s"]["one_copy"]
            >= 0.97 * entry["single_stream_tok_s"]["two_copy"]
        )
    report["checks"] = checks
    report["decision"] = (
        "go: one copy is exact and free" if all(checks.values()) else "report as measured"
    )
    return report


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import copy_published, fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    outputs = config.paths.outputs
    run = GateRun(
        output,
        "blaze_vocab",
        args.code_revision,
        config_digest=config.digest,
        limit=limit,
        research_result=limit is None,
    )
    log = output / "run.log"
    snapshot, revision = run.step("download_bf16", lambda: download_model(config))
    run.manifest["base_model"] = {"id": config.model.id, "revision": revision}
    run.manifest["traces"] = {
        "revision": run.step("traces", lambda: fetch_traces(output / "traces"))
    }
    if not (outputs / s2.SOURCE).exists():
        run.manifest["source_revision"] = run.step(
            "fetch_source", lambda: copy_published(s2.SOURCE_REPO, outputs / s2.SOURCE)
        )
    steps = plan(output, outputs, snapshot, args.plan if args.blaze_q else None, limit)
    for name, command in steps.items():
        if name.startswith("export:"):
            target = Path(command[command.index("--output") + 1])
            if (target / "blaze_addon.json").exists() or (target / MANIFEST).exists():
                run.reuse(name)
                continue
        run.step(name, lambda command=command: run_logged(command, log, ROOT))
    if limit:
        run.finish()
        return run.manifest
    bf16 = json.loads(args.bf16.read_text(encoding="utf-8"))
    models = ["blaze_s2", *(["blaze_q6"] if args.blaze_q else [])]
    report = vocab_report(output, args.reuse, bf16, models)
    write_json(output / "vocab_report.json", report)
    run.finish(checks=report["checks"], decision=report["decision"])
    print("decision=" + report["decision"], flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/vocab"))
    parser.add_argument("--bf16", type=Path, required=True, help="G2's BF16 drift score")
    parser.add_argument("--reuse", type=Path, required=True, help="The two-copy runs' results")
    parser.add_argument("--plan", type=Path, help="G2b's market plan, for --blaze-q")
    parser.add_argument("--blaze-q", action="store_true", help="Also the real 4B Blaze-Q")
    parser.add_argument("--pilot-limit", type=int, help="Run small into <output>-pilot first")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.blaze_q and args.plan is None:
        parser.error("--blaze-q needs --plan")
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    output = args.output.resolve()
    pilot_output = output.with_name(output.name + "-pilot")
    plan_file = args.plan if args.blaze_q else None

    if args.dry_run:
        outputs = load_config(args.config).paths.outputs
        print(
            json.dumps(
                {
                    "pilot": plan(pilot_output, outputs, "<4b>", plan_file, args.pilot_limit)
                    if args.pilot_limit
                    else None,
                    "full": plan(output, outputs, "<4b>", plan_file, None),
                    "estimate_on_a30": cost(ESTIMATED_HOURS["s+q" if args.blaze_q else "s"]),
                },
                indent=2,
            )
        )
        return
    for path in [args.bf16, *(args.reuse / f for pair in TWO_COPY.values() for f in pair)]:
        if not path.is_file() and (args.blaze_q or "q6" not in path.name):
            parser.error(f"missing {path}")
    if not args.skip_bootstrap:
        bootstrap_text(ROOT)
    uv = shutil.which("uv") or "uv"
    subprocess.run(
        [uv, "pip", "install", "--python", str(TEXT_PYTHON), "--no-deps", str(PLUGIN)],
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
