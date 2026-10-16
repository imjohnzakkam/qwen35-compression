#!/usr/bin/env python3
"""Blaze gate G4 on one GPU: G2b's Blaze-Q on the rest of the full suite.

G3 scored G2b's export on MATH-500, IFEval and MMMU; G4 scores the suite's other tasks with the
instruct track's settings, so the two runs together are the full suite. BF16's and Glaze v2's
full-suite results are on record and are not run again.

1. Setup: the text and vision environments, and flash-linear-attention.
2. The export, rebuilt from G2b's saved plan (the same code, calibration text and options), and
   its drift on BF16's answers, which must match G2b's to show it is the same model.
3. A pilot of both suites on N questions per task, on that export, into <output>-pilot; the full
   suites start only if it passes.
4. Text: MMLU-Pro in full, GSM8K, HellaSwag, ARC-Challenge and WikiText-2. Vision: MMBench,
   MathVista, TextVQA, OCRBench and DocVQA. MMBench and MathVista are scored after download
   (scripts/score_vision.py --run-dir <output>/vision); scripts/blaze_g4_report.py then compares
   every task with BF16 and Glaze v2.

    python scripts/run_blaze_g4.py --plan inputs/g2b_plan.json --pilot-limit 10 --dry-run
"""

from __future__ import annotations

import sys

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import _bootstrap  # noqa: E402, F401

from qwen35_compression.blaze.export import MANIFEST  # noqa: E402
from qwen35_compression.blaze.gates import cost, drift_point, excess_loss  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, vision_run_manifest  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON, score_command  # noqa: E402
from qwen35_compression.feature1 import (  # noqa: E402
    FLA_VERSION,
    build_text_eval_commands,
    build_vlm_eval_command,
    fla_install_command,
    load_benchmark_suite,
    run_logged,
)
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CENSUS = Path("data/blaze/census.json")
TEXT_SUITE = Path("configs/evaluation/blaze_g4_text.yaml")
VISION_SUITE = Path("configs/evaluation/blaze_g4_vision.yaml")
VISION_PYTHON = ROOT / ".venv-gpu-vision" / "bin" / "python"
EXPORT = "blaze_q_g2b"
# G2b's measured excess loss; the rebuilt export must land within this of it.
G2B_EXCESS = 0.001689468332214339
REBUILD_TOLERANCE = 0.0002
# On an A30, from past full-suite runs (GPTQ: text 212 min, vision 91 min) with MATH-500, IFEval
# and MMMU left out, at BF16 decode speed (the export is fake-quantized); plus setup, the
# rebuild, its drift check and the pilot.
ESTIMATED_HOURS = 5.9


def suites(output: Path, export: Path, limit: int | None) -> dict[str, list[str]]:
    """The text and vision suite commands, small with `limit`."""
    [(_, text)] = build_text_eval_commands(
        export,
        load_benchmark_suite(TEXT_SUITE),
        output / "text",
        None,
        limit_override=str(limit) if limit else None,
        python_executable=TEXT_PYTHON,
    )
    vision = build_vlm_eval_command(
        export,
        load_benchmark_suite(VISION_SUITE),
        output / "vision" / "vision",
        ROOT / "external/VLMEvalKit",
    )
    vision[0] = str(VISION_PYTHON)
    if limit:
        vision.extend(("--limit", str(limit)))
    return {"text": text, "vision": vision}


def plan(
    output: Path,
    export: Path,
    snapshot: Path | str,
    plan_file: Path,
    pilot_limit: int | None,
) -> dict[str, list[str]]:
    """Every command after setup, in run order: rebuild, drift check, pilot, full suites."""
    steps: dict[str, list[str]] = {
        "export": [
            sys.executable,
            "scripts/blaze_gptq.py",
            "--census",
            str(CENSUS),
            "--ladder",
            "--bits",
            "6",
            "--plan",
            str(plan_file),
            "--vision-bits",
            "8",
            "--output",
            str(export),
        ],
        "score:rebuild": score_command(
            export, EXPORT, output / "traces", output / "rebuild_drift.json", None, Path(snapshot)
        ),
    }
    if pilot_limit:
        pilot = output.with_name(output.name + "-pilot")
        steps.update({f"pilot:{k}": v for k, v in suites(pilot, export, pilot_limit).items()})
    steps.update(suites(output, export, None))
    return steps


def vision_record(args: argparse.Namespace, config, revision: str, export: Path, limit):
    return vision_run_manifest(
        EXPORT,
        config.model.id,
        revision,
        args.code_revision,
        config.digest,
        VISION_SUITE,
        export,
        limit,
    )


def run_gate(args: argparse.Namespace, output: Path, export: Path) -> dict[str, Any]:
    from qwen35_compression.drift import fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    run = GateRun(
        output, "blaze_g4", args.code_revision, config_digest=config.digest, research_result=True
    )
    log = output / "run.log"
    snapshot, revision = run.step("download_bf16", lambda: download_model(config))
    run.manifest["base_model"] = {"id": config.model.id, "revision": revision}
    run.manifest["traces"] = {
        "revision": run.step("traces", lambda: fetch_traces(output / "traces"))
    }
    pilot = output.with_name(output.name + "-pilot")
    for name, command in plan(
        output, export, snapshot, args.plan.resolve(), args.pilot_limit
    ).items():
        if name == "export" and (export / MANIFEST).exists():
            run.reuse(name)
            continue
        if name == "vision":
            # The pilot passed: its vision answers get the record local scoring reads.
            if args.pilot_limit:
                write_json(
                    pilot / "vision" / "run_manifest.json",
                    vision_record(args, config, revision, export, args.pilot_limit),
                )
        run.step(name, lambda command=command: run_logged(command, log, ROOT))
        if name == "score:rebuild":
            rebuilt = json.loads((output / "rebuild_drift.json").read_text(encoding="utf-8"))
            excess = excess_loss(rebuilt, json.loads(args.bf16.read_text(encoding="utf-8")))
            run.manifest["rebuild"] = {
                **drift_point(rebuilt),
                "excess_loss": excess,
                "g2b_excess_loss": G2B_EXCESS,
            }
            run.save()
            if abs(excess - G2B_EXCESS) > REBUILD_TOLERANCE:
                run.manifest["status"] = "failed"
                run.save()
                raise SystemExit(f"rebuilt export's excess loss {excess:.5f} is not G2b's")
    write_json(
        output / "vision" / "run_manifest.json", vision_record(args, config, revision, export, None)
    )
    run.finish()
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/g4"))
    parser.add_argument("--plan", type=Path, required=True, help="G2b's plan.json")
    parser.add_argument("--bf16", type=Path, required=True, help="G2's BF16 drift score")
    parser.add_argument("--pilot-limit", type=int, help="Pilot both suites on N questions first")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    output = args.output.resolve()
    export = load_config(args.config).paths.outputs / "blaze" / EXPORT

    if args.dry_run:
        print(
            json.dumps(
                {
                    "steps": plan(output, export, "<4b>", args.plan, args.pilot_limit),
                    "estimate_on_a30": cost(ESTIMATED_HOURS),
                },
                indent=2,
            )
        )
        return
    for path in (args.plan, args.bf16):
        if not path.is_file():
            parser.error(f"missing {path}")
    if not args.skip_bootstrap:
        subprocess.run(
            [sys.executable, "scripts/bootstrap_gpu.py", "--scope", "all"], cwd=ROOT, check=True
        )
        for profile, python in (("gpu-text", TEXT_PYTHON), ("gpu-vision", VISION_PYTHON)):
            subprocess.run(
                [str(python), "scripts/preflight.py", "--profile", profile], cwd=ROOT, check=True
            )
    subprocess.run(fla_install_command(sys.executable), cwd=ROOT, check=True)
    print(f"flash-linear-attention {FLA_VERSION} installed", flush=True)
    run_gate(args, output, export)


if __name__ == "__main__":
    main()
