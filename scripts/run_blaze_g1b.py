#!/usr/bin/env python3
"""Blaze gate G1b on one GPU: does a GPTQ-rounded 6-bit body, with the ladder, reach the target?

G1's 6-bit body was rounded to nearest and missed its mark narrowly (excess loss 0.00483 against
0.004). G1b rounds the same 6-bit g128 grid with GPTQ on WikiText-2's training text and holds it
to the first draft's GPTQ mark, 0.003.

1. Setup: the text-evaluation environment, flash-linear-attention, the BF16 snapshot and BF16's
   MATH-500 and MMLU-Pro answers (the drift traces).
2. The export: the ladder plus every body Linear GPTQ-rounded at 6-bit g128.
3. Drift scores of BF16 and the export on BF16's 1,501 answers.
4. The gate report (blaze.gates.g1b_report).

With --pilot-limit, a pilot runs first into <output>-pilot: GPTQ on layers 0-1 with 8 short
blocks, into its own export, and drift on N answers per task; the full gate runs only if it passes.

    python scripts/run_blaze_g1b.py --pilot-limit 10 --dry-run
"""

from __future__ import annotations

import sys

# The GPU image's base Python ships brotlicffi, which breaks httpx downloads (run_drift_study.py).
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
from qwen35_compression.blaze.gates import cost, g1b_report  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import score_command  # noqa: E402
from qwen35_compression.feature1 import FLA_VERSION, fla_install_command, run_logged  # noqa: E402
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CENSUS = Path("data/blaze/census.json")
EXPORT = "blaze_ladder_gptq6"
# G1's round-to-nearest 6-bit body, ladder included (run r_d95d713c, results/blaze/g1).
G1_RTN_SIX_BIT_EXCESS = 0.004831411646044781
PILOT_GPTQ = ["--blocks", "8", "--length", "512", "--layers", "0,1"]
# G1 on an A30: drift scoring took 5.5 minutes per model, setup about 15; GPTQ is estimated.
ESTIMATED_HOURS = {"pilot": 0.25, "full": 0.75}


def plan(
    output: Path, exports: Path, snapshot: Path | str, limit: int | None
) -> dict[str, list[str]]:
    """Every command after setup, by step name."""
    export = exports / (EXPORT + ("-pilot" if limit else ""))
    traces = output / "traces"
    return {
        "export": [
            sys.executable,
            "scripts/blaze_gptq.py",
            "--census",
            str(CENSUS),
            "--ladder",
            "--bits",
            "6",
            "--group",
            "128",
            "--output",
            str(export),
            *(PILOT_GPTQ if limit else []),
        ],
        "score:bf16": score_command(
            snapshot, "bf16", traces, output / "bf16.json", limit, Path(snapshot)
        ),
        f"score:{EXPORT}": score_command(
            export, EXPORT, traces, output / f"{EXPORT}.json", limit, Path(snapshot)
        ),
    }


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    exports = config.paths.outputs / "blaze"
    run = GateRun(
        output,
        "blaze_g1b",
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
    for name, command in plan(output, exports, snapshot, limit).items():
        export = Path(command[command.index("--output") + 1]) if name == "export" else None
        if export is not None and (export / MANIFEST).exists():
            run.reuse(name)
            continue
        run.step(name, lambda command=command: run_logged(command, log, ROOT))

    def load(name: str) -> dict[str, Any]:
        return json.loads((output / f"{name}.json").read_text(encoding="utf-8"))

    report = g1b_report(load("bf16"), load(EXPORT), G1_RTN_SIX_BIT_EXCESS if not limit else None)
    write_json(output / "g1b_report.json", report)
    run.finish(checks=report["checks"], decision=report["decision"])
    print("decision=" + report["decision"], flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/g1b"))
    parser.add_argument("--pilot-limit", type=int, help="Run small into <output>-pilot first")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-fla", action="store_true")
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
        exports = load_config(args.config).paths.outputs / "blaze"
        hours = ESTIMATED_HOURS["full"] + (ESTIMATED_HOURS["pilot"] if args.pilot_limit else 0)
        print(
            json.dumps(
                {
                    "pilot": plan(pilot_output, exports, "<bf16 snapshot>", args.pilot_limit)
                    if args.pilot_limit
                    else None,
                    "full": plan(output, exports, "<bf16 snapshot>", None),
                    "estimate_on_a30": cost(hours),
                },
                indent=2,
            )
        )
        return

    if not args.skip_bootstrap:
        bootstrap_text(ROOT)
    if not args.skip_fla:
        # GPTQ runs DeltaNet through transformers, which needs flash-linear-attention.
        subprocess.run(fla_install_command(sys.executable), cwd=ROOT, check=True)
        print(f"flash-linear-attention {FLA_VERSION} installed", flush=True)
    if args.pilot_limit:
        run_gate(args, pilot_output, args.pilot_limit)
        if args.pilot_only:
            return
    run_gate(args, output, None)


if __name__ == "__main__":
    main()
