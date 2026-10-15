#!/usr/bin/env python3
"""Blaze gate G1 on one GPU: is the vocabulary ladder free, and which body menu is worth building.

In order, stopping at the first failure:
1. Setup: the text-evaluation environment, flash-linear-attention, the BF16 snapshot and BF16's
   MATH-500 and MMLU-Pro answers (the drift traces).
2. Three fake-quantized exports, built once: the ladder alone; the ladder with every body Linear
   at 6-bit round-to-nearest; and the ladder with the body at 4-bit, the reference for the
   4x-per-bit rule.
3. Drift scores of BF16 and the three exports on BF16's 1,501 answers.
4. Prices: each FFN layer alone at 4 bits, as KL to BF16 on WikiText-2's training text.
5. The gate report (blaze.gates.g1_report): three checks and a decision for G2.

With --pilot-limit, every step first runs small (drift on N answers per task, prices on two
layers and two short blocks) into <output>-pilot, and the full gate runs only if that passes.

    python scripts/run_blaze_g1.py --pilot-limit 10 --dry-run
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
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import _bootstrap  # noqa: E402, F401

from qwen35_compression.blaze.gates import cost, g1_report  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON, score_command  # noqa: E402
from qwen35_compression.feature1 import FLA_VERSION, fla_install_command, run_logged  # noqa: E402
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CENSUS = Path("data/blaze/census.json")
# Name -> blaze_export.py arguments. Every export carries the ladder.
EXPORTS = {
    "blaze_ladder": ["--ladder"],
    "blaze_ladder_w6": ["--ladder", "--body-bits", "6"],
    "blaze_ladder_w4": ["--ladder", "--body-bits", "4"],
}
SCORED = ("bf16", *EXPORTS)
PILOT_PRICES = ["--blocks", "2", "--length", "512", "--layers", "0,1"]
# Measured times on an A30 from the drift study and Glaze v2's runs, plus the exports.
ESTIMATED_HOURS = {"pilot": 0.3, "full": 1.45}


def plan(
    output: Path, exports: Path, snapshot: Path | str, limit: int | None
) -> dict[str, list[str]]:
    """Every command after setup, by step name. Exports are shared by the pilot and full run."""
    traces = output / "traces"
    steps: dict[str, list[str]] = {}
    for name, flags in EXPORTS.items():
        steps[f"export:{name}"] = [
            sys.executable,
            "scripts/blaze_export.py",
            "--model",
            str(snapshot),
            "--census",
            str(CENSUS),
            "--output",
            str(exports / name),
            *flags,
        ]
    for name in SCORED:
        model = snapshot if name == "bf16" else exports / name
        steps[f"score:{name}"] = score_command(
            model, name, traces, output / f"{name}.json", limit, Path(snapshot)
        )
    steps["prices"] = [
        sys.executable,
        "scripts/blaze_sensitivity.py",
        "--output",
        str(output / "ffn_prices.json"),
        *(PILOT_PRICES if limit else []),
    ]
    return steps


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "run.log"
    exports = config.paths.outputs / "blaze"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "blaze_g1",
        "code_revision": args.code_revision,
        "config_digest": config.digest,
        "limit": limit,
        "research_result": limit is None,
        "status": "running",
        "steps": {},
    }

    def save() -> None:
        write_json(output / "run_manifest.json", manifest)

    def step(name: str, action: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        manifest["steps"][name] = {"status": "running"}
        save()
        try:
            value = action()
        except BaseException as error:
            manifest["steps"][name] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.perf_counter() - started),
            }
            manifest["status"] = "failed"
            save()
            raise
        manifest["steps"][name] = {
            "status": "passed",
            "seconds": round(time.perf_counter() - started),
        }
        save()
        return value

    snapshot, revision = step("download_bf16", lambda: download_model(config))
    manifest["base_model"] = {"id": config.model.id, "revision": revision}
    manifest["traces"] = {"revision": step("traces", lambda: fetch_traces(output / "traces"))}
    steps = plan(output, exports, snapshot, limit)
    for name, command in steps.items():
        if (
            name.startswith("export:")
            and (exports / name.split(":", 1)[1] / "blaze_manifest.json").exists()
        ):
            manifest["steps"][name] = {"status": "reused"}
            continue
        step(name, lambda command=command: run_logged(command, log, ROOT))

    def load(name: str) -> dict[str, Any]:
        return json.loads((output / f"{name}.json").read_text(encoding="utf-8"))

    prices = load("ffn_prices")
    report = g1_report(
        load("bf16"),
        load("blaze_ladder"),
        load("blaze_ladder_w6"),
        load("blaze_ladder_w4"),
        prices["concentration"],
    )
    write_json(output / "g1_report.json", report)
    manifest["checks"] = report["checks"]
    manifest["decision"] = report["decision"]
    manifest["status"] = "passed"
    save()
    print("decision=" + report["decision"], flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/g1"))
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
        config = load_config(args.config)
        exports = config.paths.outputs / "blaze"
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
        subprocess.run(
            [sys.executable, "scripts/bootstrap_gpu.py", "--scope", "text"], cwd=ROOT, check=True
        )
        subprocess.run(
            [str(TEXT_PYTHON), "scripts/preflight.py", "--profile", "gpu-text"],
            cwd=ROOT,
            check=True,
        )
    if not args.skip_fla:
        # The prices run DeltaNet through transformers, which needs flash-linear-attention.
        subprocess.run(fla_install_command(sys.executable), cwd=ROOT, check=True)
        print(f"flash-linear-attention {FLA_VERSION} installed", flush=True)
    if args.pilot_limit:
        pilot = run_gate(args, pilot_output, args.pilot_limit)
        if pilot["status"] != "passed":
            raise SystemExit("pilot failed; the full gate was not started")
        if args.pilot_only:
            return
    run_gate(args, output, None)


if __name__ == "__main__":
    main()
