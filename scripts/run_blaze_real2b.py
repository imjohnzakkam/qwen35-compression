#!/usr/bin/env python3
"""Blaze's real 2B export on one GPU: does vLLM serve it as its fake twin, and how fast is it?

G5 scored Blaze-Q on Qwen3.5-2B as a fake-quantized export. This run builds the real format
(compressed-tensors 8-bit body, tiered vocabulary table; `blaze.real`) and serves it with the
`blaze-vllm` plugin:

1. Setup: the text-evaluation environment with the plugin, and the 2B's BF16 snapshot.
2. Exports: Blaze-Q fake-quantized (as in G5) and real; GPTQ W4A16 g128 for the speed baseline.
3. Check (`blaze_real_check.py`): the real table equals the fake one, and vLLM's greedy tokens and
   log-probabilities agree. The run stops here if they do not.
4. Drift of BF16, fake and real Blaze-Q on BF16's G5 answers (uploaded with the run).
5. Speed and weight memory of BF16, W4A16 and real Blaze-Q (`blaze_speed.py`).
6. The report (blaze.gates.real_report).

    python scripts/run_blaze_real2b.py --traces inputs/g5_bf16_answers --dry-run
"""

from __future__ import annotations

import sys

for _name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(_name, None)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import _bootstrap  # noqa: E402, F401

from qwen35_compression.blaze.gates import cost, real_report  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON, score_command  # noqa: E402
from qwen35_compression.feature1 import run_logged  # noqa: E402
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CENSUS = Path("data/blaze/census.json")
PLUGIN = Path("plugins/blaze_vllm")
W4A16 = "gptq_w4a16_g128"
# On an A30: setup 0.3 h, W4A16 GPTQ 0.2 h, check and three drift scores 0.35 h, three speed
# runs 0.2 h.
ESTIMATED_HOURS = 1.05


def exports_of(outputs: Path) -> dict[str, Path]:
    return {
        "fake": outputs / "blaze" / "blaze_q_w8",
        "real": outputs / "blaze" / "blaze_q_w8_real",
        W4A16: outputs / W4A16,
    }


def plan(
    config_path: Path, output: Path, exports: dict[str, Path], snapshot: Path | str, traces: Path
) -> dict[str, list[str]]:
    uv = shutil.which("uv") or "uv"
    models = {"bf16": Path(snapshot), **exports}
    steps: dict[str, list[str]] = {
        "plugin": [uv, "pip", "install", "--python", str(TEXT_PYTHON), "--no-deps", str(PLUGIN)],
        "export:fake": [
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
            str(exports["fake"]),
        ],
        "export:real": [
            sys.executable,
            "scripts/blaze_real_export.py",
            "--model",
            str(snapshot),
            "--census",
            str(CENSUS),
            "--body-bits",
            "8",
            "--output",
            str(exports["real"]),
        ],
        "check": [
            str(TEXT_PYTHON),
            "scripts/blaze_real_check.py",
            "--real",
            str(exports["real"]),
            "--fake",
            str(exports["fake"]),
            "--output",
            str(output / "check.json"),
        ],
        f"export:{W4A16}": [
            sys.executable,
            "scripts/quantize.py",
            "--config",
            str(config_path),
            "--variant",
            W4A16,
        ],
    }
    for name in ("bf16", "fake", "real"):
        steps[f"score:{name}"] = score_command(
            models[name], name, traces, output / f"{name}.json", None, Path(snapshot)
        )
    for name in ("bf16", W4A16, "real"):
        steps[f"speed:{name}"] = [
            str(TEXT_PYTHON),
            "scripts/blaze_speed.py",
            "--model",
            str(models[name]),
            "--name",
            name,
            "--output",
            str(output / f"speed_{name}.json"),
        ]
    return steps


def run_gate(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    exports = exports_of(config.paths.outputs)
    run = GateRun(output, "blaze_real2b", args.code_revision, config_digest=config.digest)
    log = output / "run.log"
    snapshot, revision = run.step("download_bf16", lambda: download_model(config))
    run.manifest["base_model"] = {"id": config.model.id, "revision": revision}
    traces = args.traces.resolve()
    for name, command in plan(args.config, output, exports, snapshot, traces).items():
        run.step(name, lambda command=command: run_logged(command, log, ROOT))

    def load(name: str) -> dict[str, Any]:
        return json.loads((output / f"{name}.json").read_text(encoding="utf-8"))

    speeds = {name: load(f"speed_{name}") for name in ("bf16", W4A16, "real")}
    report = real_report(load("bf16"), load("fake"), load("real"), load("check")["summary"], speeds)
    report["export_bytes"] = {
        name: sum(p.stat().st_size for p in path.iterdir() if p.is_file())
        for name, path in exports.items()
    }
    write_json(output / "real_report.json", report)
    run.finish(checks=report["checks"], decision=report["decision"])
    print("decision=" + report["decision"], flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/blaze_2b.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/real2b"))
    parser.add_argument(
        "--traces", type=Path, required=True, help="BF16's G5 panel answers (lm-eval samples)"
    )
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    output = args.output.resolve()

    if args.dry_run:
        exports = exports_of(load_config(args.config).paths.outputs)
        print(
            json.dumps(
                {
                    "steps": plan(args.config, output, exports, "<bf16 snapshot>", args.traces),
                    "estimate_on_a30": cost(ESTIMATED_HOURS),
                },
                indent=2,
            )
        )
        return
    if not any(args.traces.rglob("samples_minerva_math500_*.jsonl")):
        parser.error(f"no BF16 MATH-500 answers under {args.traces}")
    if not args.skip_bootstrap:
        bootstrap_text(ROOT)
    run_gate(args, output)


if __name__ == "__main__":
    main()
