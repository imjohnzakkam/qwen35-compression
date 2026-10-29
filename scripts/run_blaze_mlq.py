#!/usr/bin/env python3
"""The real 4B Blaze-Q rebuilt on the multilingual census, and its drift in and outside English.

The real Blaze-Q (`run_blaze_q6.py`) ladders its vocabulary from a mostly English census, and
drifts 0.0051 on BF16's answers in 11 languages (Thai 0.028). The census with FineWeb-2 text in
30 languages, at the same 65,536 8-bit rows (equal table bytes), cut the ladder's own
multilingual cost 2.6x. This run rebuilds Blaze-Q with that census, its body unchanged (the same
G2b plan, GPTQ data and options), and scores it the way the English Blaze-Q was scored:

- BF16's 1,501 MATH-500 and MMLU-Pro answers, on vLLM's own 6-bit kernel;
- BF16's 914 HumanEval, MBPP and MGSM answers, with the 6-bit groups run as 8-bit, as before.

BF16's domain answers and their scores come from the earlier run (`--domain-traces`,
`--domain-bf16`), so nothing is generated again. Speed is not measured: the bytes and kernels
are the English Blaze-Q's.

    python scripts/run_blaze_mlq.py --plan inputs/g2b_plan.json --bf16 inputs/g2_bf16.json \
        --domain-traces inputs/domain_traces --domain-bf16 inputs/domain_bf16.json \
        --pilot-limit 10 --dry-run
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

from qwen35_compression.blaze.export import MANIFEST  # noqa: E402
from qwen35_compression.blaze.gates import cost, drift_point, excess_loss  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import score_command  # noqa: E402
from qwen35_compression.feature1 import FLA_VERSION, fla_install_command, run_logged  # noqa: E402
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPORT = "blaze_q6_ml"
# The English Blaze-Q, measured 2026-10-17 (run_blaze_q6.py): drift on BF16's 1,501 answers
# (native kernel), drift outside English (6-bit groups as 8-bit), and both files' bytes.
ENGLISH = {
    "excess_loss": 0.001689644339629759,
    "domains": {
        "code": 0.0020227261191333307,
        "ml": 0.00505165099075805,
        "ml_th": 0.02756993479172487,
    },
    "safetensors_bytes": 3_780_277_088,
}
GPTQ_BYTES = 3_780_662_272
# The multilingual ranking cost the ladder alone 0.00008 on English (0.00030 against 0.00022);
# Blaze-Q keeps its English drift if it moves no more than this.
ENGLISH_TOLERANCE = 0.0002
# On an A30: setup and downloads 0.25 h, pilot 0.12 h, GPTQ and export 0.25 h, drift 0.12 h,
# domain drift 0.08 h.
ESTIMATED_HOURS = 0.85


def plan(
    output: Path, exports: Path, snapshot: Path | str, plan_file: Path, limit: int | None
) -> dict[str, list[str]]:
    snapshot = Path(snapshot)
    export = exports / (EXPORT + ("-pilot" if limit else ""))
    command = q6.export_command(
        export, plan_file, limit, q6.CENSUS_MULTILINGUAL, q6.EQUAL_SIZE_COVER
    )
    score = score_command(
        export, EXPORT, output / "traces", output / f"{EXPORT}.json", limit, snapshot
    )
    domain_limit = q6.DOMAIN_PILOT if limit else None
    domain = q6.domain_score(export, EXPORT, output, domain_limit, snapshot)
    return {
        "export": command,
        "score": q6.with_env("native", score),
        "domain": q6.with_env("w8", domain),
    }


def domain_excess(result: dict, base: dict) -> dict[str, dict[str, float]]:
    return {
        group: {
            "excess_loss": scores["all"]["mean_nll"] - base["scores"][group]["all"]["mean_nll"],
            "flip_rate": scores["all"]["flip_rate"],
            "tokens": scores["all"]["tokens"],
        }
        for group, scores in result["scores"].items()
    }


def mlq_report(output: Path, bf16: dict, export: Path) -> dict[str, Any]:
    def load(name: str) -> dict[str, Any]:
        return json.loads((output / f"{name}.json").read_text(encoding="utf-8"))

    size = sum(p.stat().st_size for p in export.glob("*.safetensors"))
    english = {**drift_point(load(EXPORT)), "excess_loss": excess_loss(load(EXPORT), bf16)}
    domains = domain_excess(load(f"domain_{EXPORT}"), load("domain_bf16"))
    checks = {
        "equal_size": size <= GPTQ_BYTES,
        "english_kept": english["excess_loss"] <= ENGLISH["excess_loss"] + ENGLISH_TOLERANCE,
        "multilingual_better": domains["ml"]["excess_loss"] < ENGLISH["domains"]["ml"],
    }
    return {
        "safetensors_bytes": {EXPORT: size, "english_blaze_q": ENGLISH["safetensors_bytes"]},
        "english": english,
        "domains": domains,
        "english_blaze_q": ENGLISH,
        "checks": checks,
        "decision": "go: publish Blaze-Q with the multilingual ladder"
        if all(checks.values())
        else "report as measured",
    }


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    exports = config.paths.outputs / "blaze"
    run = GateRun(
        output,
        "blaze_mlq",
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
    # BF16's domain answers and, for the full run, their BF16 scores, from the earlier run.
    shutil.copytree(args.domain_traces, output / "domain_traces", dirs_exist_ok=True)
    if not limit:
        shutil.copyfile(args.domain_bf16, output / "domain_bf16.json")
    for name, command in plan(output, exports, snapshot, args.plan, limit).items():
        target = command[command.index("--output") + 1]
        if (name == "export" and (Path(target) / MANIFEST).exists()) or (
            target.endswith(".json") and Path(target).exists()
        ):
            run.reuse(name)
            continue
        run.step(name, lambda command=command: run_logged(command, log, ROOT))
    if limit:
        run.finish()
        return run.manifest
    bf16 = json.loads(args.bf16.read_text(encoding="utf-8"))
    report = mlq_report(output, bf16, exports / EXPORT)
    write_json(output / "mlq_report.json", report)
    run.finish(checks=report["checks"], decision=report["decision"])
    print("decision=" + report["decision"], flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/mlq"))
    parser.add_argument("--plan", type=Path, required=True, help="G2b's market plan")
    parser.add_argument("--bf16", type=Path, required=True, help="G2's BF16 drift score")
    parser.add_argument(
        "--domain-traces", type=Path, required=True, help="BF16's domain answers (Q6 run)"
    )
    parser.add_argument(
        "--domain-bf16", type=Path, required=True, help="BF16's own domain score (Q6 run)"
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
        exports = load_config(args.config).paths.outputs / "blaze"
        print(
            json.dumps(
                {
                    "pilot": plan(pilot_output, exports, "<4b>", args.plan, args.pilot_limit)
                    if args.pilot_limit
                    else None,
                    "full": plan(output, exports, "<4b>", args.plan, None),
                    "estimate_on_a30": cost(ESTIMATED_HOURS),
                },
                indent=2,
            )
        )
        return
    for path in (args.plan, args.bf16, args.domain_bf16, q6.CENSUS_MULTILINGUAL):
        if not path.is_file():
            parser.error(f"missing {path}")
    if not any(args.domain_traces.glob("domain_*.jsonl")):
        parser.error(f"no domain answers under {args.domain_traces}")
    if not args.skip_bootstrap:
        bootstrap_text(ROOT)
    uv = shutil.which("uv") or "uv"
    subprocess.run(
        [uv, "pip", "install", "--python", str(q6.TEXT_PYTHON), "--no-deps", str(q6.PLUGIN)],
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
