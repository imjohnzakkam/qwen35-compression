#!/usr/bin/env python3
"""Blaze-S, corrected: a published 4-bit body with the ladder and an 8-bit vision tower.

Blaze-S (run_blaze_s.py) banked the vocabulary's and the vision tower's bytes for a smaller file,
but its body was this repo's own 4-bit GPTQ, barely better than rounding (excess loss 0.090
against the published GPTQ export's 0.054). Here the body is the published AutoRound W4A16
export's, copied as it is; only its BF16 vocabulary table is laddered (the multilingual census at
65,536 8-bit rows) and its BF16 vision Linears are stored at 8-bit g128, as Glaze v2 stores them.

1. Setup: the text environment with the plugin, BF16's drift traces, the AutoRound export.
2. The export (CPU work: rounding only), and a pilot of its drift on 10 answers per task.
3. Drift on BF16's 1,501 answers (AutoRound's own: 0.0417), and speed and weight memory.
4. With --mmmu: MMMU (val) on the vision environment, as G2b ran it (AutoRound published 66.9,
   BF16 69.6), scored locally afterwards with scripts/score_vision.py --run-dir <output>/mmmu.

    python scripts/run_blaze_s2.py --bf16 inputs/g2_bf16.json --pilot-limit 10 --dry-run
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

from qwen35_compression.blaze.gates import cost, drift_point, excess_loss  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text, vision_run_manifest  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON, score_command  # noqa: E402
from qwen35_compression.feature1 import (  # noqa: E402
    FLA_VERSION,
    build_vlm_eval_command,
    fla_install_command,
    load_benchmark_suite,
    run_logged,
)
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CENSUS = Path("data/blaze/census_multilingual.json")
# The multilingual census at the English ladder's 65,536 8-bit rows (equal table bytes).
EQUAL_SIZE_COVER = "1e-9"
PLUGIN = Path("plugins/blaze_vllm")
PANEL = Path("configs/evaluation/feature1_panel.yaml")
VISION_PYTHON = ROOT / ".venv-gpu-vision" / "bin" / "python"
EXPORT = "blaze_s2"
SOURCE = "autoround_w4a16_g128"
SOURCE_REPO = "lazybrick/Qwen3.5-4B-Kiln-AutoRound-W4A16-g128"
# Recorded on BF16's 1,501 answers and in the full suite (docs/experiments/04).
RECORDED = {
    "autoround_excess": 0.0417,
    "gptq_excess": 0.0541,
    "autoround_mmmu": 66.9,
    "bf16_mmmu": 69.6,
    "gptq_w4a16_bytes": 3_780_662_272,
    "blaze_s_bytes": 2_598_007_568,
    "blaze_s_excess": 0.0899,
    "gptq_single_tok_s": 151,
    "gptq_batched_tok_s": 3128,
}
# The corrected Blaze-S keeps its source's quality if it drifts at most this much more.
MAX_EXTRA = 0.001
# On an A30: setup and downloads 0.25 h, the export 0.05 h, pilot 0.05 h, drift 0.12 h, speed
# 0.08 h; with --mmmu, vision setup 0.15 h and MMMU 0.5 h.
ESTIMATED_HOURS = {"text": 0.55, "mmmu": 1.2}


def addon_command(source: Path, export: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/blaze_addon.py",
        "--census",
        str(CENSUS),
        "--cover",
        EQUAL_SIZE_COVER,
        "--vision-bits",
        "8",
        "--source",
        str(source),
        "--output",
        str(export),
    ]


def plan(
    output: Path, outputs: Path, snapshot: Path | str, limit: int | None, mmmu: bool
) -> dict[str, list[str]]:
    export = outputs / "blaze" / EXPORT
    steps = {
        "export": addon_command(outputs / SOURCE, export),
        f"score:{EXPORT}": score_command(
            export, EXPORT, output / "traces", output / f"{EXPORT}.json", limit, Path(snapshot)
        ),
    }
    if limit:
        return steps
    steps[f"speed:{EXPORT}"] = [
        str(TEXT_PYTHON),
        "scripts/blaze_speed.py",
        "--model",
        str(export),
        "--name",
        EXPORT,
        "--output",
        str(output / f"speed_{EXPORT}.json"),
    ]
    if mmmu:
        uv = shutil.which("uv") or "uv"
        vision = build_vlm_eval_command(
            export,
            load_benchmark_suite(PANEL),
            output / "mmmu" / "vision",
            ROOT / "external/VLMEvalKit",
        )
        vision[0] = str(VISION_PYTHON)
        steps["bootstrap_vision"] = [sys.executable, "scripts/bootstrap_gpu.py", "--scope", "all"]
        steps["plugin_vision"] = [
            uv, "pip", "install", "--python", str(VISION_PYTHON), "--no-deps", str(PLUGIN)
        ]  # fmt: skip
        steps["preflight_vision"] = [
            str(VISION_PYTHON), "scripts/preflight.py", "--profile", "gpu-vision"
        ]  # fmt: skip
        steps[f"mmmu:{EXPORT}"] = vision
    return steps


def s2_report(output: Path, bf16: dict, export: Path, source: Path) -> dict[str, Any]:
    score = json.loads((output / f"{EXPORT}.json").read_text(encoding="utf-8"))
    speed = json.loads((output / f"speed_{EXPORT}.json").read_text(encoding="utf-8"))
    excess = excess_loss(score, bf16)
    sizes = {
        name: sum(p.stat().st_size for p in path.glob("*.safetensors"))
        for name, path in ((EXPORT, export), (SOURCE, source))
    }
    manifest = json.loads((export / "blaze_addon.json").read_text(encoding="utf-8"))
    checks = {
        "smaller_than_blaze_s": sizes[EXPORT] <= RECORDED["blaze_s_bytes"] * 1.01,
        "keeps_autoround_quality": excess <= RECORDED["autoround_excess"] + MAX_EXTRA,
        "faster_than_gptq": speed["single_stream"]["tokens_per_second"]
        >= RECORDED["gptq_single_tok_s"],
    }
    return {
        "bf16": drift_point(bf16),
        EXPORT: {**drift_point(score), "excess_loss": excess},
        "safetensors_bytes": sizes,
        "vision": manifest.get("vision"),
        "speed": speed,
        "recorded": RECORDED,
        "checks": checks,
        "decision": "go: the corrected Blaze-S is small, fast and keeps AutoRound's quality"
        if all(checks.values())
        else "report as measured",
    }


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import copy_published, fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    outputs = config.paths.outputs
    run = GateRun(
        output,
        "blaze_s2",
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
    source = outputs / SOURCE
    if not source.exists():
        run.manifest["source_revision"] = run.step(
            "fetch_source", lambda: copy_published(SOURCE_REPO, source)
        )
    export = outputs / "blaze" / EXPORT
    for name, command in plan(output, outputs, snapshot, limit, args.mmmu).items():
        if name == "export" and (export / "blaze_addon.json").exists():
            run.reuse(name)
            continue
        if name.startswith(("bootstrap_vision", "plugin_vision", "preflight_vision", "mmmu:")):
            # The vision check runs last; its failure leaves the text results standing.
            try:
                run.step(name, lambda command=command: run_logged(command, log, ROOT))
            except subprocess.CalledProcessError:
                run.manifest["status"] = "running"
                run.save()
                break
            if name.startswith("mmmu:"):
                write_json(
                    output / "mmmu" / "run_manifest.json",
                    vision_run_manifest(
                        EXPORT,
                        config.model.id,
                        revision,
                        args.code_revision,
                        config.digest,
                        PANEL,
                        export,
                        None,
                    ),
                )
            continue
        run.step(name, lambda command=command: run_logged(command, log, ROOT))
    if limit:
        run.finish()
        return run.manifest
    bf16 = json.loads(args.bf16.read_text(encoding="utf-8"))
    report = s2_report(output, bf16, export, source)
    write_json(output / "s2_report.json", report)
    run.finish(checks=report["checks"], decision=report["decision"])
    print("decision=" + report["decision"], flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/s2"))
    parser.add_argument("--bf16", type=Path, required=True, help="G2's BF16 drift score")
    parser.add_argument("--pilot-limit", type=int, help="Run small into <output>-pilot first")
    parser.add_argument("--mmmu", action="store_true", help="Also run MMMU (val), as G2b did")
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
                    "pilot": plan(pilot_output, outputs, "<4b>", 10, args.mmmu)
                    if args.pilot_limit
                    else None,
                    "full": plan(output, outputs, "<4b>", None, args.mmmu),
                    "estimate_on_a30": cost(ESTIMATED_HOURS["mmmu" if args.mmmu else "text"]),
                },
                indent=2,
            )
        )
        return
    if not args.bf16.is_file():
        parser.error(f"missing {args.bf16}")
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
