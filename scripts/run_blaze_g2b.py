#!/usr/bin/env python3
"""Blaze gate G2b on one GPU: the 4B at W4A16's size with an 8-bit vision tower, then G3.

G2's market had to push two FFN layers down to 4 bits to fit W4A16's size, and missed its mark.
G2b stores the vision tower at 8-bit g128, as Glaze v2 does, which frees enough for the whole body
at 6 bits and some units at 8; the market fills that room by G2's measured 4-bit prices scaled by
the bits rule (`blaze_market.py --mode fill`). Measured results are reused, not rerun: G2's prices,
G2's BF16 drift score, and BF16's and Glaze v2's published panel answers.

1. Setup: the text-evaluation environment and flash-linear-attention.
2. Market and export: the plan, then the ladder, the body GPTQ-rounded at each Linear's option,
   and the vision tower at 8 bits.
3. Drift of Blaze-Q against G2's BF16 score; the G2 mark (2.4x below Glaze v2, no larger).
4. If it passes, G3: the text panel against BF16's and Glaze v2's published answers, then MMMU
   (the vision tower changed), scored locally afterwards with the fixed extractor.
5. Last, the 2B real export served with one copy of its vocabulary table: check and speed.

With --pilot-limit, a pilot first builds a two-layer export and scores N answers per task with
it; the full run starts only if it passes.

    python scripts/run_blaze_g2b.py --prices inputs/g2_prices.json --bf16 inputs/g2_bf16.json \\
        --g5-answers inputs/g5_bf16_answers --pilot-limit 10 --dry-run
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

from qwen35_compression.blaze.export import MANIFEST  # noqa: E402
from qwen35_compression.blaze.gates import cost, g2_report, g3_report  # noqa: E402
from qwen35_compression.blaze.runs import (  # noqa: E402
    GateRun,
    bootstrap_text,
    vision_run_manifest,
)
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
TEXT_PANEL = Path("configs/evaluation/blaze_panel_text.yaml")
# The reasoning panel; its one vision task is MMMU, with the fixed extractor.
PANEL = Path("configs/evaluation/feature1_panel.yaml")
VISION_PYTHON = ROOT / ".venv-gpu-vision" / "bin" / "python"
EXPORT = "blaze_q_g2b"
CONFIG_2B = Path("configs/blaze_2b.yaml")
PLUGIN = Path("plugins/blaze_vllm")
PILOT_GPTQ = ["--blocks", "8", "--length", "512", "--layers", "0,1"]
# On an A30: setup 0.1 h, pilot 0.1 h, GPTQ and export 0.2 h, drift 0.12 h, text panel 0.5 h,
# vision setup and MMMU 0.5 h, the 2B single-copy check 0.25 h.
ESTIMATED_HOURS = 1.77


def plan(
    output: Path,
    exports: Path,
    snapshot: Path | str,
    prices: Path,
    limit: int | None,
) -> dict[str, list[str]]:
    export = exports / (EXPORT + ("-pilot" if limit else ""))
    return {
        "market": [
            sys.executable,
            "scripts/blaze_market.py",
            "--mode",
            "fill",
            "--prices",
            str(prices),
            "--lower",
            "6",
            "--upper",
            "8",
            "--vision-bits",
            "8",
            "--output",
            str(output / "plan.json"),
        ],
        "export": [
            sys.executable,
            "scripts/blaze_gptq.py",
            "--census",
            str(CENSUS),
            "--ladder",
            "--bits",
            "6",
            "--plan",
            str(output / "plan.json"),
            "--vision-bits",
            "8",
            "--output",
            str(export),
            *(PILOT_GPTQ if limit else []),
        ],
        f"score:{EXPORT}": score_command(
            export, EXPORT, output / "traces", output / f"{EXPORT}.json", limit, Path(snapshot)
        ),
    }


def panel_steps(output: Path, export: Path) -> dict[str, list[str]]:
    [(_, text)] = build_text_eval_commands(
        export,
        load_benchmark_suite(TEXT_PANEL),
        output / "panel" / EXPORT,
        None,
        python_executable=TEXT_PYTHON,
    )
    vision = build_vlm_eval_command(
        export,
        load_benchmark_suite(PANEL),
        output / "mmmu" / "vision",
        ROOT / "external/VLMEvalKit",
    )
    vision[0] = str(VISION_PYTHON)
    return {
        f"panel:{EXPORT}": text,
        "bootstrap_vision": [sys.executable, "scripts/bootstrap_gpu.py", "--scope", "all"],
        "preflight_vision": [str(VISION_PYTHON), "scripts/preflight.py", "--profile", "gpu-vision"],
        f"mmmu:{EXPORT}": vision,
    }


def single_copy_steps(output: Path, outputs_2b: Path, snapshot_2b: Path | str, answers: Path):
    uv = shutil.which("uv") or "uv"
    fake = outputs_2b / "blaze" / "blaze_q_w8"
    real = outputs_2b / "blaze" / "blaze_q_w8_real"
    single = output / "single_copy_2b"
    return {
        "plugin": [uv, "pip", "install", "--python", str(TEXT_PYTHON), "--no-deps", str(PLUGIN)],
        "export2b:fake": [
            sys.executable,
            "scripts/blaze_export.py",
            "--model",
            str(snapshot_2b),
            "--census",
            str(CENSUS),
            "--ladder",
            "--body-bits",
            "8",
            "--output",
            str(fake),
        ],
        "export2b:real": [
            sys.executable,
            "scripts/blaze_real_export.py",
            "--model",
            str(snapshot_2b),
            "--census",
            str(CENSUS),
            "--body-bits",
            "8",
            "--output",
            str(real),
        ],
        "check2b:single": [
            str(TEXT_PYTHON),
            "scripts/blaze_real_check.py",
            "--real",
            str(real),
            "--fake",
            str(fake),
            "--single-copy",
            "--output",
            str(single / "check.json"),
        ],
        "speed2b:single": [
            str(TEXT_PYTHON),
            "scripts/blaze_speed.py",
            "--model",
            str(real),
            "--name",
            "real_single_copy",
            "--single-copy",
            "--output",
            str(single / "speed.json"),
        ],
        "score2b:single": [
            *score_command(
                real, "real_single", answers, single / "drift.json", None, Path(snapshot_2b)
            ),
        ],
    }


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    exports = config.paths.outputs / "blaze"
    run = GateRun(
        output,
        "blaze_g2b",
        args.code_revision,
        config_digest=config.digest,
        limit=limit,
        research_result=limit is None,
        reused={"prices": str(args.prices), "bf16_drift": str(args.bf16)},
    )
    log = output / "run.log"

    def logged(command: list[str]):
        return lambda: run_logged(command, log, ROOT)

    snapshot, revision = run.step("download_bf16", lambda: download_model(config))
    run.manifest["base_model"] = {"id": config.model.id, "revision": revision}
    run.manifest["traces"] = {
        "revision": run.step("traces", lambda: fetch_traces(output / "traces"))
    }
    for name, command in plan(output, exports, snapshot, args.prices.resolve(), limit).items():
        export = Path(command[command.index("--output") + 1]) if name == "export" else None
        if export is not None and (export / MANIFEST).exists():
            run.reuse(name)
            continue
        run.step(name, logged(command))
    if limit:
        run.finish()
        return run.manifest

    market = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    bf16 = json.loads(args.bf16.read_text(encoding="utf-8"))
    blaze = json.loads((output / f"{EXPORT}.json").read_text(encoding="utf-8"))
    report = g2_report(bf16, blaze, market["plan_bytes"], market["target_bytes"])
    report["market"] = {k: v for k, v in market.items() if k != "linear_options"}
    write_json(output / "g2b_report.json", report)
    print("g2b decision=" + report["decision"], flush=True)
    fields: dict[str, Any] = {"checks": report["checks"], "decision": report["decision"]}
    run.manifest.update(fields)
    run.save()

    if all(report["checks"].values()):
        from run_blaze_g2 import fetch_published, panel_table

        export = exports / EXPORT
        steps = panel_steps(output, export)
        run.manifest["published"] = {
            "revision": run.step("published", lambda: fetch_published(output / "published"))
        }
        run.step(f"panel:{EXPORT}", logged(steps[f"panel:{EXPORT}"]))
        g3 = g3_report(panel_table(output, EXPORT))
        write_json(output / "g3_report.json", g3)
        fields.update(g3_checks=g3["checks"], g3_decision=g3["decision"])
        run.manifest.update(fields)
        print("g3 decision=" + g3["decision"], flush=True)
        run.step("bootstrap_vision", logged(steps["bootstrap_vision"]))
        run.step("preflight_vision", logged(steps["preflight_vision"]))
        run.step(f"mmmu:{EXPORT}", logged(steps[f"mmmu:{EXPORT}"]))
        # The record scripts/score_vision.py reads to score MMMU after download.
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

    if not args.skip_single_copy:
        config_2b = load_config(ROOT / CONFIG_2B)
        snapshot_2b, _ = run.step("download_2b", lambda: download_model(config_2b))
        for name, command in single_copy_steps(
            output, config_2b.paths.outputs, snapshot_2b, args.g5_answers.resolve()
        ).items():
            run.step(name, logged(command))
    run.finish(**fields)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/g2b"))
    parser.add_argument("--prices", type=Path, required=True, help="G2's prices.json")
    parser.add_argument("--bf16", type=Path, required=True, help="G2's BF16 drift score")
    parser.add_argument("--g5-answers", type=Path, required=True, help="BF16's 2B panel answers")
    parser.add_argument("--pilot-limit", type=int, help="Run small into <output>-pilot first")
    parser.add_argument("--skip-single-copy", action="store_true")
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
        config = load_config(args.config)
        exports = config.paths.outputs / "blaze"
        print(
            json.dumps(
                {
                    "pilot": plan(pilot_output, exports, "<4b>", args.prices, args.pilot_limit)
                    if args.pilot_limit
                    else None,
                    "full": plan(output, exports, "<4b>", args.prices, None),
                    "panel": panel_steps(output, exports / EXPORT),
                    "single_copy": single_copy_steps(
                        output, load_config(CONFIG_2B).paths.outputs, "<2b>", args.g5_answers
                    ),
                    "estimate_on_a30": cost(ESTIMATED_HOURS),
                },
                indent=2,
            )
        )
        return
    for path in (args.prices, args.bf16):
        if not path.is_file():
            parser.error(f"missing {path}")
    if not any(args.g5_answers.rglob("samples_minerva_math500_*.jsonl")):
        parser.error(f"no BF16 MATH-500 answers under {args.g5_answers}")
    if not args.skip_bootstrap:
        bootstrap_text(ROOT)
    subprocess.run(fla_install_command(sys.executable), cwd=ROOT, check=True)
    print(f"flash-linear-attention {FLA_VERSION} installed", flush=True)
    if args.pilot_limit:
        run_gate(args, pilot_output, args.pilot_limit)
    run_gate(args, output, None)


if __name__ == "__main__":
    main()
