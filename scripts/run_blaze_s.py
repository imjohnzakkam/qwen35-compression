#!/usr/bin/env python3
"""Blaze-S on one GPU: the 4B with a 4-bit body, the ladder and an 8-bit vision tower, for real.

Blaze-Q spends the vocabulary's freed bytes on the body; Blaze-S keeps a standard 4-bit body
(GPTQ, group 128) and banks them, for a smaller file and fewer bytes read per token. Its body
runs on vLLM's 4-bit Marlin kernels, so it is built directly in the real format (`blaze.real`,
served by the `blaze-vllm` plugin) and every number below is the deployable checkpoint's.

1. Setup: the text environment with the plugin, flash-linear-attention, the BF16 snapshot, BF16's
   drift traces, and the published GPTQ W4A16 g128 export (the comparison).
2. Pilot: a two-layer GPTQ export in the real format, scored on N answers per task.
3. The export, its drift (against G2's BF16 score), and the text panel against BF16's, GPTQ's and
   AutoRound's published answers to the same questions.
4. Speed and weight memory of BF16, GPTQ W4A16 and Blaze-S; then Blaze's 6-bit kernel benchmark.
5. The report: Blaze-S passes if it is smaller than GPTQ W4A16, drifts no more, and decodes a
   single stream at least as fast.

    python scripts/run_blaze_s.py --bf16 inputs/g2_bf16.json --pilot-limit 10 --dry-run
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
from qwen35_compression.blaze.gates import cost, drift_point, excess_loss  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON, TRACES_REPO, score_command  # noqa: E402
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
CENSUS = Path("data/blaze/census.json")
PANEL_SUITE = Path("configs/evaluation/blaze_panel_text.yaml")
SUBSET = ROOT / "configs" / "evaluation" / "feature1_thinking_samples.json"
PLUGIN = Path("plugins/blaze_vllm")
EXPORT = "blaze_s"
GPTQ = "gptq_w4a16_g128"
GPTQ_REPO = "lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128"
# Recorded drift of the published 4-bit exports (docs/experiments/03, 04).
RECORDED_EXCESS = {"gptq_w4a16_g128": 0.0541, "autoround_w4a16_g128": 0.0417}
PUBLISHED = {"bf16": "bf16", "gptq": "gptq_w4a16_g128", "autoround": "autoround_w4a16_g128"}
PILOT_GPTQ = ["--blocks", "8", "--length", "512", "--layers", "0,1"]
# On an A30: setup and downloads 0.25 h, pilot 0.15 h, GPTQ and export 0.2 h, drift 0.12 h, the
# panel at 4-bit speed 0.4 h, three speed runs 0.2 h, the kernel benchmark 0.1 h.
ESTIMATED_HOURS = 1.42


def export_command(export: Path, limit: int | None) -> list[str]:
    return [
        sys.executable,
        "scripts/blaze_gptq.py",
        "--census",
        str(CENSUS),
        "--ladder",
        "--bits",
        "4",
        "--vision-bits",
        "8",
        "--real",
        "--output",
        str(export),
        *(PILOT_GPTQ if limit else []),
    ]


def plan(
    output: Path, exports: Path, outputs: Path, snapshot: Path | str, limit: int | None
) -> dict[str, list[str]]:
    export = exports / (EXPORT + ("-pilot" if limit else ""))
    traces = output / "traces"
    steps = {
        "export": export_command(export, limit),
        f"score:{EXPORT}": score_command(
            export, EXPORT, traces, output / f"{EXPORT}.json", limit, Path(snapshot)
        ),
    }
    if limit:
        return steps
    [(_, panel)] = build_text_eval_commands(
        export,
        load_benchmark_suite(PANEL_SUITE),
        output / "panel" / EXPORT,
        None,
        python_executable=TEXT_PYTHON,
    )
    steps[f"panel:{EXPORT}"] = panel
    models = {"bf16": Path(snapshot), GPTQ: outputs / GPTQ, EXPORT: export}
    for name, model in models.items():
        steps[f"speed:{name}"] = [
            str(TEXT_PYTHON),
            "scripts/blaze_speed.py",
            "--model",
            str(model),
            "--name",
            name,
            "--output",
            str(output / f"speed_{name}.json"),
        ]
    steps["kernel"] = [
        str(TEXT_PYTHON),
        "scripts/blaze_kernel_bench.py",
        "--output",
        str(output / "kernel.json"),
    ]
    return steps


def fetch_published(destination: Path) -> str:
    """BF16's, GPTQ's and AutoRound's published panel answers, as lm-eval sample files."""
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().dataset_info(TRACES_REPO).sha
    folder = Path(
        snapshot_download(
            TRACES_REPO,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[
                f"{prefix}/text/samples/{task}"
                for prefix in PUBLISHED.values()
                for task in ("minerva_math500.jsonl", "ifeval.jsonl", "mmlu_pro_*.jsonl")
            ],
        )
    )
    for name, prefix in PUBLISHED.items():
        target = destination / name
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted((folder / prefix / "text" / "samples").glob("*.jsonl")):
            (target / f"samples_{path.stem}_published.jsonl").write_bytes(path.read_bytes())
    return revision


def panel_table(output: Path) -> dict[str, Any]:
    from panel_scores import paired_delta, per_question

    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    runs = {name: output / "published" / name for name in PUBLISHED}
    runs["blaze_s"] = output / "panel" / EXPORT
    rows = {name: per_question(path, subset) for name, path in runs.items()}
    table: dict[str, Any] = {}
    for task, reference in rows["bf16"].items():
        entry: dict[str, Any] = {
            name: 100 * sum(rows[name][task].values()) / len(rows[name][task]) for name in runs
        }
        entry["blaze_s_vs_bf16"] = paired_delta(rows["blaze_s"][task], reference)
        entry["blaze_s_vs_gptq"] = paired_delta(rows["blaze_s"][task], rows["gptq"][task])
        table[task] = entry
    return table


def s_report(output: Path, bf16: dict, exports: dict[str, Path]) -> dict[str, Any]:
    def load(name: str) -> dict[str, Any]:
        return json.loads((output / f"{name}.json").read_text(encoding="utf-8"))

    blaze = load(EXPORT)
    excess = excess_loss(blaze, bf16)
    speeds = {name: load(f"speed_{name}") for name in ("bf16", GPTQ, EXPORT)}
    sizes = {
        name: sum(p.stat().st_size for p in path.glob("*.safetensors"))
        for name, path in exports.items()
    }
    checks = {
        "smaller": sizes[EXPORT] < sizes[GPTQ],
        "drifts_no_more": excess <= RECORDED_EXCESS[GPTQ],
        "single_stream_no_slower": speeds[EXPORT]["single_stream"]["tokens_per_second"]
        >= speeds[GPTQ]["single_stream"]["tokens_per_second"],
    }
    return {
        "bf16": drift_point(bf16),
        "blaze_s": {**drift_point(blaze), "excess_loss": excess},
        "recorded_excess": RECORDED_EXCESS,
        "safetensors_bytes": sizes,
        "speed": speeds,
        "panel": panel_table(output),
        "checks": checks,
        "decision": "go: Blaze-S is smaller, as good and as fast"
        if all(checks.values())
        else "report as measured",
    }


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import copy_published, fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    exports = config.paths.outputs / "blaze"
    run = GateRun(
        output,
        "blaze_s",
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
    if not limit:
        gptq = config.paths.outputs / GPTQ
        if not gptq.exists():
            run.manifest["gptq_revision"] = run.step(
                "fetch_gptq", lambda: copy_published(GPTQ_REPO, gptq)
            )
        run.manifest["published"] = {
            "revision": run.step("published", lambda: fetch_published(output / "published"))
        }
    steps = plan(output, exports, config.paths.outputs, snapshot, limit)
    for name, command in steps.items():
        if name == "export" and (Path(command[command.index("--output") + 1]) / MANIFEST).exists():
            run.reuse(name)
            continue
        run.step(name, lambda command=command: run_logged(command, log, ROOT))
    if limit:
        run.finish()
        return run.manifest
    bf16 = json.loads(args.bf16.read_text(encoding="utf-8"))
    report = s_report(
        output,
        bf16,
        {"bf16": Path(snapshot), GPTQ: config.paths.outputs / GPTQ, EXPORT: exports / EXPORT},
    )
    write_json(output / "s_report.json", report)
    run.finish(checks=report["checks"], decision=report["decision"])
    print("decision=" + report["decision"], flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/s"))
    parser.add_argument("--bf16", type=Path, required=True, help="G2's BF16 drift score")
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
        config = load_config(args.config)
        exports = config.paths.outputs / "blaze"
        print(
            json.dumps(
                {
                    "pilot": plan(pilot_output, exports, config.paths.outputs, "<4b>", 10)
                    if args.pilot_limit
                    else None,
                    "full": plan(output, exports, config.paths.outputs, "<4b>", None),
                    "estimate_on_a30": cost(ESTIMATED_HOURS),
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
