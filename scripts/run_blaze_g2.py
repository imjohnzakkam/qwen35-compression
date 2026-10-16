#!/usr/bin/env python3
"""Blaze gates G2 and G3 on one GPU: the 4B's equal-size byte market, then its panel.

G1b found a GPTQ-rounded 6-bit body with the ladder at excess loss 0.0027, but an all-6-bit body
is 0.4% over a standard W4A16 export's size. G2 lets the cheapest units step down to 4 bits:

1. Setup: the text-evaluation environment, flash-linear-attention, the BF16 snapshot and BF16's
   MATH-500 and MMLU-Pro answers (the drift traces).
2. Prices: each layer's token mixer and FFN alone at 4 and 6 bits, KL to BF16 on WikiText-2.
3. Market: the cheapest set of units at 4 bits that brings the export to W4A16's size.
4. Export: the ladder, and the body GPTQ-rounded at each Linear's chosen option.
5. Drift of BF16 and Blaze-Q; G2 passes at 2.4x below Glaze v2's excess loss (0.0143), no larger.
6. G3, if G2 passes: Blaze-Q on the text panel, against BF16's and Glaze v2's published answers;
   it passes with no task more than a point below BF16.

With --pilot-limit, a pilot runs first into <output>-pilot (two layers priced and GPTQ-rounded,
drift on N answers per task, no panel); the full gate runs only if it passes.

    python scripts/run_blaze_g2.py --pilot-limit 10 --dry-run
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
from qwen35_compression.blaze.gates import cost, g2_report, g3_report  # noqa: E402
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
EXPORT = "blaze_q_g2"
# Published instruct-track answers (lazybrick/kiln-evals) the panel is compared with.
PUBLISHED = {"bf16": "bf16", "glaze_v2": "glaze_w4a16"}
PILOT_PRICES = ["--blocks", "2", "--length", "512", "--layers", "0,1"]
PILOT_GPTQ = ["--blocks", "8", "--length", "512", "--layers", "0,1"]
# On an A30: setup 0.3 h, prices 0.25 h, GPTQ and export 0.25 h, two drift scores 0.2 h, the
# panel 0.5 h; the pilot 0.2 h.
ESTIMATED_HOURS = {"pilot": 0.2, "full": 1.5}


def plan(
    output: Path, exports: Path, snapshot: Path | str, limit: int | None, panel: bool
) -> dict[str, list[str]]:
    export = exports / (EXPORT + ("-pilot" if limit else ""))
    traces = output / "traces"
    steps: dict[str, list[str]] = {
        "prices": [
            sys.executable,
            "scripts/blaze_prices.py",
            "--bits",
            "4,6",
            "--output",
            str(output / "prices.json"),
            *(PILOT_PRICES if limit else []),
        ],
        "market": [
            sys.executable,
            "scripts/blaze_market.py",
            "--prices",
            str(output / "prices.json"),
            "--output",
            str(output / "plan.json"),
            *(["--allow-over"] if limit else []),
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
    if panel and not limit:
        [(_, command)] = build_text_eval_commands(
            export,
            load_benchmark_suite(PANEL_SUITE),
            output / "panel" / EXPORT,
            None,
            python_executable=TEXT_PYTHON,
        )
        steps[f"panel:{EXPORT}"] = command
    return steps


def fetch_published(destination: Path) -> str:
    """BF16's and Glaze v2's published panel answers, renamed as lm-eval sample files."""
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


def panel_table(output: Path, name: str = EXPORT) -> dict[str, Any]:
    from panel_scores import paired_delta, per_question

    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    runs = {
        "bf16": output / "published" / "bf16",
        "glaze_v2": output / "published" / "glaze_v2",
        "blaze_q": output / "panel" / name,
    }
    rows = {name: per_question(path, subset) for name, path in runs.items()}
    table: dict[str, Any] = {}
    for task, reference in rows["bf16"].items():
        entry: dict[str, Any] = {
            name: 100 * sum(rows[name][task].values()) / len(rows[name][task]) for name in runs
        }
        entry["blaze_q_vs_bf16"] = paired_delta(rows["blaze_q"][task], reference)
        entry["blaze_q_vs_glaze_v2"] = paired_delta(rows["blaze_q"][task], rows["glaze_v2"][task])
        table[task] = entry
    return table


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    exports = config.paths.outputs / "blaze"
    run = GateRun(
        output,
        "blaze_g2",
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
    steps = plan(output, exports, snapshot, limit, panel=not args.skip_panel)
    panel_step = steps.pop(f"panel:{EXPORT}", None)
    for name, command in steps.items():
        export = Path(command[command.index("--output") + 1]) if name == "export" else None
        if export is not None and (export / MANIFEST).exists():
            run.reuse(name)
            continue
        run.step(name, lambda command=command: run_logged(command, log, ROOT))

    def load(name: str) -> dict[str, Any]:
        return json.loads((output / f"{name}.json").read_text(encoding="utf-8"))

    market = load("plan")
    report = g2_report(load("bf16"), load(EXPORT), market["plan_bytes"], market["target_bytes"])
    report["market"] = {k: v for k, v in market.items() if k != "linear_options"}
    write_json(output / "g2_report.json", report)
    print("g2 decision=" + report["decision"], flush=True)
    fields: dict[str, Any] = {"checks": report["checks"], "decision": report["decision"]}
    if panel_step is not None and all(report["checks"].values()):
        run.manifest["published"] = {
            "revision": run.step("published", lambda: fetch_published(output / "published"))
        }
        run.step(f"panel:{EXPORT}", lambda: run_logged(panel_step, log, ROOT))
        g3 = g3_report(panel_table(output))
        write_json(output / "g3_report.json", g3)
        fields["g3_checks"] = g3["checks"]
        fields["g3_decision"] = g3["decision"]
        print("g3 decision=" + g3["decision"], flush=True)
    run.finish(**fields)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/g2"))
    parser.add_argument("--pilot-limit", type=int, help="Run small into <output>-pilot first")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--skip-panel", action="store_true", help="G2 only, no G3")
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
                    "pilot": plan(pilot_output, exports, "<bf16 snapshot>", args.pilot_limit, False)
                    if args.pilot_limit
                    else None,
                    "full": plan(output, exports, "<bf16 snapshot>", None, not args.skip_panel),
                    "estimate_on_a30": cost(hours),
                },
                indent=2,
            )
        )
        return

    if not args.skip_bootstrap:
        bootstrap_text(ROOT)
    if not args.skip_fla:
        subprocess.run(fla_install_command(sys.executable), cwd=ROOT, check=True)
        print(f"flash-linear-attention {FLA_VERSION} installed", flush=True)
    if args.pilot_limit:
        run_gate(args, pilot_output, args.pilot_limit)
        if args.pilot_only:
            return
    run_gate(args, output, None)


if __name__ == "__main__":
    main()
