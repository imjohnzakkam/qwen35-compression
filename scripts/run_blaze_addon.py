#!/usr/bin/env python3
"""The ladder as an add-on, and a 4/8-bit Blaze-Q, on one GPU: both servable on today's kernels.

1. Add-on: the published GPTQ, AutoRound and Glaze v2 exports of Qwen3.5-4B with only their BF16
   vocabulary table laddered (`blaze.addon`). Each must drift as its source did (G1 found the
   ladder costs 0.0002) at about 0.86 GB less.
2. Blaze-Q 4/8: the published GPTQ export, laddered, with the units the byte market picks (by
   G2's measured 4-bit prices) re-quantized to 8-bit g128 from BF16, at exactly a standard W4A16
   export's size. Every option runs on vLLM's Marlin kernels, so it serves today.

Measured, not rerun: G2's prices and BF16 drift score (uploaded), the sources' recorded drift
(records 2, 4, 6), and BF16's, GPTQ's, AutoRound's and Glaze v2's published panel answers.
Steps: setup (text environment, the `blaze-vllm` plugin, BF16 snapshot, traces, the three
published exports); a pilot (the GPTQ add-on, N answers per task); the add-ons, the market and
Blaze-Q 4/8; drift of all four; the text panel for Blaze-Q 4/8; speed of GPTQ W4A16, its add-on
and Blaze-Q 4/8; the report.

    python scripts/run_blaze_addon.py --prices inputs/g2_prices.json --bf16 inputs/g2_bf16.json \\
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

from qwen35_compression.blaze.gates import cost, drift_point, excess_loss  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON, TRACES_REPO, score_command  # noqa: E402
from qwen35_compression.feature1 import (  # noqa: E402
    build_text_eval_commands,
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
# name: (Hugging Face repo, recorded excess loss on BF16's 1,501 answers)
SOURCES = {
    "gptq": ("lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128", 0.0541),
    "autoround": ("lazybrick/Qwen3.5-4B-Kiln-AutoRound-W4A16-g128", 0.0417),
    "glaze": ("lazybrick/Qwen3.5-4B-Kiln-Glaze-W4A16", 0.0143),
}
PUBLISHED = {
    "bf16": "bf16",
    "gptq": "gptq_w4a16_g128",
    "autoround": "autoround_w4a16_g128",
    "glaze": "glaze_w4a16",
}
Q48 = "blaze_q48"
G2B_EXCESS = 0.0017
# The add-on keeps its source's quality if it drifts at most this much more (G1: 0.0002).
ADDON_MAX_EXTRA = 0.001
# On an A30: setup and three exports' download 0.25 h, pilot 0.1 h, builds 0.15 h, four drift
# scores 0.4 h, the panel 0.4 h, three speed runs 0.2 h.
ESTIMATED_HOURS = 1.5


def paths(outputs: Path) -> dict[str, Path]:
    found = {f"src_{name}": outputs / "published" / name for name in SOURCES}
    found.update({f"addon_{name}": outputs / "blaze" / f"addon_{name}" for name in SOURCES})
    found[Q48] = outputs / "blaze" / Q48
    return found


def addon_command(source: Path, output: Path, extra: list[str] | None = None) -> list[str]:
    return [
        sys.executable,
        "scripts/blaze_addon.py",
        "--census",
        str(CENSUS),
        "--source",
        str(source),
        "--output",
        str(output),
        *(extra or []),
    ]


def plan(
    output: Path,
    outputs: Path,
    snapshot: Path | str,
    prices: Path,
    limit: int | None,
) -> dict[str, list[str]]:
    where = paths(outputs)
    traces = output / "traces"
    if limit:
        pilot = outputs / "blaze" / "addon_gptq-pilot"
        return {
            "addon:gptq": addon_command(where["src_gptq"], pilot),
            "score:addon_gptq": score_command(
                pilot, "addon_gptq", traces, output / "addon_gptq.json", limit, Path(snapshot)
            ),
        }
    steps: dict[str, list[str]] = {}
    for name in SOURCES:
        steps[f"addon:{name}"] = addon_command(where[f"src_{name}"], where[f"addon_{name}"])
    steps["market"] = [
        sys.executable,
        "scripts/blaze_market.py",
        "--mode",
        "fill",
        "--prices",
        str(prices),
        "--lower",
        "4",
        "--upper",
        "8",
        "--output",
        str(output / "plan.json"),
    ]
    steps[f"build:{Q48}"] = addon_command(
        where["src_gptq"],
        where[Q48],
        ["--bf16", str(snapshot), "--plan", str(output / "plan.json")],
    )
    for name in [f"addon_{n}" for n in SOURCES] + [Q48]:
        steps[f"score:{name}"] = score_command(
            where[name], name, traces, output / f"{name}.json", None, Path(snapshot)
        )
    [(_, panel)] = build_text_eval_commands(
        where[Q48],
        load_benchmark_suite(PANEL_SUITE),
        output / "panel" / Q48,
        None,
        python_executable=TEXT_PYTHON,
    )
    steps[f"panel:{Q48}"] = panel
    for name in ("src_gptq", "addon_gptq", Q48):
        steps[f"speed:{name}"] = [
            str(TEXT_PYTHON),
            "scripts/blaze_speed.py",
            "--model",
            str(where[name]),
            "--name",
            name,
            "--output",
            str(output / f"speed_{name}.json"),
        ]
    return steps


def fetch_published(destination: Path) -> str:
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
    runs[Q48] = output / "panel" / Q48
    rows = {name: per_question(path, subset) for name, path in runs.items()}
    table: dict[str, Any] = {}
    for task, reference in rows["bf16"].items():
        entry: dict[str, Any] = {
            name: 100 * sum(rows[name][task].values()) / len(rows[name][task]) for name in runs
        }
        for other in ("bf16", "gptq", "glaze"):
            entry[f"q48_vs_{other}"] = paired_delta(rows[Q48][task], rows[other][task])
        table[task] = entry
    return table


def report(output: Path, outputs: Path, bf16: dict[str, Any]) -> dict[str, Any]:
    where = paths(outputs)

    def load(name: str) -> dict[str, Any]:
        return json.loads((output / f"{name}.json").read_text(encoding="utf-8"))

    def size(path: Path) -> int:
        return sum(p.stat().st_size for p in path.glob("*.safetensors"))

    addons = {}
    for name, (repo, recorded) in SOURCES.items():
        excess = excess_loss(load(f"addon_{name}"), bf16)
        addons[name] = {
            "repo": repo,
            "source_excess": recorded,
            "addon": {**drift_point(load(f"addon_{name}")), "excess_loss": excess},
            "extra_excess": excess - recorded,
            "source_bytes": size(where[f"src_{name}"]),
            "addon_bytes": size(where[f"addon_{name}"]),
            "free": excess - recorded <= ADDON_MAX_EXTRA,
        }
    market = load("plan")
    q48 = excess_loss(load(Q48), bf16)
    speeds = {name: load(f"speed_{name}") for name in ("src_gptq", "addon_gptq", Q48)}
    checks = {
        "ladder_free_on_every_export": all(a["free"] for a in addons.values()),
        "q48_no_larger": size(where[Q48]) <= size(where["src_gptq"]),
        "q48_beats_glaze_v2": q48 < SOURCES["glaze"][1],
    }
    return {
        "bf16": drift_point(bf16),
        "addons": addons,
        "q48": {
            **drift_point(load(Q48)),
            "excess_loss": q48,
            "bytes": size(where[Q48]),
            "units_at_8_bits": len(market["moved_units"]),
            "plan_bytes": market["plan_bytes"],
            "target_bytes": market["target_bytes"],
            "g2b_excess": G2B_EXCESS,
        },
        "panel": panel_table(output),
        "speed": speeds,
        "checks": checks,
    }


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import copy_published, fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    outputs = config.paths.outputs
    run = GateRun(
        output,
        "blaze_addon",
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
    where = paths(outputs)
    revisions = run.manifest.setdefault("sources", {})
    for name, (repo, _) in SOURCES.items():
        if not where[f"src_{name}"].exists():
            revisions[name] = run.step(
                f"fetch:{name}", lambda r=repo, p=where[f"src_{name}"]: copy_published(r, p)
            )
    if not limit:
        run.manifest["published"] = {
            "revision": run.step("published", lambda: fetch_published(output / "published"))
        }
    for name, command in plan(output, outputs, snapshot, args.prices.resolve(), limit).items():
        target = command[command.index("--output") + 1] if "--output" in command else None
        if (
            name.startswith(("addon:", "build:"))
            and target
            and Path(target, "config.json").exists()
        ):
            run.reuse(name)
            continue
        run.step(name, lambda command=command: run_logged(command, log, ROOT))
    if limit:
        run.finish()
        return run.manifest
    result = report(output, outputs, json.loads(args.bf16.read_text(encoding="utf-8")))
    write_json(output / "addon_report.json", result)
    run.finish(checks=result["checks"])
    print("checks=" + json.dumps(result["checks"]), flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/addon"))
    parser.add_argument("--prices", type=Path, required=True, help="G2's prices.json")
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
    outputs = load_config(args.config).paths.outputs

    if args.dry_run:
        print(
            json.dumps(
                {
                    "pilot": plan(pilot_output, outputs, "<4b>", args.prices, 10)
                    if args.pilot_limit
                    else None,
                    "full": plan(output, outputs, "<4b>", args.prices, None),
                    "estimate_on_a30": cost(ESTIMATED_HOURS),
                },
                indent=2,
            )
        )
        return
    for path in (args.prices, args.bf16):
        if not path.is_file():
            parser.error(f"missing {path}")
    if not args.skip_bootstrap:
        bootstrap_text(ROOT)
    uv = shutil.which("uv") or "uv"
    subprocess.run(
        [uv, "pip", "install", "--python", str(TEXT_PYTHON), "--no-deps", str(PLUGIN)],
        cwd=ROOT,
        check=True,
    )
    if args.pilot_limit:
        run_gate(args, pilot_output, args.pilot_limit)
    run_gate(args, output, None)


if __name__ == "__main__":
    main()
