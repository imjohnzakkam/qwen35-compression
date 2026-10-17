#!/usr/bin/env python3
"""G2b's Blaze-Q as a real 4B checkpoint, and the vocabulary ladder outside English, on one GPU.

G2b measured Blaze-Q (6-bit GPTQ body, 23 units at 8 bits, the ladder, an 8-bit vision tower) as
a fake-quantized export. Its 6-bit groups are stored here densely, 6 bits a weight, so the file
is W4A16's size, and served two ways with the same codes and scales:

- native: vLLM 0.29's own kernel for 6-bit groups (Humming, chosen when Marlin cannot);
- w8: the `blaze-vllm` plugin repacks the 6-bit groups for the 8-bit Marlin kernel as they
  load (`BLAZE_W6_AS_W8=1`), so the GPU holds 8 bits a weight for them.

Both must drift as G2b did (the same model), and both are timed. Then the ladder's census,
mostly English, is tested where it is weakest: BF16 answers HumanEval, MBPP and MGSM in 11
languages, and BF16 with the ladder (as built, and from a census with FineWeb-2 text in 30
languages added), GPTQ W4A16 and Blaze-Q are scored along those answers.

1. Setup: the text environment with the plugin, flash-linear-attention, the BF16 snapshot,
   BF16's drift traces and the published GPTQ W4A16 export.
2. Pilot: a two-layer GPTQ export, both serving modes and the domain steps on a few answers.
3. The export and its size; drift and speed in both modes.
4. The domain answers, two ladder-only exports, and five domain drift scores.

    python scripts/run_blaze_q6.py --plan inputs/g2b_plan.json --bf16 inputs/g2_bf16.json \
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

from qwen35_compression.blaze.export import MANIFEST  # noqa: E402
from qwen35_compression.blaze.gates import cost, drift_point, excess_loss  # noqa: E402
from qwen35_compression.blaze.runs import GateRun, bootstrap_text  # noqa: E402
from qwen35_compression.config import load_config  # noqa: E402
from qwen35_compression.drift import TEXT_PYTHON, score_command  # noqa: E402
from qwen35_compression.feature1 import (  # noqa: E402
    FLA_VERSION,
    fla_install_command,
    run_logged,
)
from qwen35_compression.io import write_json  # noqa: E402
from qwen35_compression.provenance import git_revision  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CENSUS = Path("data/blaze/census.json")
CENSUS_MULTILINGUAL = Path("data/blaze/census_multilingual.json")
PLUGIN = Path("plugins/blaze_vllm")
EXPORT = "blaze_q6"
# Humming (vLLM's 6-bit kernel) compiles with NVRTC, whose builtins library the pip CUDA wheels
# ship outside the loader's path; without it the first 6-bit layer fails to build.
NVRTC_LIB = (
    TEXT_PYTHON.parents[1] / "lib" / "python3.13" / "site-packages" / "nvidia" / "cu13" / "lib"
)
MODES = {
    "native": {
        "LD_LIBRARY_PATH": ":".join(
            p for p in (str(NVRTC_LIB), os.environ.get("LD_LIBRARY_PATH")) if p
        )
    },
    # vLLM's compile cache is keyed on the model and its config, not on BLAZE_W6_AS_W8: the 8-bit
    # mode reusing the native mode's graph fails on its different parameters (KeyError: 'weight').
    "w8": {"BLAZE_W6_AS_W8": "1", "VLLM_CACHE_ROOT": str(Path.home() / ".cache" / "vllm-blaze-w8")},
}
GPTQ = "gptq_w4a16_g128"
GPTQ_REPO = "lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128"
LADDERS = {"ladder": CENSUS, "ladder_ml": CENSUS_MULTILINGUAL}
# The multilingual ladder keeps the English one's 65,536 8-bit rows (equal bytes); only the
# ranking that picks them changes (scripts/blaze_census_multilingual.py).
EQUAL_SIZE_COVER = "1e-9"
# G1's measured drift of the English ladder on BF16's answers (BF16 body), for comparison.
G1_LADDER_EXCESS = 0.00022
# G2b's measured excess loss; the real export must land within the tolerance of it.
G2B_EXCESS = 0.001689468332214339
REBUILD_TOLERANCE = 0.0002
# Measured on A30s with the same speed script (Blaze-S and add-on runs, 2026-10-16).
RECORDED_SPEED = {
    "bf16": {"single": 80, "batched": None, "gpu_weight_gb": None},
    GPTQ: {"single": 151, "batched": 3126, "gpu_weight_gb": 3.92},
}
PILOT_GPTQ = ["--blocks", "8", "--length", "512", "--layers", "0,1"]
DOMAIN_PILOT = 2
# On an A30: setup and downloads 0.25 h, pilot 0.15 h, GPTQ and export 0.25 h, two drift scores
# 0.25 h, two speed runs 0.15 h, domain answers 0.1 h, two ladder exports 0.1 h, five domain
# scores 0.2 h, the multilingual ladder's drift 0.1 h.
ESTIMATED_HOURS = 1.55


def with_env(mode: str, command: list[str]) -> list[str]:
    env = MODES[mode]
    return ["env", *(f"{k}={v}" for k, v in env.items()), *command] if env else command


def export_command(export: Path, plan_file: Path, limit: int | None) -> list[str]:
    return [
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
        "--real",
        "--output",
        str(export),
        *(PILOT_GPTQ if limit else []),
    ]


def domain_score(model: Path | str, name: str, output: Path, limit: int | None, snapshot: Path):
    command = score_command(
        model, name, output / "domain_traces", output / f"domain_{name}.json", limit, snapshot
    )
    return [*command, "--domains"]


def plan(
    output: Path,
    exports: Path,
    outputs: Path,
    snapshot: Path | str,
    plan_file: Path,
    limit: int | None,
) -> dict[str, list[str]]:
    snapshot = Path(snapshot)
    suffix = "-pilot" if limit else ""
    export = exports / (EXPORT + suffix)
    traces = output / "traces"
    steps: dict[str, list[str]] = {"export": export_command(export, plan_file, limit)}
    for mode in MODES:
        steps[f"score:{mode}"] = with_env(
            mode,
            score_command(
                export,
                f"{EXPORT}_{mode}",
                traces,
                output / f"{EXPORT}_{mode}.json",
                limit,
                snapshot,
            ),
        )
        if not limit:
            steps[f"speed:{mode}"] = with_env(
                mode,
                [
                    str(TEXT_PYTHON),
                    "scripts/blaze_speed.py",
                    "--model",
                    str(export),
                    "--name",
                    f"{EXPORT}_{mode}",
                    "--output",
                    str(output / f"speed_{EXPORT}_{mode}.json"),
                ],
            )
    steps["domain_answers"] = [
        str(TEXT_PYTHON),
        "scripts/blaze_domain_answers.py",
        "--model",
        str(snapshot),
        "--output",
        str(output / "domain_traces"),
        *(["--limit", str(DOMAIN_PILOT)] if limit else []),
    ]
    domain_limit = DOMAIN_PILOT if limit else None
    steps["domain:bf16"] = domain_score(snapshot, "bf16", output, domain_limit, snapshot)
    if limit:
        return steps  # BF16's own domain score shows the domain steps work
    for name, census in LADDERS.items():
        ladder = exports / name
        steps[f"build:{name}"] = [
            sys.executable,
            "scripts/blaze_export.py",
            "--model",
            str(snapshot),
            "--census",
            str(census),
            "--ladder",
            *(["--cover", EQUAL_SIZE_COVER] if name == "ladder_ml" else []),
            "--output",
            str(ladder),
        ]
        steps[f"domain:{name}"] = domain_score(ladder, name, output, None, snapshot)
        if name == "ladder_ml":
            # What the multilingual ranking costs English: drift on BF16's benchmark answers.
            steps["score:ladder_ml"] = score_command(
                ladder, name, traces, output / f"{name}.json", None, snapshot
            )
    steps[f"domain:{GPTQ}"] = domain_score(outputs / GPTQ, GPTQ, output, None, snapshot)
    steps[f"domain:{EXPORT}"] = with_env("w8", domain_score(export, EXPORT, output, None, snapshot))
    return steps


def q6_report(output: Path, bf16: dict, export: Path, gptq: Path) -> dict[str, Any]:
    def load(name: str) -> dict[str, Any] | None:
        path = output / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    sizes = {
        name: sum(p.stat().st_size for p in path.glob("*.safetensors"))
        for name, path in ((EXPORT, export), (GPTQ, gptq))
    }
    modes: dict[str, Any] = {}
    for mode in MODES:
        score, speed = load(f"{EXPORT}_{mode}"), load(f"speed_{EXPORT}_{mode}")
        modes[mode] = {
            "drift": None
            if score is None
            else {**drift_point(score), "excess_loss": excess_loss(score, bf16)},
            "speed": speed,
        }
    base = load("domain_bf16")
    domains: dict[str, Any] = {}
    if base is not None:
        for name in ("ladder", "ladder_ml", GPTQ, EXPORT):
            result = load(f"domain_{name}")
            if result is None:
                continue
            domains[name] = {
                group: {
                    "excess_loss": scores["all"]["mean_nll"]
                    - base["scores"][group]["all"]["mean_nll"],
                    "flip_rate": scores["all"]["flip_rate"],
                    "tokens": scores["all"]["tokens"],
                }
                for group, scores in result["scores"].items()
            }
        domains["bf16"] = {
            group: {"flip_rate": scores["all"]["flip_rate"], "tokens": scores["all"]["tokens"]}
            for group, scores in base["scores"].items()
        }
    ladder_ml = load("ladder_ml")
    english = {
        "g1_ladder_excess": G1_LADDER_EXCESS,
        "ladder_ml": None
        if ladder_ml is None
        else {**drift_point(ladder_ml), "excess_loss": excess_loss(ladder_ml, bf16)},
    }
    w8 = modes["w8"]["drift"]
    checks = {
        "equal_size": sizes[EXPORT] <= sizes[GPTQ],
        "same_model_w8": w8 is not None
        and abs(w8["excess_loss"] - G2B_EXCESS) <= REBUILD_TOLERANCE,
        "native_runs": modes["native"]["drift"] is not None,
    }
    return {
        "bf16": drift_point(bf16),
        "g2b_excess": G2B_EXCESS,
        "safetensors_bytes": sizes,
        "modes": modes,
        "recorded_speed": RECORDED_SPEED,
        "domains": domains,
        "english_drift_of_ladders": english,
        "checks": checks,
        "decision": "go: Blaze-Q is servable at W4A16's size"
        if checks["equal_size"] and checks["same_model_w8"]
        else "report as measured",
    }


def run_gate(args: argparse.Namespace, output: Path, limit: int | None) -> dict[str, Any]:
    from qwen35_compression.drift import copy_published, fetch_traces
    from qwen35_compression.models import download_model

    config = load_config(args.config)
    exports = config.paths.outputs / "blaze"
    run = GateRun(
        output,
        "blaze_q6",
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
    gptq = config.paths.outputs / GPTQ
    if not limit and not gptq.exists():
        run.manifest["gptq_revision"] = run.step(
            "fetch_gptq", lambda: copy_published(GPTQ_REPO, gptq)
        )
    steps = plan(output, exports, config.paths.outputs, snapshot, args.plan, limit)
    if args.reuse and not limit:
        # Results an earlier run already measured, on the same export code and plan.
        for path in sorted(args.reuse.glob("*.json")):
            if path.name != "run_manifest.json":
                shutil.copyfile(path, output / path.name)
        run.manifest["reused_from"] = str(args.reuse)
    for name, command in steps.items():
        target = command[command.index("--output") + 1] if "--output" in command else None
        if target and target.endswith(".json") and Path(target).exists():
            run.reuse(name)
            continue
        if name in ("export",) or name.startswith("build:"):
            if target and (Path(target) / MANIFEST).exists():
                run.reuse(name)
                continue
        if ":native" in name:
            # vLLM's own 6-bit kernel is new; its failure is a result, not the run's end.
            try:
                run.step(name, lambda command=command: run_logged(command, log, ROOT))
            except subprocess.CalledProcessError:
                run.manifest["status"] = "running"
                run.save()
            continue
        run.step(name, lambda command=command: run_logged(command, log, ROOT))
    if limit:
        run.finish()
        return run.manifest
    bf16 = json.loads(args.bf16.read_text(encoding="utf-8"))
    report = q6_report(output, bf16, exports / EXPORT, gptq)
    write_json(output / "q6_report.json", report)
    run.finish(checks=report["checks"], decision=report["decision"])
    print("decision=" + report["decision"], flush=True)
    return run.manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/blaze/q6"))
    parser.add_argument("--plan", type=Path, required=True, help="G2b's market plan")
    parser.add_argument("--bf16", type=Path, required=True, help="G2's BF16 drift score")
    parser.add_argument("--pilot-limit", type=int, help="Run small into <output>-pilot first")
    parser.add_argument(
        "--reuse", type=Path, help="Score and speed results of an earlier run, not measured again"
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
    pilot_output = output.with_name(output.name + "-pilot")

    if args.dry_run:
        config = load_config(args.config)
        exports = config.paths.outputs / "blaze"
        print(
            json.dumps(
                {
                    "pilot": plan(
                        pilot_output, exports, config.paths.outputs, "<4b>", args.plan, 10
                    )
                    if args.pilot_limit
                    else None,
                    "full": plan(output, exports, config.paths.outputs, "<4b>", args.plan, None),
                    "estimate_on_a30": cost(ESTIMATED_HOURS),
                },
                indent=2,
            )
        )
        return
    for path in (args.plan, args.bf16, CENSUS_MULTILINGUAL):
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
    subprocess.run(fla_install_command(sys.executable), cwd=ROOT, check=True)
    print(f"flash-linear-attention {FLA_VERSION} installed; blaze-vllm installed", flush=True)
    subprocess.run(
        [str(TEXT_PYTHON), "-c", "import humming, vllm; print('humming', vllm.__version__)"],
        cwd=ROOT,
        check=False,
    )
    if args.pilot_limit:
        run_gate(args, pilot_output, args.pilot_limit)
    run_gate(args, output, None)


if __name__ == "__main__":
    main()
