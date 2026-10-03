#!/usr/bin/env python3
"""Drift study: where and how quantization error builds up along long answers.

On one GPU: score BF16 and the published Kiln variants with scripts/drift_scores.py on BF16's
own MATH-500 and MMLU-Pro answers, then quantize and score GPTQ W4A16 g128 applied to one
language-model component at a time (DeltaNet, attention, FFN). A model that fails to quantize
or score is recorded and skipped; the rest still run. Finally, flash-linear-attention is
installed in the compression environment and checked against transformers' PyTorch DeltaNet
fallback, after every quantization has run, so all exports share the baseline's environment.

    uv run python scripts/run_drift_study.py --pilot-limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.export import verify_export
from qwen35_compression.feature1 import run_logged
from qwen35_compression.io import write_json
from qwen35_compression.models import download_model
from qwen35_compression.provenance import git_revision

ROOT = Path(__file__).resolve().parent.parent
TRACES_REPO = "lazybrick/kiln-evals"
PUBLISHED = {
    "int8_w8a8": "lazybrick/Qwen3.5-4B-Kiln-INT8-W8A8",
    "gptq_w4a16_g128": "lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128",
    "awq_w4a16_g128": "lazybrick/Qwen3.5-4B-Kiln-AWQ-W4A16-g128",
}
COMPONENTS = {
    "deltanet": "sensitivity_deltanet_w4a16_g128",
    "attention": "sensitivity_attention_w4a16_g128",
    "ffn": "sensitivity_ffn_w4a16_g128",
}
TEXT_PYTHON = ROOT / ".venv-gpu-text" / "bin" / "python"
FLA_VERSION = "0.5.2"


def fetch_published(repo: str) -> tuple[Path, str]:
    """Download a published Kiln variant without its README; return the path and revision."""
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(repo).sha
    path = snapshot_download(
        repo, revision=revision, ignore_patterns=["README.md", ".gitattributes"]
    )
    return Path(path), revision


def fetch_traces(destination: Path) -> str:
    """BF16's lm-eval answers from the published evaluation record; return its revision."""
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().dataset_info(TRACES_REPO).sha
    snapshot_download(
        TRACES_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["bf16/text/samples/minerva_math500.jsonl", "bf16/text/samples/mmlu_pro_*"],
        local_dir=destination,
    )
    return revision


def score_command(
    model: Path | str, name: str, traces: Path, output: Path, limit: int | None, tokenizer: Path
) -> list[str]:
    command = [
        str(TEXT_PYTHON),
        "scripts/drift_scores.py",
        "score",
        "--model",
        str(model),
        "--name",
        name,
        "--traces",
        str(traces),
        "--tokenizer",
        str(tokenizer),
        "--output",
        str(output),
    ]
    if limit:
        command.extend(("--limit", str(limit)))
    return command


def study(
    args: argparse.Namespace,
    output: Path,
    limit: int | None,
    components: list[str],
    published: list[str],
    fla: bool,
) -> dict:
    config = load_config(args.config)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "run.log"
    manifest: dict = {
        "schema_version": 1,
        "study": "drift",
        "code_revision": args.code_revision,
        "config_digest": config.digest,
        "limit": limit,
        "research_result": limit is None,
        "models": {},
    }

    def record(name: str, **fields) -> None:
        manifest["models"].setdefault(name, {}).update(fields)
        write_json(output / "run_manifest.json", manifest)

    snapshot, revision = download_model(config)
    manifest["base_model"] = {"id": config.model.id, "revision": revision}
    traces = output / "traces"
    manifest["traces"] = {"repo": TRACES_REPO, "revision": fetch_traces(traces)}
    write_json(output / "run_manifest.json", manifest)

    def score(name: str, model: Path | str) -> None:
        started = time.perf_counter()
        try:
            run_logged(
                score_command(model, name, traces, output / f"{name}.json", limit, snapshot),
                log,
                ROOT,
            )
            record(name, status="scored", seconds=round(time.perf_counter() - started))
        except subprocess.CalledProcessError as error:
            record(name, status="score_failed", error=str(error))

    score("bf16", snapshot)
    for name in published:
        try:
            path, repo_revision = fetch_published(PUBLISHED[name])
            verify_export(path, config.variant(name))
            record(name, source=PUBLISHED[name], revision=repo_revision, verified=True)
        except Exception as error:  # noqa: BLE001 - recorded; the study goes on without it
            record(name, status="fetch_failed", error=f"{type(error).__name__}: {error}")
            continue
        score(name, path)

    for component in components:
        variant = COMPONENTS[component]
        export = config.paths.outputs / variant
        started = time.perf_counter()
        try:
            if not (export.is_dir() and any(export.iterdir())):
                run_logged(
                    [
                        sys.executable,
                        "scripts/quantize.py",
                        "--config",
                        str(args.config),
                        "--variant",
                        variant,
                    ],
                    log,
                    ROOT,
                )
            verify_export(export, config.variant(variant))
            record(
                component, variant=variant, quantize_seconds=round(time.perf_counter() - started)
            )
        except Exception as error:  # noqa: BLE001 - recorded; the study goes on without it
            record(
                component,
                variant=variant,
                status="quantize_failed",
                error=f"{type(error).__name__}: {error}",
            )
            continue
        score(component, export)

    if fla:
        try:
            # --no-deps: fla must not replace the pinned torch; einops is its one other need.
            run_logged(
                [
                    "uv",
                    "pip",
                    "install",
                    "--no-deps",
                    "--python",
                    sys.executable,
                    f"fla-core=={FLA_VERSION}",
                    f"flash-linear-attention=={FLA_VERSION}",
                    "einops",
                ],
                log,
                ROOT,
            )
            run_logged(
                [
                    sys.executable,
                    "scripts/check_fla.py",
                    "--model",
                    str(snapshot),
                    "--output",
                    str(output / "fla_check.json"),
                ],
                log,
                ROOT,
            )
            manifest["fla_check"] = json.loads((output / "fla_check.json").read_text())
        except Exception as error:  # noqa: BLE001 - recorded; it does not affect the scores
            manifest["fla_check"] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        write_json(output / "run_manifest.json", manifest)

    scored = [
        output / f"{name}.json"
        for name, entry in manifest["models"].items()
        if entry.get("status") == "scored"
    ]
    if scored:
        report = subprocess.run(
            [sys.executable, "scripts/drift_scores.py", "report", *map(str, scored)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        (output / "report.md").write_text(report, encoding="utf-8")
        print(report, flush=True)
    manifest["status"] = (
        "passed" if manifest["models"].get("bf16", {}).get("status") == "scored" else "failed"
    )
    write_json(output / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/feature1/drift"))
    parser.add_argument("--components", default=",".join(COMPONENTS))
    parser.add_argument("--published", default=",".join(PUBLISHED))
    parser.add_argument(
        "--pilot-limit",
        type=int,
        help="First score BF16 and GPTQ on N answers per task into <output>-pilot (no "
        "quantization); run the full study only if that passes",
    )
    parser.add_argument("--skip-fla", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--code-revision", help="Producer Git revision for uploaded checkouts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.code_revision = args.code_revision or git_revision(ROOT)
    if not args.code_revision:
        raise ValueError("producer code revision is required")
    os.environ["QWEN35_CODE_REVISION"] = args.code_revision
    components = [c for c in args.components.split(",") if c]
    published = [p for p in args.published.split(",") if p]
    unknown = (set(components) - set(COMPONENTS)) | (set(published) - set(PUBLISHED))
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    output = args.output.resolve()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "pilot": None
                    if not args.pilot_limit
                    else {
                        "output": str(output.with_name(output.name + "-pilot")),
                        "limit": args.pilot_limit,
                        "published": ["gptq_w4a16_g128"],
                        "components": [],
                    },
                    "full": {
                        "output": str(output),
                        "published": published,
                        "components": components,
                        "fla_check": not args.skip_fla,
                    },
                    "score_example": score_command(
                        "MODEL",
                        "NAME",
                        output / "traces",
                        output / "NAME.json",
                        args.pilot_limit,
                        Path("TOKENIZER"),
                    ),
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
    if args.pilot_limit:
        pilot = study(
            args,
            output.with_name(output.name + "-pilot"),
            args.pilot_limit,
            [],
            ["gptq_w4a16_g128"],
            fla=False,
        )
        if (
            pilot["status"] != "passed"
            or pilot["models"].get("gptq_w4a16_g128", {}).get("status") != "scored"
        ):
            raise SystemExit("pilot failed; the full study was not started")
    manifest = study(args, output, None, components, published, fla=not args.skip_fla)
    if manifest["status"] != "passed":
        raise SystemExit("drift study failed")


if __name__ == "__main__":
    main()
