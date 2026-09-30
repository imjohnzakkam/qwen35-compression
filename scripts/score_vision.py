#!/usr/bin/env python3
"""Score downloaded vision predictions with the suite's fixed answer extractor.

GPU runs infer every vision dataset but leave the extractor-scored ones (free-form MCQ answers)
unscored. This runs VLMEvalKit's eval stage for them on the local machine, so the API key never
leaves it and every variant is scored by the same extractor. The downloaded run is not modified:
predictions are copied into a separate scoring directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.feature1 import build_vision_score_command, load_benchmark_suite
from qwen35_compression.io import write_json
from qwen35_compression.provenance import git_revision

ROOT = Path(__file__).resolve().parent.parent
SCORE_FILE_SUFFIXES = ("_acc.csv", "_score.csv", "_score.json")
# VLMEvalKit's markers for an extraction that never got an answer from the extractor. The MCQ
# path then substitutes a random option and does not count it in judge_fail_rate.
EXTRACTION_FAILURE_MARKERS = ("randomly generate one", "All 5 retries failed")
COUNT_FAILURES = """
import json, sys
from pathlib import Path
import pandas as pd
counts = {}
for path in sorted(Path(sys.argv[1]).rglob(f"*_{sys.argv[2]}*.xlsx")):
    frame = pd.read_excel(path)
    if "log" in frame.columns:
        log = frame["log"].astype(str)
        counts[path.name] = int(sum(log.str.contains(m, regex=False).sum() for m in sys.argv[3:]))
print(json.dumps(counts))
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def redact_tree(root: Path, secret: str) -> list[str]:
    """Remove the key from every file VLMEvalKit wrote (its API client logs it at INFO)."""
    needle, redacted = secret.encode(), []
    for path in root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            if needle in data:
                path.write_bytes(data.replace(needle, b"[REDACTED_OPENAI_API_KEY]"))
                redacted.append(path.relative_to(root).as_posix())
    return redacted


def run_redacted(command: list[str], cwd: Path, env: dict[str, str], secret: str) -> None:
    with subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line.replace(secret, "[REDACTED_OPENAI_API_KEY]"), end="", flush=True)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command[:2])


def find_predictions(run_dir: Path, tasks: tuple[str, ...]) -> tuple[str, dict[str, Path]]:
    vision = run_dir / "vision"
    if not vision.is_dir():
        vision = next((path for path in sorted(run_dir.rglob("vision")) if path.is_dir()), None)
    if vision is None:
        raise FileNotFoundError(f"no vision directory under {run_dir}")
    aliases = [item for item in vision.iterdir() if item.is_dir() and item.name != "logs"]
    if len(aliases) != 1:
        raise ValueError(f"expected one model directory in {vision}, found {len(aliases)}")
    alias = aliases[0].name
    found: dict[str, Path] = {}
    for task in tasks:
        candidates = sorted(aliases[0].glob(f"T*/{alias}_{task}.xlsx"))
        if not candidates:
            raise FileNotFoundError(f"no saved predictions for {task} in {aliases[0]}")
        found[task] = candidates[-1]
    return alias, found


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved vision predictions locally")
    parser.add_argument("--run-dir", type=Path, required=True, help="Downloaded run directory")
    parser.add_argument("--config", type=Path, default=Path("configs/feature1.yaml"))
    parser.add_argument("--output", type=Path, help="Defaults to <run-dir>-scored")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config" / "qwen35" / "openai.env",
        help="Provides OPENAI_API_KEY; loaded only into the scoring process",
    )
    parser.add_argument(
        "--api-nproc",
        type=int,
        default=4,
        help="Concurrent extractor requests; keep under the OpenAI account's rate limit",
    )
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.evaluation.suite_path is None:
        raise ValueError("the config has no benchmark suite")
    suite = load_benchmark_suite(config.evaluation.suite_path)
    run_dir = args.run_dir.resolve()
    manifests = list(run_dir.rglob("run_manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"expected one run_manifest.json under {run_dir}, found {len(manifests)}")
    run_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if run_manifest.get("status") != "passed":
        raise ValueError(f"run did not pass: {run_manifest.get('status')}")
    if not args.allow_pilot and run_manifest.get("research_result") is not True:
        raise ValueError("refusing to score a limited pilot as a research result")

    alias, predictions = find_predictions(run_dir, suite.vision_extracted_tasks)
    output = (args.output or run_dir.with_name(run_dir.name + "-scored")).resolve()
    toolkit_dir = ROOT / "external" / "VLMEvalKit"
    python = ROOT / ".venv-vision-score" / "bin" / "python"
    command = build_vision_score_command(
        suite, output, alias, toolkit_dir, python, api_nproc=args.api_nproc
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "output": str(output),
                    "predictions": {task: str(path) for task, path in predictions.items()},
                },
                indent=2,
            )
        )
        return

    if not python.is_file():
        raise FileNotFoundError(f"scoring environment missing: {python} (see README)")
    env = {**os.environ, **read_env_file(args.env_file)}
    if not env.get("OPENAI_API_KEY"):
        raise ValueError(f"OPENAI_API_KEY is not set in the environment or {args.env_file}")

    if output.exists():
        # --reuse would also reuse cached extractor results, including random fills.
        raise FileExistsError(f"{output} exists; move it aside to score from scratch")
    for task, source in predictions.items():
        target = output / alias / source.parent.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    secret = env["OPENAI_API_KEY"]
    try:
        run_redacted(command, toolkit_dir, env, secret)
    finally:
        redact_tree(output, secret)

    failures = json.loads(
        subprocess.run(
            [
                str(python),
                "-c",
                COUNT_FAILURES,
                str(output),
                str(suite.vision_answer_extractor),
                *EXTRACTION_FAILURE_MARKERS,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )

    scores = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and path.name.endswith(SCORE_FILE_SUFFIXES)
        and any(task in path.name for task in suite.vision_extracted_tasks)
    )
    write_json(
        output / "scoring_manifest.json",
        {
            "schema_version": 1,
            "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "answer_extractor": suite.vision_answer_extractor,
            "tasks": list(suite.vision_extracted_tasks),
            "code_revision": git_revision(ROOT),
            "vlmevalkit_revision": subprocess.run(
                ["git", "-C", str(toolkit_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "source_run": {
                "directory": str(run_dir),
                "code_revision": run_manifest.get("code_revision"),
                "model_revision": run_manifest.get("model_revision"),
                "variant": run_manifest.get("variant"),
            },
            "predictions": {
                task: {"file": path.name, "sha256": sha256(path)}
                for task, path in predictions.items()
            },
            "score_files": scores,
            "extraction_failures": failures,
        },
    )
    print(f"scored={output}")
    failed = {name: count for name, count in failures.items() if count}
    missing = [
        task for task in suite.vision_extracted_tasks if not any(task in path for path in scores)
    ]
    if failed or missing:
        raise SystemExit(
            f"scoring is not valid: extraction failures {failed}, unscored tasks {missing}"
        )


if __name__ == "__main__":
    main()
