"""Shared pieces of the drift-scoring runs: published models, BF16 traces, score commands."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES_REPO = "lazybrick/kiln-evals"
# drift_scores.py runs vLLM, which lives in the text-evaluation environment.
TEXT_PYTHON = ROOT / ".venv-gpu-text" / "bin" / "python"


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
