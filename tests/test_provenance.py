from pathlib import Path

from qwen35_compression.provenance import git_revision


def test_injected_revision_survives_uploaded_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("QWEN35_CODE_REVISION", "abc123")
    assert git_revision(tmp_path) == "abc123"
