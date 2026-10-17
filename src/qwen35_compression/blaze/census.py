"""Stage 1: how often each vocabulary row is used, as an input and as the model's own output.

Counts come from public training splits and from BF16's answers to training prompts, never from a
test split or a benchmark answer. Every Qwen3.5 size shares one 248,320-token vocabulary, so one
census serves the family. The saved form is sparse: rows that never occur are left out.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class Census:
    """Per-row token counts, kept per source so coverage can be checked source by source."""

    vocab_size: int
    counts: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, source: str, rows: Iterable[Sequence[int]]) -> int:
        """Count the token ids of each row into `source`; return how many tokens were added."""
        counts = self.counts.setdefault(source, np.zeros(self.vocab_size, dtype=np.int64))
        added = 0
        for row in rows:
            ids = np.asarray(row, dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < self.vocab_size)]
            counts += np.bincount(ids, minlength=self.vocab_size)
            added += int(ids.size)
        return added

    def total(self, sources: Iterable[str] | None = None) -> np.ndarray:
        names = list(self.counts) if sources is None else list(sources)
        out = np.zeros(self.vocab_size, dtype=np.int64)
        for name in names:
            out += self.counts[name]
        return out

    def tokens(self, sources: Iterable[str] | None = None) -> int:
        return int(self.total(sources).sum())

    def used_rows(self) -> int:
        return int(np.count_nonzero(self.total()))

    def ranking(self, sources: Iterable[str] | None = None) -> np.ndarray:
        """Row ids from most to least used; ties keep id order, so the ranking is deterministic."""
        counts = self.total(sources)
        return np.lexsort((np.arange(self.vocab_size), -counts))

    def to_json(self) -> dict[str, Any]:
        sources = {}
        for name, counts in sorted(self.counts.items()):
            used = np.flatnonzero(counts)
            sources[name] = {
                "tokens": int(counts.sum()),
                "rows": {str(int(row)): int(counts[row]) for row in used},
            }
        return {"schema_version": SCHEMA_VERSION, "vocab_size": self.vocab_size, "sources": sources}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Census:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported census schema: {data.get('schema_version')}")
        census = cls(int(data["vocab_size"]))
        for name, source in data["sources"].items():
            counts = np.zeros(census.vocab_size, dtype=np.int64)
            for row, count in source["rows"].items():
                counts[int(row)] = int(count)
            if int(counts.sum()) != int(source["tokens"]):
                raise ValueError(f"census source {name} does not add up to its token count")
            census.counts[name] = counts
        return census


def coverage(order: np.ndarray, counts: np.ndarray, rows: int) -> float:
    """Share of the tokens in `counts` that fall in the first `rows` rows of `order`."""
    total = int(counts.sum())
    if total == 0:
        raise ValueError("no tokens to cover")
    return float(counts[order[:rows]].sum()) / total


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_census(census: Census, path: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    """Write the census and its lock (the file's SHA-256 plus where every count came from)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(census.to_json(), sort_keys=True) + "\n", encoding="utf-8")
    lock = {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "vocab_size": census.vocab_size,
        "tokens": {name: int(counts.sum()) for name, counts in sorted(census.counts.items())},
        "used_rows": census.used_rows(),
        **provenance,
    }
    lock_path = path.with_name(path.stem + ".lock.json")
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def load_census(path: Path, verify: bool = True) -> Census:
    """Read a census, refusing it if its lock says the file has changed."""
    if verify:
        lock = json.loads(path.with_name(path.stem + ".lock.json").read_text(encoding="utf-8"))
        if lock["sha256"] != file_sha256(path):
            raise ValueError(f"{path} does not match its lock")
    return Census.from_json(json.loads(path.read_text(encoding="utf-8")))
