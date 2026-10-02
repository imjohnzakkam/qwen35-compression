"""Runtime adjustments to VLMEvalKit, applied by scripts/vlmeval_run.py.

Kept free of VLMEvalKit imports so the rules can be unit-tested without the toolkit installed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

# Prefix of the log VLMEvalKit's MCQ path writes when the extractor never returned a usable
# option and it substitutes a random one.
RANDOM_FILL_LOG = "Failed to predict, thus randomly generate one."
FINAL_ANSWER_RULE_LOG = "Final-answer rule after extractor failure:"

# An explicit "Final Answer: X" statement, allowing Markdown emphasis, brackets and a trailing
# option text ("**B. Not the same**"). Only the option letter is captured.
_FINAL_ANSWER = re.compile(
    r"final\s+answer\s*(?:is)?\s*[:：]?\s*[*_`>\s]*[(\[]?([A-Z])[)\]]?(?=$|[\s*_`.,:;)\]])",
    re.IGNORECASE,
)


def final_answer_letter(prediction: str, choices: Iterable[str]) -> str | None:
    """Option letter from the last explicit "Final Answer" statement, if it names a choice."""
    matches = [m.group(1) for m in _FINAL_ANSWER.finditer(str(prediction))]
    if not matches:
        return None
    letter = matches[-1]
    # The letter must be written in upper case; "final answer is a ..." is prose, not option A.
    if not letter.isupper() or letter not in set(choices):
        return None
    return letter


def with_final_answer_fallback(extract):
    """Wrap VLMEvalKit's extract_answer_from_item: replace a random fill with the model's
    explicit final answer when it states one. Prefetch and the extractor still run first."""

    def wrapped(model, item, dataset_name=None):
        result = extract(model, item, dataset_name=dataset_name)
        if not str(result.get("log", "")).startswith(RANDOM_FILL_LOG):
            return result
        choices = [ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if _present(item, ch)]
        letter = final_answer_letter(item["prediction"], choices)
        if letter is None:
            return result
        return {"opt": letter, "log": f"{FINAL_ANSWER_RULE_LOG} {letter}"}

    return wrapped


def _present(item: Mapping[str, Any], key: str) -> bool:
    if key not in item:
        return False
    value = item[key]
    return value is not None and value == value and str(value) != ""  # NaN != NaN


def subset_rows(frame, limit: int):
    """First `limit` questions of a VLMEvalKit dataset frame.

    Circular MCQ datasets (MMBench) store each rotation of a question as its own row with index
    base + k * 1e6; all rotations of a kept question are kept so circular scoring still works.
    """
    indices = [int(str(x)) if str(x).isdigit() else None for x in frame["index"]]
    if all(i is not None for i in indices) and any(i >= 1_000_000 for i in indices):
        bases: list[int] = []
        for i in indices:
            if i % 1_000_000 not in bases:
                bases.append(i % 1_000_000)
        keep = set(bases[:limit])
        mask = [i % 1_000_000 in keep for i in indices]
        return frame[mask].reset_index(drop=True)
    return frame.head(limit).reset_index(drop=True)
