from __future__ import annotations

import pandas as pd

from qwen35_compression.vlmeval_patches import (
    FINAL_ANSWER_RULE_LOG,
    RANDOM_FILL_LOG,
    final_answer_letter,
    subset_rows,
    with_final_answer_fallback,
)


def test_final_answer_letter_reads_explicit_statements() -> None:
    choices = "ABCD"
    # The three answers the extractor failed on in the 2,048-cap BF16 pass.
    mmbench = "Correct Answer: **B. Not the same**  ✅ Final Answer: **B**"
    mmmu = "> **A. [Graph showing inverted U-shape]**  ✅ Final Answer: **A**"
    assert final_answer_letter(mmbench, choices) == "B"
    assert final_answer_letter(mmmu, choices) == "A"
    assert final_answer_letter("Final answer: (C)", choices) == "C"
    assert final_answer_letter("The final answer is D.", choices) == "D"
    # The last statement wins when the model revises itself.
    assert final_answer_letter("Final Answer: A ... wait. Final Answer: C", choices) == "C"


def test_final_answer_letter_rejects_prose_and_unknown_options() -> None:
    assert final_answer_letter("The answer is B.", "ABCD") is None
    assert final_answer_letter("Final answer: The image shows a cat", "ABCD") is None
    assert final_answer_letter("the final answer is a cat", "ABCD") is None
    assert final_answer_letter("Final Answer: E", "ABCD") is None


def test_fallback_only_replaces_random_fills() -> None:
    item = {"prediction": "Final Answer: **B**", "A": "x", "B": "y", "C": float("nan")}

    def random_fill(model, item, dataset_name=None):
        return {"opt": "A", "log": RANDOM_FILL_LOG + " "}

    def extracted(model, item, dataset_name=None):
        return {"opt": "A", "log": "A"}

    assert with_final_answer_fallback(random_fill)(None, item) == {
        "opt": "B",
        "log": f"{FINAL_ANSWER_RULE_LOG} B",
    }
    assert with_final_answer_fallback(extracted)(None, item) == {"opt": "A", "log": "A"}
    # C is NaN, so "Final Answer: C" is not a valid option and the random fill stands.
    item["prediction"] = "Final Answer: C"
    assert with_final_answer_fallback(random_fill)(None, item)["log"].startswith(RANDOM_FILL_LOG)


def test_subset_keeps_every_rotation_of_kept_circular_questions() -> None:
    frame = pd.DataFrame({"index": [1, 2, 3, 1_000_001, 1_000_002, 2_000_001, 1_000_003]})
    assert list(subset_rows(frame, 2)["index"]) == [1, 2, 1_000_001, 1_000_002, 2_000_001]
    plain = pd.DataFrame({"index": ["a", "b", "c"]})
    assert list(subset_rows(plain, 2)["index"]) == ["a", "b"]
