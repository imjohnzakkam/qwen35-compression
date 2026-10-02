#!/usr/bin/env python3
"""Run VLMEvalKit's run.py with this project's fixed adjustments applied.

- Multiple-choice scoring: when prefetch and the answer extractor both fail, VLMEvalKit picks a
  random option. The model's explicit "Final Answer: X" is used instead when it states one.
- --limit N keeps the first N questions of every dataset (local validation runs only).

Usage: vlmeval_run.py --toolkit-dir DIR [--limit N] -- <run.py arguments>
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen35_compression.vlmeval_patches import (  # noqa: E402
    subset_rows,
    with_final_answer_fallback,
)


def apply_patches(limit: int | None) -> None:
    from vlmeval.dataset import image_base
    from vlmeval.dataset.utils import multiple_choice

    multiple_choice.extract_answer_from_item = with_final_answer_fallback(
        multiple_choice.extract_answer_from_item
    )
    if limit is None:
        return
    original_init = image_base.ImageBaseDataset.__init__

    def limited_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.data = subset_rows(self.data, limit)

    image_base.ImageBaseDataset.__init__ = limited_init


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toolkit-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args, rest = parser.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]
    toolkit_dir = args.toolkit_dir.resolve()
    sys.path.insert(0, str(toolkit_dir))
    apply_patches(args.limit)
    sys.argv = [str(toolkit_dir / "run.py"), *rest]
    runpy.run_path(str(toolkit_dir / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
