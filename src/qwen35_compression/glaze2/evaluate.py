"""Held-out KL to BF16 on answer tokens, per domain: the proxy's gate metric.

Every variant is scored from its written export, read back into the same student model Glaze v1
used, so the number is the stored model's, not an in-memory approximation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from qwen35_compression.glaze.losses import DistillStats, distill_step
from qwen35_compression.glaze2.data import DOMAIN_CODES, DOMAINS, PackedBlocks

IN_DOMAIN = ("math", "mcq")


def held_out_kl(
    teacher: nn.Module,
    student: nn.Module,
    head_weight: torch.Tensor,
    blocks: PackedBlocks,
    blocks_per_batch: int = 4,
    chunk_tokens: int = 512,
) -> dict[str, Any]:
    """Mean KL(BF16 || student) over the answer tokens of each domain, and in-domain overall."""
    device = head_weight.device
    stats = {name: DistillStats() for name in DOMAINS}
    for start in range(0, len(blocks), blocks_per_batch):
        stop = start + blocks_per_batch
        ids = torch.tensor(blocks.ids[start:stop], device=device)
        answer = torch.tensor(blocks.answer[start:stop], device=device).bool()
        domain = torch.tensor(blocks.domain[start:stop], device=device)
        with torch.no_grad():
            teacher_hidden = teacher(input_ids=ids, use_cache=False).last_hidden_state
            student_hidden = student(input_ids=ids, use_cache=False).last_hidden_state
            for name in DOMAINS:
                mask = answer & (domain == DOMAIN_CODES[name])
                if not mask.any():
                    continue
                stats[name].merge(
                    distill_step(
                        student_hidden,
                        teacher_hidden,
                        head_weight,
                        mask,
                        chunk_tokens,
                        float(mask.sum()),
                        backward=False,
                    )
                )
    combined = DistillStats()
    for name in IN_DOMAIN:
        combined.merge(stats[name])
    result: dict[str, Any] = {name: stats[name].summary() for name in DOMAINS}
    result["in_domain"] = combined.summary()
    return result


def evaluate_export(
    teacher: nn.Module,
    head_weight: torch.Tensor,
    export_dir: Path,
    blocks: PackedBlocks,
    blocks_per_batch: int = 4,
) -> dict[str, Any]:
    """Build the student from an export (any mix of 4- and 8-bit groups) and score it."""
    from qwen35_compression.glaze.student import build_student, read_init_export

    student = build_student(teacher, read_init_export(export_dir, allow_mixed=True), False)
    student.eval()
    try:
        return held_out_kl(teacher, student, head_weight, blocks, blocks_per_batch)
    finally:
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def reduction(baseline: dict[str, Any], candidate: dict[str, Any], key: str) -> float | None:
    """How much of the baseline's KL the candidate removes (0.15 = 15% lower); None when the
    domain has no scored tokens."""
    before, after = baseline[key]["mean_kl"], candidate[key]["mean_kl"]
    if before is None or after is None or before == 0:
        return None
    return 1 - after / before
