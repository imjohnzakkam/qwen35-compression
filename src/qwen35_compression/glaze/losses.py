"""Glaze's objective: KL(teacher || student) on next-token distributions, computed in chunks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DistillStats:
    """Token count, summed KL, top-1 flips and summed NLL of the teacher's top token."""

    tokens: int = 0
    kl: float = 0.0
    flips: int = 0
    nll: float = 0.0

    def merge(self, other: DistillStats) -> None:
        self.tokens += other.tokens
        self.kl += other.kl
        self.flips += other.flips
        self.nll += other.nll

    @property
    def mean_kl(self) -> float:
        return self.kl / self.tokens if self.tokens else float("nan")

    def summary(self) -> dict[str, float | int | None]:
        if not self.tokens:
            return {"tokens": 0, "mean_kl": None, "flip_rate": None, "mean_nll": None}
        return {
            "tokens": self.tokens,
            "mean_kl": self.kl / self.tokens,
            "flip_rate": self.flips / self.tokens,
            "mean_nll": self.nll / self.tokens,
        }


def _widened(logits: torch.Tensor) -> torch.Tensor:
    """BF16 logits in fp32 for the softmax; wider inputs keep their precision."""
    return logits.to(torch.promote_types(logits.dtype, torch.float32))


def distill_step(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    head_weight: torch.Tensor,
    mask: torch.Tensor,
    chunk_tokens: int,
    normalizer: float,
    backward: bool,
) -> DistillStats:
    """KL(teacher || student) at the masked positions, a chunk of positions at a time.

    Logits over the full vocabulary exist for one chunk at a time. With `backward`, each chunk's
    KL divided by `normalizer` is backpropagated to the final hidden states as it is computed,
    and the accumulated gradient then flows once through the student. A flip is a position where
    the student's top token differs from the teacher's; NLL is the student's for the teacher's
    top token, as the drift study measures them along BF16's own tokens.
    """
    if chunk_tokens <= 0 or normalizer <= 0:
        raise ValueError("chunk_tokens and normalizer must be positive")
    if student_hidden.shape != teacher_hidden.shape or mask.shape != student_hidden.shape[:-1]:
        raise ValueError("student, teacher and mask shapes do not match")
    width = student_hidden.shape[-1]
    flat_student = student_hidden.reshape(-1, width)
    flat_teacher = teacher_hidden.reshape(-1, width)
    positions = mask.reshape(-1).nonzero().squeeze(-1)
    leaf = flat_student.detach().requires_grad_(backward)

    zero = torch.zeros((), dtype=torch.float64, device=student_hidden.device)
    kl_total, nll_total, flips_total = zero.clone(), zero.clone(), zero.clone()
    for start in range(0, positions.numel(), chunk_tokens):
        index = positions[start : start + chunk_tokens]
        with torch.no_grad():
            teacher_logits = _widened(F.linear(flat_teacher[index], head_weight))
            teacher_logp = torch.log_softmax(teacher_logits, -1)
            teacher_top = teacher_logp.argmax(dim=-1)
        student_logp = torch.log_softmax(_widened(F.linear(leaf[index], head_weight)), -1)
        kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
        if backward:
            (kl.sum() / normalizer).backward()
        with torch.no_grad():
            kl_total += kl.sum().double()
            flips_total += (student_logp.argmax(dim=-1) != teacher_top).sum().double()
            nll_total -= student_logp.gather(-1, teacher_top.unsqueeze(-1)).sum().double()
    if backward and leaf.grad is not None:
        flat_student.backward(leaf.grad)
    return DistillStats(
        tokens=int(positions.numel()),
        kl=float(kl_total),
        flips=int(flips_total),
        nll=float(nll_total),
    )
