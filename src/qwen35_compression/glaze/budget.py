"""Memory and time estimates for Glaze training, and the guards that stop a run early."""

from __future__ import annotations

GIB = 2**30


class BudgetExceeded(RuntimeError):
    """A run would run out of GPU memory or time; it stops before spending more."""


def memory_estimate_gib(
    *,
    quantized_params: float,
    scales: float,
    embedding_params: float,
    layers: int,
    hidden: int,
    intermediate: int,
    vocab: int,
    micro_batch_tokens: int,
    chunk_tokens: int,
    deltanet_layer_gib: float = 1.7,
    overhead_gib: float = 1.5,
) -> dict[str, float]:
    """GPU memory of a training step with the BF16 teacher on the same GPU, item by item.

    `deltanet_layer_gib` is one DeltaNet layer's forward and backward with flash-linear-attention
    at 8,192 tokens, from the drift study's fla check.
    """
    items = {
        "teacher_quantizable_weights": quantized_params * 2 / GIB,
        "shared_embedding": embedding_params * 2 / GIB,
        "student_codes": quantized_params / GIB,
        "scale_parameters_adam_grad": scales * 4 * 4 / GIB,
        "base_scales": scales * 2 / GIB,
        "checkpointed_layer_inputs": layers * micro_batch_tokens * hidden * 2 / GIB,
        "layer_recompute_backward": micro_batch_tokens * intermediate * 2 * 4 / GIB
        + deltanet_layer_gib,
        # Student logits (bf16), fp32 log-softmax and its gradient; the teacher's fp32 log-probs.
        "logit_chunk": chunk_tokens * vocab * (2 + 4 + 4 + 4) / GIB,
        "context_and_allocator": overhead_gib,
    }
    items["total"] = sum(items.values())
    return items


def check_memory(peak_bytes: int, total_bytes: int, fraction: float) -> None:
    if total_bytes > 0 and peak_bytes > fraction * total_bytes:
        raise BudgetExceeded(
            f"peak GPU memory {peak_bytes / GIB:.1f} GiB exceeds {fraction:.0%} of "
            f"{total_bytes / GIB:.1f} GiB"
        )


def projected_minutes(elapsed_seconds: float, steps_done: int, total_steps: int) -> float:
    if steps_done <= 0:
        raise ValueError("no steps done yet")
    return elapsed_seconds / steps_done * total_steps / 60


def check_time(elapsed_seconds: float, steps_done: int, total_steps: int, budget: float) -> None:
    minutes = projected_minutes(elapsed_seconds, steps_done, total_steps)
    if minutes > budget:
        raise BudgetExceeded(
            f"training would take {minutes:.1f} min at the measured pace, over its "
            f"{budget:.0f}-minute budget"
        )
