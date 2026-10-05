"""One BF16 forward and backward pass over the calibration blocks, recording sensitivities.

Labels are sampled from BF16's own next-token distribution, so the squared gradients estimate
the Fisher information of the model's predictions (the curvature of KL to BF16 at BF16). It
records, per Linear layer, the mean squared input per input channel and the mean squared output
gradient per output channel (the Kronecker factors the allocation uses), and per decoder layer
the squared gradient of its output hidden state, per token and per channel (the block-loss
weights of the rounding stage).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from qwen35_compression.glaze2.allocate import FisherFactors

# Token weights are clipped so a few tokens cannot dominate a block's loss.
TOKEN_WEIGHT_RANGE = (0.1, 10.0)


@dataclass
class BlockWeights:
    """Per decoder layer: token weights [blocks, tokens] and channel weights [hidden], both with
    mean 1."""

    tokens: list[torch.Tensor]
    channels: list[torch.Tensor]


def _sampled_label_backward(
    hidden: torch.Tensor, head_weight: torch.Tensor, chunk: int, generator: torch.Generator
) -> None:
    """Backpropagate sum_t -log p(y_t), y_t drawn from p itself, from the final hidden state.

    Logits exist for one chunk of positions at a time, as in Glaze v1's distillation loss.
    """
    flat = hidden.reshape(-1, hidden.shape[-1])
    leaf = flat.detach().requires_grad_(True)
    for start in range(0, flat.shape[0], chunk):
        logits = F.linear(leaf[start : start + chunk], head_weight).float()
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            labels = torch.multinomial(probs, 1, generator=generator)
        loss = F.cross_entropy(logits, labels.squeeze(-1), reduction="sum")
        loss.backward()
    flat.backward(leaf.grad)


def fisher_pass(
    text_model: nn.Module,
    head_weight: torch.Tensor,
    blocks: Sequence[Sequence[int]],
    linears: dict[str, nn.Linear],
    chunk_tokens: int = 512,
    seed: int = 42,
) -> tuple[dict[str, FisherFactors], BlockWeights]:
    """Sensitivities from one pass over `blocks` (one block per forward), parameters frozen."""
    device = head_weight.device
    inputs = {name: torch.zeros(m.in_features, device=device) for name, m in linears.items()}
    outputs = {name: torch.zeros(m.out_features, device=device) for name, m in linears.items()}
    layers = list(text_model.layers)
    hidden_size = text_model.config.hidden_size
    channels = [torch.zeros(hidden_size, device=device) for _ in layers]
    tokens = [torch.zeros(len(blocks), len(blocks[0])) for _ in layers]
    current = {"block": 0}
    handles = []

    def input_hook(name: str):
        def hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
            x = args[0].detach().float()
            inputs[name] += x.reshape(-1, x.shape[-1]).square().sum(0)

        return hook

    def output_hook(name: str):
        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            g = grad_output[0].detach().float()
            outputs[name] += g.reshape(-1, g.shape[-1]).square().sum(0)

        return hook

    def layer_hook(index: int):
        def hook(module: nn.Module, args: tuple, kwargs: dict, output: torch.Tensor):
            def grad_hook(grad: torch.Tensor) -> None:
                g2 = grad.detach().float().square()
                channels[index] += g2.reshape(-1, g2.shape[-1]).sum(0)
                tokens[index][current["block"]] = g2.sum(-1).reshape(-1).cpu()

            if output.requires_grad:
                output.register_hook(grad_hook)

        return hook

    for name, module in linears.items():
        handles.append(module.register_forward_hook(input_hook(name)))
        handles.append(module.register_full_backward_hook(output_hook(name)))
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(layer_hook(index), with_kwargs=True))

    # Sampled where the logits are: copying a chunk's probabilities to the CPU costs ~0.5 GB.
    generator = torch.Generator(device=device).manual_seed(seed)
    count = 0
    try:
        for block_index, block in enumerate(blocks):
            current["block"] = block_index
            ids = torch.tensor([list(block)], device=device)
            embeds = text_model.get_input_embeddings()(ids).detach().requires_grad_(True)
            hidden = text_model(inputs_embeds=embeds, use_cache=False).last_hidden_state
            _sampled_label_backward(hidden, head_weight, chunk_tokens, generator)
            count += ids.numel()
    finally:
        for handle in handles:
            handle.remove()

    factors = {
        name: FisherFactors(
            output=(outputs[name] / count).cpu(), input=(inputs[name] / count).cpu()
        )
        for name in linears
    }
    low, high = TOKEN_WEIGHT_RANGE
    weights = BlockWeights(
        tokens=[(t / t.mean().clamp_min(1e-30)).clamp(low, high) for t in tokens],
        channels=[(c / c.mean().clamp_min(1e-30)).cpu() for c in channels],
    )
    return factors, weights
