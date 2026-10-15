"""GPTQ on Blaze's grid: the body's Linears rounded with second-order error feedback.

Each Linear's inputs on generic text give H = X^T X. Columns are rounded left to right on the
same symmetric group grid as round-to-nearest (`glaze2.quant`: BF16 scale = group max over
2^(k-1) - 1, codes times scale in BF16), and each column's rounding error is spread over the
columns not yet rounded through the inverse Hessian's Cholesky factor (Frantar et al., 2022).
A group's scale is set when its first column is reached, from the weights as updated so far.

Decoder layers are quantized in order, each on the outputs of the already quantized layers.
The calibration text is generic (WikiText-2's training split), not the model's own answers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import nn

from qwen35_compression.glaze2.quant import SCALE_DTYPE, Option, code_range, group_scales
from qwen35_compression.glaze2.rounding import capture_inputs

DAMP = 0.01


@torch.no_grad()
def gptq_weight(
    weight: torch.Tensor, hessian: torch.Tensor, option: Option, damp: float = DAMP
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ for one [out, in] weight: (BF16 weight as stored, int8 codes, BF16 scales)."""
    rows, columns = weight.shape
    group = option.group
    if columns % group:
        raise ValueError(f"{columns} inputs are not a multiple of group {group}")
    low, high = code_range(option.bits)
    work = weight.float().clone()
    h = hessian.float().clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    work[:, dead] = 0
    h += damp * torch.mean(torch.diag(h)) * torch.eye(columns, device=h.device)
    factor = torch.linalg.cholesky(h)
    inverse = torch.linalg.cholesky(torch.cholesky_inverse(factor), upper=True)

    codes = torch.zeros(rows, columns, dtype=torch.int8, device=work.device)
    scales = torch.zeros(rows, columns // group, dtype=SCALE_DTYPE, device=work.device)
    stored = torch.zeros(rows, columns, dtype=SCALE_DTYPE, device=work.device)
    # Blocks of one group: a group's scale comes from its columns as updated by every earlier
    # block, and the in-block updates stay inside the group.
    for start in range(0, columns, group):
        end = start + group
        block = work[:, start:end].clone()
        errors = torch.zeros_like(block)
        local = inverse[start:end, start:end]
        scale = group_scales(block, option)[:, 0]
        scales[:, start // group] = scale.to(SCALE_DTYPE)
        for i in range(group):
            column = block[:, i]
            code = torch.clamp(torch.round(column / scale), low, high)
            value = (code.to(SCALE_DTYPE) * scale.to(SCALE_DTYPE)).float()
            codes[:, start + i] = code.to(torch.int8)
            stored[:, start + i] = value.to(SCALE_DTYPE)
            error = (column - value) / local[i, i]
            block[:, i:] -= error.unsqueeze(1) * local[i, i:].unsqueeze(0)
            errors[:, i] = error
        work[:, end:] -= errors @ inverse[start:end, end:]
    return stored, codes, scales


class HessianCollector:
    """Accumulates X^T X over a Linear's inputs while registered."""

    def __init__(self, module: nn.Linear) -> None:
        columns = module.in_features
        self.hessian = torch.zeros(columns, columns, device=module.weight.device)
        self.tokens = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        self.hessian.addmm_(x.T, x)
        self.tokens += x.shape[0]

    def remove(self) -> torch.Tensor:
        self.handle.remove()
        if not self.tokens:
            raise RuntimeError("no inputs reached this Linear")
        return self.hessian / self.tokens


def _layer_output(layer: nn.Module, hidden: torch.Tensor, kwargs: dict) -> torch.Tensor:
    out = layer(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def gptq_body(
    text_model: nn.Module,
    blocks: Sequence[Sequence[int]],
    option: Option | Mapping[str, Option],
    batch: int = 8,
    layers: Sequence[int] | None = None,
    damp: float = DAMP,
    log: Callable[[str], None] = print,
) -> dict[str, torch.Tensor]:
    """GPTQ every Linear of the decoder layers in order, in place; return module path (relative
    to `text_model`, e.g. `layers.3.mlp.up_proj`) -> stored BF16 weight on the CPU.

    `option` is one option for every Linear, or module path -> option.
    `layers` limits the work to those layer indices (a pilot); every later layer still runs, on
    the quantized chain, only when it is quantized itself.
    """
    if len(blocks) % batch:
        raise ValueError(f"{len(blocks)} blocks are not a multiple of batch {batch}")
    device = next(text_model.parameters()).device
    hidden: list[torch.Tensor] = []
    kwargs_by_layer: list[dict] = []
    for start in range(0, len(blocks), batch):
        ids = torch.tensor([list(b) for b in blocks[start : start + batch]], device=device)
        first, captured = capture_inputs(text_model, ids)
        hidden.append(first.cpu())
        kwargs_by_layer = captured
    keep = set(range(len(text_model.layers))) if layers is None else set(layers)
    last = max(keep)
    results: dict[str, torch.Tensor] = {}
    for index, layer in enumerate(text_model.layers):
        if index > last:
            break
        kwargs = kwargs_by_layer[index]
        linears = {
            f"layers.{index}.{local}": module
            for local, module in layer.named_modules()
            if isinstance(module, nn.Linear)
        }
        if index in keep:
            collectors = {name: HessianCollector(module) for name, module in linears.items()}
            for x in hidden:
                _layer_output(layer, x.to(device), kwargs)
            worst = 0.0
            for name, module in linears.items():
                h = collectors[name].remove()
                original = module.weight.detach().float()
                chosen = option if isinstance(option, Option) else option[name]
                stored, _, _ = gptq_weight(original, h, chosen, damp)
                # Output error on the calibration inputs, relative to the weight's own output.
                delta = stored.float() - original
                ratio = float(((delta @ h) * delta).sum() / ((original @ h) * original).sum())
                worst = max(worst, ratio)
                module.weight.copy_(stored.to(module.weight.dtype))
                results[name] = stored.cpu()
                del h
            log(
                f"blaze gptq layer {index}: {len(linears)} linears, "
                f"worst relative output error {worst:.2e}"
            )
        hidden = [_layer_output(layer, x.to(device), kwargs).cpu() for x in hidden]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
