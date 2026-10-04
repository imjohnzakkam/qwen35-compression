"""Stage 4: sensitivity-weighted block reconstruction with learned rounding.

Decoder layers are quantized in order. Layer b sees the outputs of the already quantized layers
(the quantized chain) and is fitted to BF16's own layer output on BF16's inputs (the BF16 chain),
with each token and channel's squared error weighted by the Fisher pass. The rounding offsets and
clip factors move by signed gradient descent, as in AutoRound, and the state with the lowest loss
on held-out blocks is kept.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from qwen35_compression.glaze2.allocate import Allocation
from qwen35_compression.glaze2.fisher import BlockWeights
from qwen35_compression.glaze2.quant import LearnedQuantLinear, Option


@dataclass(frozen=True)
class RoundingSettings:
    blocks_per_batch: int = 8
    max_iters: int = 400
    min_iters: int = 50
    eval_every: int = 25
    lr: float = 5e-3
    seed: int = 42


@dataclass
class QuantizedLinear:
    codes: torch.Tensor  # int8, [out, in]
    scales: torch.Tensor  # BF16, [out, in / group]
    option: Option


@dataclass
class LayerRecord:
    index: int
    iterations: int
    best_iteration: int
    dev_loss_rtn: float
    dev_loss_best: float
    history: list[tuple[int, float]] = field(default_factory=list)


class StopForward(Exception):
    pass


def capture_inputs(text_model: nn.Module, ids: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
    """Layer 0's input hidden state for `ids`, and the keyword arguments every decoder layer is
    called with (rotary embeddings, masks), captured from one forward pass."""
    layers = list(text_model.layers)
    captured: list[dict] = [{} for _ in layers]
    first: dict[str, torch.Tensor] = {}
    handles = []

    def hook(index: int):
        def pre_hook(module: nn.Module, args: tuple, kwargs: dict):
            hidden = args[0] if args else kwargs["hidden_states"]
            if index == 0:
                first["hidden"] = hidden.detach()
            captured[index] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            if index == len(layers) - 1:
                raise StopForward

        return pre_hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(hook(index), with_kwargs=True))
    try:
        with torch.no_grad():
            text_model(input_ids=ids, use_cache=False)
    except StopForward:
        pass
    finally:
        for handle in handles:
            handle.remove()
    return first["hidden"], captured


def _quantizable(layer: nn.Module, prefix: str, names: set[str]) -> dict[str, str]:
    """Submodule path inside the layer -> full name, for the layer's Linears being quantized."""
    found = {}
    for local, module in layer.named_modules():
        full = f"{prefix}.{local}" if local else prefix
        if isinstance(module, nn.Linear) and full in names:
            found[local] = full
    return found


def _swap(layer: nn.Module, local: str, module: nn.Module) -> None:
    parent_name, _, child = local.rpartition(".")
    parent = layer.get_submodule(parent_name) if parent_name else layer
    setattr(parent, child, module)


def _weighted_loss(
    output: torch.Tensor, target: torch.Tensor, tokens: torch.Tensor, channels: torch.Tensor
) -> torch.Tensor:
    error = (output.float() - target.float()).square()
    return (error * tokens.unsqueeze(-1) * channels).mean()


def quantize_layers(
    text_model: nn.Module,
    calibration: Sequence[Sequence[int]],
    held_out: Sequence[Sequence[int]],
    allocation: Allocation,
    linear_names: Sequence[str],
    weights: BlockWeights,
    settings: RoundingSettings,
    prefix: str = "layers",
    log: Callable[[str], None] = print,
) -> tuple[dict[str, QuantizedLinear], list[LayerRecord]]:
    """Quantize every decoder layer's chosen Linears; the model itself is left unchanged.

    `linear_names` are paths relative to `text_model` (e.g. `layers.3.mlp.up_proj`); `weights`
    carries one token row per calibration block, in order.
    """
    device = next(text_model.parameters()).device
    size = settings.blocks_per_batch
    if len(calibration) % size or len(held_out) % size:
        raise ValueError(f"block counts must be multiples of {size}")
    names = set(linear_names)
    generator = torch.Generator().manual_seed(settings.seed)

    def chains(blocks: Sequence[Sequence[int]]) -> tuple[list[torch.Tensor], list[dict]]:
        states, kwargs = [], []
        for start in range(0, len(blocks), size):
            ids = torch.tensor([list(b) for b in blocks[start : start + size]], device=device)
            hidden, captured = capture_inputs(text_model, ids)
            states.append(hidden.cpu())
            kwargs = captured
        return states, kwargs

    cal_fp, layer_kwargs = chains(calibration)
    dev_fp, _ = chains(held_out)
    cal_q = [s.clone() for s in cal_fp]
    dev_q = [s.clone() for s in dev_fp]
    dev_tokens = torch.ones(size, len(held_out[0]))

    results: dict[str, QuantizedLinear] = {}
    records: list[LayerRecord] = []
    for index, layer in enumerate(text_model.layers):
        kwargs = layer_kwargs[index]
        local_names = _quantizable(layer, f"{prefix}.{index}", names)
        channels = weights.channels[index].to(device)
        token_rows = weights.tokens[index]

        def run(module: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
            return module(hidden.to(device), **kwargs)

        with torch.no_grad():
            cal_target = [run(layer, x).cpu() for x in cal_fp]
            dev_target = [run(layer, x).cpu() for x in dev_fp]
        student = copy.deepcopy(layer)
        learned: dict[str, LearnedQuantLinear] = {}
        for local, full in local_names.items():
            module = LearnedQuantLinear(layer.get_submodule(local), allocation.option_of(full))
            _swap(student, local, module)
            learned[local] = module
        params = [p for m in learned.values() for p in (m.offsets, m.alpha)]

        def dev_loss(module: nn.Module) -> float:
            with torch.no_grad():
                losses = [
                    _weighted_loss(
                        run(module, x), y.to(device), dev_tokens.to(device), channels
                    ).item()
                    for x, y in zip(dev_q, dev_target, strict=True)
                ]
            return sum(losses) / len(losses)

        def snapshot(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
            return [p.detach().clone() for p in tensors]

        rtn = dev_loss(student) if params else 0.0
        best, best_iter, best_state = rtn, 0, snapshot(params)
        history = [(0, rtn)]
        iterations = 0
        if params:
            batches = len(cal_q)
            for step in range(1, settings.max_iters + 1):
                pick = int(torch.randint(batches, (1,), generator=generator))
                rows = token_rows[pick * size : (pick + 1) * size].to(device)
                loss = _weighted_loss(
                    run(student, cal_q[pick]), cal_target[pick].to(device), rows, channels
                )
                for p in params:
                    p.grad = None
                loss.backward()
                lr = settings.lr * (1 - (step - 1) / settings.max_iters)
                with torch.no_grad():
                    for p in params:
                        if p.grad is not None:
                            p.sub_(lr * torch.sign(p.grad))
                for m in learned.values():
                    m.clamp_()
                iterations = step
                if step % settings.eval_every == 0 or step == settings.max_iters:
                    current = dev_loss(student)
                    history.append((step, current))
                    if current < best:
                        best, best_iter, best_state = current, step, snapshot(params)
                    elif step >= settings.min_iters and step - best_iter >= 4 * settings.eval_every:
                        break  # held-out loss stopped improving
            with torch.no_grad():
                for p, saved in zip(params, best_state, strict=True):
                    p.copy_(saved)
        for local, module in learned.items():
            codes, scales = module.export()
            results[local_names[local]] = QuantizedLinear(codes.cpu(), scales.cpu(), module.option)
        with torch.no_grad():
            cal_q = [run(student, x).cpu() for x in cal_q]
            dev_q = [run(student, x).cpu() for x in dev_q]
        cal_fp, dev_fp = cal_target, dev_target
        record = LayerRecord(index, iterations, best_iter, rtn, best, history)
        records.append(record)
        drop = 0.0 if not math.isfinite(rtn) or rtn == 0 else 1 - best / rtn
        log(
            f"glaze2 layer {index}: {len(learned)} linears, held-out loss {rtn:.3e} -> "
            f"{best:.3e} ({drop:.1%}) at iteration {best_iter}/{iterations}"
        )
        del student, learned, params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results, records
