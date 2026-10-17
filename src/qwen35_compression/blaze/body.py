"""The transformer body: which checkpoint tensors are its Linear layers, and in-place swaps that
fake-quantize one unit at a time so its price (the KL it adds) can be measured."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from torch import nn

from qwen35_compression.glaze2.quant import Option, fake_quantize

TEXT_PREFIX = "model.language_model."
# A decoder layer's Linear weight in a Qwen3.5 checkpoint (2-D; norms, conv1d, A_log are not).
BODY_WEIGHT = re.compile(r"^model\.language_model\.layers\.\d+\.[\w.]+\.weight$")
FFN_LINEAR = re.compile(r"^layers\.(\d+)\.mlp\.\w+$")
# A decoder layer's token mixer (DeltaNet or attention) or its FFN, as one unit.
PART_LINEAR = re.compile(r"^layers\.(\d+)\.(linear_attn|self_attn|mlp)\.\w+$")


def is_body_linear(name: str, shape: Sequence[int]) -> bool:
    return len(shape) == 2 and bool(BODY_WEIGHT.match(name))


def ffn_layer_units(model: nn.Module) -> dict[str, list[str]]:
    """Each decoder layer's feed-forward Linears (gate, up and down) as one unit, `layers.N.mlp`,
    in layer order."""
    units: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        match = FFN_LINEAR.match(name)
        if match and isinstance(module, nn.Linear):
            units.setdefault(f"layers.{match.group(1)}.mlp", []).append(name)
    return dict(sorted(units.items(), key=lambda item: int(item[0].split(".")[1])))


def layer_part_units(model: nn.Module) -> dict[str, list[str]]:
    """Each decoder layer's token-mixer Linears as `layers.N.mixer` and its FFN Linears as
    `layers.N.mlp`, in layer order. Units never split a projection vLLM fuses (qkv, in_proj_qkvz,
    in_proj_ba, gate_up), so each can take one precision in a real export."""
    units: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        match = PART_LINEAR.match(name)
        if match and isinstance(module, nn.Linear):
            part = "mlp" if match.group(2) == "mlp" else "mixer"
            units.setdefault(f"layers.{match.group(1)}.{part}", []).append(name)

    def key(item: tuple[str, list[str]]) -> tuple[int, int]:
        _, index, part = item[0].split(".")
        return int(index), part == "mlp"

    return dict(sorted(units.items(), key=key))


def unit_size(model: nn.Module, names: Sequence[str]) -> int:
    return sum(model.get_submodule(name).weight.numel() for name in names)


@contextmanager
def swapped(model: nn.Module, names: Sequence[str], option: Option) -> Iterator[None]:
    """Inside the block the named Linears hold round-to-nearest weights at `option`; afterwards
    they hold exactly their original values again."""
    saved: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for name in names:
                weight = model.get_submodule(name).weight
                saved[name] = weight.detach().clone()
                weight.copy_(fake_quantize(weight, option))
        yield
    finally:
        with torch.no_grad():
            for name, original in saved.items():
                model.get_submodule(name).weight.copy_(original)
