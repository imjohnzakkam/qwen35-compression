"""The Glaze student: the BF16 text model with its quantized layers taken from an export."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize

from qwen35_compression.glaze.quant_linear import GroupQuantLinear, unpack_int4

# Where Qwen3_5ForConditionalGeneration keeps the text model's tensors in a saved export.
LANGUAGE_MODEL_PREFIX = "model.language_model."
# RMSNorm modules whose weights Glaze trains, matched by class name.
NORM_CLASSES = frozenset({"Qwen3_5RMSNorm", "Qwen3_5RMSNormGated"})
QUANTIZATION_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_shape")


@dataclass
class InitExport:
    """The language-model tensors of a symmetric INT4 group-quantized compressed-tensors export."""

    directory: Path
    group_size: int
    # Module paths inside the text model, e.g. "layers.0.mlp.gate_proj".
    quantized: tuple[str, ...]
    # Tensors without the language-model prefix; the embedding is left out (the teacher's is used).
    tensors: dict[str, torch.Tensor]


def init_group_size(config_json: dict[str, Any]) -> int:
    """Group size of a W4A16 symmetric pack-quantized export; anything else is rejected."""
    quantization = config_json.get("quantization_config") or {}
    if (
        quantization.get("quant_method") != "compressed-tensors"
        or quantization.get("format") != "pack-quantized"
    ):
        raise ValueError("Glaze needs a pack-quantized compressed-tensors export")
    groups = quantization.get("config_groups") or {}
    if len(groups) != 1:
        raise ValueError(f"Glaze needs exactly one quantization group, found {len(groups)}")
    (group,) = groups.values()
    weights = group.get("weights") or {}
    if (
        weights.get("num_bits") != 4
        or weights.get("type") != "int"
        or weights.get("symmetric") is not True
        or weights.get("strategy") != "group"
        or not weights.get("group_size")
    ):
        raise ValueError("Glaze needs symmetric INT4 group-quantized weights")
    if group.get("input_activations") is not None:
        raise ValueError("Glaze needs weight-only quantization (W4A16)")
    return int(weights["group_size"])


def read_init_export(directory: Path, prefix: str = LANGUAGE_MODEL_PREFIX) -> InitExport:
    from safetensors import safe_open

    config_json = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    group_size = init_group_size(config_json)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {directory}")
    tensors: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(prefix) and not key.endswith("embed_tokens.weight"):
                    tensors[key.removeprefix(prefix)] = handle.get_tensor(key)
    quantized = tuple(
        sorted(
            key.removesuffix(".weight_packed") for key in tensors if key.endswith(".weight_packed")
        )
    )
    if not quantized:
        raise ValueError(f"no quantized language-model layers under {prefix!r} in {directory}")
    for name in quantized:
        for suffix in (".weight_scale", ".weight_shape"):
            if name + suffix not in tensors:
                raise ValueError(f"init export has {name}.weight_packed but no {name}{suffix}")
        if name + ".weight" in tensors:
            raise ValueError(f"init export has both packed and dense weights for {name}")
    return InitExport(directory, group_size, quantized, tensors)


def layer_shape(init: InitExport, name: str) -> tuple[int, int]:
    rows, columns = (int(value) for value in init.tensors[name + ".weight_shape"].tolist())
    return rows, columns


class StorageCast(nn.Module):
    """Parametrization: an fp32 master weight, seen by the module in its storage dtype."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, master: torch.Tensor) -> torch.Tensor:
        return master.to(self.dtype)


def build_student(teacher: nn.Module, init: InitExport, train_norms: bool = True) -> nn.Module:
    """Copy the teacher's text model, swapping each quantized Linear for its export's codes.

    The embedding is shared with the teacher (it is not quantized). Every other parameter is
    loaded from the init export, so with all log-scales at zero the student computes exactly what
    the init export does. Only the log-scales, and with `train_norms` the RMSNorm weights, train.
    """
    modules = dict(teacher.named_modules())
    embedding = teacher.get_input_embeddings().weight
    shared: dict[int, Any] = {id(embedding): embedding}
    for name in init.quantized:
        layer = modules.get(name)
        if not isinstance(layer, nn.Linear) or layer.bias is not None:
            raise ValueError(f"{name} is not a bias-free Linear in the teacher")
        if tuple(layer.weight.shape) != layer_shape(init, name):
            raise ValueError(
                f"{name}: teacher weight {tuple(layer.weight.shape)} does not match the export's "
                f"{layer_shape(init, name)}"
            )
        # Shared rather than copied: these weights are replaced below and must not be duplicated.
        shared[id(layer.weight)] = layer.weight
    student = copy.deepcopy(teacher, memo=shared)

    device = embedding.device
    for name in init.quantized:
        parent_name, _, child = name.rpartition(".")
        parent = student.get_submodule(parent_name) if parent_name else student
        # Unpacked where the model lives: on a GPU this is much faster than on the CPU.
        packed = init.tensors[name + ".weight_packed"].to(device)
        codes = unpack_int4(packed, layer_shape(init, name))
        scales = init.tensors[name + ".weight_scale"].to(device)
        setattr(parent, child, GroupQuantLinear(codes, scales, init.group_size))

    expected = {key for key in init.tensors if not key.endswith(QUANTIZATION_SUFFIXES)}
    loaded = set()
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if parameter is embedding or name.endswith("log_scale"):
                continue
            source = init.tensors.get(name)
            if source is None:
                raise ValueError(f"init export has no tensor for the student's {name}")
            if source.shape != parameter.shape or source.dtype != parameter.dtype:
                raise ValueError(
                    f"{name}: export {source.dtype} {tuple(source.shape)} does not match the "
                    f"model's {parameter.dtype} {tuple(parameter.shape)}"
                )
            parameter.copy_(source.to(parameter.device))
            loaded.add(name)
    unused = sorted(expected - loaded)
    if unused:
        raise ValueError(f"init export tensors the student does not use: {unused[:5]}")

    student.requires_grad_(False)
    for module in student.modules():
        if isinstance(module, GroupQuantLinear):
            module.log_scale.requires_grad_(True)
    if train_norms:
        for module in student.modules():
            if type(module).__name__ in NORM_CLASSES:
                stored = module.weight
                module.weight = nn.Parameter(stored.detach().to(torch.float32))
                parametrize.register_parametrization(
                    module, "weight", StorageCast(stored.dtype), unsafe=True
                )
    return student


def trainable_state(student: nn.Module) -> dict[str, torch.Tensor]:
    """A CPU copy of every trainable parameter (log-scales and norm masters)."""
    return {
        name: parameter.detach().to("cpu", copy=True)
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(student: nn.Module, state: dict[str, torch.Tensor]) -> None:
    trainable = {name: p for name, p in student.named_parameters() if p.requires_grad}
    if set(trainable) != set(state):
        raise ValueError("saved state does not match the student's trainable parameters")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state[name].to(parameter.device))


def exported_tensors(student: nn.Module, prefix: str = LANGUAGE_MODEL_PREFIX) -> dict[str, Any]:
    """The tensors Glaze changes, named and typed as the export stores them."""
    tensors: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, module in student.named_modules():
            if isinstance(module, GroupQuantLinear):
                tensors[f"{prefix}{name}.weight_scale"] = module.stored_scales().cpu()
            elif parametrize.is_parametrized(module, "weight"):
                tensors[f"{prefix}{name}.weight"] = module.weight.cpu()
    return tensors
