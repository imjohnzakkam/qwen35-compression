"""A tiny Qwen3.5 text model and a compressed-tensors export of it, for Glaze's CPU tests.

Three Gated DeltaNet layers and one full-attention layer, with the same module names, packed INT4
format and export layout as Qwen3.5-4B's AutoRound export, at sizes that run in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from qwen35_compression.glaze.quant_linear import dequantize_groups, pack_int4, unpack_int4
from qwen35_compression.glaze.student import LANGUAGE_MODEL_PREFIX

GROUP = 32
BLOCK = 16
VOCAB = 96


def tiny_config():
    from transformers import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        # Wider than the default 0.02, so logits are peaked and 4-bit error shows in the KL.
        initializer_range=0.1,
    )


def tiny_teacher(seed: int = 0) -> nn.Module:
    """A frozen BF16 text model with non-trivial norm weights."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(seed)
    model = Qwen3_5TextModel(tiny_config()).to(torch.bfloat16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.add_(0.1 * torch.randn_like(parameter))
    model.requires_grad_(False)
    model.eval()
    return model


def rtn_quantize(weight: torch.Tensor, group: int = GROUP) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric round-to-nearest INT4 with BF16 group scales."""
    rows, columns = weight.shape
    grouped = weight.float().view(rows, columns // group, group)
    scales = (grouped.abs().amax(dim=-1) / 7).clamp(min=1e-6).to(torch.bfloat16)
    codes = torch.round(grouped / scales.float().unsqueeze(-1)).clamp(-8, 7)
    return codes.to(torch.int8).view(rows, columns), scales


def quantized_names(teacher: nn.Module) -> list[str]:
    return [name for name, module in teacher.named_modules() if isinstance(module, nn.Linear)]


def write_tiny_export(teacher: nn.Module, directory: Path, group: int = GROUP) -> list[str]:
    """Save `teacher` as a pack-quantized export, laid out like the real one."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    names = quantized_names(teacher)
    tensors: dict[str, torch.Tensor] = {}
    for name in names:
        weight = teacher.get_submodule(name).weight.detach()
        codes, scales = rtn_quantize(weight, group)
        tensors[f"{LANGUAGE_MODEL_PREFIX}{name}.weight_packed"] = pack_int4(codes)
        tensors[f"{LANGUAGE_MODEL_PREFIX}{name}.weight_scale"] = scales
        tensors[f"{LANGUAGE_MODEL_PREFIX}{name}.weight_shape"] = torch.tensor(list(weight.shape))
    quantized_weights = {f"{name}.weight" for name in names}
    for name, parameter in teacher.named_parameters():
        if name not in quantized_weights:
            tensors[f"{LANGUAGE_MODEL_PREFIX}{name}"] = parameter.detach().clone().contiguous()
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": group,
                    },
                    "input_activations": None,
                    "output_activations": None,
                }
            },
            "ignore": ["lm_head"],
        },
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (directory / "tokenizer.json").write_text('{"version": "tiny"}', encoding="utf-8")
    (directory / "recipe.yaml").write_text("tiny: true\n", encoding="utf-8")
    return names


def dequantized_reference(teacher: nn.Module, directory: Path) -> nn.Module:
    """The teacher with each Linear weight replaced by the export's dequantized weight."""
    import copy

    from safetensors.torch import load_file

    tensors = load_file(str(directory / "model.safetensors"))
    reference = copy.deepcopy(teacher)
    with torch.no_grad():
        for name in quantized_names(teacher):
            key = f"{LANGUAGE_MODEL_PREFIX}{name}"
            shape = tuple(tensors[key + ".weight_shape"].tolist())
            codes = unpack_int4(tensors[key + ".weight_packed"], shape)
            weight = dequantize_groups(codes, tensors[key + ".weight_scale"], GROUP)
            reference.get_submodule(name).weight.copy_(weight)
    return reference


def random_blocks(count: int, seed: int = 0, length: int = BLOCK) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (count, length), generator=generator).tolist()
