from __future__ import annotations

from typing import Any

from qwen35_compression.config import VariantConfig


def _scheme(name: str, group_size: int | None = None) -> Any:
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme(name, ["Linear"])
    if group_size is not None:
        if scheme.weights is None:
            raise ValueError(f"scheme {name} has no weight quantization")
        scheme.weights.group_size = group_size
    return scheme


def build_recipe(variant: VariantConfig) -> list[Any]:
    if variant.method == "bf16":
        return []

    if variant.method == "gptq":
        from llmcompressor.modifiers.gptq import GPTQModifier

        return [
            GPTQModifier(
                config_groups={"group_0": _scheme(variant.scheme or "W4A16", variant.group_size)},
                ignore=list(variant.ignore),
            )
        ]

    if variant.method == "awq":
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.transform.awq import AWQModifier

        return [
            AWQModifier(duo_scaling="both"),
            QuantizationModifier(
                config_groups={
                    "group_0": _scheme(variant.scheme or "W4A16_ASYM", variant.group_size)
                },
                ignore=list(variant.ignore),
            ),
        ]

    if variant.method == "int8":
        from llmcompressor.modifiers.gptq import GPTQModifier
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

        return [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(
                config_groups={"group_0": _scheme(variant.scheme or "W8A8")},
                ignore=list(variant.ignore),
            ),
        ]

    raise ValueError(f"unsupported quantization method: {variant.method}")
