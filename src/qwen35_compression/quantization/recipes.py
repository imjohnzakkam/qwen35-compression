from __future__ import annotations

from typing import Any

from qwen35_compression.config import VariantConfig


def _scheme(name: str, targets: tuple[str, ...], group_size: int | None = None) -> Any:
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme(name, list(targets))
    if group_size is not None:
        if scheme.weights is None:
            raise ValueError(f"scheme {name} has no weight quantization")
        scheme.weights.group_size = group_size
    return scheme


def allocation_groups(variant: VariantConfig) -> dict[str, Any]:
    """Config groups for a per-Linear allocation: one symmetric int group scheme per option,
    targeting its exact module names (the same layout as Glaze v2's export)."""
    import json

    from qwen35_compression.glaze2.quant import OPTIONS

    assert variant.allocation is not None
    allocation = json.loads(variant.allocation.read_text(encoding="utf-8"))
    options = {option.name: option for option in OPTIONS}
    groups = {}
    for name, modules in sorted(allocation["options"].items()):
        if not modules:
            continue
        option = options[name]
        scheme = _scheme("W4A16", tuple(modules), option.group)
        scheme.weights.num_bits = option.bits
        groups[f"group_{name}"] = scheme
    return groups


def build_recipe(variant: VariantConfig) -> list[Any]:
    if variant.method == "bf16":
        return []

    if variant.method == "gptq":
        from llmcompressor.modifiers.gptq import GPTQModifier

        return [
            GPTQModifier(
                config_groups={
                    "group_0": _scheme(
                        variant.scheme or "W4A16", variant.targets, variant.group_size
                    )
                },
                ignore=list(variant.ignore),
            )
        ]

    if variant.method == "awq":
        import torch
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.transform.awq import AWQModifier

        return [
            # AWQ caches each layer's calibration inputs (512 x 2,048 tokens) for its scale search;
            # on the GPU that filled a 24 GB A30 by the 4B model's second layer. Offloading the
            # cache to the CPU changes where it lives, not the scales AWQ finds.
            AWQModifier(duo_scaling="both", offload_device=torch.device("cpu")),
            QuantizationModifier(
                config_groups={
                    "group_0": _scheme(
                        variant.scheme or "W4A16_ASYM", variant.targets, variant.group_size
                    )
                },
                ignore=list(variant.ignore),
            ),
        ]

    if variant.method == "autoround":
        from llmcompressor.modifiers.autoround import AutoRoundModifier

        return [
            # Tunes each block's weight rounding and clipping by signed gradient descent against
            # the BF16 block's outputs, with AutoRound's defaults (200 steps per block, batch 8;
            # a pilot sets fewer steps).
            # torch.compile is off: it only speeds up tuning and is untested on Qwen3.5's
            # linear-attention layers. It caches every packed calibration sample's block inputs
            # and BF16 outputs on the GPU, which needs more than a 24 GB card with this model
            # (DeltaNet runs in transformers' PyTorch fallback): run it on a 40 GB GPU.
            # With an allocation, each Linear is tuned at its own precision.
            AutoRoundModifier(
                config_groups=(
                    allocation_groups(variant)
                    if variant.allocation is not None
                    else {
                        "group_0": _scheme(
                            variant.scheme or "W4A16", variant.targets, variant.group_size
                        )
                    }
                ),
                ignore=list(variant.ignore),
                iters=variant.autoround_iters,
                enable_torch_compile=False,
            )
        ]

    if variant.method == "int8":
        from llmcompressor.modifiers.gptq import GPTQModifier
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

        return [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(
                config_groups={"group_0": _scheme(variant.scheme or "W8A8", variant.targets)},
                ignore=list(variant.ignore),
            ),
        ]

    if variant.method == "mixed":
        from llmcompressor.modifiers.gptq import GPTQModifier

        return [
            GPTQModifier(
                config_groups={
                    group.name: _scheme(group.scheme, group.targets, group.group_size)
                    for group in variant.groups
                },
                ignore=list(variant.ignore),
            )
        ]

    raise ValueError(f"unsupported quantization method: {variant.method}")
