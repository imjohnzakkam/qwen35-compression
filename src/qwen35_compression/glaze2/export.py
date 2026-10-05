"""Stage 5: write a compressed-tensors export stock vLLM serves, and count its bytes.

Tensors are copied from the BF16 checkpoint except: language-model Linears Glaze v2 quantized
(packed codes, BF16 scales, shape), vision-tower Linears quantized to 8 bits by round-to-nearest,
and the multi-token-prediction layer, which the exports of every method leave out. Each storage
option becomes one config group whose targets are the exact module names.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from qwen35_compression.glaze2.quant import Option, fake_quantize, quantize
from qwen35_compression.glaze2.rounding import QuantizedLinear

LANGUAGE_PREFIX = "model.language_model."
VISION_PREFIX = "model.visual."
SKIP_PREFIXES = ("mtp.",)
# The vision tower's Linear layers, by their weight names under model.visual.
VISION_LINEAR = re.compile(
    r"^(blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.linear_fc1|mlp\.linear_fc2)"
    r"|merger\.linear_fc[12]|deepstack_merger_list\.\d+\.linear_fc[12])\.weight$"
)
VISION_OPTION = Option(8, 128)
# Small files copied as they are: tokenizer, processor and generation settings.
COPIED_SUFFIXES = (".json", ".jinja", ".txt", ".model")
SKIPPED_FILES = {"config.json", "model.safetensors.index.json"}


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

    return pack_to_int32(codes.to(torch.int8), bits)


def unpack(packed: torch.Tensor, bits: int, shape: tuple[int, int]) -> torch.Tensor:
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    return unpack_from_int32(packed, bits, torch.Size(shape))


def quantized_tensors(name: str, item: QuantizedLinear) -> dict[str, torch.Tensor]:
    rows, columns = item.codes.shape
    return {
        f"{name}.weight_packed": pack(item.codes, item.option.bits),
        f"{name}.weight_scale": item.scales.to(torch.bfloat16).contiguous(),
        f"{name}.weight_shape": torch.tensor([rows, columns]),
    }


def vision_linear(weight_name: str) -> bool:
    if not weight_name.startswith(VISION_PREFIX):
        return False
    return bool(VISION_LINEAR.match(weight_name.removeprefix(VISION_PREFIX)))


def rtn_quantized(weight: torch.Tensor, option: Option) -> QuantizedLinear:
    codes, scales = quantize(weight.float(), option)
    return QuantizedLinear(codes.to(torch.int8), scales.to(torch.bfloat16), option)


def _safetensor_files(source: Path) -> list[Path]:
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {source}")
    return files


@dataclass
class ByteBudget:
    """Bytes of the export before the language model's choices: everything copied, the vision
    tower at 8 bits, and every quantized language-model Linear at 4-bit g128."""

    fixed: int
    language_base: int

    def allocation_budget(self, target: int, margin: int) -> int:
        return target - self.fixed - self.language_base - margin


def plan_bytes(source: Path, language_linears: dict[str, tuple[int, int]]) -> ByteBudget:
    """`language_linears`: module names relative to the text model -> (out, in)."""
    from safetensors import safe_open

    from qwen35_compression.glaze2.quant import BASE

    weight_names = {f"{LANGUAGE_PREFIX}{n}.weight" for n in language_linears}
    fixed = 0
    for path in _safetensor_files(source):
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(SKIP_PREFIXES) or key in weight_names:
                    continue
                piece = handle.get_slice(key)
                shape = piece.get_shape()
                numel = 1
                for d in shape:
                    numel *= d
                if vision_linear(key) and shape[-1] % VISION_OPTION.group == 0:
                    fixed += VISION_OPTION.tensor_bytes(shape[0], shape[1]) + 16
                else:
                    fixed += numel * _itemsize(piece.get_dtype())
    for path in source.iterdir():
        if path.suffix in COPIED_SUFFIXES and path.name not in SKIPPED_FILES:
            fixed += path.stat().st_size
    base = sum(BASE.tensor_bytes(o, i) + 16 for o, i in language_linears.values())
    return ByteBudget(fixed=fixed, language_base=base)


def _itemsize(dtype: str) -> int:
    return {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1}[dtype]


def quantization_config(groups: dict[Option, list[str]]) -> dict[str, Any]:
    config_groups = {}
    for index, (option, targets) in enumerate(sorted(groups.items())):
        config_groups[f"group_{index}"] = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": sorted(targets),
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": option.group,
                "num_bits": option.bits,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "group",
                "symmetric": True,
                "type": "int",
            },
        }
    return {
        "config_groups": config_groups,
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
    }


def write_export(
    source: Path,
    output: Path,
    language: dict[str, QuantizedLinear],
    quantize_vision: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Write the export to `output` (which must not exist or be empty); return its record.

    Without `quantize_vision` the vision tower stays BF16, as in every other method's export.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output}")
    output.mkdir(parents=True, exist_ok=True)
    weight_names = {f"{LANGUAGE_PREFIX}{name}.weight": name for name in language}
    tensors: dict[str, torch.Tensor] = {}
    groups: dict[Option, list[str]] = {}
    vision_count = 0
    for path in _safetensor_files(source):
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(SKIP_PREFIXES):
                    continue
                module = key.removesuffix(".weight")
                if key in weight_names:
                    item = language[weight_names[key]]
                    tensors.update(quantized_tensors(module, item))
                    groups.setdefault(item.option, []).append(module)
                    continue
                tensor = handle.get_tensor(key)
                if (
                    quantize_vision
                    and vision_linear(key)
                    and tensor.shape[-1] % VISION_OPTION.group == 0
                ):
                    item = rtn_quantized(tensor, VISION_OPTION)
                    tensors.update(quantized_tensors(module, item))
                    groups.setdefault(VISION_OPTION, []).append(module)
                    vision_count += 1
                    continue
                tensors[key] = tensor.contiguous()
    missing = set(weight_names) - {f"{m}.weight" for ms in groups.values() for m in ms}
    if missing:
        raise ValueError(f"quantized layers missing from the checkpoint: {sorted(missing)[:5]}")
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = quantization_config(groups)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for path in source.iterdir():
        if path.suffix in COPIED_SUFFIXES and path.name not in SKIPPED_FILES:
            shutil.copyfile(path, output / path.name)
    total = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    record = {
        "total_bytes": total,
        "groups": {o.name: len(names) for o, names in sorted(groups.items())},
        "vision_linears": vision_count,
    }
    log(f"glaze2 export: {total:,} bytes, groups {record['groups']}")
    return record


def dequantized(item: QuantizedLinear) -> torch.Tensor:
    from qwen35_compression.glaze2.quant import dequantize

    return dequantize(item.codes.float(), item.scales, item.option)


__all__ = ["fake_quantize", "plan_bytes", "write_export", "unpack", "pack"]
