"""Real Blaze exports: the stored format itself, served by vLLM with the `blaze-vllm` plugin.

The body's Linears are compressed-tensors pack-quantized groups, as any W4A16/W8A16 export, so
vLLM's Marlin kernels run them. The tied vocabulary table is stored as two quantized blocks in
usage order plus an index map:

- `embed_tokens.hot.*`: the busiest rows, at the ladder's upper rung (8-bit g128);
- `embed_tokens.cold.*`: every other row, at the lower rung (4-bit g128);
- `embed_tokens.position`: for each token id, its row in hot-then-cold order.

Each block is a compressed-tensors tensor triple (`weight_packed`, `weight_scale`,
`weight_shape`) with its own config group, so vLLM's quantized embedding (a dequantizing gather)
and quantized LM head (Marlin) load them as they are. The stored values are exactly the
fake-quantized export's: the same codes and BF16 scales.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from qwen35_compression.blaze.body import is_body_linear
from qwen35_compression.blaze.export import HEAD, TIED_TABLE, _shards, _tensor_names
from qwen35_compression.blaze.ladder import LadderPlan
from qwen35_compression.glaze2.export import (
    COPIED_SUFFIXES,
    SKIP_PREFIXES,
    SKIPPED_FILES,
    quantization_config,
    quantized_tensors,
    rtn_quantized,
    unpack,
)
from qwen35_compression.glaze2.quant import Option, dequantize
from qwen35_compression.glaze2.rounding import QuantizedLinear

ARCHITECTURE_KEY = "blaze_vocab"
TABLE_MODULE = TIED_TABLE.removesuffix(".weight")
HOT = f"{TABLE_MODULE}.hot"
COLD = f"{TABLE_MODULE}.cold"
POSITION = f"{TABLE_MODULE}.position"
# vLLM pads vocabulary blocks to multiples of 64 rows; whole blocks need no padding.
ROW_MULTIPLE = 64


def table_parts(table: torch.Tensor, plan: LadderPlan) -> dict[str, QuantizedLinear]:
    """The two quantized blocks of a laddered table, rows in usage order."""
    order = torch.tensor(plan.order, dtype=torch.long, device=table.device)
    ordered = table.index_select(0, order)
    return {
        HOT: rtn_quantized(ordered[: plan.hot_rows], plan.hot_option),
        COLD: rtn_quantized(ordered[plan.hot_rows :], plan.rest_option),
    }


def positions(plan: LadderPlan) -> torch.Tensor:
    """Token id -> its row in hot-then-cold order."""
    position = torch.empty(plan.vocab_size, dtype=torch.int32)
    position[torch.tensor(plan.order, dtype=torch.long)] = torch.arange(
        plan.vocab_size, dtype=torch.int32
    )
    return position


def write_real_export(
    source: Path,
    output: Path,
    ladder: LadderPlan,
    body: Option | Mapping[str, Option],
    device: str = "cpu",
    log: Callable[[str], None] = print,
    row_multiple: int = ROW_MULTIPLE,
) -> dict[str, Any]:
    """Write the real export of `source`: the laddered tied table and every body Linear at
    `body` (one option, or checkpoint key -> option), rounded to nearest. The MTP head is left
    out, as every quantized export leaves it."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output}")
    if HEAD in _tensor_names(source):
        raise ValueError("real exports support tied vocabulary tables only (0.8B-4B)")
    if ladder.hot_rows % row_multiple or ladder.vocab_size % row_multiple:
        raise ValueError(f"vocabulary blocks must be multiples of {row_multiple} rows")
    output.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    groups: dict[Option, list[str]] = {}
    hidden = 0
    for path in _shards(source):
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(SKIP_PREFIXES):
                    continue
                tensor = handle.get_tensor(key)
                if key == TIED_TABLE:
                    hidden = tensor.shape[1]
                    for module, item in table_parts(tensor.to(device), ladder).items():
                        item = QuantizedLinear(item.codes.cpu(), item.scales.cpu(), item.option)
                        tensors.update(quantized_tensors(module, item))
                    tensors[POSITION] = positions(ladder)
                    continue
                if is_body_linear(key, tensor.shape):
                    option = body if isinstance(body, Option) else body[key]
                    item = rtn_quantized(tensor.to(device), option)
                    item = QuantizedLinear(item.codes.cpu(), item.scales.cpu(), option)
                    module = key.removesuffix(".weight")
                    tensors.update(quantized_tensors(module, item))
                    groups.setdefault(option, []).append(module)
                    continue
                tensors[key] = tensor.contiguous()
    if not hidden:
        raise ValueError(f"no vocabulary table ({TIED_TABLE}) in {source}")
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    quant = quantization_config(groups)
    for name, module, option in (
        ("blaze_hot", HOT, ladder.hot_option),
        ("blaze_cold", COLD, ladder.rest_option),
    ):
        group = json.loads(json.dumps(next(iter(quant["config_groups"].values()))))
        group["targets"] = [f"re:.*{module.split('.')[-2]}\\.{module.split('.')[-1]}$"]
        group["weights"]["num_bits"] = option.bits
        group["weights"]["group_size"] = option.group
        quant["config_groups"][name] = group
    config["quantization_config"] = quant
    config[ARCHITECTURE_KEY] = {
        "hot_rows": ladder.hot_rows,
        "hot_bits": ladder.hot_option.bits,
        "cold_bits": ladder.rest_option.bits,
        "group_size": ladder.hot_option.group,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for path in source.iterdir():
        if path.suffix in COPIED_SUFFIXES and path.name not in SKIPPED_FILES:
            shutil.copyfile(path, output / path.name)
    total = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    record = {
        "total_bytes": total,
        "safetensors_bytes": (output / "model.safetensors").stat().st_size,
        "hot_rows": ladder.hot_rows,
        "body_groups": {o.name: len(names) for o, names in sorted(groups.items())},
    }
    log(f"blaze real export: {total:,} bytes, body groups {record['body_groups']}")
    return record


def read_table(export: Path) -> torch.Tensor:
    """The vocabulary table a real export stores, dequantized back to token-id order (BF16)."""
    from safetensors.torch import load_file

    tensors = load_file(str(export / "model.safetensors"))
    config = json.loads((export / "config.json").read_text(encoding="utf-8"))[ARCHITECTURE_KEY]
    parts = []
    for module, bits in ((HOT, config["hot_bits"]), (COLD, config["cold_bits"])):
        shape = tuple(int(v) for v in tensors[f"{module}.weight_shape"])
        codes = unpack(tensors[f"{module}.weight_packed"], bits, shape)
        option = Option(bits, config["group_size"])
        parts.append(dequantize(codes.float(), tensors[f"{module}.weight_scale"], option))
    ordered = torch.cat(parts)
    return ordered.index_select(0, tensors[POSITION].long())
