"""The vocabulary ladder as an add-on to an existing compressed-tensors export.

Any 4-bit Qwen3.5 export (GPTQ, AutoRound, Glaze v2) stores its tied vocabulary table in BF16.
`ladder_export` rewrites only that table into Blaze's real tiered form (`blaze.real`): every body
tensor, the vision tower and the export's own quantization groups are copied as they are, so
the result isolates what the ladder costs and saves. With `upgrade`, named body Linears are
re-quantized from the BF16 checkpoint at a wider option (round-to-nearest) in a group of their
own, which vLLM matches by name ahead of a generic `Linear` target.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from qwen35_compression.blaze.export import TIED_TABLE, _shards
from qwen35_compression.blaze.ladder import LadderPlan
from qwen35_compression.blaze.real import (
    ARCHITECTURE_KEY,
    COLD,
    HOT,
    POSITION,
    ROW_MULTIPLE,
    positions,
    table_parts,
)
from qwen35_compression.glaze2.export import quantized_tensors, rtn_quantized
from qwen35_compression.glaze2.quant import Option
from qwen35_compression.glaze2.rounding import QuantizedLinear

PACKED_SUFFIXES = (
    ".weight_packed",
    ".weight_scale",
    ".weight_shape",
    ".weight_zero_point",
    ".weight_g_idx",
)
COPIED = (".json", ".jinja", ".txt", ".model")
SKIPPED = {"config.json", "model.safetensors.index.json", "compression_manifest.json"}
MANIFEST = "blaze_addon.json"


def _group(template: dict[str, Any], targets: list[str], option: Option) -> dict[str, Any]:
    group = copy.deepcopy(template)
    group["targets"] = targets
    group["weights"]["num_bits"] = option.bits
    group["weights"]["group_size"] = option.group
    group["weights"]["strategy"] = "group"
    group["weights"]["symmetric"] = True
    group["weights"]["actorder"] = None
    return group


def ladder_export(
    source: Path,
    output: Path,
    ladder: LadderPlan,
    bf16: Path | None = None,
    upgrade: Mapping[str, Option] | None = None,
    device: str = "cpu",
    log: Callable[[str], None] = print,
    row_multiple: int = ROW_MULTIPLE,
) -> dict[str, Any]:
    """Write `source` (a compressed-tensors export with a BF16 tied table) to `output` with the
    table laddered. `upgrade` maps body module names (`model.language_model.layers.N....`) to a
    wider option, quantized to nearest from the BF16 checkpoint `bf16`."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output}")
    if ladder.hot_rows % row_multiple or ladder.vocab_size % row_multiple:
        raise ValueError(f"vocabulary blocks must be multiples of {row_multiple} rows")
    upgrade = dict(upgrade or {})
    if upgrade and bf16 is None:
        raise ValueError("upgraded modules need the BF16 checkpoint")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config")
    if not quant or quant.get("quant_method") != "compressed-tensors":
        raise ValueError(f"{source} is not a compressed-tensors export")
    if ARCHITECTURE_KEY in config:
        raise ValueError(f"{source} is already laddered")

    output.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    dropped = 0
    table_found = False
    for path in _shards(source):
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                module = next((key[: -len(s)] for s in PACKED_SUFFIXES if key.endswith(s)), None)
                if module is not None and module in upgrade:
                    dropped += 1
                    continue
                if key == TIED_TABLE:
                    table = handle.get_tensor(key)
                    for name, item in table_parts(table.to(device), ladder).items():
                        item = QuantizedLinear(item.codes.cpu(), item.scales.cpu(), item.option)
                        tensors.update(quantized_tensors(name, item))
                    tensors[POSITION] = positions(ladder)
                    table_found = True
                    continue
                if key == "lm_head.weight":
                    raise ValueError("ladder add-ons support tied vocabulary tables only")
                tensors[key] = handle.get_tensor(key).contiguous()
    if not table_found:
        raise ValueError(f"no BF16 vocabulary table ({TIED_TABLE}) in {source}")
    missing = [m for m in upgrade if f"{m}.weight_packed" not in _keys(source)]
    if missing:
        raise KeyError(f"upgraded modules are not quantized in {source}: {missing[:3]}")

    if upgrade:
        assert bf16 is not None
        weights = _bf16_weights(bf16, upgrade)
        for module, option in upgrade.items():
            item = rtn_quantized(weights[module].to(device), option)
            item = QuantizedLinear(item.codes.cpu(), item.scales.cpu(), option)
            tensors.update(quantized_tensors(module, item))

    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "pt"})
    template = next(iter(quant["config_groups"].values()))
    for name, module, option in (
        ("blaze_hot", HOT, ladder.hot_option),
        ("blaze_cold", COLD, ladder.rest_option),
    ):
        tail = module.split(".")
        quant["config_groups"][name] = _group(template, [f"re:.*{tail[-2]}\\.{tail[-1]}$"], option)
    by_option: dict[Option, list[str]] = {}
    for module, option in upgrade.items():
        by_option.setdefault(option, []).append(module)
    for option, modules in sorted(by_option.items()):
        quant["config_groups"][f"blaze_upgrade_{option.name}"] = _group(
            template, sorted(modules), option
        )
    config["quantization_config"] = quant
    config[ARCHITECTURE_KEY] = {
        "hot_rows": ladder.hot_rows,
        "hot_bits": ladder.hot_option.bits,
        "cold_bits": ladder.rest_option.bits,
        "group_size": ladder.hot_option.group,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for path in source.iterdir():
        if path.is_file() and path.suffix in COPIED and path.name not in SKIPPED:
            shutil.copyfile(path, output / path.name)
    if (source / "compression_manifest.json").exists():
        shutil.copyfile(
            source / "compression_manifest.json", output / "source_compression_manifest.json"
        )
    record = {
        "schema_version": 1,
        "source": str(source),
        "hot_rows": ladder.hot_rows,
        "upgraded": {o.name: len(m) for o, m in sorted(by_option.items())},
        "dropped_tensors": dropped,
        "source_safetensors_bytes": sum(p.stat().st_size for p in _shards(source)),
        "safetensors_bytes": (output / "model.safetensors").stat().st_size,
    }
    (output / MANIFEST).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log(
        f"blaze add-on: {record['source_safetensors_bytes']:,} -> "
        f"{record['safetensors_bytes']:,} bytes, upgraded {record['upgraded']}"
    )
    return record


def _keys(source: Path) -> set[str]:
    from safetensors import safe_open

    keys: set[str] = set()
    for path in _shards(source):
        with safe_open(str(path), framework="pt") as handle:
            keys.update(handle.keys())
    return keys


def _bf16_weights(bf16: Path, modules: Mapping[str, Option]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    wanted = {f"{m}.weight": m for m in modules}
    found: dict[str, torch.Tensor] = {}
    for path in _shards(bf16):
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key in wanted:
                    found[wanted[key]] = handle.get_tensor(key)
    absent = sorted(set(modules) - set(found))
    if absent:
        raise KeyError(f"BF16 weights missing for {absent[:3]}")
    return found
