#!/usr/bin/env python3
"""AutoRound's own mixed-precision allocation (AutoScheme) at Glaze v2's language-model budget.

The control for Glaze v2's allocator: AutoScheme (auto-round's delta-loss scoring) chooses one
option per Linear from Glaze's menu (4-bit with group size 128, 64 or 32, 8-bit with group size
128), with vLLM's fused units tied, on Glaze's calibration blocks. Its target is the average bits
per weight of Glaze's own language-model allocation, counted the way AutoScheme counts them, so
both spend the same budget. The vision tower is left out: it does not take part in text tasks.

Writes the variant's allocation file (module names per option), which AutoRound then tunes.

python scripts/autoscheme_allocate.py --config configs/feature1.yaml \\
    --variant autoround_autoscheme_w4a16 --reference outputs/feature1/glaze2_w4a16_g128
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config
from qwen35_compression.glaze2.allocate import FUSED, group_units
from qwen35_compression.glaze2.quant import OPTIONS, Option

LAYER = re.compile(r"layers\.(\d+)\.(.+)$")
PREFIX = "model.language_model."


def canonical(name: str) -> str:
    """A language-model Linear's name as the exports spell it, whatever prefix the loader used."""
    match = LAYER.search(name)
    if match is None:
        raise ValueError(f"not a decoder-layer Linear: {name}")
    return f"{PREFIX}layers.{match.group(1)}.{match.group(2)}"


def reference_allocation(config_json: dict[str, Any]) -> dict[str, Option]:
    """Each language-model Linear's option in a compressed-tensors export (vision left out)."""
    groups = config_json["quantization_config"]["config_groups"]
    allocation = {}
    for group in groups.values():
        weights = group["weights"]
        option = Option(weights["num_bits"], weights["group_size"])
        if option not in OPTIONS or not weights["symmetric"] or weights["type"] != "int":
            raise ValueError(f"unexpected reference scheme: {weights}")
        for name in group["targets"]:
            if "visual" not in name:
                allocation[name] = option
    return allocation


def autoscheme_bits(shape: tuple[int, int], option: Option) -> int:
    """Bits of one Linear as AutoScheme counts them: codes, plus a 16-bit scale and a zero point
    of the code width per group (auto_round.auto_scheme.utils.compute_layer_bits)."""
    rows, columns = shape
    groups = rows * math.ceil(columns / option.group)
    return option.bits * rows * columns + groups * (16 + option.bits)


def average_bits(allocation: dict[str, Option], shapes: dict[str, tuple[int, int]]) -> float:
    bits = sum(autoscheme_bits(shapes[name], option) for name, option in allocation.items())
    return bits / sum(rows * columns for rows, columns in (shapes[n] for n in allocation))


def export_bytes(allocation: dict[str, Option], shapes: dict[str, tuple[int, int]]) -> int:
    """Bytes the allocation takes in a compressed-tensors export (codes and BF16 scales)."""
    return sum(option.tensor_bytes(*shapes[name]) for name, option in allocation.items())


def check_units(allocation: dict[str, Option]) -> None:
    """vLLM fuses these Linears into one kernel, so they must share an option."""
    for unit, members in group_units(allocation).items():
        if len({allocation[name] for name in members}) > 1:
            raise ValueError(f"fused unit {unit} has mixed options")


def from_layer_config(layer_config: dict[str, dict[str, Any]]) -> dict[str, Option]:
    """AutoScheme's chosen options for the language model's quantized Linears."""
    chosen = {}
    for name, scheme in layer_config.items():
        if "visual" in name or name.startswith("mtp.") or not LAYER.search(name):
            continue
        if scheme.get("bits", 16) >= 16:
            continue
        option = Option(scheme["bits"], scheme["group_size"])
        if option not in OPTIONS or not scheme.get("sym", True):
            raise ValueError(f"AutoScheme chose an option outside the menu for {name}: {scheme}")
        chosen[canonical(name)] = option
    return chosen


def linear_shapes(snapshot: Path, names: set[str]) -> dict[str, tuple[int, int]]:
    """(out, in) of each named Linear, read from the BF16 checkpoint's safetensors headers."""
    from safetensors import safe_open

    shapes = {}
    for path in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key.endswith(".weight") and key[: -len(".weight")] in names:
                    rows, columns = handle.get_slice(key).get_shape()
                    shapes[key[: -len(".weight")]] = (rows, columns)
    missing = names - shapes.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} Linears not in the checkpoint, e.g. {sorted(missing)[:3]}"
        )
    return shapes


def write_text_blocks(blocks_path: Path, tokenizer: Any, output: Path) -> int:
    """Glaze's calibration blocks as text, the form AutoScheme's dataset loader reads."""
    from qwen35_compression.glaze2.data import load_blocks

    blocks = load_blocks(blocks_path)
    with output.open("w", encoding="utf-8") as handle:
        for ids in blocks.ids:
            text = tokenizer.decode(ids, skip_special_tokens=False)
            handle.write(json.dumps({"text": text}) + "\n")
    return len(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Glaze v2's export")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=2048)
    args = parser.parse_args()
    config = load_config(args.config)
    variant = config.variant(args.variant)
    if variant.allocation is None or variant.calibration_blocks is None:
        parser.error(f"{variant.name} needs allocation and calibration_blocks")
    output = variant.allocation
    if output.exists():
        parser.error(f"refusing to overwrite {output}")

    from auto_round import AutoRound
    from auto_round.auto_scheme.gen_auto_scheme import AutoScheme
    from auto_round.schemes import QuantizationScheme
    from transformers import AutoTokenizer

    from qwen35_compression.models import download_model

    snapshot, revision = download_model(config)
    reference_config = json.loads((args.reference / "config.json").read_text(encoding="utf-8"))
    reference = reference_allocation(reference_config)
    shapes = linear_shapes(snapshot, set(reference))
    target = average_bits(reference, shapes)
    print(f"autoscheme target {target:.4f} bits per weight over {len(reference)} Linears")

    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    text_path = output.with_name("calibration_text.jsonl")
    write_text_blocks(variant.calibration_blocks, tokenizer, text_path)
    # As auto-round's own W4A16 and W8A16 presets, at each group size.
    options = [
        QuantizationScheme.from_dict(
            {"bits": o.bits, "sym": True, "group_size": o.group, "data_type": "int", "act_bits": 16}
        )
        for o in OPTIONS
    ]
    scheme = AutoScheme(
        avg_bits=target,
        options=options,
        shared_layers=[list(members) for members in FUSED],
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        dataset=str(text_path),
    )
    ar = AutoRound(
        str(snapshot),
        scheme=scheme,
        dataset=str(text_path),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        iters=0,
    )
    ar.post_init()
    chosen = from_layer_config(ar.layer_config)
    if set(chosen) != set(reference):
        extra, missing = set(chosen) - set(reference), set(reference) - set(chosen)
        raise ValueError(
            f"AutoScheme's Linears differ from Glaze's: {len(extra)} extra "
            f"{sorted(extra)[:3]}, {len(missing)} missing {sorted(missing)[:3]}"
        )
    check_units(chosen)
    counts = {o.name: sum(1 for v in chosen.values() if v == o) for o in OPTIONS}
    record = {
        "schema_version": 1,
        "method": "autoscheme",
        "base_model": {"id": config.model.id, "revision": revision},
        "reference": str(args.reference),
        "target_avg_bits": target,
        "avg_bits": average_bits(chosen, shapes),
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "linears": counts,
        "language_model_bytes": export_bytes(chosen, shapes),
        "reference_language_model_bytes": export_bytes(reference, shapes),
        "options": {o.name: sorted(name for name, v in chosen.items() if v == o) for o in OPTIONS},
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in record.items() if k != "options"}
    print("allocation=" + json.dumps(summary))


if __name__ == "__main__":
    main()
