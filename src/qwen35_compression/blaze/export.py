"""Fake-quantized exports: the BF16 checkpoint with chosen tensors replaced by their stored values.

A fake-quantized export holds exactly the numbers a real Blaze export would compute with, written
back in BF16, so stock vLLM loads it like the BF16 model and its drift and benchmark scores are
the real format's. Sizes are recorded as the real format would store them; the files themselves
stay BF16-sized. Real (tiered) exports come with the kernels, after the quality gates.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from qwen35_compression.blaze.body import is_body_linear
from qwen35_compression.blaze.ladder import REST_OPTION, LadderPlan, ladder_table
from qwen35_compression.glaze2.quant import Option, fake_quantize

TIED_TABLE = "model.language_model.embed_tokens.weight"
HEAD = "lm_head.weight"
MANIFEST = "blaze_manifest.json"


def _shards(source: Path) -> list[Path]:
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {source}")
    return files


def _tensor_names(source: Path) -> set[str]:
    from safetensors import safe_open

    names: set[str] = set()
    for path in _shards(source):
        with safe_open(str(path), framework="pt") as handle:
            names.update(handle.keys())
    return names


def write_fake_export(
    source: Path,
    output: Path,
    ladder: LadderPlan | None = None,
    body: Option | None = None,
    device: str = "cpu",
    log: Callable[[str], None] = print,
    replaced: Mapping[str, torch.Tensor] | None = None,
    options: Mapping[str, Option] | None = None,
    vision: Option | None = None,
) -> dict[str, Any]:
    """Copy `source` into `output` with the vocabulary laddered and/or every body Linear
    stored at `body`, and return the export's record.

    Body Linears are rounded to nearest, unless `replaced` (checkpoint key -> stored BF16 weight,
    e.g. from GPTQ at `body`) gives their values; every replaced key must be a body Linear.
    `options` (checkpoint key -> option) overrides `body` per Linear, for rounding and bytes.
    `vision` rounds the vision tower's Linears to nearest at that option (Glaze v2 stores them
    at 8-bit g128); left out, the vision tower stays BF16.

    Tied tables (0.8B to 4B) are laddered once. Untied models ladder the LM head and keep the
    input table at the ladder's lower rung, as Blaze stores them.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output}")
    output.mkdir(parents=True, exist_ok=True)
    names = _tensor_names(source)
    tied = HEAD not in names
    replaced = dict(replaced or {})
    given = len(replaced)
    if replaced and body is None:
        raise ValueError("replaced weights need the body option they were stored at")
    missing = sorted(key for key in replaced if key not in names)
    if missing:
        raise KeyError(f"replaced weights not in {source}: {missing[:3]}")
    hidden = 0
    body_linears = 0
    body_bytes = 0
    vision_linears = 0
    for path in _shards(source):
        tensors: dict[str, torch.Tensor] = {}
        metadata: dict[str, str] = {"format": "pt"}
        with safe_open(str(path), framework="pt") as handle:
            metadata.update(handle.metadata() or {})
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if ladder is not None and key == (TIED_TABLE if tied else HEAD):
                    hidden = tensor.shape[1]
                    tensor = ladder_table(tensor.to(device), ladder).cpu()
                    log(f"blaze export: laddered {key} ({ladder.hot_rows:,} rows at 8 bits)")
                elif ladder is not None and not tied and key == TIED_TABLE:
                    tensor = fake_quantize(tensor.to(device), REST_OPTION).cpu()
                elif body is not None and is_body_linear(key, tensor.shape):
                    option = (options or {}).get(key, body)
                    if tensor.shape[1] % option.group:
                        raise ValueError(f"{key}: {tensor.shape[1]} inputs, group {option.group}")
                    if key in replaced:
                        stored = replaced.pop(key)
                        if stored.shape != tensor.shape:
                            raise ValueError(f"{key}: replaced shape {tuple(stored.shape)}")
                        tensor = stored.to(tensor.dtype).cpu()
                    else:
                        tensor = fake_quantize(tensor.to(device), option).cpu()
                    body_linears += 1
                    body_bytes += option.tensor_bytes(tensor.shape[0], tensor.shape[1])
                elif vision is not None and _vision_linear(key, tensor.shape, vision):
                    tensor = fake_quantize(tensor.to(device), vision).cpu()
                    vision_linears += 1
                tensors[key] = tensor.contiguous()
        save_file(tensors, str(output / path.name), metadata=metadata)
    if replaced:
        raise ValueError(f"replaced weights that are not body Linears: {sorted(replaced)[:3]}")
    if ladder is not None and not hidden:
        raise ValueError(f"no vocabulary table ({TIED_TABLE if tied else HEAD}) in {source}")
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copyfile(path, output / path.name)
    record: dict[str, Any] = {
        "schema_version": 1,
        "fake_quantized": True,
        "source": str(source),
        "tied_table": tied,
        "ladder": ladder.record(hidden) if ladder is not None else None,
        "input_table": (
            {
                "option": REST_OPTION.name,
                "bytes": REST_OPTION.tensor_bytes(ladder.vocab_size, hidden),
            }
            if ladder is not None and not tied
            else None
        ),
        "vision": (
            {"option": vision.name, "linears": vision_linears} if vision is not None else None
        ),
        "body": (
            {
                "option": body.name,
                "linears": body_linears,
                "replaced_linears": given,
                "stored_bytes": body_bytes,
            }
            if body is not None
            else None
        ),
    }
    (output / MANIFEST).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log(f"blaze export: {output.name} ladder={ladder is not None} body={body and body.name}")
    return record


def _vision_linear(key: str, shape: Any, option: Option) -> bool:
    from qwen35_compression.glaze2.export import vision_linear

    return vision_linear(key) and len(shape) == 2 and shape[-1] % option.group == 0


def size_plan(
    source: Path,
    body: Option,
    ladder: LadderPlan | None = None,
    options: Mapping[str, Option] | None = None,
    vision: Option | None = None,
) -> dict[str, Any]:
    """Bytes of an export of `source` as the real format stores it, by part: the vocabulary
    table(s) laddered (or BF16 without a ladder), body Linears at `body`, everything else BF16.

    The MTP head (`mtp.*`) is left out: vLLM does not load it unless asked, and quantized
    exports drop it, so both sides of a size comparison count the same tensors. `options`
    (checkpoint key -> option) overrides `body` per Linear; `vision` stores the vision tower's
    Linears at that option (the rest of the tower stays BF16).
    """
    from safetensors import safe_open

    parts = {"table": 0, "body": 0, "vision": 0, "other": 0}
    skipped = 0
    hidden = 0
    names = _tensor_names(source)
    tied = HEAD not in names
    for path in _shards(source):
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                shape = tensor.get_shape()
                count = 1
                for size in shape:
                    count *= size
                bf16 = count * 2
                if key.startswith("mtp."):
                    skipped += bf16
                elif key in (TIED_TABLE, HEAD):
                    hidden = shape[1]
                    if ladder is None:
                        parts["table"] += bf16
                    elif key == (TIED_TABLE if tied else HEAD):
                        parts["table"] += ladder.table_bytes(hidden)
                    else:
                        parts["table"] += REST_OPTION.tensor_bytes(shape[0], shape[1])
                elif is_body_linear(key, shape):
                    option = (options or {}).get(key, body)
                    parts["body"] += option.tensor_bytes(shape[0], shape[1])
                elif vision is not None and _vision_linear(key, shape, vision):
                    parts["vision"] += vision.tensor_bytes(shape[0], shape[1])
                elif "visual" in key:
                    parts["vision"] += bf16
                else:
                    parts["other"] += bf16
    if not hidden:
        raise ValueError(f"no vocabulary table in {source}")
    return {
        "body_option": body.name,
        "vision_option": vision.name if vision is not None else None,
        "ladder": ladder is not None,
        "parts": parts,
        "total_bytes": sum(parts.values()),
        "mtp_bytes_excluded": skipped,
    }
