"""Fake-quantized exports: the BF16 checkpoint with chosen tensors replaced by their stored values.

A fake-quantized export holds exactly the numbers a real Blaze export would compute with, written
back in BF16, so stock vLLM loads it like the BF16 model and its drift and benchmark scores are
the real format's. Sizes are recorded as the real format would store them; the files themselves
stay BF16-sized. Real (tiered) exports come with the kernels, after the quality gates.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
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
) -> dict[str, Any]:
    """Copy `source` into `output` with the vocabulary laddered and/or every body Linear
    round-to-nearest at `body`, and return the export's record.

    Tied tables (0.8B to 4B) are laddered once. Untied models ladder the LM head and keep the
    input table at the ladder's lower rung, as Blaze stores them.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tied = HEAD not in _tensor_names(source)
    hidden = 0
    body_linears = 0
    body_bytes = 0
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
                    if tensor.shape[1] % body.group:
                        raise ValueError(f"{key}: {tensor.shape[1]} inputs, group {body.group}")
                    tensor = fake_quantize(tensor.to(device), body).cpu()
                    body_linears += 1
                    body_bytes += body.tensor_bytes(tensor.shape[0], tensor.shape[1])
                tensors[key] = tensor.contiguous()
        save_file(tensors, str(output / path.name), metadata=metadata)
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
        "body": (
            {"option": body.name, "linears": body_linears, "stored_bytes": body_bytes}
            if body is not None
            else None
        ),
    }
    (output / MANIFEST).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log(f"blaze export: {output.name} ladder={ladder is not None} body={body and body.name}")
    return record
