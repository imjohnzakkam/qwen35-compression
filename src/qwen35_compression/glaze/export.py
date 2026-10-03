"""Writing a Glaze export: the init export with its scales and norm weights replaced."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

import torch

from qwen35_compression.export import MANIFEST_NAME

# Files of a downloaded init that do not belong to the export itself.
SKIPPED_FILES = frozenset({MANIFEST_NAME, "README.md", ".gitattributes"})


def write_refined_export(
    init_dir: Path, output_dir: Path, replacements: Mapping[str, torch.Tensor]
) -> None:
    """Copy the init export to `output_dir`, replacing the named tensors.

    Every replacement must name an existing tensor of the same shape and dtype, so the result has
    exactly the init's tensor names, dtypes, shapes, file layout and quantization config.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = dict(replacements)
    for source in sorted(init_dir.iterdir()):
        if source.name in SKIPPED_FILES:
            continue
        if not source.is_file():
            raise ValueError(f"unexpected entry in the init export: {source}")
        target = output_dir / source.name
        if source.suffix != ".safetensors":
            shutil.copyfile(source, target)
            continue
        with safe_open(str(source), framework="pt") as handle:
            metadata = handle.metadata()
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        for key in list(tensors):
            if key not in remaining:
                continue
            new = remaining.pop(key)
            old = tensors[key]
            if new.shape != old.shape or new.dtype != old.dtype:
                raise ValueError(
                    f"{key}: replacement {new.dtype} {tuple(new.shape)} does not match the init's "
                    f"{old.dtype} {tuple(old.shape)}"
                )
            tensors[key] = new.detach().to("cpu").contiguous()
        save_file(tensors, str(target), metadata=metadata)
    if remaining:
        raise ValueError(f"replacements not found in the init export: {sorted(remaining)[:5]}")
