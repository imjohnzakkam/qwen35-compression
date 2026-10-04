#!/usr/bin/env python3
"""Score the proxy's variants on held-out KL to BF16 and apply phase 1's gate.

Each export is read back into the student model and compared with BF16 on the held-out blocks'
answer tokens, per domain. The gate (fixed in the plan before any code): Glaze v2 in full (D) at
least 15% lower in-domain KL than AutoRound (A), chat KL no more than 5% higher, and an export no
larger than A's.

python scripts/glaze2_evaluate.py --config configs/glaze2_proxy.yaml --output OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen35_compression.config import load_config

BASELINE = "autoround_w4a16_g128"
DATA_ONLY = "autoround_glaze2_data_w4a16_g128"
UNIFORM = "glaze2_uniform_w4a16_g128"
FULL = "glaze2_w4a16_g128"
VARIANTS = (BASELINE, DATA_ONLY, UNIFORM, FULL)
GATE_IN_DOMAIN = 0.15
GATE_CHAT_WORSE = 0.05


def gate(scores: dict[str, dict[str, Any]], sizes: dict[str, int]) -> dict[str, Any]:
    from qwen35_compression.glaze2.evaluate import reduction

    a, d = scores[BASELINE], scores[FULL]
    in_domain = reduction(a, d, "in_domain")
    chat = reduction(a, d, "chat")
    ablations = {
        "H1_allocation (D vs C)": reduction(scores[UNIFORM], d, "in_domain"),
        "H2_data (B vs A)": reduction(a, scores[DATA_ONLY], "in_domain"),
        "H3_weighting (C vs B)": reduction(scores[DATA_ONLY], scores[UNIFORM], "in_domain"),
        "H4_all (D vs A)": in_domain,
    }
    passed = (
        in_domain is not None
        and in_domain >= GATE_IN_DOMAIN
        and (chat is None or chat >= -GATE_CHAT_WORSE)
        and sizes[FULL] <= sizes[BASELINE]
    )
    return {
        "in_domain_reduction": in_domain,
        "chat_reduction": chat,
        "size_ok": sizes[FULL] <= sizes[BASELINE],
        "ablations": ablations,
        "decision": "go" if passed else "stop",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks-per-batch", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite {output}")

    from qwen35_compression.glaze2.data import load_blocks
    from qwen35_compression.glaze2.evaluate import evaluate_export
    from qwen35_compression.glaze2.pipeline import directory_bytes, load_teacher

    config = load_config(args.config)
    settings = config.variant(FULL).glaze2
    assert settings is not None
    blocks = load_blocks(settings.held_out_blocks)
    teacher, model, _, _, _ = load_teacher(config)
    head = model.get_output_embeddings().weight
    scores, sizes = {}, {}
    for name in VARIANTS:
        export = config.paths.outputs / name
        sizes[name] = directory_bytes(export)
        scores[name] = evaluate_export(teacher, head, export, blocks, args.blocks_per_batch)
        kl = {k: scores[name][k]["mean_kl"] for k in ("in_domain", "chat")}
        print(f"glaze2 eval {name}: KL {kl}, {sizes[name]:,} bytes")
    result = {"scores": scores, "bytes": sizes, "gate": gate(scores, sizes)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("gate=" + json.dumps(result["gate"]))


if __name__ == "__main__":
    main()
