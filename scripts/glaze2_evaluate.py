#!/usr/bin/env python3
"""Score the proxy's variants on held-out KL to BF16 and apply phase 1's gate.

Each export is read back into the student model and compared with BF16 on the held-out blocks'
answer tokens, per domain. The gate (fixed in the plan before any code): Glaze v2 in full (D) at
least 15% lower in-domain KL than AutoRound (A), chat KL no more than 5% higher, and an export no
larger than A's.

python scripts/glaze2_evaluate.py --config configs/glaze2_proxy.yaml --output OUT

On the 4B only AutoRound and Glaze v2 are built: --variants autoround_w4a16_g128,glaze2_w4a16_g128
scores those two, and the ablations whose variants are missing are left out.
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


def gate(
    scores: dict[str, dict[str, Any]],
    sizes: dict[str, int],
    baseline: str = BASELINE,
    candidate: str = FULL,
) -> dict[str, Any]:
    from qwen35_compression.glaze2.evaluate import reduction

    a, d = scores[baseline], scores[candidate]
    in_domain = reduction(a, d, "in_domain")
    chat = reduction(a, d, "chat")
    pairs = {
        "H1_allocation (D vs C)": (UNIFORM, candidate),
        "H2_data (B vs A)": (baseline, DATA_ONLY),
        "H3_weighting (C vs B)": (DATA_ONLY, UNIFORM),
        "H4_all (D vs A)": (baseline, candidate),
    }
    ablations = {
        name: reduction(scores[before], scores[after], "in_domain")
        for name, (before, after) in pairs.items()
        if before in scores and after in scores
    }
    size_ok = sizes[candidate] <= sizes[baseline]
    passed = (
        in_domain is not None
        and in_domain >= GATE_IN_DOMAIN
        and (chat is None or chat >= -GATE_CHAT_WORSE)
        and size_ok
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "in_domain_reduction": in_domain,
        "chat_reduction": chat,
        "size_ok": size_ok,
        "ablations": ablations,
        "decision": "go" if passed else "stop",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks-per-batch", type=int, default=4)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated exports to score; the first is the baseline, the last the "
        "candidate (default: the proxy's A, B, C, D)",
    )
    args = parser.parse_args()
    variants = [name for name in args.variants.split(",") if name]
    if len(variants) < 2:
        parser.error("--variants needs a baseline and a candidate")
    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite {output}")

    from qwen35_compression.glaze2.data import load_blocks
    from qwen35_compression.glaze2.evaluate import evaluate_export
    from qwen35_compression.glaze2.pipeline import directory_bytes, load_teacher

    config = load_config(args.config)
    settings = config.variant(variants[-1]).glaze2
    if settings is None:
        parser.error(f"the candidate {variants[-1]} is not a glaze2 variant")
    blocks = load_blocks(settings.held_out_blocks)
    teacher, model, _, _, _ = load_teacher(config)
    head = model.get_output_embeddings().weight
    scores, sizes = {}, {}
    for name in variants:
        export = config.paths.outputs / name
        sizes[name] = directory_bytes(export)
        scores[name] = evaluate_export(teacher, head, export, blocks, args.blocks_per_batch)
        kl = {k: scores[name][k]["mean_kl"] for k in ("in_domain", "chat")}
        print(f"glaze2 eval {name}: KL {kl}, {sizes[name]:,} bytes")
    result = {
        "scores": scores,
        "bytes": sizes,
        "gate": gate(scores, sizes, variants[0], variants[-1]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("gate=" + json.dumps(result["gate"]))


if __name__ == "__main__":
    main()
