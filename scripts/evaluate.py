#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from pathlib import Path

from qwen35_compression.config import load_config
from qwen35_compression.evaluation import evaluate_smoke
from qwen35_compression.export import verify_export
from qwen35_compression.io import write_json
from qwen35_compression.models import load_resolved_model, model_source, resolve_revision
from qwen35_compression.provenance import environment_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    variant = config.variant(args.variant)
    source = model_source(config, variant.name)
    if variant.method != "bf16":
        verify_export(Path(source), variant)

    model, processor = load_resolved_model(source, config)
    metrics = evaluate_smoke(model, processor, config.evaluation)
    root = config.source_path.parent.parent
    result = {
        "schema_version": 1,
        "phase": config.phase,
        "variant": variant.name,
        "method": variant.method,
        "model_id": config.model.id,
        "model_revision": resolve_revision(config),
        "config_digest": config.digest,
        "environment": environment_record(root),
        "metrics": metrics,
    }
    result_path = config.paths.results / f"{variant.name}.json"
    write_json(result_path, result)
    print(f"result={result_path}")
    print(f"perplexity={metrics['perplexity']:.6f}")
    print(f"prefix_accuracy={metrics['prefix_accuracy']:.6f}")
    del model, processor
    gc.collect()


if __name__ == "__main__":
    main()
