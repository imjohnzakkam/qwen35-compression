# qwen35-compression

Reproducible compression experiments for the dense Qwen3.5 vision-language family.

## Phases

- Phase 0: `Qwen/Qwen3.5-0.8B` pipeline smoke only.
- Phase 1: `Qwen/Qwen3.5-4B` full experiment matrix.
- Phase 2: `Qwen/Qwen3.5-9B` validation of strong baselines and the best Phase 1 recipe only.

The 9B model is not downloaded or run by any Phase 0 command.

## Setup

```bash
uv sync
uv run pytest
```

`llmcompressor` is Linux-only in this project because compression runs target NVIDIA GPUs. The
configuration, manifest, and unit tests remain usable on macOS.

## Phase 0

Run the complete smoke matrix:

```bash
uv run python scripts/run_phase0.py --config configs/phase0.yaml
```

Or run individual steps:

```bash
uv run python scripts/download_model.py --config configs/phase0.yaml
uv run python scripts/evaluate.py --config configs/phase0.yaml --variant bf16
uv run python scripts/quantize.py --config configs/phase0.yaml --variant gptq_w4a16_g128
uv run python scripts/evaluate.py --config configs/phase0.yaml --variant gptq_w4a16_g128
uv run python scripts/verify_export.py --config configs/phase0.yaml --variant gptq_w4a16_g128
```

Outputs and results are intentionally ignored by Git. Every result records the resolved model
revision, exact experiment config digest, package versions, CUDA/GPU details, and Git revision.
Every compressed checkpoint contains `compression_manifest.json` with an SHA-256 inventory.

Phase 0 uses small fixed local fixtures to test plumbing; its scores are not research results and
must not be compared with the full Phase 1/2 benchmark suite.

