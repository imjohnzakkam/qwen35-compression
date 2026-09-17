# qwen35-compression

Reproducible compression experiments for the dense Qwen3.5 vision-language family.

## Features

- Feature 0: `Qwen/Qwen3.5-0.8B` pipeline smoke only.
- Feature 1: `Qwen/Qwen3.5-4B` full experiment matrix.
- Feature 2: `Qwen/Qwen3.5-9B` validation of strong baselines and the best Feature 1 recipe only.

The 9B model is not downloaded or run by any Feature 0 command.

## Setup

```bash
uv sync
uv run pytest
```

`llmcompressor` is Linux-only in this project because compression runs target NVIDIA GPUs. The
configuration, manifest, and unit tests remain usable on macOS.

Validate the complete Feature 1 contract on macOS without CUDA or GPU billing:

```bash
UV_CACHE_DIR=/private/tmp/qwen35-uv-cache uv sync --frozen
UV_CACHE_DIR=/private/tmp/qwen35-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/qwen35-uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/qwen35-uv-cache uv run python scripts/preflight.py \
  --profile local --output logs/preflight/macbook.json
UV_CACHE_DIR=/private/tmp/qwen35-uv-cache uv run python scripts/run_feature1_bf16.py --dry-run
```

The dry run validates and prints the complete BF16 text-and-vision workflow. CUDA execution remains
blocked until the two remote preflights pass. Text, vision, and compression use separate
environments because their pinned Torch, `compressed-tensors`, and ANTLR requirements conflict.
The real Mac smoke uses the cached 0.8B checkpoint for both fixed text prompts and one pinned image;
it does not download the 4B checkpoint.

## Feature 0

Run the complete smoke matrix:

```bash
uv run python scripts/run_feature0.py --config configs/feature0.yaml
```

Or run individual steps:

```bash
uv run python scripts/download_model.py --config configs/feature0.yaml
uv run python scripts/evaluate.py --config configs/feature0.yaml --variant bf16
uv run python scripts/quantize.py --config configs/feature0.yaml --variant gptq_w4a16_g128
uv run python scripts/evaluate.py --config configs/feature0.yaml --variant gptq_w4a16_g128
uv run python scripts/verify_export.py --config configs/feature0.yaml --variant gptq_w4a16_g128
```

Outputs and results are intentionally ignored by Git. Every result records the resolved model
revision, exact experiment config digest, package versions, CUDA/GPU details, and Git revision.
Every compressed checkpoint contains `compression_manifest.json` with an SHA-256 inventory.

Feature 0 uses small fixed local fixtures to test plumbing; its scores are not research results and
must not be compared with the full Feature 1/2 benchmark suite.

## Feature 1 BF16 baseline

`scripts/run_feature1_bf16.py` provisions the two isolated CUDA evaluator environments, verifies
their exact package versions and CUDA access, downloads the pinned model revision, runs the text and
vision suites, and writes raw results plus `run.log` and `run_manifest.json` beneath
`results/feature1/bf16/`.

```bash
uv run python scripts/run_feature1_bf16.py
```

For a text-only, non-research pilot, add `--text-only --limit 1`. A limited run is recorded with
`research_result: false`; it cannot be mistaken for the full baseline.

After a JarvisLabs run completes, copy the entire result directory into the ignored local log store:

```bash
uv run python scripts/fetch_jarvis_results.py --instance-id INSTANCE_ID
```

## Feature 1 quantized variants

`scripts/run_feature1_variant.py` quantizes one variant from `configs/variants/feature1.yaml` in
the compression environment, verifies the export manifest, and then scores the exported checkpoint
with exactly the BF16 baseline's smoke, text, and vision commands. Results land beneath
`results/feature1/<variant>/`. An existing non-empty export is reused rather than rebuilt.

```bash
uv run python scripts/run_feature1_variant.py --variant gptq_w4a16_g128
```

Pass `--skip-bootstrap` on a machine whose evaluator environments were already built by a previous
run, and `--text-only --limit 1` for a non-research pilot.

Both Feature 1 drivers pin `enable_thinking=False`: the 4B chat template thinks by default while the
0.8B does not, and a 256-token cap on a thinking trace scores zero on every generative task.

The default single-GPU vLLM backend is deliberate. Four spot L4 GPUs cost four times as much per
hour; `torchrun` reduces elapsed time but cannot reduce total cost for a 4B model that fits on one
L4.
