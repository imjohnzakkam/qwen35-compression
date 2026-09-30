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

## Feature 1 vision protocol

`scripts/vlmeval_qwen35.py` serves the checkpoint with vLLM's OpenAI-compatible server and drives
VLMEvalKit's API path (`LMDeployAPI`) with 32 concurrent workers, because VLMEvalKit's in-process
Qwen3-VL path generates one sample at a time (about five seconds per sample on an L4, or more than
a day for the six datasets). Inference runs first for every dataset (`--mode infer`, resumable
with `--reuse`), then scoring:

- TextVQA, OCRBench, and DocVQA are scored by rules on the GPU (`--judge exact_matching`).
- MMBench, MMMU, and MathVista are inferred on the GPU but scored after download by one fixed
  answer extractor, `gpt-4o-mini` (VLMEvalKit's and the OpenCompass leaderboard's default). The
  4B model answers multiple-choice questions with explanations that the rules cannot parse; under
  rules alone MMMU scored below chance. A variant must never extract its own answers, or
  quantization would change the extractor as well as the model.

The extractor, and which datasets use it, are set in `configs/evaluation/feature1.yaml` and
recorded in the run manifest under `vision_protocol`. Score a downloaded run on the local machine;
the OpenAI key is read from `~/.config/qwen35/openai.env` into the scoring process only and never
leaves the machine:

```bash
uv venv .venv-vision-score --python 3.11
uv pip install --python .venv-vision-score/bin/python -r requirements/vision-score.lock
uv pip install --python .venv-vision-score/bin/python --no-deps -e external/VLMEvalKit
uv run python scripts/score_vision.py --run-dir logs/jarvis/feature1-bf16-<timestamp>
```

The downloaded run is left untouched; predictions are copied to `<run-dir>-scored/`, which also
holds `scoring_manifest.json` (extractor, VLMEvalKit revision, prediction SHA-256s). One pass costs
about $0.40 in API credit per model.

Pass `--vision-only` to either driver to run just this stage, for example after a text-only run.

## Feature 1 evaluation tracks

Two protocols, kept apart on purpose. The instruct track is cheap enough to run on every variant.
The thinking track measures the model the way Qwen publishes it, on a subset, for BF16 and the
final few variants.

| | Instruct track (main) | Thinking track |
| --- | --- | --- |
| Suite | `configs/evaluation/feature1.yaml` | `configs/evaluation/feature1_thinking.yaml` |
| Mode | `enable_thinking=False` | thinking, as the model card recommends |
| Decoding | greedy | `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` |
| Answer cap | 8,192 tokens, text and vision | 32,768 tokens |
| Tasks | full text and vision suite | MMLU-Pro (fixed 1,001 questions), MATH-500, IFEval |
| Runs | one | one per seed (42, 43), averaged |
| Used for | every variant and the sensitivity sweep | BF16 and the final 2–3 variants |

The 8,192-token cap replaced 2,048, which cut off 8.2% of MMLU-Pro, 14.6% of MATH-500, and 10.8% of
MMMU answers (most then scored wrong). A variant that writes longer answers would have lost accuracy
to truncation rather than quality. The text context is 12,288 tokens because lm-eval silently
truncates the start of any prompt that does not fit beside the answer budget. Suites that leave
fewer than 2,048 prompt tokens (16,384 for vision) are rejected. The larger context also changes
WikiText perplexity, which is computed over windows of that length.

Run the thinking track with `--suite`. It is text-only and writes to `<variant>-thinking/`:

```bash
uv run python scripts/run_feature1_bf16.py --suite configs/evaluation/feature1_thinking.yaml
uv run python scripts/run_feature1_variant.py --variant gptq_w4a16_g128 \
  --suite configs/evaluation/feature1_thinking.yaml
uv run python scripts/fetch_jarvis_results.py --instance-id INSTANCE_ID \
  --remote-path /home/qwen35-compression/results/feature1/bf16-thinking
```

The MMLU-Pro subset is stratified by subject with seed 42, and `scripts/make_thinking_subset.py`
regenerates it. lm-eval strips everything up to `</think>` before extracting answers, and in thinking
mode it cannot score likelihood tasks (HellaSwag, ARC, WikiText). Every run manifest records the
suite's path and SHA-256 under `benchmark_suite`.

**Why these numbers differ from the model card.** Qwen reports thinking mode with 32,768–81,920
output tokens, sampling, and answer-format prompts: for example MMLU-Pro 79.1 and IFEval 89.8. The
instruct track scored 67.0 and 82.8 under a 2,048-token cap; the gap comes from the mode, the cap,
and the answer format, not from a different checkpoint. The thinking track should land near the card
on MMLU-Pro and IFEval. It differs on purpose in using lm-eval's prompts, a subset, and no GPQA
Diamond (`Idavidrein/gpqa` is gated on Hugging Face).

The default single-GPU vLLM backend is deliberate. Four spot L4 GPUs cost four times as much per
hour; `torchrun` reduces elapsed time but cannot reduce total cost for a 4B model that fits on one
L4.
