# qwen35-compression

Reproducible compression experiments for the dense Qwen3.5 vision-language family.

## Features

- Feature 0: `Qwen/Qwen3.5-0.8B` pipeline smoke only.
- Feature 1: `Qwen/Qwen3.5-4B` full experiment matrix.
- Feature 2: `Qwen/Qwen3.5-9B` validation of strong baselines and the best Feature 1 recipe only.

The 9B model is not downloaded or run by any Feature 0 command.

The Feature 0/1/2 names will become `smoke-0.8b`, `study-4b`, and `validate-9b` once the 4B study
is finished. Renaming earlier would change the config digest halfway through the 4B results.

## Status

Last updated 2026-10-01. Stages follow `plan.md`.

| Stage | Status |
| --- | --- |
| Feature 0: 0.8B pipeline smoke (BF16, INT8, GPTQ, AWQ, export) | Done. `results/phase0/` |
| Feature 1: BF16 text baseline | Done |
| Feature 1: BF16 vision baseline | Inference done. 3 of 6 datasets wait for extractor scoring |
| Feature 1: frozen calibration set | Done. `data/calibration/feature1*.lock.json` |
| Feature 1: INT8, GPTQ, AWQ baselines | Not started. Driver ready; needs GPU budget |
| Feature 1: component sensitivity, mixed precision | Not started |
| Feature 2: 9B validation | Not started |

### Feature 1 BF16 baseline: Qwen3.5-4B, instruct mode

All scores use `enable_thinking=False`. Do not compare them with thinking-mode numbers.

Text: lm-evaluation-harness on vLLM, 0-shot, chat template, code `ffc8f10`.

| Task | Metric | Score |
| --- | --- | ---: |
| MMLU-Pro | exact match, custom extract | 67.0 |
| GSM8K | exact match, flexible extract | 82.9 |
| MATH-500 | math_verify | 73.2 |
| IFEval | prompt-level strict | 82.8 |
| HellaSwag | acc_norm | 65.4 |
| ARC-Challenge | acc_norm | 50.6 |
| WikiText-2 | word perplexity | 11.47 |

HellaSwag and ARC-Challenge read lower than typical reports for this model size. They are
likelihood-scored tasks run through the chat template, which is the likely cause. This still needs
checking; comparisons between variants are unaffected because every variant uses the same protocol.

Vision: VLMEvalKit through a vLLM server, code `1e6abe0`, 0 rejected requests.

| Dataset | Score | Scoring |
| --- | ---: | --- |
| DocVQA (val) | 95.4 | rules (ANLS) |
| OCRBench | 86.3 | rules |
| TextVQA (val) | 83.0 | rules |
| MMBench (dev, EN v1.1) | pending | gpt-4o-mini extractor |
| MMMU (val) | pending | gpt-4o-mini extractor |
| MathVista (mini) | pending | gpt-4o-mini extractor |

The first extractor pass was invalid: an OpenAI rate limit made VLMEvalKit fill 391 MMMU and 30
MMBench answers with random options. It is kept only as a record at
`logs/jarvis/feature1-bf16-20260930T030752Z-scored-ratelimited-invalid/`. A throttled pass is due
once the account's daily request limit has reset.

Raw results are ignored by Git and live under `logs/jarvis/`:

- text: `feature1-bf16-full-20260917T232611Z/`
- vision: `feature1-bf16-20260930T030752Z/`
- extractor scores: `feature1-bf16-20260930T030752Z-scored/`, once the pass is done

### GPU spend (JarvisLabs, on-demand L4 at ₹41.31/h)

| Run | Cost |
| --- | ---: |
| BF16 vision, including one aborted attempt | ₹145.57 |
| Balance on 2026-10-01 | ₹835.24 |

Every run pairs with a teardown watcher. It downloads the results and destroys the instance when the
run ends or fails, and it stops the run at a hard budget deadline.

### Protocol decisions so far

- **Instruct mode.** The 4B chat template thinks by default and the 0.8B does not. Every size and
  variant is scored with `enable_thinking=False` and a 2048-token generation cap.
- **Vision context.** The vision server uses `vision_max_model_len: 32768`. At the text suite's
  4096, image prompts over 2048 tokens were rejected, and VLMEvalKit counted each rejection as a
  wrong answer.
- **Multiple-choice extraction.** MMBench, MMMU, and MathVista are scored by one fixed
  `gpt-4o-mini` extractor after download, never by the variant itself. See the vision protocol
  section below. Check the per-item `log` column for random fills, not VLMEvalKit's
  `judge_fail_rate`, which reports 0% even when answers were randomly filled.

### Next

1. Finish BF16 vision scoring for MMBench, MMMU, and MathVista (about $0.40 of API credit).
2. Run INT8 W8A8, GPTQ W4A16 g128, and AWQ W4A16 g128. The estimate is ₹900–1,000 of GPU time,
   which is more than the current balance.
3. Build the comparison table from `plan.md`: change from BF16 and compression ratio, for text and
   vision.
4. Component sensitivity and a mixed-precision recipe, then the 9B validation.

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

`scripts/vlmeval_qwen35.py` serves the checkpoint with vLLM's OpenAI-compatible server (context
`vision_max_model_len`) and drives VLMEvalKit's API path (`LMDeployAPI`) with 32 concurrent workers, because VLMEvalKit's in-process
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
holds `scoring_manifest.json` (extractor, VLMEvalKit revision, prediction SHA-256s, extraction
failures). One pass costs about $0.40 in API credit per model. The extractor runs 4 requests at a
time (`--api-nproc`) to stay under the OpenAI account's rate limit. The script exits non-zero if
any answer was randomly filled or any task is unscored, and it redacts the API key, which
VLMEvalKit's client logs, from its output and from every file it writes.

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
