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

Last updated 2026-10-03. Stages follow `plan.md`.

| Stage | Status |
| --- | --- |
| Feature 0: 0.8B pipeline smoke (BF16, INT8, GPTQ, AWQ, export) | Done. `results/phase0/` |
| Feature 1: BF16 baseline, instruct track, 8,192-token cap | Done (text and vision), 2026-10-02 |
| Feature 1: BF16 thinking track | Not started. MMLU-Pro subset, MATH-500, IFEval |
| Feature 1: frozen calibration set | Done. `data/calibration/feature1*.lock.json` |
| Feature 1: INT8, GPTQ, AWQ baselines | Running, 2026-10-03 |
| Feature 1: component sensitivity, mixed precision | Not started |
| Feature 2: 9B validation | Not started |

### Feature 1 BF16 baseline: Qwen3.5-4B, instruct mode, 8,192-token cap

The reference every variant is compared with. `enable_thinking=False`, greedy decoding, code
`9fd9b46`, one A30 (run `r_e8b44978`). Raw results: `logs/jarvis/feature1-bf16-a30-8192-20261002b/`,
extractor scores: `logs/jarvis/feature1-bf16-a30-8192-20261002b-scored/`.

| Task | Metric | 2,048 cap | 8,192 cap |
| --- | --- | ---: | ---: |
| MMLU-Pro | exact match, custom extract | 67.0 | 74.6 |
| GSM8K | exact match, flexible extract | 82.9 | 83.2 |
| MATH-500 | math_verify | 73.2 | 83.4 |
| IFEval | prompt-level strict | 82.8 | 82.3 |
| HellaSwag | acc_norm | 65.4 | 65.4 |
| ARC-Challenge | acc_norm | 50.6 | 51.1 |
| WikiText-2 | word perplexity (lower is better) | 11.47 | 10.95 |
| DocVQA (val) | ANLS, rules | 95.4 | 95.3 |
| OCRBench | rules | 86.3 | 86.3 |
| TextVQA (val) | rules | 83.0 | 82.8 |
| MMBench (dev, EN v1.1) | gpt-4o-mini extractor | 84.8 | 85.4 |
| MMMU (val) | gpt-4o-mini extractor | 63.4 | 69.6 |
| MathVista (mini) | gpt-4o-mini extractor | 80.8 | 81.0 |

At 8,192 tokens, 4.8% of MMLU-Pro and 3.6% of MATH-500 answers still reach the cap, almost all of
them repetition loops. Four MMMU answers end at the cap without an answer, and VLMEvalKit fills
them with random options (at most 0.4 points). How to score them is still open. WikiText
perplexity changed because its windows grew from 4,096 to 12,288 tokens.

**Against the model card.** Qwen reports MMLU-Pro 79.1, IFEval 89.8, MMMU 77.6, MathVista 85.1,
MMBench 89.4 and OCRBench 85.0. The card's numbers come from thinking mode (the model's default),
sampling at `temperature=1.0`, a 32,768–81,920-token budget, answer-format prompts and Qwen's own
harness. These are the same BF16 weights (the checkpoint's own dtype), so precision is not the
difference. OCRBench, which needs a short answer and no reasoning, matches the card (86.3 against
85.0), while the gaps appear only on reasoning-heavy tasks. The thinking track (below) reproduces
the card's protocol for BF16 and the final variants.

### Earlier BF16 record: 2,048-token cap (superseded)

All scores use `enable_thinking=False`. Do not compare them with thinking-mode numbers, including the
model card's (for example MMLU-Pro 79.1, IFEval 89.8). The 2,048-token cap cut off 8.2% of MMLU-Pro,
14.6% of MATH-500 and 10.8% of MMMU answers. These numbers are kept as a record and will be replaced
by the 8,192-token rerun before any variant is compared with them.

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
| MMBench (dev, EN v1.1) | 84.8 | gpt-4o-mini extractor |
| MMMU (val) | 63.4 | gpt-4o-mini extractor |
| MathVista (mini) | 80.8 | gpt-4o-mini extractor |

The throttled extractor pass (2026-10-02) left 3 of about 6,900 answers randomly filled: 1 MMBench and
2 MMMU questions where the model wrote an explicit `Final Answer: **B**` that the extractor still
failed to map. That moves the scores by at most 0.1 and 0.2 points. The final-answer rule (see
protocol decisions) now handles these cases. The first, rate-limited pass is kept only as a record
at `logs/jarvis/feature1-bf16-20260930T030752Z-scored-ratelimited-invalid/`.

Raw results are ignored by Git and live under `logs/jarvis/`:

- text: `feature1-bf16-full-20260917T232611Z/`
- vision: `feature1-bf16-20260930T030752Z/`
- extractor scores: `feature1-bf16-20260930T030752Z-scored/`

### GPU spend (JarvisLabs, on-demand)

| Run | GPU | Cost |
| --- | --- | ---: |
| BF16 vision at 2,048, including one aborted attempt | L4, ₹41.31/h | ₹145.57 (~$1.51) |
| BF16 limit-10 pilot, and a full run lost when a watcher misread a status glitch | A30, ₹38.88/h | ₹112.49 (~$1.17) |
| BF16 full run at 8,192 | A30, ₹38.88/h | ₹212.27 (~$2.20) |

USD at ₹96.3. The A30 replaced the L4: generation here is limited by memory bandwidth, and the A30
has about 3× the L4's for less per hour. Measured: about 1,600 output tokens/s, 1.75–1.9× the L4.

Every run pairs with a teardown watcher. It downloads the results and destroys the instance when the
run ends or fails, and it stops the run at a hard budget deadline.

### Protocol decisions so far

- **Two tracks.** The main track is instruct mode (`enable_thinking=False`, greedy, 8,192-token cap)
  for every variant. A thinking track follows the model card's settings on a subset, for BF16 and the
  final 2–3 variants, with two seeds averaged. Running thinking mode on every variant would cost about
  ₹2,000+ per model, and sampling noise (about ±1.5 points per run) is as large as the INT8 and
  GPTQ effects being measured.
- **Answer cap.** 2,048 → 8,192 tokens. The old cap truncated enough answers that a more verbose
  variant would have lost accuracy to truncation alone.
- **Vision context.** The vision server uses `vision_max_model_len: 32768`. At the text suite's
  4096, image prompts over 2048 tokens were rejected, and VLMEvalKit counted each rejection as a
  wrong answer.
- **Multiple-choice extraction.** MMBench, MMMU, and MathVista are scored by one fixed
  `gpt-4o-mini` extractor after download, never by the variant itself. See the vision protocol
  section below. Check the per-item `log` column for random fills, not VLMEvalKit's
  `judge_fail_rate`, which reports 0% even when answers were randomly filled.
- **Final-answer rule.** When rule matching and the extractor both fail on a multiple-choice answer,
  VLMEvalKit picks a random option. Before that happens, the answer is taken from the model's last
  explicit `Final Answer: X` statement, if X is a valid option. The rule is fixed and the same for
  every variant. `scoring_manifest.json` counts how often it was used.
- **Pilot first.** Every full GPU run starts with a `--limit 10` pilot (text and vision) on the same
  instance, so setup is paid for once. The pilot writes to `results/feature1/bf16-pilot`, so the full
  run cannot reuse its predictions. A 16 GB Mac is too small to validate the 4B model at these
  settings in reasonable time, so `scripts/validate_local.py` (below) is optional.

### Next

1. Run INT8 W8A8, GPTQ W4A16 g128, and AWQ W4A16 g128 on the instruct track (running: three A30s in
   parallel, each a limit-10 pilot and then the full suite with `--pilot-limit 10`).
2. Build the comparison table from `plan.md`: change from BF16 and compression ratio, for text and
   vision.
3. Run the thinking track for BF16 and the best variant on one instance.
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
run. `--limit N` runs a non-research pilot on the first N questions of every text task and vision
dataset. `--pilot-limit N` runs that pilot into `<output>-pilot` first and starts the full run only
if it passes, on the same machine: the export the pilot builds is reused and setup is paid once.

```bash
uv run python scripts/run_feature1_variant.py --variant gptq_w4a16_g128 --pilot-limit 10
```

## Local validation (optional)

`scripts/validate_local.py` runs the same suite end to end on the Mac on the first N questions of
every task (default 10): lm-eval tasks, prompts, chat template and `enable_thinking`, VLMEvalKit's API
path with the same datasets, and the fixed extractor. It then checks the outputs and writes
`validation_report.json` with a PASS/FAIL verdict. Results go to `results/local-validate/`. They are
not research results.

```bash
uv run python scripts/validate_local.py --limit 10 --answer-cap 1024
```

What differs from the GPU run, and why:

- **Backend.** vLLM does not run on macOS. Text uses lm-eval's `hf` backend, and vision uses
  `scripts/local_vlm_server.py`, a small OpenAI-compatible transformers server that stands in for
  vLLM's. VLMEvalKit's client side is unchanged.
- **Device.** The Apple GPU (`--device mps`). On the CPU the 4B model generates under 1 token/s.
- **Answer cap.** `--answer-cap` lowers the cap for the local run only. On the Apple GPU (about
  32 tokens/s at batch size 8), one answer that runs to 8,192 tokens holds up its whole batch for
  about 30 minutes. The GPU run keeps the suite's cap, and the report records both values.

The GPU-only pieces (CUDA environments, vLLM) are covered by the GPU driver's preflight, which fails
within minutes. The teardown watcher then destroys the instance.

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
