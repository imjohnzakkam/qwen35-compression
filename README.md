# qwen35-compression

Reproducible post-training compression of the Qwen3.5 vision-language models, measured against BF16
under one fixed evaluation protocol.

## Results

`Qwen/Qwen3.5-4B`, instruct mode (`enable_thinking=False`), greedy decoding, up to 8,192 generated
tokens. The embeddings and `lm_head` stay in BF16 in every variant, and the vision encoder in every
variant but Glaze v2, which stores it at 8 bits.

| Task | BF16 | INT8 W8A8 | GPTQ W4A16 g128 | AWQ W4A16 g128 | AutoRound W4A16 g128 | Glaze v2 W4A16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro | 74.6 | 74.3 | 71.4 | 71.9 | 72.8 | 73.7 |
| GSM8K | 83.2 | 83.5 | 81.9 | 82.6 | 82.3 | 82.8 |
| MATH-500 | 83.4 | 82.6 | 75.8 | 73.4 | 73.2 | **83.2** |
| IFEval | 82.3 | 82.1 | 80.8 | 79.3 | 80.6 | 82.8 |
| HellaSwag | 65.4 | 65.2 | 64.3 | 64.8 | 64.8 | 64.8 |
| ARC-Challenge | 51.1 | 49.7 | 50.8 | 50.0 | 50.0 | 49.6 |
| WikiText-2 perplexity (lower is better) | 10.95 | 11.09 | 11.43 | 11.55 | 11.45 | 11.30 |
| MMBench (dev, EN v1.1) | 85.4 | 83.7 | 84.1 | 82.9 | 84.7 | 85.7 |
| MMMU (val) | 69.6 | 67.7 | 64.9 | 65.7 | 66.9 | 66.7 |
| MathVista (mini) | 81.0 | 81.8 | 78.0 | 78.0 | 80.3 | 80.6 |
| OCRBench | 86.3 | 87.5 | 86.0 | 87.1 | 87.2 | 86.3 |
| DocVQA (val) | 95.3 | 95.3 | 94.8 | 95.1 | 95.3 | 95.4 |
| TextVQA (val) | 82.8 | 82.2 | 81.7 | 81.8 | 82.5 | 82.2 |
| Checkpoint size | 9.32 GB | 5.51 GB | 3.78 GB | 3.79 GB | 3.80 GB | 3.80 GB |

- **INT8 W8A8 is effectively lossless.**
- **Uniform 4-bit weights hold short answers but lose long reasoning** (GPTQ, AWQ, AutoRound:
  MATH-500 −7.6 to −10.2, MMMU −2.7 to −4.7), much of it answers that loop until the token limit.
- **The 4-bit error is local and spread across the network.** It does not compound along an answer,
  and no single component dominates.
- **Glaze v2 keeps long reasoning at 4 bits.** At AutoRound's size it scores 83.2 on MATH-500 (BF16
  83.4, AutoRound 73.2) and stays within 1 point of BF16 on every task but ARC-Challenge and MMMU. It
  quantizes from BF16 on BF16's own answers to in-domain prompts, learns its rounding against a
  Fisher-weighted objective, and spends the bytes an 8-bit vision tower frees on the most
  sensitive layers ([record](docs/experiments/06-glaze-v2.md)).

Details are in [`docs/`](docs/README.md): the evaluation protocol, one record per experiment, and
engineering notes.

**Models:** [Kiln collection](https://huggingface.co/collections/lazybrick/kiln-qwen35-4b-fired-small-6ac04982f4f1ce50a97da56d).
**Per-sample records:** [`lazybrick/kiln-evals`](https://huggingface.co/datasets/lazybrick/kiln-evals).

## Setup

```bash
uv sync
uv run pytest
```

Configuration, dry runs and tests work on any platform. Quantization and evaluation need Linux and
an NVIDIA GPU (24 GB is enough). The drivers build the separate pinned environments they need (text
evaluation, vision evaluation, compression) on first use.

## Reproduce

```bash
# BF16 baseline: text and vision suites
uv run python scripts/run_feature1_bf16.py

# A quantized variant from configs/variants/feature1.yaml: a 10-question pilot, then the full suite
uv run python scripts/run_feature1_variant.py --variant gptq_w4a16_g128 --pilot-limit 10

# Score MMBench, MMMU and MathVista with the fixed answer extractor (reads OPENAI_API_KEY from
# ~/.config/qwen35/openai.env)
uv venv .venv-vision-score --python 3.11
uv pip install --python .venv-vision-score/bin/python -r requirements/vision-score.lock
uv pip install --python .venv-vision-score/bin/python --no-deps -e external/VLMEvalKit
uv run python scripts/score_vision.py --run-dir results/feature1/gptq_w4a16_g128

# Reasoning-panel scores and paired intervals against BF16
uv run python scripts/panel_scores.py --run bf16=results/feature1/bf16 \
  --run gptq=results/feature1/gptq_w4a16_g128 --vs bf16 --loops

# Glaze v2 on the 4B: BF16 answers, a pilot, quantization, a held-out gate against AutoRound's
# published export, drift scores and the full suite (one 40 GB GPU)
uv run python scripts/run_glaze2_4b.py

# Drift study: token-level divergence from BF16, by position and by component
uv run python scripts/run_drift_study.py --pilot-limit 20
```

Every run writes a manifest with the model revision, code revision, config and suite digests and
package versions. Every compressed checkpoint carries a SHA-256 inventory
(`compression_manifest.json`).

## Layout

| Path | Contents |
| --- | --- |
| `configs/` | Model, calibration and variant definitions (`feature1.yaml` is the Qwen3.5-4B study); evaluation suites in `configs/evaluation/` |
| `data/calibration/` | Frozen calibration sets and their SHA-256 locks |
| `scripts/` | Drivers for quantization, evaluation, scoring and analysis |
| `src/qwen35_compression/` | Library code: configuration, recipes, evaluation commands, manifests |
| `tests/` | Unit tests |
| `docs/` | Protocol, experiment records and engineering notes |
