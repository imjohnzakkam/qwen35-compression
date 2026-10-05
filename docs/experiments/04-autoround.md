# 4. AutoRound

[AutoRound](https://aclanthology.org/2024.findings-emnlp.662.pdf) tunes each block's weight rounding
and clipping by signed gradient descent against the BF16 block's outputs. It is a strong
post-training baseline for 4-bit weights.

**Status:** quantized, scored with the token-level measure of the [drift study](03-drift-study.md),
and run on the full benchmark suite. [Glaze v2](06-glaze-v2.md) compares against it.

## Configuration

`autoround_w4a16_g128` in `configs/variants/feature1.yaml`:
- llm-compressor's `AutoRoundModifier` (auto-round 0.14.2)
- W4A16, symmetric, group size 128, the same layers left unquantized as GPTQ and AWQ
- AutoRound's defaults: 200 tuning steps per block, batch size 8, 128 calibration samples of 2,048
  tokens
- `torch.compile` off

**Calibration.** The same 512 conversations as the other methods, tokenized, joined in order and cut
into 2,048-token blocks (285 blocks, 585k tokens). AutoRound uses the first 128 blocks
(`calibration_samples: 128`). It stacks every sample's cached inputs into one tensor, so all samples
need the same length.

**Environment.** flash-linear-attention 0.5.2 is installed before quantizing (see
[engineering notes](../engineering-notes.md#flash-linear-attention)), and quantization runs with
`PYTORCH_ALLOC_CONF=expandable_segments:True`.

| | |
| --- | --- |
| Code | `c3a16a0` |
| Hardware | One NVIDIA A100 (40 GB) |
| Quantization | 16 min, peak GPU memory 34.9 GiB |
| Checkpoint | 3.80 GB (GPTQ 3.78 GB, AWQ 3.79 GB) |

## Results

Drift on BF16's 1,501 MATH-500 and MMLU-Pro answers (2.3M tokens). Excess loss is the mean
negative log-likelihood above BF16's own.

| Model | Flip rate | Loss | Excess loss |
| --- | ---: | ---: | ---: |
| BF16 (noise floor) | 0.35% | 0.128 | — |
| INT8 W8A8 | 2.42% | 0.139 | 0.011 |
| AutoRound W4A16 g128 | 4.96% | 0.170 | 0.042 |
| AWQ W4A16 g128 | 5.42% | 0.178 | 0.050 |
| GPTQ W4A16 g128 | 5.70% | 0.182 | 0.054 |

- **AutoRound is the strongest 4-bit baseline.** At the same size it removes 23% of GPTQ's excess
  loss and 16% of AWQ's. The ordering holds on both task groups: flip rate 3.5% (MATH-500) and 5.6%
  (MMLU-Pro), against 3.7% and 6.1% for AWQ and 4.1% and 6.4% for GPTQ.
- **4-bit headroom remains.** AutoRound's excess loss is still about 4 times INT8's, and its flip
  rate is flat along the answer, like the other methods' (5.7% in the first 256 tokens, 5.2% beyond
  4,096).

### Benchmarks

The full suite with the [instruct-track protocol](../evaluation.md#instruct-track), on one A30
(code `96438ec`; DocVQA completed separately with `configs/evaluation/feature1_docvqa.yaml`, code
`9f50193`, after the first run reached its time limit during that task). Change from BF16
in parentheses.

| Task | BF16 | GPTQ | AWQ | AutoRound |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro | 74.6 | 71.4 | 71.9 | 72.8 (−1.8) |
| GSM8K | 83.2 | 81.9 | 82.6 | 82.3 (−0.9) |
| MATH-500 | 83.4 | 75.8 | 73.4 | 73.2 (−10.2) |
| IFEval | 82.3 | 80.8 | 79.3 | 80.6 (−1.7) |
| HellaSwag | 65.4 | 64.3 | 64.8 | 64.8 (−0.6) |
| ARC-Challenge | 51.1 | 50.8 | 50.0 | 50.0 (−1.1) |
| WikiText-2 (lower is better) | 10.95 | 11.43 | 11.55 | 11.45 |
| MMBench (dev, EN v1.1) | 85.4 | 84.1 | 82.9 | 84.7 (−0.7) |
| MMMU (val) | 69.6 | 64.9 | 65.7 | 66.9 (−2.7) |
| MathVista (mini) | 81.0 | 78.0 | 78.0 | 80.3 (−0.7) |
| OCRBench | 86.3 | 86.0 | 87.1 | 87.2 (+0.9) |
| DocVQA (val) | 95.3 | 94.8 | 95.1 | 95.3 (0.0) |
| TextVQA (val) | 82.8 | 81.7 | 81.8 | 82.5 (−0.3) |

- **Lowest drift is not the best MATH-500.** AutoRound has the least excess loss of the three 4-bit
  baselines, yet scores 73.2 on MATH-500 (−10.2, paired interval −14.0 to −6.6), level with AWQ.
  7.0% of its MATH-500 answers reach the 8,192-token limit (BF16 3.6%).
- **It is the strongest 4-bit baseline on vision reasoning** (MMMU, MathVista, MMBench).
- **Run-to-run spread:** an earlier MATH-500 run of the same export on an A100 scored 74.6.

## Memory

Four earlier attempts ran out of memory on the first decoder layer:

| GPU | Calibration | Batch | Outcome |
| --- | --- | ---: | --- |
| A30, 24 GB | 512 conversations, variable length | 8 | Stopped: cached inputs of different lengths cannot be stacked (`Sizes of tensors must match`) |
| A30, 24 GB | 285 packed blocks | 8 | Out of memory computing the BF16 reference outputs (21.0 GiB allocated) |
| A30, 24 GB | 285 packed blocks | 2 | Out of memory at the same step (20.1 GiB allocated) |
| A100, 40 GB | 285 packed blocks | 8 | Out of memory in the first tuning step (36.4 GiB allocated) |
| A100, 40 GB | 128 packed blocks, flash-linear-attention | 8 | Completed, peak 34.9 GiB |

Three things occupy the GPU at once:
- **The whole model.** The model is loaded with `device_map="auto"`, so all 9.3 GB of weights stay
  on the GPU.
- **Per-sample caches.** AutoRound keeps every calibration sample's block inputs, BF16 reference
  outputs and quantized-block outputs on the GPU. The cache scales with the sample count, not with
  the batch size, which is why batch 2 did not help.
- **The tuning step.** Without flash-linear-attention, the forward and backward pass of one DeltaNet
  layer on 8 × 2,048 tokens needs about 9 GiB beyond the weights. With it, about 3 GiB.

Even with 128 samples and flash-linear-attention, the peak exceeds a 24 GB GPU.
