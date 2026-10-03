# 2. Quantization baselines

Three established post-training quantization methods, applied to the language model of
`Qwen/Qwen3.5-4B` and scored with the [instruct-track protocol](../evaluation.md#instruct-track).

## Methods

| Variant | Method | Weights | Activations |
| --- | --- | --- | --- |
| INT8 W8A8 | SmoothQuant (strength 0.8), then GPTQ | INT8, symmetric, per output channel | INT8, symmetric, dynamic per token |
| GPTQ W4A16 g128 | GPTQ (static activation ordering, dampening 0.01) | INT4, symmetric, group size 128 | BF16 |
| AWQ W4A16 g128 | AWQ (duo scaling), then round-to-nearest | INT4, asymmetric, group size 128 | BF16 |

Shared settings:

- **Not quantized:** the vision encoder, the token embeddings and `lm_head`.
- **Calibration:** 512 conversations from
  [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
  (`train_sft`, revision `8049631`), sampled with seed 42, chat template applied, truncated to
  2,048 tokens. The set is frozen with a SHA-256 lock (`data/calibration/feature1.lock.json`).
- **Toolkit:** llm-compressor 0.13.0, compressed-tensors 0.18.0, torch 2.10.0.
- **Format:** compressed-tensors safetensors, served by vLLM 0.29.0.
- **Code:** `f4e2433` for INT8 and GPTQ; `0d41716` for AWQ, which adds CPU offload of AWQ's
  calibration cache (see [engineering notes](../engineering-notes.md#llm-compressor)).

Quantization on one NVIDIA A30 (24 GB):

| Variant | Time | Peak GPU memory | Checkpoint |
| --- | ---: | ---: | ---: |
| INT8 W8A8 | 30 min | 9.9 GiB | 5.51 GB |
| GPTQ W4A16 g128 | 24 min | 10.0 GiB | 3.78 GB |
| AWQ W4A16 g128 | 90 min | 14.8 GiB | 3.79 GB |

BF16 is 9.32 GB.

## Results

Change from BF16 in parentheses.

| Task | BF16 | INT8 W8A8 | GPTQ W4A16 g128 | AWQ W4A16 g128 |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro | 74.6 | 74.3 (−0.3) | 71.4 (−3.2) | 71.9 (−2.7) |
| GSM8K | 83.2 | 83.5 (+0.3) | 81.9 (−1.3) | 82.6 (−0.6) |
| MATH-500 | 83.4 | 82.6 (−0.8) | 75.8 (−7.6) | 73.4 (−10.0) |
| IFEval | 82.3 | 82.1 (−0.2) | 80.8 (−1.5) | 79.3 (−3.0) |
| HellaSwag | 65.4 | 65.2 (−0.2) | 64.3 (−1.1) | 64.8 (−0.6) |
| ARC-Challenge | 51.1 | 49.7 (−1.4) | 50.8 (−0.3) | 50.0 (−1.1) |
| WikiText-2 (lower is better) | 10.95 | 11.09 | 11.43 | 11.55 |
| MMBench (dev, EN v1.1) | 85.4 | 83.7 (−1.7) | 84.1 (−1.3) | 82.9 (−2.5) |
| MMMU (val) | 69.6 | 67.7 (−1.9) | 64.9 (−4.7) | 65.7 (−3.9) |
| MathVista (mini) | 81.0 | 81.8 (+0.8) | 78.0 (−3.0) | 78.0 (−3.0) |
| OCRBench | 86.3 | 87.5 (+1.2) | 86.0 (−0.3) | 87.1 (+0.8) |
| DocVQA (val) | 95.3 | 95.3 (0.0) | 94.8 (−0.5) | 95.1 (−0.2) |
| TextVQA (val) | 82.8 | 82.2 (−0.6) | 81.7 (−1.1) | 81.8 (−1.0) |
| Checkpoint size | 9.32 GB | 5.51 GB (1.69×) | 3.78 GB (2.47×) | 3.79 GB (2.46×) |

### Reasoning panel, with paired 95% intervals

Change from BF16 on the same questions (paired bootstrap, 2,000 resamples):

| Variant | MMLU-Pro (1,001) | MATH-500 | IFEval |
| --- | ---: | ---: | ---: |
| INT8 W8A8 | +0.1 [−2.1, +2.2] | −0.8 [−3.8, +2.2] | −0.2 [−3.0, +2.4] |
| GPTQ W4A16 g128 | −3.5 [−5.9, −1.2] | −7.6 [−11.2, −3.8] | −1.5 [−4.6, +1.5] |
| AWQ W4A16 g128 | −2.5 [−4.8, −0.3] | −10.0 [−13.8, −6.0] | −3.0 [−6.1, +0.0] |

### Answer length

On MATH-500, quantized models write longer answers and loop until the limit more often:

| Variant | Mean answer length | Answers at the 8,192-token limit | Correct among those |
| --- | ---: | ---: | ---: |
| BF16 | 1,384 tokens | 18 (3.6%) | 6 |
| INT8 W8A8 | 1,500 | 27 (5.4%) | 10 |
| AWQ W4A16 g128 | 1,663 | 32 (6.4%) | 4 |
| GPTQ W4A16 g128 | 1,696 | 40 (8.0%) | 3 |

On MMLU-Pro's math questions, the share of answers at the limit rises from 3.0% (BF16) to 5.3%
(INT8) and 7.3% (GPTQ).

## Findings

- **INT8 W8A8 is effectively lossless.** No panel change is distinguishable from zero. Its largest
  drops are on vision reasoning (MMBench −1.7, MMMU −1.9), although the vision encoder is not
  quantized: the language model reasons over the image tokens.
- **4-bit weights cost long reasoning, not short answers.** Both W4A16 variants hold short-answer
  and OCR tasks within about 1.5 points but lose 7.6–10 points on MATH-500 and 3–5 points on MMMU
  and MathVista.
- **Much of the 4-bit loss is answers that never finish.** For GPTQ, the extra answers that loop
  into the limit account for about 5 of its 7.6 MATH-500 points. AWQ loops less but gives more wrong
  final answers.
- **AWQ is not better than GPTQ on this model.** The two are statistically indistinguishable on the
  panel. AWQ is slightly ahead on MMLU-Pro and GSM8K and behind on MATH-500 and IFEval.

## Artifacts

- Models:
  - [Qwen3.5-4B-Kiln-INT8-W8A8](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-INT8-W8A8)
  - [Qwen3.5-4B-Kiln-GPTQ-W4A16-g128](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128)
  - [Qwen3.5-4B-Kiln-AWQ-W4A16-g128](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-AWQ-W4A16-g128)
- Records: [`lazybrick/kiln-evals`](https://huggingface.co/datasets/lazybrick/kiln-evals), one
  directory per variant.
