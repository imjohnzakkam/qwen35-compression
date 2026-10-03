# 1. BF16 baseline

The reference every compressed model is measured against: `Qwen/Qwen3.5-4B` in its native BF16, on
the [instruct track](../evaluation.md#instruct-track).

| | |
| --- | --- |
| Code | `9fd9b46` |
| Hardware | One NVIDIA A30 (24 GB) |
| Records | [`lazybrick/kiln-evals/bf16`](https://huggingface.co/datasets/lazybrick/kiln-evals/tree/main/bf16) |

## Results

| Task | Metric | Score |
| --- | --- | ---: |
| MMLU-Pro | exact match | 74.6 |
| GSM8K | exact match, flexible extract | 83.2 |
| MATH-500 | math_verify | 83.4 |
| IFEval | prompt-level strict | 82.3 |
| HellaSwag | acc_norm | 65.4 |
| ARC-Challenge | acc_norm | 51.1 |
| WikiText-2 | word perplexity (lower is better) | 10.95 |
| MMBench (dev, EN v1.1) | accuracy | 85.4 |
| MMMU (val) | accuracy | 69.6 |
| MathVista (mini) | accuracy | 81.0 |
| OCRBench | score | 86.3 |
| DocVQA (val) | ANLS | 95.3 |
| TextVQA (val) | accuracy | 82.8 |

## Answer limit

The baseline was first run with a 2,048-token answer limit. That limit cut off 8.2% of MMLU-Pro,
14.6% of MATH-500 and 10.8% of MMMU answers, most of which were then scored wrong. A compressed
model that writes longer answers would have lost accuracy to truncation alone, so the limit was
raised to 8,192 tokens and the baseline rerun.

| Task | 2,048-token limit | 8,192-token limit |
| --- | ---: | ---: |
| MMLU-Pro | 67.0 | 74.6 |
| GSM8K | 82.9 | 83.2 |
| MATH-500 | 73.2 | 83.4 |
| IFEval | 82.8 | 82.3 |
| HellaSwag | 65.4 | 65.4 |
| ARC-Challenge | 50.6 | 51.1 |
| WikiText-2 | 11.47 | 10.95 |
| MMBench (dev, EN v1.1) | 84.8 | 85.4 |
| MMMU (val) | 63.4 | 69.6 |
| MathVista (mini) | 80.8 | 81.0 |
| OCRBench | 86.3 | 86.3 |
| DocVQA (val) | 95.4 | 95.3 |
| TextVQA (val) | 83.0 | 82.8 |

At 8,192 tokens, 4.8% of MMLU-Pro and 3.6% of MATH-500 answers still reach the limit. Almost all of
these are repetition loops that more tokens would not resolve. WikiText-2 perplexity changed
because the evaluation context, and so the perplexity window, grew from 4,096 to 12,288 tokens.

## Notes

- **Model card.** The weights are the same BF16 weights behind Qwen's model card, but the card
  reports thinking mode with sampling and much larger token budgets (see
  [Relation to the model card](../evaluation.md#relation-to-the-model-card)). Instruct-track scores
  are lower on reasoning tasks and match on OCR.
- **Likelihood tasks.** HellaSwag and ARC-Challenge read lower than commonly reported for this model
  size. Both are likelihood-scored and run here through the chat template, which is the likely
  cause. Every model uses the same protocol, so comparisons between models are unaffected.
