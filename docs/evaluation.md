# Evaluation protocol

Every model, BF16 and compressed alike, is evaluated with the same settings, so the difference
between two models measures the effect of compression alone.

## Model

| | |
| --- | --- |
| Model | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Architecture | Hybrid vision-language model: 24 Gated DeltaNet (linear attention) layers and 8 full-attention layers, plus a vision encoder |
| Reference precision | BF16, the checkpoint's own dtype |

## Instruct track

The track every compressed model is scored on. Configuration: `configs/evaluation/feature1.yaml`.

| Setting | Value |
| --- | --- |
| Mode | `enable_thinking=False` (Qwen3.5-4B thinks by default; pinned so every model answers directly) |
| Decoding | Greedy (`do_sample=False`), seed 42 |
| Answer limit | 8,192 generated tokens, for text and vision |
| Text context | 12,288 tokens: 4,096 for the prompt beside the 8,192-token answer |
| Vision context | 32,768 tokens (prompts can carry up to 8 images) |

### Text tasks

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.13 on vLLM 0.29.0,
0-shot, with the chat template applied.

| Task | Metric |
| --- | --- |
| MMLU-Pro | exact match, custom extract |
| GSM8K | exact match, flexible extract |
| MATH-500 (`minerva_math500`) | math_verify |
| IFEval | prompt-level strict accuracy |
| HellaSwag | acc_norm |
| ARC-Challenge | acc_norm |
| WikiText-2 | word perplexity (lower is better), windows of 12,288 tokens |

### Vision tasks

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) at revision `34a64e6`. The model is served
with vLLM's OpenAI-compatible server and queried by VLMEvalKit's API client (`LMDeployAPI`) with 32
concurrent requests.

| Dataset | Scoring |
| --- | --- |
| DocVQA (val) | VLMEvalKit rules (ANLS) |
| OCRBench | VLMEvalKit rules |
| TextVQA (val) | VLMEvalKit rules |
| MMBench (dev, EN v1.1) | Fixed answer extractor |
| MMMU (val) | Fixed answer extractor |
| MathVista (mini) | Fixed answer extractor |

**Answer extraction.** In instruct mode the model answers multiple-choice questions with a
worked explanation that VLMEvalKit's rules often cannot parse (under rules alone, MMMU scored
below chance). MMBench, MMMU and MathVista are therefore scored by one fixed extractor,
`gpt-4o-mini` (VLMEvalKit's default judge), after inference. The extractor is identical for every
model and is never the model under evaluation.

**Final-answer rule.** When VLMEvalKit's rule matching and the extractor both fail to identify a
multiple-choice option, VLMEvalKit substitutes a random option. Before that happens, the answer is
taken from the model's last explicit `Final Answer: X` statement, if `X` is a valid option.
Answers that reach the token limit without stating any option still receive a random option. That
affects 4–7 of 900 MMMU validation answers per model, worth at most 0.8 points. Each scoring
manifest counts both cases.

## Reasoning panel

A shorter screen for candidate compressions, run before the full suite. Configuration:
`configs/evaluation/feature1_panel.yaml`.

- **Tasks:** MMLU-Pro (a fixed, subject-stratified subset of 1,001 questions), MATH-500, IFEval and
  MMMU (val), where 4-bit weights lose the most.
- **Settings:** identical to the instruct track. A panel score therefore equals the same questions
  scored from a full run, and `scripts/panel_scores.py` computes it either way.

## Comparing methods

- **Equal size.** Methods are compared at equal checkpoint size (within about 2%). A method that is
  more accurate only because it is larger sits at a different point on the size-accuracy curve.
- **Paired intervals.** `scripts/panel_scores.py --vs <reference>` reports each model's change from a
  reference on the same questions, with paired bootstrap 95% intervals. On the panel these
  intervals span roughly ±2 points (MMLU-Pro) to ±3.5 points (MATH-500), so smaller differences
  need the token-level measure of the [drift study](experiments/03-drift-study.md).

## Thinking track

`configs/evaluation/feature1_thinking.yaml` defines the protocol behind Qwen's published scores:
- thinking mode
- `temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`
- a 32,768-token budget
- two seeds, averaged
- MMLU-Pro (the same 1,001 questions), MATH-500 and IFEval

No results from this track are reported yet.

## Relation to the model card

Qwen's model card reports thinking-mode results produced with sampling, 32,768–81,920-token
budgets, benchmark-specific answer prompts and Qwen's own harness. Scores on the instruct track are
lower on reasoning tasks (for example MMLU-Pro 74.6 against the card's 79.1, MMMU 69.6 against
77.6) but match on tasks that need no reasoning (OCRBench 86.3 against 85.0). The weights are the
same BF16 weights. The gap comes from the protocol, not from numerical precision.
