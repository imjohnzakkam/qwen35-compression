# Documentation

Records of the Qwen3.5 compression experiments: what was run, how it was measured, and what was
found. Every result here was produced by the code in this repository at the revision stated in
each record.

| Document | Contents |
| --- | --- |
| [Evaluation protocol](evaluation.md) | Model, tasks, decoding, harnesses, answer extraction, and how methods are compared |
| [1. BF16 baseline](experiments/01-bf16-baseline.md) | The reference every compressed model is measured against |
| [2. Quantization baselines](experiments/02-quantization-baselines.md) | INT8 W8A8, GPTQ W4A16 and AWQ W4A16 on the full suite |
| [3. Drift study](experiments/03-drift-study.md) | Where and how quantization error appears, token by token and by component |
| [4. AutoRound](experiments/04-autoround.md) | AutoRound W4A16 g128: the strongest 4-bit baseline on the drift measure, and its memory needs |
| [Engineering notes](engineering-notes.md) | Pitfalls when evaluating and quantizing Qwen3.5 with vLLM, lm-eval, VLMEvalKit and llm-compressor |

## Summary

All results are for `Qwen/Qwen3.5-4B` (revision `851bf6e`) in instruct mode with greedy decoding
and up to 8,192 generated tokens.

- **INT8 W8A8 is effectively lossless** (5.51 GB, 1.69× smaller): no task changes by more than
  2 points, and none of the panel changes is statistically distinguishable from zero.
- **4-bit weights (GPTQ, AWQ; 3.8 GB, 2.5× smaller) keep short answers but lose long reasoning:**
  MATH-500 drops 7.6–10 points and MMMU 4–5, while OCR and short-answer tasks stay within about
  1.5 points. Much of the drop is answers that loop until the token limit.
- **The 4-bit error is local:** replayed along BF16's own answers, quantized models disagree with
  BF16 at a constant rate from the first token to the 8,000th. Nothing compounds through the
  DeltaNet recurrent state.
- **No component dominates the 4-bit error:** by excess loss, attention contributes 18% from 9% of
  the weights, DeltaNet 27% from 27.5%, and FFN 56% from 63.5%.

## Released artifacts

| Artifact | Contents |
| --- | --- |
| [lazybrick/Qwen3.5-4B-Kiln-INT8-W8A8](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-INT8-W8A8) | INT8 weights and activations (SmoothQuant + GPTQ) |
| [lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128) | 4-bit weights, GPTQ, group size 128 |
| [lazybrick/Qwen3.5-4B-Kiln-AWQ-W4A16-g128](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-AWQ-W4A16-g128) | 4-bit weights, AWQ, group size 128 |
| [lazybrick/kiln-evals](https://huggingface.co/datasets/lazybrick/kiln-evals) | Per-sample outputs, predictions, scores and run manifests for BF16 and each variant |
