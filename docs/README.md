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
| [4. AutoRound](experiments/04-autoround.md) | AutoRound W4A16 g128: the strongest 4-bit baseline on the drift measure, its full suite, and its memory needs |
| [5. Glaze v1](experiments/05-glaze-v1.md) | Tuning AutoRound's scales and norms end to end: why it did not beat AutoRound, and what that showed |
| [6. Glaze v2](experiments/06-glaze-v2.md) | Quantizing from BF16 with in-domain data, Fisher-weighted rounding and byte-neutral mixed precision: MATH-500 within 0.2 points of BF16 at AutoRound's size |
| [7. Glaze v2 controls](experiments/07-glaze-controls.md) | AutoRound on Glaze v2's calibration data: the data alone gives 7.0 of Glaze v2's 10.0-point MATH-500 gain |
| [8. Blaze](experiments/08-blaze.md) | Moving the vocabulary table's bytes into the body: the usage census and gates G1-G4 on the 4B (at W4A16's size, 12 of 13 tasks within a point of BF16 and 8.5x below Glaze v2's excess loss), G5 on the 2B, the real 2B and 4B served by vLLM, and drift on code and 11 languages |
| [Engineering notes](engineering-notes.md) | Pitfalls when evaluating and quantizing Qwen3.5 with vLLM, lm-eval, VLMEvalKit and llm-compressor |

## Summary

All results are for `Qwen/Qwen3.5-4B` (revision `851bf6e`), except Blaze's 2B check, in instruct
mode with greedy decoding and up to 8,192 generated tokens.

- **INT8 W8A8 is effectively lossless** (5.51 GB, 1.69× smaller): no task changes by more than
  2 points, and none of the panel changes is statistically distinguishable from zero.
- **4-bit weights (GPTQ, AWQ; 3.8 GB, 2.5× smaller) keep short answers but lose long reasoning:**
  MATH-500 drops 7.6–10 points and MMMU 4–5, while OCR and short-answer tasks stay within about
  1.5 points. Much of the drop is answers that loop until the token limit.
- **The 4-bit error is local:** replayed along BF16's own answers, quantized models disagree with
  BF16 at a constant rate from the first token to the 8,000th. Nothing compounds through the
  DeltaNet recurrent state.
- **No component dominates the 4-bit error:** by excess loss, attention contributes 18% from 8.2% of
  the weights, DeltaNet 27% from 28.3%, and FFN 56% from 63.5%.
- **Glaze v2 closes most of the 4-bit gap at the same size** (3.80 GB): MATH-500 83.2 against BF16's
  83.4 and AutoRound's 73.2 (+10.0 points over AutoRound, paired interval +6.6 to +14.0), and two
  thirds less excess loss than AutoRound on the drift measure, near INT8 W8A8's.
- **Most of that MATH-500 gain is Glaze v2's calibration data:** AutoRound tuned on the same
  in-domain blocks reaches 80.2 (+7.0, paired interval +2.8 to +11.0). Glaze v2's own stages add
  about 3 more points there, and 3.5 on IFEval, where the data alone does not help.
- **Blaze moves the vocabulary table's bytes into the body, with no in-domain data:** storing the
  busiest 65,536 vocabulary rows at 8 bits and the rest at 4 costs almost nothing (excess loss
  0.0002). On the 4B at a standard W4A16 export's size, with the vision tower at 8 bits as in Glaze
  v2, Blaze-Q (a 6/8-bit GPTQ body) reaches excess loss 0.0017, 8.5x below Glaze v2's, and on the
  full suite is within a point of BF16 on 12 of 13 tasks: MMLU-Pro 75.2 (BF16 74.6), MATH-500 83.0
  (83.4), ARC-Challenge 51.0 (51.1), MMMU 68.7 (69.6); MathVista is 79.9 against 81.0 (paired
  interval −2.6 to +0.5). The real 4B checkpoint, 385 KB under GPTQ W4A16's, drifts exactly as
  the export those scores came from and is served by vLLM 0.29's own 6-bit kernel at 147 tokens per
  second single stream (GPTQ W4A16 151, BF16 80), holding GPTQ W4A16's 3.92 GB of GPU weights now
  that the plugin keeps its vocabulary once. On code and math in 11 languages it drifts 15x and
  13x less than GPTQ W4A16; the ladder's English-only census is its weak spot there, and a census
  with FineWeb-2 in 30 languages cuts the ladder's multilingual cost 2.6x at the same size. On
  Qwen3.5-2B at W4A16's size it buys an all-8-bit body: MATH-500 67.8 and
  MMLU-Pro 56.7 against BF16's 64.6 and 56.4 and W4A16's 52.4 and 45.9. That 2B export is served for
  real by vLLM with a small plugin, with the same scores, W4A16's size on disk, and 229 tokens per
  second single stream (W4A16 240, BF16 164).
- **The ladder alone shrinks any 4-bit export for free:** applied to the published GPTQ, AutoRound
  and Glaze v2 checkpoints, changing only the BF16 vocabulary table, it cuts each from 3.78 to 2.92
  GB (−23%) with excess loss unchanged to 0.0001, and decodes a single stream 16% faster. Glaze v2's
  quality at 2.91 GB serves today on vLLM with a small plugin. With the vision tower also at 8 bits,
  AutoRound's body becomes Blaze-S: 2.60 GB (−31%), excess loss 0.0418 against AutoRound's 0.0417,
  174 tokens per second single stream against 151, 2.74 GB of GPU weights against 3.92, and MMMU
  within noise of AutoRound's (−1.1, interval −3.3 to +1.0).

## Released artifacts

| Artifact | Contents |
| --- | --- |
| [lazybrick/Qwen3.5-4B-Kiln-INT8-W8A8](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-INT8-W8A8) | INT8 weights and activations (SmoothQuant + GPTQ) |
| [lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-GPTQ-W4A16-g128) | 4-bit weights, GPTQ, group size 128 |
| [lazybrick/Qwen3.5-4B-Kiln-AWQ-W4A16-g128](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-AWQ-W4A16-g128) | 4-bit weights, AWQ, group size 128 |
| [lazybrick/Qwen3.5-4B-Kiln-AutoRound-W4A16-g128](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-AutoRound-W4A16-g128) | 4-bit weights, AutoRound, group size 128 |
| [lazybrick/Qwen3.5-4B-Kiln-Glaze-W4A16](https://huggingface.co/lazybrick/Qwen3.5-4B-Kiln-Glaze-W4A16) | Glaze v2: 4-bit weights (group size 128, 64 or 32) with the most sensitive layers and the vision tower at 8 bits, at AutoRound's size |
| [lazybrick/kiln-evals](https://huggingface.co/datasets/lazybrick/kiln-evals) | Per-sample outputs, predictions, scores and run manifests for BF16 and each variant |
