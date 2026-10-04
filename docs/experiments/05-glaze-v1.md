# 5. Glaze v1: refining AutoRound's scales and norms

Glaze v1 kept the INT4 codes of [AutoRound](04-autoround.md)'s export and tuned only its group scales
and RMSNorm weights, end to end, to bring the model's next-token distribution closer to BF16's. It
did not beat AutoRound: on the drift measure its excess loss was 2.4% higher.

**Status:** closed. Gate 1 (at least 5% less excess loss than AutoRound, paired interval excluding
zero) returned *stop*. The findings shape [Glaze v2](#what-it-taught).

## Method

- **Student:** AutoRound's W4A16 g128 export with its codes frozen. Trainable: 27,883,520 group
  scales and 170,496 norm weights, all stored in BF16.
- **Objective:** forward KL from the BF16 teacher over the full vocabulary, on packed 2,048-token
  calibration blocks, 16,384 tokens per step.
- **Optimizer (`GridDescent`):** each step moves the values with the largest predicted decrease
  (gradient average × one grid step) by exactly one BF16 grid step, so every intermediate model is
  one the export can store. A guard refuses any move that changes a value's multiplier
  (`1 + w` for Qwen3.5's RMSNorm) by more than 1%.
- **Selection:** the checkpoint with the lowest KL on 32 held-out blocks (128–159) is exported; the
  initial model counts. A study stops before scoring if that KL falls by less than 2%.
- **Scoring:** the [drift study](03-drift-study.md)'s replay of BF16's 1,501 MATH-500 and MMLU-Pro
  answers, and MATH-500 free generation, for AutoRound and Glaze side by side.

| | |
| --- | --- |
| Code | `17838bf` and `6dc9a2a` (attempts 1–2), `78e2c0d` (diagnosis), `551f906` (B1), `c6339d8` (B1b) |
| Hardware | One NVIDIA A100 (40 GB) |
| Variants | `glaze_v1_w4a16_g128` (B1), `glaze_v1b_w4a16_g128` (B1b) |

## Results

### Optimizer

The first two attempts failed their five-step pilot on a fixed batch, so neither trained.

| Attempt | Update | Fixed-batch KL, steps 1–5 |
| --- | --- | --- |
| 1 | Adam on fp32 masters of BF16 values | 0.0074, 0.0075, 0.0076, 0.0870, 0.0612 |
| 2 | One grid step for the 2,805 most promising values per step | 0.0075, 0.0099, 0.0089, 0.0094, 0.0089 |

Adam moved every value by about the same amount, so nearly all of them crossed a grid point in the
same step. A diagnosis run measured each set of one-step moves both ways from the init:

| Moves (one grid step each) | KL change down | KL change up | Best fraction of a step |
| --- | ---: | ---: | ---: |
| Top 28 by predicted gain | −0.000014 | +0.000404 | 0.30 |
| Top 280 | +0.000008 | +0.001147 | 0.39 |
| Top 2,805 | +0.002419 | +0.005191 | 0.21 |
| 2,805 random scales | +0.000125 | +0.000064 | — |

- **The gradient was right, the step too big.** The measured slope along the moves matched the
  gradient's prediction (0.82–1.27× for the larger sets), but the best move was 0.2–0.4 of a BF16 grid step.
- **281 moves per step trained;** 2,805 did not, with or without norm weights.
- **Changes under about 1e-4 are noise** on a 16,384-token batch: moving random scales shifted the KL
  that much where the gradient predicted 1e-5.

### B1: AutoRound's own calibration blocks

Trained on the 128 blocks AutoRound calibrated on (0–127), no checkpoint lowered the held-out KL
(0.034554 at the init): every probe and every evaluation stayed between +0.06% and +2.0%. The study
stopped before scoring.

AutoRound's export has a KL of about 0.0075–0.009 on those blocks against 0.035 on held-out blocks
from the same data: it fits its calibration data about four to five times better than new text.

### B1b: blocks AutoRound never saw

Trained on blocks 160–279, the starting KL was 0.040, and the held-out KL fell 2.65% (best at step 48
of 60; 11,775 scales and 583 norm weights changed). That was enough to score:

| Measure | AutoRound | Glaze B1b | Glaze − AutoRound (95% interval) |
| --- | ---: | ---: | --- |
| Drift excess loss | 0.04170 | 0.04268 | +0.00098 (+0.00092 to +0.00104) |
| Drift flip rate | 4.96% | 4.95% | −0.016 points (−0.025 to −0.008) |
| MATH-500 | 74.6% | 75.0% | +0.4 points (−3.0 to +3.6) |
| MATH-500 answers at the token limit | 26 | 33 | |

Excess loss rose 2.4%, so the gate returned *stop*. MATH-500 did not change measurably.

## What it taught

- **AutoRound overfits its calibration data.** About 5× lower KL on its own 128 blocks than on
  unseen blocks.
- **Calibration domain decides what improves.** A 2.65% gain on held-out chat text became a 2.4% loss
  on math and multiple-choice answers.
- **BF16 scales are too coarse to tune afterwards.** The best moves are a fraction of one grid step.
- **The rounding decisions are the large lever,** and Glaze v1 froze them.

Glaze v2 therefore quantizes from BF16: it learns codes and scales together with scales at BF16
precision, calibrates on BF16's own answers to in-domain prompts with held-out early stopping, and
spends the same bytes where the error is.
