# 7. Glaze v2 controls: how much of the gain is the data?

[Glaze v2](06-glaze-v2.md) scores 83.2 on MATH-500 against AutoRound's 73.2 at the same size, but
it changes several things at once: the calibration data, the rounding objective, and the precision
of each layer. This record tests the first of these. AutoRound is run unchanged on Glaze v2's own
calibration blocks.

**Status:** the data control is done. Glaze v2's calibration data alone takes AutoRound from 73.2 to
80.2 on MATH-500, 7.0 of the 10.0 points between them. The allocation control (AutoRound's own
mixed precision at Glaze v2's budget) passed its pilot but has not been run in full.

## Design

Two controls, both tuned by AutoRound on Glaze v2's calibration blocks:

| Control | Variant | What changes from AutoRound's export |
| --- | --- | --- |
| Data | `autoround_glaze2_data_w4a16_g128` | Calibration data only: Glaze v2's blocks instead of UltraChat. Every Linear stays at 4-bit g128. |
| Allocation | `autoround_autoscheme_w4a16` | Data as above, plus a per-Linear precision chosen by AutoRound's AutoScheme from Glaze v2's options, at Glaze v2's language-model budget |

- **Data.** The blocks are rebuilt from the BF16 answers saved by Glaze v2's run. Their SHA-256
  digests match that run's for all four sets (512 calibration and 64 validation blocks, and the
  pilot's 16 and 8).
- **Volume.** AutoRound keeps every calibration sample's block inputs and outputs on the GPU: 128
  blocks peak at 34.9 GiB on a 40 GB A100 and 285 run out of memory
  ([AutoRound record](04-autoround.md#memory)). Both controls therefore use the first 128 of Glaze
  v2's 512 blocks. The blocks interleave the domains, so these 128 keep the 40/40/20 mix of math,
  multiple-choice and chat tokens. AutoRound's own export also uses 128 blocks.
- **Tuning.** AutoRound's defaults, as in its export: 200 steps per block, batch 8, through
  llm-compressor 0.13.0.
- **Allocation budget.** AutoScheme targets the average bits per weight of Glaze v2's own
  language-model allocation, counted as AutoScheme counts them: 4.909 over the same 248 Linears.
  Its options are Glaze v2's (4-bit with group size 128, 64 or 32; 8-bit with group size 128), with
  vLLM's fused Linears tied to one option. The vision tower stays in BF16, as in AutoRound's
  export. Text tasks never run it, so the text comparison is at equal language-model bytes.
- **Evaluation.** MATH-500 and IFEval with the
  [instruct-track protocol](../evaluation.md#instruct-track), after a two-question pilot.
  MATH-500 is where Glaze v2 and AutoRound differ, and IFEval checks that in-domain calibration does
  not cost instruction following. Validation KL to BF16 is on Glaze v2's 64 validation blocks, as in
  [record 6](06-glaze-v2.md#validation-kl).

| | |
| --- | --- |
| Code | `20849ad` (`scripts/run_glaze_controls.py`) |
| Hardware | One NVIDIA A100 (40 GB) |
| Data control quantization | 14.5 min |
| Data control checkpoint | 3,800,831,116 bytes (AutoRound: 3,800,831,215) |
| Published exports scored | AutoRound `091cfd6`, Glaze v2 `aee82c5` |

## Results

| Model | MATH-500 | IFEval | In-domain KL | Chat KL |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 83.4 | 82.3 | — | — |
| AutoRound W4A16 g128 | 73.2 | 80.6 | 0.0305 | 0.0435 |
| **AutoRound on Glaze v2's data** | **80.2** | **79.3** | **0.0178** | **0.0440** |
| Glaze v2 | 83.2 | 82.8 | 0.0123 | 0.0310 |

Glaze v2's validation KL reproduces its record exactly (0.0123 and 0.0310), so the validation
blocks are the same.

Paired on the same questions (bootstrap, 2,000 resamples, 95% intervals):

| Comparison | MATH-500 | IFEval |
| --- | ---: | ---: |
| Glaze data − AutoRound | **+7.0 [+2.8, +11.0]** | −1.3 [−4.1, +1.7] |
| Glaze v2 − Glaze data | +3.0 [−0.2, +6.2] | **+3.5 [+0.7, +6.3]** |
| Glaze data − BF16 | −3.2 [−6.8, +0.2] | −3.0 [−5.7, −0.2] |

On MATH-500, the data control alone solves 73 questions AutoRound misses, and AutoRound alone
solves 38. Glaze v2 alone solves 44 that the data control misses, and the data control alone 29.

### Allocation control (pilot only)

The pilot ran every stage on 16 blocks with 10 steps per block. AutoScheme scored the layers on 2
samples of 512 tokens:
- **Allocation:** 4.909 bits per weight against a target of 4.909. The language model takes
  2,145,971,200 bytes against Glaze v2's 2,156,400,640.
- **Choices, per Linear:** 131 at group size 32, 47 at 64, 13 at 128, and 57 at 8 bits. Glaze v2
  chose 65, 61, 54 and 68.
- **Serving:** the mixed export was quantized by AutoRound through llm-compressor, and stock vLLM
  0.29 loaded it with its Marlin kernels and answered both check prompts.

The full control did not run, for two reasons in the run scripts:
- the vLLM check had loaded the export and answered, but failed writing its result into a folder
  nothing had created. The driver then skipped the control's full run. This is fixed in `c6d9761`.
- the follow-up run on the same machine was launched with a flag that `jl` accepts only for new
  instances, so it was refused and the instance was destroyed. The driver can now run one control
  alone (`--controls allocation`).

## Findings

- **Most of Glaze v2's MATH-500 gain is its calibration data.** With nothing else changed,
  AutoRound on Glaze v2's blocks gains 7.0 of the 10.0 points, and its in-domain KL falls 42%. The
  data is in-domain by design (MATH training problems answered by BF16 itself), so this gain is
  domain adaptation, not a better quantizer. It matches prior work: calibrating on a model's own
  generations ([Williams et al., 2025](https://arxiv.org/abs/2410.17170)), and for reasoning models
  on their own answers to math problems, which gained GPTQ 9.8 points over WikiText2
  ([Liu et al., 2025](https://arxiv.org/abs/2504.04823)).
- **The data alone does not help outside its domain.** Chat KL is unchanged (0.0440 against 0.0435),
  and IFEval is 1.3 points lower than AutoRound's (not significant).
- **Glaze v2's own stages add a smaller, broader gain.** Over the data control they add 3.0 points
  on MATH-500 (interval touching zero), 3.5 on IFEval (significant), and cut chat KL by 29%. Glaze
  v2 is the only one of the three that matches BF16 on both tasks.
- **That remainder is not yet attributed.** It combines Glaze v2's four times larger calibration
  volume (512 blocks, which AutoRound cannot hold on this GPU), its Fisher-weighted rounding, and its
  allocation with the 8-bit vision tower that pays for it.

## Caveats

- **Different GPUs.** The data control and Glaze v2 ran on A100s, AutoRound and BF16 on A30s.
  Repeated runs of one model differ by about 1–2 points.
- **One seed per model.** The intervals resample questions, not calibration or tuning seeds.
- **Validation KL** is on the blocks that selected Glaze v2's rounding state, so it favours Glaze
  v2 (see [record 6](06-glaze-v2.md#validation-kl)). The data control and AutoRound never saw them.

## Artifacts

- Driver: `scripts/run_glaze_controls.py`; AutoScheme allocation: `scripts/autoscheme_allocate.py`
- Variants: `autoround_glaze2_data_w4a16_g128` and `autoround_autoscheme_w4a16` in
  `configs/variants/feature1.yaml`; evaluation suite `configs/evaluation/feature1_controls.yaml`
- The control exports were not kept or published. Per-sample outputs, validation KL and the pilot
  allocation are in the run's local records, not yet in
  [`lazybrick/kiln-evals`](https://huggingface.co/datasets/lazybrick/kiln-evals).
