# 3. Drift study

Where quantization error appears in Qwen3.5-4B: along an answer, and by component of the network.

## Method

`scripts/drift_scores.py` replays BF16's own instruct-track answers through a model:
- **Answers:** all 500 MATH-500 answers and the 1,001 MMLU-Pro subset answers, 2,304,479 answer
  tokens in total.
- **How they're fed in:** each prompt and BF16 answer goes through the model in one vLLM prefill
  with `prompt_logprobs`. No text is generated, so a model is scored in minutes.

At every answer token it records:

- **Flip:** the model's top choice is not BF16's token.
- **Loss:** the model's negative log-likelihood of BF16's token.

Scored on BF16 itself, the flip rate is the noise floor: near-ties resolved differently under
different batching. Because the input is always BF16's tokens, a flip rate that grows with position
would mean error accumulating in the model's state (the recurrent DeltaNet state or the attention
context). A flat rate means the error is local to each token.

Positions are reported for the 187 long answers (at least 2,048 tokens, finished), so early and
late positions are compared within the same answers rather than across easy and hard questions.

The component variants apply GPTQ W4A16 g128, with the same calibration as the full GPTQ baseline,
to the Linear layers of one component and keep the rest in BF16:

| Component | Layers | Share of the 4-bit weights |
| --- | --- | ---: |
| DeltaNet | 24 linear-attention layers: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` | 28.3% |
| Attention | 8 full-attention layers: `q_proj`, `k_proj`, `v_proj`, `o_proj` | 8.2% |
| FFN | 32 MLPs: `gate_proj`, `up_proj`, `down_proj` | 63.5% |

| | |
| --- | --- |
| Code | `0172f24` (BF16 and the published variants), `8c6cd54` (component variants), `c3a16a0` (AutoRound) |
| Hardware | One NVIDIA A30 (24 GB); AutoRound on one A100 (40 GB) |

## Results

| Model | Flip rate | Loss | 0–256 | 256–512 | 512–1k | 1k–2k | 2k–4k | 4k–8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 (noise floor) | 0.35% | 0.128 | 0.47 | 0.34 | 0.40 | 0.36 | 0.35 | 0.32 |
| INT8 W8A8 | 2.42% | 0.139 | 3.07 | 2.77 | 2.59 | 2.55 | 2.45 | 2.30 |
| AutoRound W4A16 g128 | 4.96% | 0.170 | 5.70 | 5.37 | 5.26 | 5.34 | 5.33 | 5.24 |
| AWQ W4A16 g128 | 5.42% | 0.178 | 6.55 | 6.14 | 5.84 | 5.65 | 5.59 | 5.43 |
| GPTQ W4A16 g128 | 5.70% | 0.182 | 6.68 | 6.32 | 6.07 | 6.12 | 6.14 | 5.95 |
| GPTQ, DeltaNet only | 3.03% | 0.143 | 3.52 | 3.36 | 3.31 | 3.35 | 3.27 | 3.17 |
| GPTQ, attention only | 2.47% | 0.138 | 2.52 | 2.55 | 2.53 | 2.77 | 2.92 | 2.93 |
| GPTQ, FFN only | 4.20% | 0.158 | 5.28 | 4.79 | 4.56 | 4.46 | 4.39 | 4.16 |

The position columns are flip rates (%) by answer-token position, over the long answers.

### Attribution by component

Excess loss over BF16 adds up across the three components (27% + 18% + 56% = 101% of whole-model
GPTQ), so it attributes the error. Flip rates do not add up (the three components sum to 162% of
whole-model GPTQ), because the same tokens flip under several components, and they overstate small
components.

| Component | Share of 4-bit weights | Share of GPTQ's excess loss | Per parameter |
| --- | ---: | ---: | ---: |
| Attention | 8.2% | 18% | 2.2× |
| DeltaNet | 28.3% | 27% | 1.0× |
| FFN | 63.5% | 56% | 0.9× |

## Findings

- **The error does not compound.** With BF16's tokens as input, disagreement is flat or slightly
  falling from the first answer token to the 8,000th, for INT8, GPTQ and AWQ alike. The DeltaNet
  recurrent state does not accumulate quantization error. 4-bit damage is a local, per-token
  divergence: GPTQ disagrees with BF16 on about 1 token in 18. In free generation each
  disagreement can steer the answer onto a different path, which is how the longer and looping
  answers of the [quantization baselines](02-quantization-baselines.md#answer-length) arise.
- **Attention is the one long-context effect.** With only the attention layers quantized,
  disagreement rises along the answer (2.5% to 2.9%). This fits softmax attention over a growing
  context of quantized keys and values. It is small next to the whole-model error.
- **No component dominates.** Only the full-attention layers are disproportionately sensitive, and
  they are 8% of the weights. Keeping them at 8 bits would add about 0.15 GB and remove at most
  about 18% of GPTQ's excess loss.
- **The measure is more sensitive than the benchmarks.** INT8 changes 2.4% of top choices (7× the
  noise floor) with no measurable benchmark change. AWQ (5.42%) and GPTQ (5.70%) separate cleanly
  here but not on the benchmark panel. Whether small drift differences predict benchmark
  differences has not yet been tested beyond these four models.

## flash-linear-attention

Without [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), transformers
runs Qwen3.5's Gated DeltaNet layers in a PyTorch fallback. `scripts/check_fla.py` compares the two
on one DeltaNet layer, forward and backward, on 8 × 2,048 tokens:

| Implementation | Peak memory | Time |
| --- | ---: | ---: |
| PyTorch fallback | 17.1 GiB | 8.9 s |
| flash-linear-attention 0.5.2 | 11.3 GiB | 93 s |

- **Accuracy:** the outputs agree to 0.3% (BF16 rounding).
- **Time:** the flash-linear-attention figure is from a single, first call and likely includes
  Triton compilation; steady-state speed was not measured.

## Reproduce

```bash
uv run python scripts/run_drift_study.py --pilot-limit 20
uv run python scripts/drift_scores.py report results/feature1/drift/*.json
```
