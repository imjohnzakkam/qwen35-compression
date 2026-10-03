# 4. AutoRound

[AutoRound](https://aclanthology.org/2024.findings-emnlp.662.pdf) tunes each block's weight rounding
and clipping by signed gradient descent against the BF16 block's outputs. It is a strong
post-training baseline for 4-bit weights.

**Status: not yet evaluated on Qwen3.5-4B.** Every attempt so far stopped during quantization, as
recorded below.

## Configuration

`autoround_w4a16_g128` in `configs/variants/feature1.yaml`:
- llm-compressor's `AutoRoundModifier` (auto-round 0.14.2)
- W4A16, symmetric, group size 128, the same layers left unquantized as GPTQ and AWQ
- 200 tuning steps per block, `torch.compile` off

## Attempts

| GPU | Calibration | Batch | Outcome |
| --- | --- | ---: | --- |
| A30, 24 GB | 512 conversations, variable length | 8 | Stopped at the first block: AutoRound stacks every sample's cached inputs into one tensor, which needs one sequence length (`Sizes of tensors must match`) |
| A30, 24 GB | 285 packed blocks of 2,048 tokens | 8 | Out of memory while computing the BF16 reference outputs for block 1 (21.0 GiB allocated) |
| A30, 24 GB | 285 packed blocks | 2 | Out of memory at the same step (20.1 GiB allocated) |
| A100, 40 GB | 285 packed blocks | 8 | Reference outputs fit; out of memory in the first tuning step (36.4 GiB allocated) |

**Packed calibration.** For AutoRound only, the same 512 conversations are tokenized as for the
other methods, joined in order and cut into 2,048-token blocks: 285 blocks, 585k tokens. GPTQ and
AWQ calibration is unchanged.

**Memory.** Usage barely changed between batch sizes 8 and 2, so it is dominated by cached
per-sample tensors, not by the tuning batch. Two causes add up:
- AutoRound keeps the block inputs and BF16 reference outputs of every calibration sample on the
  GPU.
- Without flash-linear-attention, transformers runs the 24 DeltaNet layers in a PyTorch fallback
  that needs about a third more memory for forward and backward passes (see the
  [drift study](03-drift-study.md#flash-linear-attention)).

Running AutoRound on this model therefore needs a GPU with more than 40 GB, flash-linear-attention,
or fewer cached samples.
