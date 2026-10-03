"""Glaze: end-to-end distillation of a W4A16 export's group scales and norm weights.

Glaze starts from a symmetric INT4 group-quantized export (AutoRound or GPTQ), keeps its codes
frozen, and trains only the per-group scales and the language model's RMSNorm weights so that the
4-bit model matches BF16's next-token distribution. The refined export has the same tensors,
dtypes, shapes and quantization config as its init, so it is served by the same vLLM kernels.
"""
