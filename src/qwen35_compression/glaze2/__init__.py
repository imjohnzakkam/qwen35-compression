"""Glaze v2: a byte-budgeted, generalization-aware W4A16 quantizer.

It quantizes from BF16 in five stages: in-domain self-generated calibration data, a Fisher pass,
a byte-neutral precision allocation, sensitivity-weighted block reconstruction with learned
rounding, and a compressed-tensors export stock vLLM serves.
"""
