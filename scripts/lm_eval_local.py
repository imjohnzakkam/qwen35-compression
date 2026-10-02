#!/usr/bin/env python3
"""Run lm-eval for local validation on the Apple GPU.

The MPS allocator keeps freed blocks cached. Likelihood tasks allocate a differently shaped
[window, vocab] logits tensor per WikiText window, and the cache grew until a 16 GB Mac ran out
after a few windows. Releasing it after every forward pass keeps memory flat. Arguments are
passed to lm-eval unchanged.
"""

from __future__ import annotations

import runpy
import sys

import torch
from lm_eval.models.huggingface import HFLM

_model_call = HFLM._model_call


def _model_call_releasing_cache(self, *args, **kwargs):
    try:
        return _model_call(self, *args, **kwargs)
    finally:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


HFLM._model_call = _model_call_releasing_cache

if __name__ == "__main__":
    sys.argv[0] = "lm_eval"
    runpy.run_module("lm_eval", run_name="__main__", alter_sys=True)
