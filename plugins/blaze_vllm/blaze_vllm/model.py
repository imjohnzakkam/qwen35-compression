"""Blaze's tiered vocabulary table in vLLM 0.29 (tensor parallel size 1).

The table is stored in usage order as two compressed-tensors blocks: `hot` (the busiest rows,
8-bit g128) and `cold` (the rest, 4-bit g128), with `position`, each token id's row in that
order. Logits run both blocks through vLLM's quantized LM head (Marlin), which repacks them
into its own layout. `BLAZE_VOCAB` picks how lookups find their rows:

- `marlin` (default): gathered and dequantized from the Marlin-repacked head itself
  (`blaze_vllm.marlin_rows`), so each block is held once, at Marlin's logit speed;
- `twice`: vLLM's quantized-embedding kernel on a second copy in the stored layout;
- `packed` (or `BLAZE_SINGLE_COPY=1`): one copy in the stored layout, with logits from a
  dequantizing Triton matmul (`blaze_vllm.head`), about four times slower.

With `BLAZE_W6_AS_W8=1`, a checkpoint's 6-bit body groups run on the 8-bit kernels with the same
codes (`blaze_vllm.widen`); without it, vLLM serves them with its own 6-bit kernel.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from blaze_vllm import marlin_rows, widen

CONFIG_KEY = "blaze_vocab"
LAYOUT_ENV = "BLAZE_VOCAB"
SINGLE_COPY_ENV = "BLAZE_SINGLE_COPY"
LAYOUTS = ("marlin", "twice", "packed")
# Logits are written into one output in row chunks, so the two blocks' partial logits never
# need more than a chunk's memory beside it.
LOGIT_CHUNK = 1024


class _TieredHead(QuantizeMethodBase):
    """The LM head's method for the tiered table: both blocks' logits, in token-id order."""

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        pass

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        hot_ids, cold_ids = layer.order[: layer.hot_rows], layer.order[layer.hot_rows :]
        logits = None
        for start in range(0, flat.shape[0], LOGIT_CHUNK):
            rows = flat[start : start + LOGIT_CHUNK]
            if layer.layout == "packed":
                from blaze_vllm.head import packed_logits

                hot = packed_logits(
                    rows, layer.hot.weight_packed, layer.hot.weight_scale, layer.hot_bits
                )
                cold = packed_logits(
                    rows, layer.cold.weight_packed, layer.cold.weight_scale, layer.cold_bits
                )
            else:
                hot = layer.hot_head.quant_method.apply(layer.hot_head, rows)
                cold = layer.cold_head.quant_method.apply(layer.cold_head, rows)
            if logits is None:
                logits = hot.new_empty(flat.shape[0], layer.vocab_size)
            part = logits[start : start + rows.shape[0]]
            part.index_copy_(1, hot_ids, hot)
            part.index_copy_(1, cold_ids, cold)
        logits = logits.reshape(*x.shape[:-1], layer.vocab_size)
        return logits if bias is None else logits + bias

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Row in usage order -> token id: hot rows first, then cold.
        layer.order = torch.argsort(layer.position)


class TieredVocab(nn.Module):
    """A tied vocabulary table in two quantized blocks: embedding lookups and LM head."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        hot_rows: int,
        quant_config,
        prefix: str,
        hot_bits: int = 8,
        cold_bits: int = 4,
        layout: str = "marlin",
    ):
        super().__init__()
        if layout not in LAYOUTS:
            raise ValueError(f"unknown blaze vocabulary layout {layout!r}; one of {LAYOUTS}")
        self.hot_bits, self.cold_bits = hot_bits, cold_bits
        self.layout = layout
        self.hidden = hidden
        cold_rows = vocab_size - hot_rows
        self.vocab_size = vocab_size
        self.org_vocab_size = vocab_size
        self.num_embeddings = vocab_size
        self.hot_rows = hot_rows
        self.tp_size = 1
        parts = []
        if layout != "marlin":
            self.hot = VocabParallelEmbedding(
                hot_rows, hidden, quant_config=quant_config, prefix=f"{prefix}.hot"
            )
            self.cold = VocabParallelEmbedding(
                cold_rows, hidden, quant_config=quant_config, prefix=f"{prefix}.cold"
            )
            parts += [self.hot, self.cold]
        if layout != "packed":
            self.hot_head = ParallelLMHead(
                hot_rows, hidden, quant_config=quant_config, prefix=f"{prefix}.hot"
            )
            self.cold_head = ParallelLMHead(
                cold_rows, hidden, quant_config=quant_config, prefix=f"{prefix}.cold"
            )
            parts += [self.hot_head, self.cold_head]
        if layout == "marlin":
            # Each code's place in a 16 x 64 Marlin block, per bit width.
            for name, bits in (("hot_table", hot_bits), ("cold_table", cold_bits)):
                self.register_buffer(name, marlin_rows.position_table(bits), persistent=False)
        for part in parts:
            if part.num_embeddings_padded != part.org_vocab_size:
                raise ValueError("blaze vocabulary blocks must be multiples of 64 rows")
        self.register_buffer("position", torch.zeros(vocab_size, dtype=torch.long))
        self.register_buffer("order", torch.arange(vocab_size, dtype=torch.long))
        self.quant_method = _TieredHead()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        position = self.position[input_ids]
        is_hot = (position < self.hot_rows).unsqueeze(-1)
        hot_ids = position.clamp(max=self.hot_rows - 1)
        cold_ids = (position - self.hot_rows).clamp(min=0)
        if self.layout == "marlin":
            hot = self._marlin_rows(self.hot_head, self.hot_table, self.hot_bits, hot_ids)
            cold = self._marlin_rows(self.cold_head, self.cold_table, self.cold_bits, cold_ids)
        else:
            hot = self.hot(hot_ids)
            cold = self.cold(cold_ids)
        return torch.where(is_hot, hot, cold)

    def _marlin_rows(
        self, head: nn.Module, table: torch.Tensor, bits: int, ids: torch.Tensor
    ) -> torch.Tensor:
        rows = marlin_rows.gather(
            ids, head.weight_packed, head.weight_scale, table, bits, self.hidden
        )
        return rows.reshape(*ids.shape, self.hidden)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if name == "position":
                self.position.copy_(weight.to(self.position.dtype))
                loaded.add(name)
                continue
            part, rest = name.split(".", 1)
            if part not in ("hot", "cold"):
                raise KeyError(f"unexpected blaze vocabulary tensor: {name}")
            # Each block feeds the modules its layout keeps: the lookup, the LM head, or both.
            modules = {
                "marlin": (f"{part}_head",),
                "twice": (part, f"{part}_head"),
                "packed": (part,),
            }[self.layout]
            for module in modules:
                key = f"{module}.{rest}"
                param = params[key]
                param.weight_loader(param, weight)
                loaded.add(key)
        return loaded


class BlazeQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 with Blaze's tiered vocabulary table when the config names one."""

    def __init__(self, *, vllm_config, prefix: str = "model"):
        config = vllm_config.model_config.hf_config
        six_bit: set[str] = set()
        if os.environ.get(widen.ENV) == "1":
            six_bit = widen.six_bit_modules(getattr(config, "quantization_config", None) or {})
            if six_bit and not widen.widen_schemes(vllm_config.quant_config):
                raise ValueError("the checkpoint's 6-bit groups are not in vLLM's quant config")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.six_bit = six_bit
        blaze = getattr(config, CONFIG_KEY, None)
        if blaze is None:
            return
        text = config.text_config
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError("blaze vocabulary tables run at tensor parallel size 1")
        language = self.language_model
        vocab = TieredVocab(
            text.vocab_size,
            text.hidden_size,
            int(blaze["hot_rows"]),
            vllm_config.quant_config,
            prefix=f"{prefix}.language_model.model.embed_tokens".removeprefix("."),
            hot_bits=int(blaze["hot_bits"]),
            cold_bits=int(blaze["cold_bits"]),
            layout="packed"
            if os.environ.get(SINGLE_COPY_ENV) == "1"
            else os.environ.get(LAYOUT_ENV, "marlin"),
        )
        language.model.embed_tokens = vocab
        language.lm_head = vocab

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.six_bit:
            weights = widen.widened(weights, self.six_bit)
        return super().load_weights(weights)
