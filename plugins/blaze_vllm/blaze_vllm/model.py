"""Blaze's tiered vocabulary table in vLLM 0.29 (tensor parallel size 1).

The table is stored in usage order as two compressed-tensors blocks: `hot` (the busiest rows,
8-bit g128) and `cold` (the rest, 4-bit g128), with `position`, each token id's row in that
order. Lookups gather and dequantize rows with vLLM's quantized-embedding kernel; logits run
both blocks through vLLM's quantized LM head (Marlin) and are put back in token-id order. Each
block is held twice on the GPU, once per kernel layout; the checkpoint holds it once.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

CONFIG_KEY = "blaze_vocab"
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

    def __init__(self, vocab_size: int, hidden: int, hot_rows: int, quant_config, prefix: str):
        super().__init__()
        cold_rows = vocab_size - hot_rows
        self.vocab_size = vocab_size
        self.org_vocab_size = vocab_size
        self.num_embeddings = vocab_size
        self.hot_rows = hot_rows
        self.tp_size = 1
        self.hot = VocabParallelEmbedding(
            hot_rows, hidden, quant_config=quant_config, prefix=f"{prefix}.hot"
        )
        self.cold = VocabParallelEmbedding(
            cold_rows, hidden, quant_config=quant_config, prefix=f"{prefix}.cold"
        )
        self.hot_head = ParallelLMHead(
            hot_rows, hidden, quant_config=quant_config, prefix=f"{prefix}.hot"
        )
        self.cold_head = ParallelLMHead(
            cold_rows, hidden, quant_config=quant_config, prefix=f"{prefix}.cold"
        )
        for part in (self.hot, self.cold, self.hot_head, self.cold_head):
            if part.num_embeddings_padded != part.org_vocab_size:
                raise ValueError("blaze vocabulary blocks must be multiples of 64 rows")
        self.register_buffer("position", torch.zeros(vocab_size, dtype=torch.long))
        self.register_buffer("order", torch.arange(vocab_size, dtype=torch.long))
        self.quant_method = _TieredHead()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        position = self.position[input_ids]
        is_hot = (position < self.hot_rows).unsqueeze(-1)
        hot = self.hot(position.clamp(max=self.hot_rows - 1))
        cold = self.cold((position - self.hot_rows).clamp(min=0))
        return torch.where(is_hot, hot, cold)

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
            # Each block feeds two modules: the lookup and the LM head.
            for module in (part, f"{part}_head"):
                key = f"{module}.{rest}"
                param = params[key]
                param.weight_loader(param, weight)
                loaded.add(key)
        return loaded


class BlazeQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 with Blaze's tiered vocabulary table when the config names one."""

    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
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
        )
        language.model.embed_tokens = vocab
        language.lm_head = vocab
