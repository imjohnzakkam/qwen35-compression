"""vLLM plugin for Blaze's real exports.

It re-registers `Qwen3_5ForConditionalGeneration` with a subclass that, when the checkpoint's
config carries `blaze_vocab`, replaces the tied BF16 vocabulary table with Blaze's two quantized
blocks and index map (see `blaze_vllm.model`). Any other Qwen3.5 checkpoint loads unchanged.
"""

ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
MODEL = "blaze_vllm.model:BlazeQwen3_5ForConditionalGeneration"


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, MODEL)
