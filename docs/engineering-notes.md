# Engineering notes

Problems met while evaluating and quantizing Qwen3.5 with vLLM, lm-evaluation-harness, VLMEvalKit
and llm-compressor, with the fixes used in this repository. Versions are those pinned here: vLLM
0.29.0, lm-eval 0.4.13, VLMEvalKit `34a64e6`, llm-compressor 0.13.0, transformers 5.12.1.

## Model behaviour

- **Thinking is on by default for Qwen3.5-4B, but not for Qwen3.5-0.8B.** Results across sizes are
  comparable only with `enable_thinking` pinned. Here it is set in the suite configuration and
  passed as a chat-template argument by both harnesses.
- **Greedy instruct answers loop.** With greedy decoding a few percent of long answers repeat until
  the token limit, and quantized models do so more often. Answer limits must be generous (here
  8,192 tokens) so that verbosity does not turn into truncation errors.

## lm-evaluation-harness

- **Silent prompt truncation.** lm-eval cuts the start of any prompt that does not fit beside
  `max_gen_toks`. Set the model context to the answer limit plus room for prompts: here 12,288
  tokens for an 8,192-token limit.
- **`--samples` and `--limit` are mutually exclusive.** Tasks missing from a `--samples` file run in
  full, which lets one suite combine a fixed MMLU-Pro subset with complete MATH-500 and IFEval.
- **Per-sample files.** lm-eval writes one JSON object per `\n`-terminated line, and some answers
  contain Unicode line separators. Read the files line by line; Python's `str.splitlines()` also
  splits on those separators.

## vLLM

- **Prompt log-probabilities need bounded prefill chunks.** Likelihood tasks (WikiText, HellaSwag,
  ARC) and teacher-forced scoring compute log-probabilities over the 248k-token vocabulary for
  every prompt token in a prefill chunk. With the default chunk of 8,192 tokens that is about 8 GB
  of fp32 logits. `max_num_batched_tokens=4096` keeps the peak within a 24 GB GPU.
- **compressed-tensors targets by name.** vLLM names Qwen3.5's text layers
  `language_model.model.layers.N…`, where transformers uses `model.language_model.layers.N…`. vLLM
  also fuses projections (`in_proj_qkvz`, `in_proj_ba`, `qkv_proj`, `gate_up_proj`) and matches a
  fused layer only if all its parts match. Regex targets written against transformers' names
  silently fail to match in vLLM. vLLM then loads packed 4-bit weights into a layer it takes for
  BF16 (`'MergedColumnParallelLinear' object has no attribute 'data'`). Match on
  `layers.N.<component>.<projection>`, as the sensitivity variants here do. Targets by layer type
  (`Linear`) are unaffected.

## VLMEvalKit

- **Image prompts need a large context.** With MMMU's multi-image prompts and full-page DocVQA
  scans, a 4,096-token context rejected prompts, and VLMEvalKit scored each rejection as a wrong
  answer. The vision server here uses 32,768 tokens.
- **Throughput.** The in-process Qwen3-VL path generates one sample at a time. Serving the model with
  vLLM's OpenAI-compatible server and using VLMEvalKit's API client with 32 concurrent requests
  makes the six datasets practical.
- **Random fills are not counted as judge failures.** When the extractor cannot map an answer to an
  option, VLMEvalKit substitutes a random option and still reports `judge_fail_rate` as 0%.
  Rate-limit errors from the extractor's API produce the same silent random fills. Count them from
  the per-question `log` column, and keep extractor concurrency low (4 here).

## llm-compressor

- **AWQ memory.** AWQ caches every layer's calibration inputs for its scale search, and keeps the
  cache on the GPU unless `offload_device` is set (it moves it to the CPU automatically only for
  mixture-of-experts models). With 512 × 2,048-token samples it filled a 24 GB GPU at the second
  layer. `AWQModifier(offload_device=torch.device("cpu"))` moves the cache without changing the
  scales.
- **AutoRound** needs calibration samples of one length, and a 40 GB GPU with flash-linear-attention on
  this model; see [AutoRound](experiments/04-autoround.md).

## flash-linear-attention

transformers selects Qwen3.5's DeltaNet kernels independently: installing
`flash-linear-attention` alone replaces the PyTorch fallback for the chunked delta rule, without
`causal-conv1d`. The package depends on torch, so install it with `--no-deps` into a pinned
environment:

```bash
uv pip install --no-deps fla-core==0.5.2 flash-linear-attention==0.5.2 einops
```

## Hub downloads

On a GPU image whose base Python provides `brotlicffi`, httpx 0.28 (used by huggingface_hub) can
fail mid-download while decoding Brotli-compressed responses (`decoder process called with data
when 'can_accept_more_data()' is False`). Hiding the module before httpx is imported makes it
request gzip instead:

```python
import sys

for name in ("brotli", "brotlicffi"):
    sys.modules.setdefault(name, None)
```
