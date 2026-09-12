# Qwen3.5-9B Compression Plan

Goal: quantize `Qwen/Qwen3.5-9B` (multimodal image-text, post-trained checkpoint, not `-Base`),
measure exactly what each method costs in accuracy on both text and vision, find which components
are quantization-sensitive, and use that to design a mixed-precision recipe that beats uniform INT4.

First milestone, and nothing else counts until it is reproducible end to end:

> BF16 vs INT8 vs GPTQ-W4A16 vs AWQ-W4A16, evaluated on identical text and multimodal benchmarks.

---

## Environment and tooling

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. Both are committed. There is
no hand-maintained `requirements.txt`; if an external system needs one, generate it from the lock:

```bash
uv export --format requirements-txt > requirements.txt
```

Qwen3.5 requires Transformers 5.x, and VLMEvalKit recommends `transformers>=5.2.0` for this family,
so the floor is 5.2. `llmcompressor` 0.13.0 is the current release and exposes a `qwen` extra.
`lm-eval` deliberately keeps model backends out of its base install, so the Hugging Face backend
comes from `lm-eval[hf]`.

Setup:

```bash
uv sync
uv run python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

`uv run` handles the virtualenv, so `.venv` never needs manual activation.

VLMEvalKit stays out of the main dependency list. Its dependency surface is large and
model-specific, and the main environment should stay clean for compression plus text evaluation.
Install it separately, or later as a dedicated `vlm` dependency group:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git external/VLMEvalKit
uv pip install -e external/VLMEvalKit
```

---

## Repository layout

A `src` layout with a real installed package, so imports are
`from qwen35_compression.quantization.gptq import quantize` rather than relative imports from
loose scripts.

```text
qwen35-compression/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── bf16.yaml
│   ├── int8.yaml
│   ├── gptq_w4a16_g128.yaml
│   └── awq_w4a16_g128.yaml
│
├── src/
│   └── qwen35_compression/
│       ├── __init__.py
│       ├── models.py
│       ├── calibration.py
│       │
│       ├── quantization/
│       │   ├── gptq.py
│       │   ├── awq.py
│       │   └── int8.py
│       │
│       ├── evaluation/
│       │   ├── text.py
│       │   └── vlm.py
│       │
│       └── analysis/
│           └── sensitivity.py
│
├── scripts/
│   ├── download_model.py
│   ├── quantize.py
│   ├── evaluate_text.py
│   └── evaluate_vlm.py
│
├── outputs/     # quantized checkpoints
└── results/     # bf16.json, int8.json, gptq_w4a16_g128.json, ...
```

Packaging is setuptools with `where = ["src"]`.

Every quantized checkpoint gets a stable name used consistently across configs, checkpoint
directories, results files, and tables: `bf16`, `int8_w8a8`, `gptq_w4a16_g128`, `gptq_w4a16_g32`,
`awq_w4a16_g128`, `awq_w4a16_g32`, then the mixed-precision variants.

The working loop:

```bash
uv sync

uv run python scripts/download_model.py

uv run python scripts/evaluate_text.py --model Qwen/Qwen3.5-9B

uv run python scripts/quantize.py --method gptq --bits 4 --group-size 128
uv run python scripts/quantize.py --method awq  --bits 4 --group-size 128

uv run python scripts/evaluate_text.py --model outputs/qwen35-9b-gptq-w4a16-g128
```

---

## Stage 1 — Baseline setup

Download the post-trained `Qwen/Qwen3.5-9B` checkpoint and pin its revision hash. Record the hash,
`transformers` version, torch version, GPU type, and driver in `results/environment.json`. Every
result file references that environment record so a number can always be traced back to the stack
that produced it.

Confirm the model loads and generates for both a text-only prompt and an image-text prompt before
touching quantization. A broken multimodal path discovered after three quantization runs is three
wasted runs.

Record the on-disk size of the BF16 weights. That is the denominator for every compression ratio
reported later.

---

## Stage 2 — BF16 evaluation

Establish the reference numbers before any weights change.

Text, via EleutherAI `lm-evaluation-harness`:

- MMLU-Pro
- GSM8K and MATH-500
- IFEval
- HellaSwag and ARC
- WikiText-2 perplexity

Vision, via VLMEvalKit (it supports Qwen3.5 and expects `transformers` ≥ 5.2 for this family):

- MMBench
- MMMU
- MathVista
- TextVQA and OCRBench
- DocVQA

Write everything to `results/bf16.json` in one schema shared by all later runs: task name, metric
name, value, stderr, number of samples, harness version, generation settings. Fix a seed and record
it. Fix few-shot counts and record them. Later comparisons are only meaningful if the harness config
is byte-identical across models, so keep it in `configs/` rather than in command-line flags.

---

## Stage 3 — Calibration data

Build one calibration set and freeze it. Every method uses the same one, or GPTQ-vs-AWQ comparisons
measure the data, not the algorithm.

```text
512–1024 samples
sequence length 2048 or 4096
fixed random seed
```

Store the dataset name, split, sample indices, tokenizer settings, and seed in
`configs/calibration.yaml`, loaded by `calibration.py`. For the multimodal path, decide explicitly whether calibration includes
image-text samples and record that decision, because it changes what the vision tower sees during
GPTQ and AWQ scale estimation.

---

## Stage 4 — Quantization baselines

Do not reimplement the algorithms. Use `llm-compressor` from vLLM (it ships a Qwen3.5 GPTQ/AWQ
example to adapt) and GPTQModel (explicit Qwen3.5 support). The point of this stage is a trustworthy
reference line, not novel code.

Produce:

```text
W8A8 / INT8            simple round-trip quantization
GPTQ W4A16 g128
GPTQ W4A16 g32 or g64
AWQ  W4A16 g128
AWQ  W4A16 g32 or g64
```

`models.py` and `calibration.py` own model loading, calibration loading, checkpoint naming, and
size accounting so the per-method quantizers stay thin. Each run writes a small manifest next to the
checkpoint: method, bit width, group size, which modules were skipped, wall-clock time, peak memory.

Record disk size for each checkpoint as it is produced.

---

## Stage 5 — Evaluate every quantized model

Re-run the Stage 2 text suite unchanged on each checkpoint, then the Stage 6 vision suite. Same
harness config, same seed, same few-shot counts.

Collate into one table:

| Model        | MMLU-Pro | MATH | IFEval | PPL | Size |
| ------------ | -------: | ---: | -----: | --: | ---: |
| BF16         | baseline | baseline | baseline | baseline | ~18 GB |
| INT8         | | | | | |
| GPTQ W4 g128 | | | | | |
| GPTQ W4 g32  | | | | | |
| AWQ W4 g128  | | | | | |
| AWQ W4 g32   | | | | | |

Two columns carry the whole argument: delta accuracy against BF16, and compression ratio against
BF16. Report absolute scores too, but read the deltas.

---

## Stage 6 — Vision evaluation

The model is multimodal, so a text-only verdict is half an answer. Run VLMEvalKit across BF16, INT8,
GPTQ, and AWQ on MMBench, MMMU, MathVista, TextVQA, OCRBench, and DocVQA.

The specific question here: does the vision score degrade at the same rate as the text score under
the same bit width? If vision falls faster, the vision tower is a mixed-precision candidate before
anything in the language model is.

---

## Stage 7 — Component-wise quantization

With the uniform baselines trusted, start splitting the model. Each experiment changes one axis and
is evaluated with the full text and vision suite.

```text
1   Vision encoder BF16   | LLM INT4
2   Vision encoder INT8   | LLM INT4
3   Vision encoder INT4   | LLM INT4
4   Attention INT8        | FFN INT4
5   DeltaNet INT8         | Attention INT4 | FFN INT4
```

The output is a ranking of which component costs the most accuracy when it drops to low precision.

---

## Stage 8 — Sensitivity analysis

Invert the previous stage. Hold the model at BF16, quantize exactly one component or layer group,
evaluate, record the delta, restore, repeat.

```text
BF16 model → quantize component X → evaluate → record Δaccuracy → restore
```

Target shape of the result:

```text
Component          Δ accuracy
Vision encoder          -1.8
Attention               -0.4
FFN                     -0.2
DeltaNet                -1.1
LM head                 -2.6
```

This is the most expensive stage in GPU hours, so use a reduced but fixed evaluation subset for the
sweep, then confirm the top few findings on the full suite. Record which subset was used; a
sensitivity number from a subset is not comparable to a headline number from the full suite.

---

## Stage 9 — Mixed-precision recipe

Turn the sensitivity ranking into an allocation:

```text
Sensitive layers      → INT8 or BF16
Normal layers         → INT4
Robust layers         → INT3 or INT4
```

A plausible starting recipe, to be replaced by whatever Stage 8 actually says:

```text
Vision       INT8
DeltaNet     INT8
Attention    INT4
FFN          INT4
LM head      BF16
Embeddings   BF16
```

Express recipes as declarative YAML in `configs/recipes/` so a recipe is a diffable artifact, not a
set of edits buried in a script. Iterate: adjust the allocation, re-evaluate, keep what improves the
accuracy-per-byte frontier.

---

## Stage 10 — Final comparison

The write-up answers five questions directly:

- How much smaller is the model?
- How much accuracy was lost?
- Which components are quantization-sensitive?
- Does text quality degrade differently from vision quality?
- Does mixed precision beat uniform INT4?

| Method               | Bits  | Text Δ | Vision Δ | Model size | Compression |
| -------------------- | ----: | -----: | -------: | ---------: | ----------: |
| BF16                 | 16    | 0 | 0 | X GB | 1× |
| INT8                 | 8     | | | | |
| GPTQ                 | W4A16 | | | | |
| AWQ                  | W4A16 | | | | |
| **Mixed precision**  | mixed | | | | |

The mixed-precision row is only interesting if it sits above the uniform INT4 rows at comparable
size. If it does not, that is a real result and gets reported as one.

---

## Execution order

```text
1. uv sync, verify torch and transformers import
2. Download Qwen3.5-9B and pin the revision
3. Run BF16 text and vision evaluations
4. Freeze the calibration set
5. Integrate llm-compressor / GPTQModel
6. GPTQ W4A16
7. AWQ W4A16
8. INT8 baseline
9. Run lm-evaluation-harness across all checkpoints
10. Run VLMEvalKit across all checkpoints
11. Component-wise experiments
12. Sensitivity sweep
13. Design and evaluate mixed precision
14. Final accuracy and size comparison
```

Steps 1 through 10 are the first milestone. Get them reproducible from a clean checkout before
starting step 11.
