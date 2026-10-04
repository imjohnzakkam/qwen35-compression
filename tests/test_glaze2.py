"""Glaze v2: quantization math, allocation, Fisher pass, rounding, export, evaluation, data."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import pytest
import torch
from glaze_tiny import WIDE, random_blocks, tiny_teacher
from torch import nn

from qwen35_compression.glaze2.allocate import (
    FisherFactors,
    Unit,
    allocate,
    build_units,
    group_units,
    uniform,
    unit_of,
)
from qwen35_compression.glaze2.data import (
    DOMAIN_CODES,
    Decontaminator,
    PackedBlocks,
    Sample,
    chat_sample,
    interleave,
    is_held_out,
    load_blocks,
    math_prompt,
    mcq_prompt,
    pack_samples,
    save_blocks,
    sciq_options,
)
from qwen35_compression.glaze2.export import (
    LANGUAGE_PREFIX,
    pack,
    plan_bytes,
    unpack,
    vision_linear,
    write_export,
)
from qwen35_compression.glaze2.fisher import fisher_pass
from qwen35_compression.glaze2.quant import (
    BASE,
    OPTIONS,
    LearnedQuantLinear,
    Option,
    code_range,
    dequantize,
    fake_quantize,
    quantize,
)
from qwen35_compression.glaze2.rounding import RoundingSettings, capture_inputs, quantize_layers

# ---------------------------------------------------------------- quantization math


def test_option_bytes_match_the_plan() -> None:
    assert Option(4, 128).bytes_per_weight() == 0.515625
    assert Option(4, 32).bytes_per_weight() == 0.5625
    assert Option(8, 128).bytes_per_weight() == 1.015625
    assert Option(4, 128).tensor_bytes(256, 512) == 256 * 512 // 2 + 256 * 4 * 2
    assert [o.name for o in OPTIONS] == ["w4g128", "w4g64", "w4g32", "w8g128"]


@pytest.mark.parametrize("option", OPTIONS)
def test_codes_survive_compressed_tensors_packing(option: Option) -> None:
    weight = torch.randn(64, 256) * 0.02
    codes, scales = quantize(weight, option)
    low, high = code_range(option.bits)
    assert codes.min() >= low and codes.max() <= high
    packed = pack(codes, option.bits)
    assert packed.dtype is torch.int32
    assert torch.equal(unpack(packed, option.bits, (64, 256)), codes.to(torch.int8))
    # Scales are BF16 numbers; the weight is their product rounded to BF16, as the kernel does.
    assert torch.equal(scales.to(torch.bfloat16).float(), scales)
    weight_bf16 = dequantize(codes, scales, option)
    assert weight_bf16.dtype is torch.bfloat16
    error = (weight_bf16.float() - weight).abs().reshape(64, -1, option.group)
    # Half a step, plus the clamp a BF16-rounded-down scale forces on the largest weight, plus
    # rounding the product to BF16 (8 significant bits: a real share of an 8-bit step).
    peak = weight.abs().reshape(64, -1, option.group).amax(-1)
    assert (error.amax(-1) <= scales * 0.6 + peak * 2**-8 + 1e-7).all()


def test_eight_bits_are_finer_than_four() -> None:
    weight = torch.randn(32, 256) * 0.02
    four = (fake_quantize(weight, Option(4, 128)).float() - weight).square().mean()
    finer = (fake_quantize(weight, Option(4, 32)).float() - weight).square().mean()
    eight = (fake_quantize(weight, Option(8, 128)).float() - weight).square().mean()
    assert eight < finer < four


def test_learned_linear_starts_at_round_to_nearest_and_exports_what_it_computes() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(256, 16, bias=False).to(torch.bfloat16)
    learned = LearnedQuantLinear(linear, BASE)
    assert torch.equal(
        learned.quantized_weight().to(torch.bfloat16), fake_quantize(linear.weight, BASE)
    )
    with torch.no_grad():
        learned.offsets.uniform_(-0.5, 0.5)
        learned.alpha.uniform_(0.6, 1.0)
    codes, scales = learned.export()
    assert torch.equal(
        dequantize(codes.float(), scales, BASE), learned.quantized_weight().to(torch.bfloat16)
    )
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    learned(x).float().square().sum().backward()
    assert learned.offsets.grad is not None and learned.alpha.grad is not None
    assert learned.alpha.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="multiple of group"):
        LearnedQuantLinear(nn.Linear(100, 4, bias=False), BASE)


# ---------------------------------------------------------------- allocation


def test_units_follow_vllm_fusion() -> None:
    names = [
        "layers.3.self_attn.q_proj",
        "layers.3.self_attn.k_proj",
        "layers.3.self_attn.v_proj",
        "layers.3.self_attn.o_proj",
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.linear_attn.in_proj_z",
        "layers.0.linear_attn.in_proj_b",
        "layers.0.linear_attn.in_proj_a",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
    ]
    units = group_units(names)
    assert len(units) == 6
    assert units["layers.3.self_attn.{q_proj,k_proj,v_proj}"] == names[:3]
    assert unit_of("layers.3.self_attn.o_proj") == "layers.3.self_attn.o_proj"
    assert unit_of("layers.0.linear_attn.in_proj_a") == "layers.0.linear_attn.{in_proj_b,in_proj_a}"


def _brute_force(units: list[Unit], budget: int) -> float:
    best = -float("inf")
    for combo in itertools.product(*[u.options() for u in units]):
        extra = sum(u.bytes(o) - u.bytes(BASE) for u, o in zip(units, combo, strict=True))
        if extra <= budget:
            best = max(best, sum(u.saving[o] for u, o in zip(units, combo, strict=True)))
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_knapsack_finds_the_best_choice_within_budget(seed: int) -> None:
    rng = random.Random(seed)
    units = []
    for index in range(5):
        unit = Unit(f"u{index}", [f"u{index}"], [(rng.choice([64, 128]), 256)])
        for option in OPTIONS:
            unit.saving[option] = 0.0 if option == BASE else rng.uniform(0, 10)
        units.append(unit)
    budget = rng.randint(1000, 40000)
    result = allocate(units, budget, resolution=1)
    assert result.extra_bytes <= budget
    assert result.predicted_saving == pytest.approx(_brute_force(units, budget))
    # A coarse resolution rounds costs up, so it never overspends.
    assert allocate(units, budget, resolution=4096).extra_bytes <= budget
    assert set(uniform(units).choice.values()) == {BASE}
    with pytest.raises(ValueError):
        allocate(units, -1)


def test_saving_prefers_sensitive_tensors() -> None:
    torch.manual_seed(0)
    weights = {
        "layers.0.mlp.down_proj": torch.randn(64, 256),
        "layers.1.mlp.down_proj": torch.randn(64, 256),
    }
    quiet = FisherFactors(output=torch.full((64,), 1e-3), input=torch.ones(256))
    loud = FisherFactors(output=torch.ones(64), input=torch.ones(256))
    units = build_units(weights, {"layers.0.mlp.down_proj": quiet, "layers.1.mlp.down_proj": loud})
    savings = {u.name: u.saving[Option(8, 128)] for u in units}
    assert savings["layers.1.mlp.down_proj"] > 100 * savings["layers.0.mlp.down_proj"] > 0
    budget = units[0].bytes(Option(8, 128)) - units[0].bytes(BASE)
    choice = allocate(units, budget, resolution=1).choice
    assert choice["layers.1.mlp.down_proj"] == Option(8, 128)
    assert choice["layers.0.mlp.down_proj"] == BASE


# ---------------------------------------------------------------- Fisher pass and rounding


@pytest.fixture(scope="module")
def wide():
    teacher = tiny_teacher(**WIDE)
    head = teacher.get_input_embeddings().weight
    linears = {
        name: module
        for name, module in teacher.named_modules()
        if isinstance(module, nn.Linear) and name.startswith("layers.")
    }
    return teacher, head, linears


def test_fisher_pass_records_every_linear_and_layer(wide) -> None:
    teacher, head, linears = wide
    blocks = random_blocks(3, seed=1)
    factors, weights = fisher_pass(teacher, head, blocks, linears, chunk_tokens=7)
    assert set(factors) == set(linears)
    for name, module in linears.items():
        assert factors[name].input.shape == (module.in_features,)
        assert factors[name].output.shape == (module.out_features,)
        assert (factors[name].input > 0).all() and (factors[name].output >= 0).all()
        assert factors[name].output.sum() > 0
    assert len(weights.tokens) == len(weights.channels) == len(teacher.layers)
    assert weights.tokens[0].shape == (3, len(blocks[0]))
    assert weights.channels[0].mean().item() == pytest.approx(1.0, rel=1e-5)
    assert weights.tokens[0].min() >= 0.1 and weights.tokens[0].max() <= 10.0
    # The pass leaves no hooks or gradients behind.
    assert all(p.grad is None for p in teacher.parameters())
    assert not any(m._forward_hooks or m._backward_hooks for m in teacher.modules())


def test_captured_inputs_replay_the_model(wide) -> None:
    teacher, _, _ = wide
    ids = torch.tensor(random_blocks(2, seed=2))
    hidden, kwargs = capture_inputs(teacher, ids)
    with torch.no_grad():
        for index, layer in enumerate(teacher.layers):
            hidden = layer(hidden, **kwargs[index])
        expected = teacher(input_ids=ids, use_cache=False).last_hidden_state
        assert torch.allclose(teacher.norm(hidden).float(), expected.float(), atol=1e-2)


def test_rounding_quantizes_every_linear_and_never_ends_worse(wide) -> None:
    teacher, head, linears = wide
    calibration, held_out = random_blocks(4, seed=3), random_blocks(2, seed=4)
    factors, weights = fisher_pass(teacher, head, calibration, linears, chunk_tokens=16)
    units = build_units({n: m.weight for n, m in linears.items()}, factors)
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    settings = RoundingSettings(blocks_per_batch=2, max_iters=20, min_iters=5, eval_every=5)
    quantized, records = quantize_layers(
        teacher,
        calibration,
        held_out,
        uniform(units),
        list(linears),
        weights,
        settings,
        log=lambda _: None,
    )
    assert set(quantized) == set(linears)
    for name, item in quantized.items():
        low, high = code_range(item.option.bits)
        assert (
            item.codes.dtype is torch.int8 and low <= item.codes.min() <= item.codes.max() <= high
        )
        assert item.scales.dtype is torch.bfloat16 and item.option == BASE
    assert all(r.dev_loss_best <= r.dev_loss_rtn for r in records)
    assert any(r.dev_loss_best < r.dev_loss_rtn for r in records)
    after = teacher.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    with pytest.raises(ValueError, match="multiples"):
        quantize_layers(
            teacher, calibration[:3], held_out, uniform(units), list(linears), weights, settings
        )


# ---------------------------------------------------------------- export


def _snapshot(teacher: nn.Module, directory: Path) -> Path:
    """A BF16 checkpoint laid out like Qwen3.5's: the text model, a vision Linear, an MTP tensor."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True)
    tensors = {LANGUAGE_PREFIX + k: v.contiguous() for k, v in teacher.state_dict().items()}
    tensors["model.visual.blocks.0.attn.qkv.weight"] = torch.randn(384, 128).to(torch.bfloat16)
    tensors["model.visual.blocks.0.attn.qkv.bias"] = torch.randn(384).to(torch.bfloat16)
    tensors["model.visual.pos_embed.weight"] = torch.randn(16, 128).to(torch.bfloat16)
    tensors["mtp.fc.weight"] = torch.randn(8, 8).to(torch.bfloat16)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    return directory


def test_vision_linears_are_recognised() -> None:
    assert vision_linear("model.visual.blocks.3.mlp.linear_fc2.weight")
    assert vision_linear("model.visual.merger.linear_fc1.weight")
    assert vision_linear("model.visual.deepstack_merger_list.1.linear_fc2.weight")
    assert not vision_linear("model.visual.pos_embed.weight")
    assert not vision_linear("model.visual.blocks.3.attn.qkv.bias")
    assert not vision_linear("model.language_model.layers.0.mlp.up_proj.weight")


def test_export_round_trips_into_the_student(wide, tmp_path: Path) -> None:
    from safetensors import safe_open

    from qwen35_compression.glaze.student import build_student, read_init_export

    teacher, head, linears = wide
    calibration, held_out = random_blocks(2, seed=5), random_blocks(2, seed=6)
    factors, weights = fisher_pass(teacher, head, calibration, linears, chunk_tokens=16)
    units = build_units({n: m.weight for n, m in linears.items()}, factors)
    snapshot = _snapshot(teacher, tmp_path / "bf16")
    shapes = {n: tuple(m.weight.shape) for n, m in linears.items()}
    plan = plan_bytes(snapshot, shapes)
    allocation = allocate(units, 40_000, resolution=1)
    assert len(set(allocation.choice.values())) > 1  # a real mix of options
    settings = RoundingSettings(blocks_per_batch=2, max_iters=5, min_iters=5, eval_every=5)
    quantized, _ = quantize_layers(
        teacher,
        calibration,
        held_out,
        allocation,
        list(linears),
        weights,
        settings,
        log=lambda _: None,
    )
    out = tmp_path / "export"
    record = write_export(snapshot, out, quantized, log=lambda _: None)
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    targets = [t for g in config["config_groups"].values() for t in g["targets"]]
    assert "model.visual.blocks.0.attn.qkv" in targets and len(targets) == len(linears) + 1
    with safe_open(str(out / "model.safetensors"), "pt") as handle:
        keys = set(handle.keys())
    assert "mtp.fc.weight" not in keys and "model.visual.pos_embed.weight" in keys
    assert "model.visual.blocks.0.attn.qkv.bias" in keys
    assert "model.visual.blocks.0.attn.qkv.weight_packed" in keys
    # The byte plan predicts the export to within its config and header overhead.
    predicted = plan.fixed + plan.language_base + allocation.extra_bytes
    assert 0 <= record["total_bytes"] - predicted < 1_000_000
    # Read back, the student computes exactly the dequantized weights.
    init = read_init_export(out, allow_mixed=True)
    student = build_student(teacher, init, train_norms=False)
    for name, item in quantized.items():
        layer = student.get_submodule(name)
        assert (layer.bits, layer.group_size) == (item.option.bits, item.option.group)
        assert torch.equal(layer.codes, item.codes)
    with pytest.raises(FileExistsError):
        write_export(snapshot, out, quantized, log=lambda _: None)


# ---------------------------------------------------------------- evaluation


def test_held_out_kl_is_zero_for_bf16_and_split_by_domain(wide) -> None:
    import copy

    from qwen35_compression.glaze2.evaluate import held_out_kl, reduction

    teacher, head, _ = wide
    ids = random_blocks(2, seed=7)
    blocks = PackedBlocks(
        ids=ids,
        answer=[[0] * 4 + [1] * (len(ids[0]) - 4) for _ in ids],
        domain=[[DOMAIN_CODES["math"]] * 8 + [DOMAIN_CODES["chat"]] * (len(ids[0]) - 8)] * 2,
    )
    same = held_out_kl(teacher, copy.deepcopy(teacher), head, blocks, blocks_per_batch=1)
    assert same["math"]["mean_kl"] == pytest.approx(0, abs=1e-6)
    assert same["math"]["tokens"] == 2 * 4 and same["mcq"]["tokens"] == 0
    assert same["in_domain"]["tokens"] == 8
    noisy = copy.deepcopy(teacher)
    with torch.no_grad():
        for name, module in noisy.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(fake_quantize(module.weight, Option(4, 32)))
    worse = held_out_kl(teacher, noisy, head, blocks, blocks_per_batch=2)
    assert worse["chat"]["mean_kl"] > 0
    assert reduction(worse, same, "chat") == pytest.approx(1.0)


# ---------------------------------------------------------------- data


def test_decontamination_drops_shared_13_grams() -> None:
    test = "A train leaves the station at noon traveling at sixty miles per hour toward the city"
    clean = Decontaminator([test])
    assert not clean.clean("Q: " + test.upper() + " How long?")
    assert clean.clean(
        "A train leaves the station at noon. Something else entirely different here."
    )


def test_prompts_read_like_the_tasks() -> None:
    assert "\\boxed{}" in math_prompt("What is 2+2?") and math_prompt(" x ").endswith("x")
    text = mcq_prompt("Which is a gas?", ["Iron", "Oxygen"])
    assert "(A) Iron\n(B) Oxygen" in text and '"The answer is (X)"' in text
    row = {
        "question": "q",
        "correct_answer": "c",
        "distractor1": "d1",
        "distractor2": "d2",
        "distractor3": "d3",
    }
    assert sorted(sciq_options(row, 1)) == ["c", "d1", "d2", "d3"]
    assert sciq_options(row, 1) == sciq_options(row, 1)
    held = [is_held_out(f"{i:016x}"[::-1], 0.25) for i in range(4000)]
    assert 0.2 < sum(held) / len(held) < 0.3


def test_packing_keeps_answer_masks_and_domains(tmp_path: Path) -> None:
    samples = [
        Sample("math", [1, 2], [3, 4, 5]),
        Sample("chat", [6], [7, 8]),
        Sample("mcq", [9, 9], [10]),
    ]
    blocks = pack_samples(samples, length=4, count=2)
    assert blocks.ids == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert blocks.answer == [[0, 0, 1, 1], [1, 0, 1, 1]]
    assert blocks.domain == [[0, 0, 0, 0], [0, 2, 2, 2]]
    assert blocks.tokens_by_domain() == {"math": 5, "mcq": 0, "chat": 3}
    save_blocks(tmp_path / "b.jsonl", blocks)
    loaded = load_blocks(tmp_path / "b.jsonl")
    assert loaded.ids == blocks.ids and loaded.answer == blocks.answer
    with pytest.raises(ValueError, match="needed"):
        pack_samples(samples, length=4, count=5)


def test_interleaving_keeps_domain_shares() -> None:
    samples = [Sample(d, [0] * 10, [1] * 10) for d in ("math", "mcq", "chat") for _ in range(50)]
    order = interleave(samples, {"math": 0.4, "mcq": 0.4, "chat": 0.2}, seed=0)
    first = order[:20]
    counts = {d: sum(s.domain == d for s in first) for d in ("math", "mcq", "chat")}
    assert counts == {"math": 8, "mcq": 8, "chat": 4}
    assert len(order) == 150


class _Tokenizer:
    """Chat template: <u> prompt </u> then <a> answer </a>, one token per character."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role'][0]}>{m['content']}</{m['role'][0]}>" for m in messages)
        if add_generation_prompt:
            text += "<a>"
        return [ord(c) for c in text]


def test_chat_samples_split_prompt_from_answer() -> None:
    sample = chat_sample(_Tokenizer(), "math", "hi", "42", {})
    assert "".join(map(chr, sample.prompt_ids)) == "<u>hi</u><a>"
    assert "".join(map(chr, sample.answer_ids)) == "42</a>"


def _cached_tokenizer():
    """Qwen3.5-0.8B's real tokenizer when the Hub cache has it (the chat template matters)."""
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        path = snapshot_download(
            "Qwen/Qwen3.5-0.8B",
            revision="2fc06364715b967f1860aea9cf38778875588b17",
            local_files_only=True,
            allow_patterns=["tokenizer*", "*.jinja", "*.json"],
        )
        return AutoTokenizer.from_pretrained(path)
    except Exception:
        return None


def test_chat_samples_with_the_real_qwen_template() -> None:
    tokenizer = _cached_tokenizer()
    if tokenizer is None:
        pytest.skip("Qwen3.5-0.8B tokenizer not cached")
    sample = chat_sample(tokenizer, "math", "What is 2+2?", "It is 4.", {"enable_thinking": False})
    prompt = tokenizer.decode(sample.prompt_ids)
    answer = tokenizer.decode(sample.answer_ids)
    assert "What is 2+2?" in prompt and "It is 4." in answer
    assert "What is 2+2?" not in answer and all(isinstance(t, int) for t in sample.answer_ids)


def test_token_ids_come_out_of_every_return_shape() -> None:
    from transformers import BatchEncoding

    from qwen35_compression.glaze2.data import _ids

    assert _ids([1, 2]) == [1, 2]
    assert _ids(BatchEncoding({"input_ids": [3, 4]})) == [3, 4]
    assert _ids({"input_ids": [[5, 6]]}) == [5, 6]
    assert _ids(torch.tensor([[7, 8]])) == [7, 8]
    with pytest.raises(ValueError):
        _ids([[1], [2]])
