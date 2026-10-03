"""Glaze: format fidelity, objective, student construction, training, export and guards."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from glaze_tiny import (
    BLOCK,
    GROUP,
    dequantized_reference,
    quantized_names,
    random_blocks,
    rtn_quantize,
    tiny_teacher,
    write_tiny_export,
)

from qwen35_compression.config import GlazeConfig
from qwen35_compression.glaze.budget import (
    BudgetExceeded,
    check_memory,
    check_time,
    memory_estimate_gib,
    projected_minutes,
)
from qwen35_compression.glaze.data import blocks_digest, epoch_order, groups, split_blocks
from qwen35_compression.glaze.export import write_refined_export
from qwen35_compression.glaze.grid import GridDescent, grid_neighbor
from qwen35_compression.glaze.losses import DistillStats, distill_step
from qwen35_compression.glaze.quant_linear import (
    GroupQuantLinear,
    dequantize_groups,
    pack_int4,
    unpack_int4,
)
from qwen35_compression.glaze.student import (
    LANGUAGE_MODEL_PREFIX,
    build_student,
    exported_tensors,
    init_group_size,
    load_trainable_state,
    read_init_export,
    trainable_state,
)
from qwen35_compression.glaze.train import (
    GlazeTrainer,
    PilotFailed,
    learning_rate_factor,
    make_schedule,
)

# ---------------------------------------------------------------- format and the quantized layer


def _codes(rows: int, columns: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-8, 8, (rows, columns), generator=generator, dtype=torch.int8)


def _scales(rows: int, groups_: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(rows, groups_, generator=generator) * 0.02 + 1e-3).to(torch.bfloat16)


@pytest.mark.parametrize("shape", [(4, 64), (32, 256), (3, 128)])
def test_int4_codes_survive_packing(shape: tuple[int, int]) -> None:
    codes = _codes(*shape)
    packed = pack_int4(codes)
    assert packed.dtype is torch.int32
    assert torch.equal(unpack_int4(packed, shape), codes)


@pytest.mark.parametrize("group", [32, 128])
def test_dequantization_matches_compressed_tensors_bit_for_bit(group: int) -> None:
    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.lifecycle.forward import dequantize

    codes = _codes(16, 4 * group)
    scales = _scales(16, 4)
    args = QuantizationArgs(
        num_bits=4, type="int", symmetric=True, strategy="group", group_size=group
    )
    expected = dequantize(codes, scales, None, args)
    actual = dequantize_groups(codes, scales, group)
    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual, expected)


def test_quantized_layer_starts_exactly_at_the_export() -> None:
    codes, scales = _codes(8, 64), _scales(8, 2)
    layer = GroupQuantLinear(codes, scales, 32)
    assert torch.equal(layer.stored_scales(), scales)
    inputs = torch.randn(3, 5, 64, dtype=torch.bfloat16)
    expected = F.linear(inputs, dequantize_groups(codes, scales, 32))
    assert torch.equal(layer(inputs), expected)
    assert [name for name, _ in layer.named_parameters()] == ["scales"]
    assert layer.scales.dtype is torch.float32 and torch.equal(layer.initial_scales, scales)


def test_scale_gradients_match_finite_differences() -> None:
    codes = _codes(4, 16)
    scales = (torch.rand(4, 2, dtype=torch.float64) + 0.5) * 0.1
    layer = GroupQuantLinear(codes, scales, 8)
    inputs = torch.randn(3, 16, dtype=torch.float64)

    def output(values: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(layer, {"scales": values}, (inputs,))

    start = scales.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(output, (start,))


def test_grid_neighbor_moves_one_bf16_point() -> None:
    values = torch.tensor([1.0, -1.0, 0.0123, -0.0123, 3.0e-3]).to(torch.bfloat16).float()
    for sign in (1.0, -1.0):
        limit = torch.full_like(values, sign * float("inf")).to(torch.bfloat16)
        expected = torch.nextafter(values.to(torch.bfloat16), limit).float()
        moved = grid_neighbor(values, torch.full_like(values, sign), torch.bfloat16)
        assert torch.equal(moved, expected)
        # Results lie on the grid and keep their sign.
        assert torch.equal(moved.to(torch.bfloat16).float(), moved)
        assert torch.equal(torch.sign(moved), torch.sign(values))
    up = grid_neighbor(values, torch.ones_like(values), torch.bfloat16)
    assert up[0].item() == 1.0078125 and up[1].item() == -0.99609375


def test_grid_neighbor_refuses_zero_sign_changes_and_overflow() -> None:
    smallest = torch.tensor([9.183549615799121e-41])  # the smallest positive BF16 subnormal
    assert torch.isnan(grid_neighbor(smallest, torch.tensor([-1.0]), torch.bfloat16)).all()
    largest = torch.tensor([3.3895313892515355e38])
    assert torch.isnan(grid_neighbor(largest, torch.tensor([1.0]), torch.bfloat16)).all()
    zero, one = torch.tensor([0.0]), torch.tensor([1.0])
    assert torch.isnan(grid_neighbor(zero, one, torch.bfloat16)).all()
    assert torch.isnan(grid_neighbor(one, zero, torch.bfloat16)).all()
    double = torch.tensor([1.0], dtype=torch.float64)
    assert grid_neighbor(double, torch.tensor([1.0]), torch.float64).item() == 1 + 2**-52


def _on_grid(*values: float) -> torch.Tensor:
    return torch.tensor(values).to(torch.bfloat16).float()


def test_grid_descent_moves_only_the_most_promising_values() -> None:
    first = _on_grid(1.0, 1.0, 1.0).requires_grad_(True)
    second = _on_grid(-0.5, 2.0).requires_grad_(True)
    optimizer = GridDescent([first, second], torch.bfloat16, momentum=0.0)
    first.grad = torch.tensor([0.1, -3.0, 0.0])
    second.grad = torch.tensor([2.0, 0.5])
    # Predicted decrease = |gradient| x one grid step: first[1] (3 x 2^-7) and second[0]
    # (2 x 2^-8) beat second[1] (0.5 x 2^-7) and first[0] (0.1 x 2^-8).
    assert optimizer.step(2) == 2
    assert first.tolist() == [1.0, 1.0078125, 1.0]  # negative gradient: up
    assert second.tolist() == [-0.50390625, 2.0]  # positive gradient: down, away from zero
    assert optimizer.averages[0][1] == 0 and optimizer.averages[1][0] == 0
    assert optimizer.averages[1][1] == 0.5  # not moved: its evidence is kept
    # With no gradient there is no predicted gain, and nothing moves.
    rested = GridDescent([_on_grid(1.0, 2.0).requires_grad_(True)], torch.bfloat16)
    assert rested.step(10) == 0


def test_grid_descent_needs_values_on_the_grid() -> None:
    with pytest.raises(ValueError, match="grid"):
        GridDescent([torch.tensor([1.001], requires_grad=True)], torch.bfloat16)
    with pytest.raises(ValueError, match="momentum"):
        GridDescent([_on_grid(1.0)], torch.bfloat16, momentum=1.0)
    with pytest.raises(ValueError, match="nothing"):
        GridDescent([], torch.bfloat16)


def test_quantized_layer_rejects_bad_tensors() -> None:
    codes, scales = _codes(4, 64), _scales(4, 2)
    with pytest.raises(ValueError, match="int8"):
        GroupQuantLinear(codes.to(torch.int32), scales, 32)
    with pytest.raises(ValueError, match="shape"):
        GroupQuantLinear(codes, _scales(4, 4), 32)
    with pytest.raises(ValueError, match="multiple"):
        GroupQuantLinear(codes, scales, 48)
    with pytest.raises(ValueError, match="INT4 range"):
        GroupQuantLinear(torch.full((4, 64), 9, dtype=torch.int8), scales, 32)
    with pytest.raises(ValueError, match="finite"):
        GroupQuantLinear(codes, torch.full_like(scales, float("nan")), 32)


def test_negative_scales_keep_their_sign() -> None:
    # AutoRound's full-range symmetric INT4 stores negative scales for some groups.
    codes, scales = _codes(4, 64), _scales(4, 2)
    scales[0, 1] = -scales[0, 1]
    layer = GroupQuantLinear(codes, scales, 32)
    assert torch.equal(layer.stored_scales(), scales)
    with torch.no_grad():
        layer.scales.copy_(layer.scales.to(torch.bfloat16).float())
    optimizer = GridDescent([layer.scales], torch.bfloat16, momentum=0.0)
    for sign in (1.0, -1.0):
        layer.scales.grad = torch.full_like(layer.scales, sign)
        assert optimizer.step(layer.scales.numel()) == layer.scales.numel()
    assert torch.equal(torch.sign(layer.stored_scales()), torch.sign(scales))


# ---------------------------------------------------------------- objective


def _direct(student: torch.Tensor, teacher: torch.Tensor, head: torch.Tensor, mask: torch.Tensor):
    s = torch.log_softmax(student[mask] @ head.T, -1)
    t = torch.log_softmax(teacher[mask] @ head.T, -1)
    kl = (t.exp() * (t - s)).sum(-1)
    top = t.argmax(-1)
    flips = int((s.argmax(-1) != top).sum())
    nll = float(-s.gather(-1, top[:, None]).sum().detach())
    return kl, flips, nll


def _hidden(seed: int, requires_grad: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(2, 5, 8, dtype=torch.float64, generator=generator)
    return value.requires_grad_(requires_grad)


@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_chunked_kl_equals_the_direct_computation(chunk: int) -> None:
    head = torch.randn(11, 8, dtype=torch.float64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 1] = mask[1, 4] = False
    student, teacher = _hidden(0, True), _hidden(1)
    stats = distill_step(student, teacher, head, mask, chunk, normalizer=4.0, backward=True)

    reference = _hidden(0, True)
    kl, flips, nll = _direct(reference, teacher, head, mask)
    (kl.sum() / 4.0).backward()
    assert stats.tokens == 8
    assert stats.kl == pytest.approx(float(kl.sum().detach()), rel=1e-12)
    assert stats.flips == flips
    assert stats.nll == pytest.approx(nll, rel=1e-12)
    assert torch.allclose(student.grad, reference.grad, atol=1e-14)
    # Masked positions take no gradient.
    assert torch.all(student.grad[0, 1] == 0) and torch.all(student.grad[1, 4] == 0)


def test_kl_is_zero_only_when_the_student_matches() -> None:
    head = torch.randn(11, 8, dtype=torch.float64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    same = _hidden(2, True)
    stats = distill_step(same, _hidden(2), head, mask, 4, 10.0, backward=True)
    assert stats.kl == pytest.approx(0.0, abs=1e-12) and stats.flips == 0
    assert same.grad.abs().max() < 1e-12
    other = distill_step(_hidden(3), _hidden(2), head, mask, 4, 10.0, backward=False)
    assert other.kl > 0
    assert other.summary()["mean_kl"] == pytest.approx(other.kl / 10)


def test_distill_step_rejects_mismatched_inputs() -> None:
    head = torch.randn(11, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="positive"):
        distill_step(
            _hidden(0), _hidden(1), head, torch.ones(2, 5, dtype=torch.bool), 0, 1.0, False
        )
    with pytest.raises(ValueError, match="shapes"):
        distill_step(
            _hidden(0), _hidden(1), head, torch.ones(2, 4, dtype=torch.bool), 2, 1.0, False
        )


def test_stats_merge_and_summarise() -> None:
    total = DistillStats()
    total.merge(DistillStats(tokens=4, kl=2.0, flips=1, nll=3.0))
    total.merge(DistillStats(tokens=6, kl=1.0, flips=2, nll=1.0))
    assert total.summary() == {"tokens": 10, "mean_kl": 0.3, "flip_rate": 0.3, "mean_nll": 0.4}
    assert DistillStats().summary()["mean_kl"] is None
    assert math.isnan(DistillStats().mean_kl)


# ---------------------------------------------------------------- student and export


@pytest.fixture()
def tiny(tmp_path: Path):
    teacher = tiny_teacher()
    init_dir = tmp_path / "init"
    names = write_tiny_export(teacher, init_dir)
    return teacher, init_dir, names


def test_student_reproduces_the_init_export_exactly(tiny) -> None:
    teacher, init_dir, _ = tiny
    student = build_student(teacher, read_init_export(init_dir))
    reference = dequantized_reference(teacher, init_dir)
    ids = torch.tensor(random_blocks(2, seed=5))
    with torch.no_grad():
        got = student(input_ids=ids, use_cache=False).last_hidden_state
        expected = reference(input_ids=ids, use_cache=False).last_hidden_state
        bf16 = teacher(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(got, expected)
    # Quantization changes the output, so the test can tell the student from the teacher.
    assert not torch.equal(got, bf16)


def test_student_trains_only_scales_and_norms(tiny) -> None:
    teacher, init_dir, names = tiny
    student = build_student(teacher, read_init_export(init_dir))
    trainable = [name for name, p in student.named_parameters() if p.requires_grad]
    scales = [name for name in trainable if name.endswith(".scales")]
    norms = [name for name in trainable if name.endswith("parametrizations.weight.original")]
    assert sorted(scales) == sorted(f"{name}.scales" for name in names)
    assert len(norms) == 4 * 2 + 3 + 2 + 1  # layer norms, DeltaNet norms, q/k norms, final norm
    assert sorted(trainable) == sorted(scales + norms)
    assert all(p.dtype is torch.float32 for n, p in student.named_parameters() if n in norms)
    # The embedding is shared with the teacher, not copied.
    assert student.get_input_embeddings().weight is teacher.get_input_embeddings().weight
    without_norms = build_student(teacher, read_init_export(init_dir), train_norms=False)
    assert all(
        n.endswith(".scales") for n, p in without_norms.named_parameters() if p.requires_grad
    )


def test_untouched_student_exports_every_tensor_unchanged(tiny, tmp_path: Path) -> None:
    from safetensors.torch import load_file

    teacher, init_dir, _ = tiny
    (init_dir / "compression_manifest.json").write_text("{}", encoding="utf-8")
    (init_dir / "README.md").write_text("card", encoding="utf-8")
    student = build_student(teacher, read_init_export(init_dir))
    output = tmp_path / "out"
    write_refined_export(init_dir, output, exported_tensors(student))
    before = load_file(str(init_dir / "model.safetensors"))
    after = load_file(str(output / "model.safetensors"))
    assert before.keys() == after.keys()
    for key in before:
        assert before[key].dtype == after[key].dtype and torch.equal(before[key], after[key]), key
    for name in ("config.json", "tokenizer.json", "recipe.yaml"):
        assert (output / name).read_bytes() == (init_dir / name).read_bytes()
    # The init's manifest and card describe the init; they are not carried over.
    assert not (output / "compression_manifest.json").exists()
    assert not (output / "README.md").exists()


def test_trained_export_changes_only_scales_and_norms(tiny, tmp_path: Path) -> None:
    from safetensors.torch import load_file

    teacher, init_dir, _ = tiny
    student = build_student(teacher, read_init_export(init_dir))
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if parameter.requires_grad:
                parameter.add_(0.05)
    output = tmp_path / "out"
    write_refined_export(init_dir, output, exported_tensors(student))
    before = load_file(str(init_dir / "model.safetensors"))
    after = load_file(str(output / "model.safetensors"))
    changed = {key for key in before if not torch.equal(before[key], after[key])}
    assert changed and all(key.endswith((".weight_scale", "norm.weight")) for key in changed)
    for key in before:
        assert before[key].dtype == after[key].dtype and before[key].shape == after[key].shape
    # A rebuilt student from the new export computes what the trained student computes.
    rebuilt = build_student(teacher, read_init_export(output))
    ids = torch.tensor(random_blocks(2, seed=6))
    with torch.no_grad():
        assert torch.equal(
            rebuilt(input_ids=ids, use_cache=False).last_hidden_state,
            student(input_ids=ids, use_cache=False).last_hidden_state,
        )


def test_export_refuses_unknown_or_reshaped_tensors(tiny, tmp_path: Path) -> None:
    teacher, init_dir, names = tiny
    key = f"{LANGUAGE_MODEL_PREFIX}{names[0]}.weight_scale"
    with pytest.raises(ValueError, match="not found"):
        write_refined_export(init_dir, tmp_path / "a", {"model.nope": torch.zeros(1)})
    with pytest.raises(ValueError, match="does not match"):
        write_refined_export(init_dir, tmp_path / "b", {key: torch.zeros(1, dtype=torch.bfloat16)})
    occupied = tmp_path / "c"
    occupied.mkdir()
    (occupied / "x").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_refined_export(init_dir, occupied, {})


def test_init_export_must_be_symmetric_int4_groups() -> None:
    good = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 128,
                    },
                    "input_activations": None,
                }
            },
        }
    }
    assert init_group_size(good) == 128
    for path, value in (
        (("weights", "symmetric"), False),
        (("weights", "num_bits"), 8),
        (("weights", "strategy"), "channel"),
        (("input_activations",), {"num_bits": 8}),
    ):
        bad = json.loads(json.dumps(good))
        target = bad["quantization_config"]["config_groups"]["group_0"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError):
            init_group_size(bad)
    with pytest.raises(ValueError, match="pack-quantized"):
        init_group_size({"quantization_config": {"format": "float-quantized"}})


def test_student_rejects_an_export_that_does_not_fit(tiny, tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    teacher, init_dir, names = tiny
    tensors = load_file(str(init_dir / "model.safetensors"))
    missing = dict(tensors)
    del missing[f"{LANGUAGE_MODEL_PREFIX}norm.weight"]
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text((init_dir / "config.json").read_text(), encoding="utf-8")
    save_file(missing, str(broken / "model.safetensors"))
    with pytest.raises(ValueError, match="no tensor"):
        build_student(teacher, read_init_export(broken))

    extra = dict(tensors)
    extra[f"{LANGUAGE_MODEL_PREFIX}layers.0.unexpected.weight"] = torch.zeros(2)
    save_file(extra, str(broken / "model.safetensors"))
    with pytest.raises(ValueError, match="does not use"):
        build_student(teacher, read_init_export(broken))

    reshaped = dict(tensors)
    reshaped[f"{LANGUAGE_MODEL_PREFIX}{names[0]}.weight_shape"] = torch.tensor([1, 64])
    save_file(reshaped, str(broken / "model.safetensors"))
    with pytest.raises(ValueError, match="does not match"):
        build_student(teacher, read_init_export(broken))


# ---------------------------------------------------------------- training


def _glaze(**overrides) -> GlazeConfig:
    settings = {
        "train_blocks": 8,
        "dev_blocks": 2,
        "epochs": 4,
        "tokens_per_step": 4 * BLOCK,
        "micro_batch_tokens": 2 * BLOCK,
        "flip_fractions": (0.05,),
        "probe_steps": 0,
        "warmup_steps": 1,
        "logit_chunk_tokens": 7,
        "eval_every_steps": 2,
        "max_train_minutes": 60.0,
    }
    settings.update(overrides)
    return GlazeConfig(**settings)


def _trainer(tiny, glaze: GlazeConfig) -> tuple[GlazeTrainer, torch.nn.Module]:
    teacher, init_dir, _ = tiny
    student = build_student(teacher, read_init_export(init_dir))
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    head = teacher.get_input_embeddings().weight
    return GlazeTrainer(student, teacher, head, glaze, log=lambda _: None), student


def test_training_moves_the_student_toward_the_teacher(tiny) -> None:
    glaze = _glaze()
    trainer, student = _trainer(tiny, glaze)
    teacher = tiny[0]
    frozen = {
        name: tensor.clone()
        for name, tensor in student.state_dict().items()
        if not name.endswith(".scales") and "original" not in name
    }
    train, _ = split_blocks(random_blocks(10, seed=7), glaze)
    schedule = make_schedule(glaze, BLOCK)
    # Scored on training blocks: random tokens share no structure a held-out set could test.
    result = trainer.fit(train, train[:2], schedule, 0.05)
    assert result.best_step > 0
    assert result.best_dev["mean_kl"] < 0.9 * result.initial_dev["mean_kl"]
    assert len(result.history) == schedule.total_steps == 8
    assert [entry["step"] for entry in result.history if "dev" in entry] == [2, 4, 6, 8]
    assert all(entry["moved"] >= 1 for entry in result.history)
    # Every trained value is still a BF16 number the export can store exactly.
    for parameter in trainer.parameters:
        assert torch.equal(parameter.to(torch.bfloat16).float(), parameter)
    # Codes, base scales, embedding, conv and decay parameters never move.
    after = student.state_dict()
    for name, tensor in frozen.items():
        assert torch.equal(after[name], tensor), name
    assert student.get_input_embeddings().weight is teacher.get_input_embeddings().weight


def test_training_is_deterministic(tiny) -> None:
    glaze = _glaze(epochs=1)
    blocks = random_blocks(10, seed=8)
    train, dev = split_blocks(blocks, glaze)
    states = []
    for _ in range(2):
        trainer, student = _trainer(tiny, glaze)
        trainer.fit(train, dev, make_schedule(glaze, BLOCK), 0.05)
        states.append(trainable_state(student))
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


def test_probe_picks_the_better_rate_and_restores_the_init(tiny) -> None:
    glaze = _glaze(flip_fractions=(1e-9, 0.05), probe_steps=2)
    trainer, student = _trainer(tiny, glaze)
    initial = trainable_state(student)
    train, _ = split_blocks(random_blocks(10, seed=9), glaze)
    dev = train[:2]  # fit data, so the useful rate is the one that lowers it
    rate, results = trainer.probe(train, dev, make_schedule(glaze, BLOCK))
    assert rate == 0.05 and set(results) == {repr(1e-9), repr(0.05)}
    after = trainable_state(student)
    assert all(torch.equal(initial[key], after[key]) for key in initial)
    single = GlazeTrainer(student, tiny[0], tiny[0].get_input_embeddings().weight, _glaze())
    assert single.probe(train, dev, make_schedule(glaze, BLOCK)) == (0.05, {})


def test_pilot_passes_when_kl_falls_and_fails_when_it_does_not(tiny) -> None:
    glaze = _glaze()
    trainer, _ = _trainer(tiny, glaze)
    train, _ = split_blocks(random_blocks(10, seed=10), glaze)
    schedule = make_schedule(glaze, BLOCK)
    losses = trainer.pilot(train, schedule, 4, 0.05)
    assert len(losses) == 4 and min(losses[1:]) < losses[0]

    class NoMoves:
        def zero_grad(self) -> None:
            pass

        def step(self, flips: int) -> int:
            return 0

    stuck, _ = _trainer(tiny, glaze)
    stuck.make_optimizer = NoMoves
    with pytest.raises(PilotFailed):
        stuck.pilot(train, schedule, 3, 0.05)
    with pytest.raises(ValueError):
        stuck.pilot(train, schedule, 1, 0.05)


def test_saved_state_round_trips(tiny) -> None:
    trainer, student = _trainer(tiny, _glaze())
    state = trainable_state(student)
    with torch.no_grad():
        for parameter in student.parameters():
            if parameter.requires_grad:
                parameter.add_(1.0)
    load_trainable_state(student, state)
    assert all(torch.equal(state[k], v) for k, v in trainable_state(student).items())
    with pytest.raises(ValueError):
        load_trainable_state(student, {"nope": torch.zeros(1)})


def test_schedule_and_learning_rate() -> None:
    glaze = GlazeConfig()
    schedule = make_schedule(glaze, 2048)
    assert (schedule.blocks_per_micro, schedule.micro_per_step) == (4, 2)
    assert (schedule.steps_per_epoch, schedule.total_steps) == (16, 64)
    factors = [learning_rate_factor(step, 3, 64) for step in range(64)]
    assert factors[:4] == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0])
    assert all(a >= b for a, b in zip(factors[2:], factors[3:], strict=False))
    assert 0 < factors[-1] < 0.01


def test_data_split_and_order() -> None:
    glaze = GlazeConfig(train_blocks=4, dev_blocks=2)
    blocks = [[i] * 3 for i in range(7)]
    train, dev = split_blocks(blocks, glaze)
    assert train == blocks[:4] and dev == blocks[4:6]
    with pytest.raises(ValueError, match="needed"):
        split_blocks(blocks[:5], glaze)
    with pytest.raises(ValueError, match="length"):
        split_blocks([[1, 2]] + blocks[1:], glaze)
    assert epoch_order(8, 0, 42) == epoch_order(8, 0, 42) != epoch_order(8, 1, 42)
    assert sorted(epoch_order(8, 3, 42)) == list(range(8))
    assert groups([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        groups([1, 2, 3], 2)
    assert blocks_digest(blocks) == blocks_digest([list(b) for b in blocks]) != blocks_digest(dev)


# ---------------------------------------------------------------- guards and estimates


def test_memory_estimate_reproduces_the_plan() -> None:
    from qwen35_compression.glaze.train import QWEN35_4B_SHAPES

    plan_estimate = memory_estimate_gib(
        **QWEN35_4B_SHAPES, micro_batch_tokens=8192, chunk_tokens=1024
    )
    assert plan_estimate["total"] == pytest.approx(19.95, abs=0.01)
    assert plan_estimate["logit_chunk"] == pytest.approx(3.32, abs=0.01)
    configured = memory_estimate_gib(**QWEN35_4B_SHAPES, micro_batch_tokens=8192, chunk_tokens=512)
    assert configured["total"] < plan_estimate["total"]
    assert configured["total"] * 1.4 < 0.85 * 39.5  # fits an A100 40 GB with margin


def test_guards_stop_runs_that_would_overrun() -> None:
    check_memory(30 * 2**30, 40 * 2**30, 0.85)
    with pytest.raises(BudgetExceeded, match="memory"):
        check_memory(35 * 2**30, 40 * 2**30, 0.85)
    assert projected_minutes(60, 2, 64) == pytest.approx(32.0)
    check_time(50, 2, 64, 30)
    with pytest.raises(BudgetExceeded, match="budget"):
        check_time(60, 2, 64, 30)
    with pytest.raises(ValueError):
        projected_minutes(10, 0, 64)


def test_rtn_fixture_is_a_real_quantization() -> None:
    weight = torch.randn(4, 64, dtype=torch.bfloat16)
    codes, scales = rtn_quantize(weight)
    error = (dequantize_groups(codes, scales, GROUP).float() - weight.float()).abs()
    assert 0 < error.max() <= scales.float().max() / 2 + 1e-3
    assert len(quantized_names(tiny_teacher())) == 3 * 8 + 7


@pytest.mark.parametrize("flips", [1, 7, 50])
def test_grid_descent_bounds_each_step(flips: int) -> None:
    # The pilot that failed moved every scale together; a step here moves at most `flips`
    # values, each exactly one grid step against its gradient average.
    generator = torch.Generator().manual_seed(flips)
    values = (torch.randn(4, 25, generator=generator) * 0.02).to(torch.bfloat16).float()
    parameter = values.clone().requires_grad_(True)
    optimizer = GridDescent([parameter], torch.bfloat16, momentum=0.0)
    parameter.grad = torch.randn(4, 25, generator=generator)
    expected = grid_neighbor(values, -torch.sign(parameter.grad), torch.bfloat16)
    moved = optimizer.step(flips)
    changed = parameter.detach() != values
    assert moved == int(changed.sum()) <= flips
    assert torch.equal(parameter.detach()[changed], expected[changed])
