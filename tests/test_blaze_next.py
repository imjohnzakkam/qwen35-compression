"""Blaze's real export, its vLLM plugin glue, the byte market, and the G2/G3/real reports."""

from __future__ import annotations

import importlib
import itertools
import json
import random
import sys
import types
from pathlib import Path

import pytest
import torch
from glaze_tiny import WIDE, random_blocks, tiny_teacher
from test_blaze import BODY, QUIET, UNTOUCHED, _checkpoint, _drift, _load_script

from qwen35_compression.blaze.body import layer_part_units
from qwen35_compression.blaze.export import TIED_TABLE, size_plan, write_fake_export
from qwen35_compression.blaze.gates import g2_report, g3_report, real_report
from qwen35_compression.blaze.gptq import gptq_body
from qwen35_compression.blaze.ladder import LadderPlan, ladder_table
from qwen35_compression.blaze.market import Step, cheapest_cover, steps
from qwen35_compression.blaze.real import (
    ARCHITECTURE_KEY,
    COLD,
    HOT,
    POSITION,
    positions,
    read_table,
    write_real_export,
)
from qwen35_compression.glaze2.export import unpack
from qwen35_compression.glaze2.quant import Option, dequantize, fake_quantize

PLAN = LadderPlan((5, 1, 9, 0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15), 4)


def _real(tmp_path: Path) -> tuple[dict[str, torch.Tensor], Path]:
    tensors = _checkpoint(tmp_path / "bf16")
    out = tmp_path / "real"
    write_real_export(tmp_path / "bf16", out, PLAN, Option(8, 128), row_multiple=1, **QUIET)
    return tensors, out


def test_real_export_stores_the_fake_exports_values(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    tensors, out = _real(tmp_path)
    assert torch.equal(read_table(out), ladder_table(tensors[TIED_TABLE], PLAN))
    stored = load_file(str(out / "model.safetensors"))
    module = BODY.removesuffix(".weight")
    codes = unpack(stored[f"{module}.weight_packed"], 8, (128, 256))
    body = dequantize(codes.float(), stored[f"{module}.weight_scale"], Option(8, 128))
    assert torch.equal(body, fake_quantize(tensors[BODY], Option(8, 128)))
    assert TIED_TABLE not in stored and "mtp.fc.weight" not in stored
    assert all(torch.equal(stored[k], tensors[k]) for k in UNTOUCHED if not k.startswith("mtp"))
    config = json.loads((out / "config.json").read_text())
    assert config[ARCHITECTURE_KEY] == {
        "hot_rows": 4,
        "hot_bits": 8,
        "cold_bits": 4,
        "group_size": 128,
    }
    groups = config["quantization_config"]["config_groups"]
    assert groups["blaze_hot"]["targets"] == ["re:.*embed_tokens\\.hot$"]
    assert groups["blaze_cold"]["weights"]["num_bits"] == 4
    assert any(module in g["targets"] for g in groups.values())


def test_positions_invert_the_usage_order() -> None:
    position = positions(PLAN)
    assert [int(position[token]) for token in PLAN.order] == list(range(16))


def test_real_export_refuses_untied_tables_and_ragged_blocks(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "untied", tied=False)
    with pytest.raises(ValueError, match="tied"):
        write_real_export(tmp_path / "untied", tmp_path / "a", PLAN, Option(8, 128), **QUIET)
    _checkpoint(tmp_path / "bf16")
    with pytest.raises(ValueError, match="multiples of 64"):
        write_real_export(tmp_path / "bf16", tmp_path / "b", PLAN, Option(8, 128), **QUIET)


def _stub_vllm(bits_by_prefix: dict[str, int]) -> None:
    """Just enough of vLLM 0.29 for blaze_vllm.model: quantized blocks dequantized in torch."""

    class Param(torch.nn.Parameter):
        def __new__(cls, data):
            param = super().__new__(cls, data, requires_grad=False)
            param.weight_loader = lambda p, w: p.data.copy_(w)
            return param

    def bits_of(prefix: str) -> int:
        return next(b for p, b in bits_by_prefix.items() if prefix.endswith(p))

    class Block(torch.nn.Module):
        def __init__(self, rows, hidden, quant_config=None, prefix=""):
            super().__init__()
            self.bits = bits_of(prefix)
            self.org_vocab_size = self.num_embeddings_padded = rows
            self.hidden = hidden
            self.weight_packed = Param(
                torch.zeros(rows, hidden * self.bits // 32, dtype=torch.int32)
            )
            self.weight_scale = Param(torch.zeros(rows, hidden // 128, dtype=torch.bfloat16))
            self.weight_shape = Param(torch.zeros(2, dtype=torch.int64))
            outer = self

            class Method:
                def apply(self, layer, x):
                    return x @ outer.table().float().T

            self.quant_method = Method()

        def table(self) -> torch.Tensor:
            shape = (self.org_vocab_size, self.hidden)
            codes = unpack(self.weight_packed.data, self.bits, shape)
            return dequantize(codes.float(), self.weight_scale.data, Option(self.bits, 128))

        def forward(self, ids):
            return self.table()[ids]

    class Base(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.language_model = types.SimpleNamespace(model=types.SimpleNamespace())

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.quantization": types.ModuleType("q"),
        "vllm.model_executor.layers.quantization.base_config": types.ModuleType("b"),
        "vllm.model_executor.layers.vocab_parallel_embedding": types.ModuleType("v"),
        "vllm.model_executor.models": types.ModuleType("m"),
        "vllm.model_executor.models.qwen3_5": types.ModuleType("q35"),
    }
    modules["vllm.model_executor.layers.quantization.base_config"].QuantizeMethodBase = object
    modules["vllm.model_executor.layers.vocab_parallel_embedding"].VocabParallelEmbedding = Block
    modules["vllm.model_executor.layers.vocab_parallel_embedding"].ParallelLMHead = Block
    modules["vllm.model_executor.models.qwen3_5"].Qwen3_5ForConditionalGeneration = Base
    sys.modules.update(modules)


def test_plugin_serves_the_real_table_as_the_fake_one(tmp_path: Path, monkeypatch) -> None:
    from safetensors.torch import load_file

    tensors, out = _real(tmp_path)
    saved = dict(sys.modules)
    _stub_vllm({"hot": 8, "cold": 4})
    monkeypatch.syspath_prepend("plugins/blaze_vllm")
    try:
        model = importlib.import_module("blaze_vllm.model")
        vocab = model.TieredVocab(16, 128, 4, None, prefix="language_model.model.embed_tokens")
        stored = load_file(str(out / "model.safetensors"))
        prefix = TIED_TABLE.removesuffix("weight")
        weights = [(k.removeprefix(prefix), v) for k, v in stored.items() if k.startswith(prefix)]
        loaded = vocab.load_weights(iter(weights))
        assert loaded == {
            f"{m}.{p}"
            for m in ("hot", "cold", "hot_head", "cold_head")
            for p in ("weight_packed", "weight_scale", "weight_shape")
        } | {"position"}
        fake = ladder_table(tensors[TIED_TABLE], PLAN).float()
        ids = torch.tensor([0, 5, 9, 15, 1, 4])
        assert torch.equal(vocab(ids).float(), fake[ids])
        vocab.quant_method.process_weights_after_loading(vocab)
        hidden = torch.randn(3, 128)
        logits = vocab.quant_method.apply(vocab, hidden)
        assert torch.allclose(logits, hidden @ fake.T, atol=1e-4)
        model.LOGIT_CHUNK = 2  # several chunks, one output
        assert torch.allclose(vocab.quant_method.apply(vocab, hidden), logits)

        # One copy: logits straight from the lookup blocks, no LM-head modules.
        single = model.TieredVocab(16, 128, 4, None, prefix="x.embed_tokens", single_copy=True)
        assert not hasattr(single, "hot_head")
        assert single.load_weights(iter(weights)) == {
            f"{m}.{p}"
            for m in ("hot", "cold")
            for p in ("weight_packed", "weight_scale", "weight_shape")
        } | {"position"}
        single.quant_method.process_weights_after_loading(single)
        assert torch.equal(single(ids).float(), fake[ids])
        assert torch.allclose(single.quant_method.apply(single, hidden), logits, atol=1e-4)
    finally:
        for name in list(sys.modules):
            if name not in saved:
                del sys.modules[name]
    assert HOT.endswith("hot") and COLD.endswith("cold") and POSITION.endswith("position")


def test_layer_parts_are_mixers_and_ffns_in_layer_order() -> None:
    teacher = tiny_teacher(**WIDE)
    units = layer_part_units(teacher)
    layers = len(teacher.layers)
    assert list(units)[:2] == ["layers.0.mixer", "layers.0.mlp"] and len(units) == 2 * layers
    assert all("mlp." in m for m in units["layers.0.mlp"])
    assert all(".linear_attn." in m or ".self_attn." in m for m in units["layers.0.mixer"])


def test_cheapest_cover_matches_brute_force() -> None:
    rng = random.Random(0)
    for _ in range(40):
        options = [
            Step(f"u{i}", rng.random(), rng.randint(1, 9) * 4096) for i in range(rng.randint(2, 7))
        ]
        deficit = rng.randint(0, sum(s.saved for s in options))
        chosen = cheapest_cover(options, deficit, bin_bytes=4096)
        assert sum(s.saved for s in options if s.unit in chosen) >= deficit
        best = min(
            sum(s.cost for s in combo)
            for r in range(len(options) + 1)
            for combo in itertools.combinations(options, r)
            if sum(s.saved for s in combo) >= deficit
        )
        assert sum(s.cost for s in options if s.unit in chosen) == pytest.approx(best)
    with pytest.raises(ValueError, match="saves less"):
        cheapest_cover([Step("a", 1.0, 4096)], 10 * 4096, bin_bytes=4096)
    assert steps({"a": 0.2}, {"a": 0.1}, {"a": 8})[0].cost == 0.0


def test_gptq_and_exports_take_per_linear_options(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    teacher = tiny_teacher(**WIDE)
    names = [n for n, m in teacher.named_modules() if isinstance(m, torch.nn.Linear)]
    mixed = {name: Option(4 if ".mlp." in name else 6, 128) for name in names}
    stored = gptq_body(teacher, random_blocks(2), mixed, batch=2, log=lambda _: None)
    assert set(stored) == set(names)
    tensors = _checkpoint(tmp_path / "bf16")
    options = {BODY: Option(4, 128)}
    record = write_fake_export(
        tmp_path / "bf16", tmp_path / "out", None, Option(6, 128), options=options, **QUIET
    )
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert torch.equal(out[BODY], fake_quantize(tensors[BODY], Option(4, 128)))
    assert record["body"]["stored_bytes"] == Option(4, 128).tensor_bytes(128, 256)
    plan = size_plan(tmp_path / "bf16", Option(6, 128), None, options)
    assert plan["parts"]["body"] == Option(4, 128).tensor_bytes(128, 256)


def test_g2_g3_and_real_reports() -> None:
    bf16 = _drift(0.128, 0.0035)
    passed = g2_report(bf16, _drift(0.1325, 0.012), 99, 100)
    assert passed["checks"] == {"beats_glaze_v2": True, "no_larger": True}
    assert passed["decision"].startswith("go")
    assert g2_report(bf16, _drift(0.135, 0.012), 99, 100)["decision"].startswith("stop")
    assert g2_report(bf16, _drift(0.1325, 0.012), 101, 100)["decision"].startswith("stop")
    panel = {
        "MATH-500": {"blaze_q_vs_bf16": {"delta": -0.4}},
        "IFEval": {"blaze_q_vs_bf16": {"delta": 0.5}},
    }
    assert g3_report(panel)["checks"] == {"panel": True}
    panel["MATH-500"]["blaze_q_vs_bf16"]["delta"] = -1.4
    assert g3_report(panel)["decision"].startswith("stop")
    fake, real = _drift(0.1285, 0.007), _drift(0.1286, 0.007)
    good = real_report(bf16, fake, real, {"passed": True}, {})
    assert good["checks"] == {"serves_like_fake": True, "same_drift": True}
    assert real_report(bf16, fake, _drift(0.131, 0.01), {"passed": True}, {})[
        "decision"
    ].startswith("stop")


def test_g2_and_real2b_plans(tmp_path: Path) -> None:
    g2 = _load_script("run_blaze_g2")
    full = g2.plan(tmp_path / "g2", tmp_path / "exports", "SNAPSHOT", None, panel=True)
    assert list(full) == [
        "prices",
        "market",
        "export",
        "score:bf16",
        "score:blaze_q_g2",
        "panel:blaze_q_g2",
    ]
    assert full["export"][full["export"].index("--plan") + 1] == str(tmp_path / "g2" / "plan.json")
    pilot = g2.plan(tmp_path / "g2-pilot", tmp_path / "exports", "SNAPSHOT", 10, panel=True)
    assert "panel:blaze_q_g2" not in pilot and pilot["market"][-1] == "--allow-over"
    assert pilot["export"][-2:] == ["--layers", "0,1"]
    real = _load_script("run_blaze_real2b")
    exports = real.exports_of(tmp_path / "outputs")
    steps_ = real.plan(Path("configs/blaze_2b.yaml"), tmp_path / "r", exports, "S", tmp_path / "t")
    order = list(steps_)
    assert order.index("check") < order.index("export:gptq_w4a16_g128") < order.index("score:real")
    assert steps_["export:real"][-1] == str(exports["real"])
    assert steps_["score:real"][steps_["score:real"].index("--traces") + 1] == str(tmp_path / "t")


def test_best_fill_matches_brute_force() -> None:
    from qwen35_compression.blaze.market import best_fill

    rng = random.Random(1)
    for _ in range(40):
        names = [f"u{i}" for i in range(rng.randint(1, 7))]
        gains = {n: rng.random() for n in names}
        costs = {n: rng.randint(1, 9) * 4096 for n in names}
        budget = rng.randint(0, sum(costs.values()))
        chosen = best_fill(gains, costs, budget, bin_bytes=4096)
        assert sum(costs[n] for n in chosen) <= budget
        best = max(
            sum(gains[n] for n in combo)
            for r in range(len(names) + 1)
            for combo in itertools.combinations(names, r)
            if sum(costs[n] for n in combo) <= budget
        )
        assert sum(gains[n] for n in chosen) == pytest.approx(best)


def test_vision_tower_at_8_bits_in_exports_and_sizes(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    tensors = _checkpoint(tmp_path / "bf16")
    vision_key = "model.visual.blocks.0.mlp.linear_fc1.weight"
    record = write_fake_export(
        tmp_path / "bf16", tmp_path / "out", None, Option(6, 128), vision=Option(8, 128), **QUIET
    )
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert torch.equal(out[vision_key], fake_quantize(tensors[vision_key], Option(8, 128)))
    assert record["vision"] == {"option": "w8g128", "linears": 1}
    plan = size_plan(tmp_path / "bf16", Option(6, 128), vision=Option(8, 128))
    assert plan["parts"]["vision"] == Option(8, 128).tensor_bytes(256, 128)
    assert size_plan(tmp_path / "bf16", Option(6, 128))["parts"]["vision"] == 256 * 128 * 2


def test_g2b_plan_fills_from_reused_prices(tmp_path: Path) -> None:
    driver = _load_script("run_blaze_g2b")
    prices = tmp_path / "prices.json"
    full = driver.plan(tmp_path / "g2b", tmp_path / "exports", "S", prices, None)
    market = full["market"]
    assert market[market.index("--mode") + 1] == "fill"
    assert market[market.index("--prices") + 1] == str(prices)
    assert full["export"][full["export"].index("--vision-bits") + 1] == "8"
    assert "score:bf16" not in full  # G2's BF16 score is reused
    pilot = driver.plan(tmp_path / "p", tmp_path / "exports", "S", prices, 10)
    assert pilot["export"][-2:] == ["--layers", "0,1"]
    assert pilot["score:blaze_q_g2b"][-2:] == ["--limit", "10"]
    steps_ = driver.panel_steps(tmp_path / "g2b", tmp_path / "e")
    assert steps_["mmmu:blaze_q_g2b"][0].endswith(".venv-gpu-vision/bin/python")
    assert "MMMU_DEV_VAL" in steps_["mmmu:blaze_q_g2b"]
    single = driver.single_copy_steps(tmp_path, tmp_path / "o2b", "S2", tmp_path / "a")
    assert (
        "--single-copy" in single["check2b:single"] and "--single-copy" in single["speed2b:single"]
    )
