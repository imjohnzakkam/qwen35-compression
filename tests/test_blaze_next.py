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
from qwen35_compression.glaze2.export import pack, quantization_config, unpack
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


def test_g4_plan_rebuilds_checks_then_pilots(tmp_path: Path) -> None:
    driver = _load_script("run_blaze_g4")
    steps_ = driver.plan(tmp_path / "g4", tmp_path / "export", "S", tmp_path / "plan.json", 10)
    assert list(steps_) == [
        "export",
        "score:rebuild",
        "pilot:text",
        "pilot:vision",
        "text",
        "vision",
    ]
    assert steps_["export"][steps_["export"].index("--plan") + 1] == str(tmp_path / "plan.json")
    assert steps_["pilot:text"][-2:] == ["--limit", "10"]
    assert steps_["pilot:vision"][-2:] == ["--limit", "10"]
    text = " ".join(steps_["text"])
    assert "--limit" not in text and "mmlu_pro,gsm8k,hellaswag,arc_challenge,wikitext" in text
    assert "minerva_math500" not in text and "MMMU_DEV_VAL" not in steps_["vision"]
    assert steps_["vision"][0].endswith(".venv-gpu-vision/bin/python")


def test_g4_report_needs_every_task_near_bf16() -> None:
    report = _load_script("blaze_g4_report")
    scores = {task: bf16 for task, (bf16, _) in report.RECORDED.items()}
    assert report.g4_report(scores)["checks"] == {"suite": True}
    scores["MMMU"] -= 1.9
    assert report.g4_report(scores)["checks"] == {"suite": True}
    scores["ARC-Challenge"] -= 1.2
    result = report.g4_report(scores)
    assert result["checks"] == {"suite": False}
    assert result["tasks"]["ARC-Challenge"]["within"] is False
    scores = {task: bf16 for task, (bf16, _) in report.RECORDED.items()}
    scores["WikiText-2"] += 5  # perplexity is reported, not gated
    assert report.g4_report(scores)["checks"] == {"suite": True}


def test_real_export_takes_gptq_codes_and_an_8_bit_vision_tower(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    from qwen35_compression.glaze2.export import rtn_quantized
    from qwen35_compression.glaze2.rounding import QuantizedLinear

    tensors = _checkpoint(tmp_path / "bf16")
    item = rtn_quantized(tensors[BODY] * 0.5, Option(4, 128))
    given = {BODY: QuantizedLinear(item.codes, item.scales, Option(4, 128))}
    write_real_export(
        tmp_path / "bf16",
        tmp_path / "real",
        PLAN,
        Option(4, 128),
        row_multiple=1,
        quantized=given,
        vision=Option(8, 128),
        **QUIET,
    )
    stored = load_file(str(tmp_path / "real" / "model.safetensors"))
    module = BODY.removesuffix(".weight")
    assert torch.equal(unpack(stored[f"{module}.weight_packed"], 4, (128, 256)), item.codes.int())
    vision = "model.visual.blocks.0.mlp.linear_fc1"
    assert f"{vision}.weight_packed" in stored and f"{vision}.weight" not in stored
    groups = json.loads((tmp_path / "real" / "config.json").read_text())["quantization_config"]
    assert any(vision in g["targets"] for g in groups["config_groups"].values())


def test_six_bit_planes_round_trip_and_match_the_reference() -> None:
    sys.path.insert(0, "plugins/blaze_vllm")
    from blaze_vllm import w6

    generator = torch.Generator().manual_seed(3)
    codes = torch.randint(-32, 32, (32, 256), generator=generator)
    lo, hi = w6.pack(codes)
    assert lo.shape == (32, 32) and hi.shape == (32, 16) and lo.dtype == torch.int32
    assert torch.equal(w6.unpack(lo, hi), codes.int())
    scale = (torch.rand(32, 2, generator=generator) * 0.01 + 1e-3).to(torch.bfloat16)
    x = torch.randn(3, 256, generator=generator)
    weight = (codes.float().reshape(32, 2, 128) * scale.float().unsqueeze(-1)).to(torch.bfloat16)
    assert torch.allclose(w6.matmul(x, lo, hi, scale), x @ weight.float().reshape(32, 256).T)
    assert w6.choose(1, 2560, 9216, 56)[2] > 1  # small outputs split the inputs
    assert w6.choose(256, 18432, 2560, 56)[2] == 1


def test_blaze_s_plan_builds_the_real_format_and_times_it(tmp_path: Path) -> None:
    driver = _load_script("run_blaze_s")
    full = driver.plan(tmp_path / "s", tmp_path / "e", tmp_path / "o", "S", None)
    assert list(full) == [
        "export",
        "score:blaze_s",
        "panel:blaze_s",
        "speed:bf16",
        "speed:gptq_w4a16_g128",
        "speed:blaze_s",
        "kernel",
    ]
    export = full["export"]
    assert "--real" in export and export[export.index("--bits") + 1] == "4"
    assert export[export.index("--vision-bits") + 1] == "8"
    pilot = driver.plan(tmp_path / "p", tmp_path / "e", tmp_path / "o", "S", 10)
    assert list(pilot) == ["export", "score:blaze_s"] and pilot["export"][-2:] == [
        "--layers",
        "0,1",
    ]


def _four_bit_export(tmp_path: Path) -> tuple[dict[str, torch.Tensor], Path]:
    """A tiny compressed-tensors export: the body at 4-bit g128, the table left BF16."""
    from qwen35_compression.glaze2.export import rtn_quantized, write_export

    tensors = _checkpoint(tmp_path / "bf16")
    module = BODY.removesuffix(".weight").removeprefix("model.language_model.")
    item = rtn_quantized(tensors[BODY], Option(4, 128))
    write_export(
        tmp_path / "bf16",
        tmp_path / "w4",
        {module: item},
        quantize_vision=False,
        log=lambda _: None,
    )
    return tensors, tmp_path / "w4"


def test_ladder_add_on_changes_only_the_table(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    from qwen35_compression.blaze.addon import ladder_export

    tensors, source = _four_bit_export(tmp_path)
    record = ladder_export(source, tmp_path / "out", PLAN, row_multiple=1, log=lambda _: None)
    before = load_file(str(source / "model.safetensors"))
    after = load_file(str(tmp_path / "out" / "model.safetensors"))
    for key, value in before.items():
        if key != TIED_TABLE:
            assert torch.equal(after[key], value), key
    assert TIED_TABLE not in after
    assert torch.equal(read_table(tmp_path / "out"), ladder_table(tensors[TIED_TABLE], PLAN))
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    groups = config["quantization_config"]["config_groups"]
    assert {"blaze_hot", "blaze_cold"} <= set(groups) and config[ARCHITECTURE_KEY]["hot_rows"] == 4
    assert record["upgraded"] == {}
    with pytest.raises(ValueError, match="already laddered"):
        ladder_export(tmp_path / "out", tmp_path / "again", PLAN, row_multiple=1)


def test_ladder_add_on_widens_planned_units_from_bf16(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    from qwen35_compression.blaze.addon import ladder_export

    tensors, source = _four_bit_export(tmp_path)
    module = BODY.removesuffix(".weight")
    ladder_export(
        source,
        tmp_path / "out",
        PLAN,
        bf16=tmp_path / "bf16",
        upgrade={module: Option(8, 128)},
        row_multiple=1,
        log=lambda _: None,
    )
    after = load_file(str(tmp_path / "out" / "model.safetensors"))
    codes = unpack(after[f"{module}.weight_packed"], 8, (128, 256))
    widened = dequantize(codes.float(), after[f"{module}.weight_scale"], Option(8, 128))
    assert torch.equal(widened, fake_quantize(tensors[BODY], Option(8, 128)))
    groups = json.loads((tmp_path / "out" / "config.json").read_text())["quantization_config"]
    assert groups["config_groups"]["blaze_upgrade_w8g128"]["targets"] == [module]
    with pytest.raises(ValueError, match="BF16 checkpoint"):
        ladder_export(
            source, tmp_path / "x", PLAN, upgrade={module: Option(8, 128)}, row_multiple=1
        )


def test_addon_plan_reuses_published_exports(tmp_path: Path) -> None:
    driver = _load_script("run_blaze_addon")
    full = driver.plan(tmp_path / "a", tmp_path / "o", "S", tmp_path / "prices.json", None)
    assert [n for n in full if n.startswith("addon:")] == [
        "addon:gptq",
        "addon:autoround",
        "addon:glaze",
    ]
    market = full["market"]
    assert market[market.index("--lower") + 1] == "4" and market[market.index("--upper") + 1] == "8"
    assert "--vision-bits" not in market  # the vision tower stays as each source stores it
    build = full["build:blaze_q48"]
    assert build[build.index("--source") + 1] == str(tmp_path / "o" / "published" / "gptq")
    assert "--plan" in build and "--bf16" in build
    assert [n for n in full if n.startswith("score:")] == [
        "score:addon_gptq",
        "score:addon_autoround",
        "score:addon_glaze",
        "score:blaze_q48",
    ]
    pilot = driver.plan(tmp_path / "p", tmp_path / "o", "S", tmp_path / "prices.json", 10)
    assert list(pilot) == ["addon:gptq", "score:addon_gptq"]


def test_six_bit_groups_run_on_eight_bit_words_with_the_same_codes() -> None:
    sys.path.insert(0, "plugins/blaze_vllm")
    from blaze_vllm import widen
    from compressed_tensors.quantization import QuantizationArgs

    generator = torch.Generator().manual_seed(5)
    codes = torch.randint(-32, 32, (5, 384), generator=generator)
    six = pack(codes, 6)
    assert six.shape == (5, 72)  # dense: exactly 6 bits a weight
    assert torch.equal(widen.widen_packed(six), pack(codes, 8))
    with pytest.raises(ValueError):
        widen.widen_packed(six[:, :70])

    config = quantization_config({Option(6, 128): ["a.q_proj", "a.k_proj"], Option(8, 128): ["b"]})
    assert widen.six_bit_modules(config) == {"a.q_proj", "a.k_proj"}
    with pytest.raises(ValueError):
        group = {"targets": ["Linear"], "weights": {"num_bits": 6}}
        widen.six_bit_modules({"config_groups": {"g": group}})

    def args(bits: int) -> QuantizationArgs:
        return QuantizationArgs(num_bits=bits, group_size=128, strategy="group", symmetric=True)

    schemes = types.SimpleNamespace(
        target_scheme_map={"a.q_proj": {"weights": args(6)}, "b": {"weights": args(8)}}
    )
    assert widen.widen_schemes(schemes) == 1
    assert [s["weights"].num_bits for s in schemes.target_scheme_map.values()] == [8, 8]

    weights = [
        ("a.q_proj.weight_packed", six),
        ("a.q_proj.weight_scale", torch.ones(5, 3)),
        ("b.weight_packed", pack(codes, 8)),
    ]
    out = dict(widen.widened(iter(weights), {"a.q_proj"}))
    assert torch.equal(out["a.q_proj.weight_packed"], pack(codes, 8))
    assert out["b.weight_packed"] is weights[2][1]


def test_q6_plan_serves_both_ways_and_scores_domains(tmp_path: Path) -> None:
    driver = _load_script("run_blaze_q6")
    plan_file = tmp_path / "plan.json"
    full = driver.plan(tmp_path / "q", tmp_path / "e", tmp_path / "o", "S", plan_file, None)
    assert list(full) == [
        "export",
        "score:native",
        "speed:native",
        "score:w8",
        "speed:w8",
        "domain_answers",
        "domain:bf16",
        "build:ladder",
        "domain:ladder",
        "build:ladder_ml",
        "domain:ladder_ml",
        "score:ladder_ml",
        "domain:gptq_w4a16_g128",
        "domain:blaze_q6",
    ]
    export = full["export"]
    assert export[export.index("--bits") + 1] == "6" and "--real" in export
    assert export[export.index("--plan") + 1] == str(plan_file)
    native = full["score:native"]
    assert native[0] == "env" and native[1].startswith("LD_LIBRARY_PATH=") and "cu13" in native[1]
    assert full["score:w8"][:2] == ["env", "BLAZE_W6_AS_W8=1"]
    assert full["domain:blaze_q6"][:2] == ["env", "BLAZE_W6_AS_W8=1"]
    assert all(full[k][-1] == "--domains" for k in full if k.startswith("domain:"))
    census = full["build:ladder_ml"]
    assert census[census.index("--census") + 1] == "data/blaze/census_multilingual.json"
    assert census[census.index("--cover") + 1] == "1e-9"  # 65,536 rows, as the English ladder
    assert "--cover" not in full["build:ladder"]
    pilot = driver.plan(tmp_path / "p", tmp_path / "e", tmp_path / "o", "S", plan_file, 10)
    assert list(pilot) == ["export", "score:native", "score:w8", "domain_answers", "domain:bf16"]
    assert pilot["export"][-2:] == ["--layers", "0,1"]

    scores = _load_script("drift_scores")
    assert scores.groups("domain_ml_zh") == ("ml", "ml_zh", "both")
    assert scores.groups("minerva_math500") == ("math500", "both")


def test_ladder_add_on_can_store_the_vision_tower_at_8_bits(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    from qwen35_compression.blaze.addon import ladder_export
    from qwen35_compression.glaze2.export import rtn_quantized

    tensors, source = _four_bit_export(tmp_path)
    vision = "model.visual.blocks.0.mlp.linear_fc1"
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"]["ignore"] = ["lm_head", vision]  # as published exports list it
    (source / "config.json").write_text(json.dumps(config))
    record = ladder_export(
        source, tmp_path / "out", PLAN, row_multiple=1, log=lambda _: None, vision=Option(8, 128)
    )
    after = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert f"{vision}.weight" not in after
    expected = rtn_quantized(tensors[f"{vision}.weight"], Option(8, 128))
    assert torch.equal(
        unpack(after[f"{vision}.weight_packed"], 8, (256, 128)), expected.codes.int()
    )
    quant = json.loads((tmp_path / "out" / "config.json").read_text())["quantization_config"]
    assert quant["config_groups"]["blaze_vision_w8g128"]["targets"] == [vision]
    assert quant["config_groups"]["blaze_vision_w8g128"]["weights"]["num_bits"] == 8
    assert quant["ignore"] == ["lm_head"]
    assert record["vision"] == {"option": "w8g128", "modules": 1}


def test_corrected_blaze_s_plan_reuses_the_autoround_body(tmp_path: Path) -> None:
    driver = _load_script("run_blaze_s2")
    full = driver.plan(tmp_path / "s2", tmp_path / "o", "S", None, mmmu=False)
    assert list(full) == ["export", "score:blaze_s2", "speed:blaze_s2"]
    export = full["export"]
    assert export[export.index("--source") + 1] == str(tmp_path / "o" / "autoround_w4a16_g128")
    assert export[export.index("--vision-bits") + 1] == "8"
    assert export[export.index("--cover") + 1] == "1e-9"
    assert export[export.index("--census") + 1] == "data/blaze/census_multilingual.json"
    with_mmmu = driver.plan(tmp_path / "s2", tmp_path / "o", "S", None, mmmu=True)
    assert list(with_mmmu)[-4:] == [
        "bootstrap_vision",
        "plugin_vision",
        "preflight_vision",
        "mmmu:blaze_s2",
    ]
    pilot = driver.plan(tmp_path / "p", tmp_path / "o", "S", 10, mmmu=True)
    assert list(pilot) == ["export", "score:blaze_s2"]
