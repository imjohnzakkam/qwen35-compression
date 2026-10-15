"""Blaze: the usage census, the vocabulary ladder, fake-quantized exports, unit prices, gate G1."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from glaze_tiny import WIDE, random_blocks, tiny_teacher

from qwen35_compression.blaze.body import ffn_layer_units, is_body_linear, swapped
from qwen35_compression.blaze.census import Census, coverage, load_census, save_census
from qwen35_compression.blaze.export import HEAD, MANIFEST, TIED_TABLE, write_fake_export
from qwen35_compression.blaze.gates import cost, excess_loss, g1_report
from qwen35_compression.blaze.ladder import (
    LadderPlan,
    ladder_table,
    plan_ladder,
    renumber,
    restore,
)
from qwen35_compression.blaze.sensitivity import concentration, measure_units, pack_text
from qwen35_compression.glaze2.quant import Option, fake_quantize

QUIET = {"log": lambda _: None}


def _census() -> Census:
    census = Census(8)
    census.add("text", [[0, 0, 0, 3, 3, 5], [3, 0]])
    census.add("answers", [[5, 5, 1]])
    return census


def test_census_counts_ranks_and_ignores_out_of_range_ids() -> None:
    census = _census()
    assert census.total().tolist() == [4, 1, 0, 3, 0, 3, 0, 0]
    assert census.tokens() == 11 and census.used_rows() == 4
    # Rows 3 and 5 tie at 3 tokens; the lower id ranks first.
    assert census.ranking().tolist()[:4] == [0, 3, 5, 1]
    assert census.add("text", [[8, -1, 2]]) == 1


def test_census_round_trips_and_refuses_an_edited_file(tmp_path: Path) -> None:
    census = _census()
    path = tmp_path / "census.json"
    lock = save_census(census, path, {"sources": {}})
    assert lock["used_rows"] == 4 and lock["tokens"] == {"answers": 3, "text": 8}
    loaded = load_census(path)
    assert all((loaded.counts[name] == census.counts[name]).all() for name in census.counts)
    path.write_text(path.read_text().replace('"3": 3', '"3": 2'))
    with pytest.raises(ValueError, match="lock"):
        load_census(path)


def test_coverage_counts_tokens_inside_the_busiest_rows() -> None:
    census = _census()
    order = census.ranking()
    assert coverage(order, census.total(), 1) == pytest.approx(4 / 11)
    assert coverage(order, census.counts["answers"], 3) == pytest.approx(2 / 3)
    assert coverage(order, census.counts["answers"], 4) == pytest.approx(1.0)


def test_plan_ladder_covers_the_census_and_respects_the_floor() -> None:
    census = _census()
    plan = plan_ladder(census, cover=0.5, at_least=1)
    assert plan.hot_rows == 2 and set(plan.order[:2]) == {0, 3}
    assert plan_ladder(census, cover=0.5, at_least=5).hot_rows == 5
    assert plan_ladder(census, cover=1.0, at_least=1).hot_rows == 4
    assert plan_ladder(census, cover=1.0, at_least=100).hot_rows == 8
    with pytest.raises(ValueError):
        plan_ladder(census, cover=0.0)


def test_ladder_plan_checks_its_order() -> None:
    with pytest.raises(ValueError):
        LadderPlan((0, 0, 1), 1)
    with pytest.raises(ValueError):
        LadderPlan((0, 1), 3)


def test_ladder_bytes_for_the_4b_table() -> None:
    plan = LadderPlan(tuple(range(248_320)), 65_536)
    expected = (
        Option(8, 128).tensor_bytes(65_536, 2560)
        + Option(4, 128).tensor_bytes(182_784, 2560)
        + 4 * 248_320
    )
    assert plan.table_bytes(2560) == expected == 412_661_760
    assert plan.record(2560)["bf16_bytes"] == 1_271_398_400


def test_ladder_table_quantizes_each_row_at_its_rung() -> None:
    torch.manual_seed(0)
    weight = torch.randn(6, 256, dtype=torch.bfloat16)
    plan = LadderPlan((4, 1, 0, 2, 3, 5), 2)
    table = ladder_table(weight, plan, chunk_rows=1)
    for row in range(6):
        option = Option(8, 128) if row in (4, 1) else Option(4, 128)
        assert torch.equal(table[row], fake_quantize(weight[row : row + 1], option)[0])
    with pytest.raises(ValueError):
        ladder_table(weight[:5], plan)


def test_renumbering_round_trips() -> None:
    weight = torch.arange(12.0).reshape(6, 2)
    plan = LadderPlan((4, 1, 0, 2, 3, 5), 2)
    stored, index = renumber(weight, plan)
    assert stored[0].tolist() == weight[4].tolist() and stored[1].tolist() == weight[1].tolist()
    assert index.dtype == torch.int32 and int(index[4]) == 0 and int(index[1]) == 1
    assert torch.equal(restore(stored, index), weight)


def test_body_linears_are_decoder_layer_matrices_only() -> None:
    assert is_body_linear("model.language_model.layers.3.mlp.down_proj.weight", (2560, 9216))
    assert is_body_linear("model.language_model.layers.0.linear_attn.in_proj_b.weight", (32, 2560))
    for name, shape in (
        ("model.language_model.layers.3.input_layernorm.weight", (2560,)),
        ("model.language_model.layers.0.linear_attn.conv1d.weight", (8192, 1, 4)),
        ("model.language_model.embed_tokens.weight", (248_320, 2560)),
        ("mtp.layers.0.mlp.down_proj.weight", (2560, 9216)),
        ("model.visual.blocks.0.mlp.linear_fc1.weight", (4096, 1024)),
    ):
        assert not is_body_linear(name, shape)


def test_ffn_units_and_swaps_restore_the_weights() -> None:
    teacher = tiny_teacher(**WIDE)
    units = ffn_layer_units(teacher)
    assert list(units) == [f"layers.{index}.mlp" for index in range(4)]
    assert sorted(units["layers.0.mlp"]) == [
        "layers.0.mlp.down_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
    ]
    before = teacher.get_submodule("layers.1.mlp.up_proj").weight.clone()
    with swapped(teacher, units["layers.1.mlp"], Option(4, 128)):
        inside = teacher.get_submodule("layers.1.mlp.up_proj").weight.clone()
    assert torch.equal(inside, fake_quantize(before, Option(4, 128)))
    assert torch.equal(teacher.get_submodule("layers.1.mlp.up_proj").weight, before)


def test_pack_text_cuts_fixed_blocks() -> None:
    assert pack_text([[1, 2, 3], [4, 5], [6, 7, 8, 9]], 4, 2) == [[1, 2, 3, 4], [5, 6, 7, 8]]
    with pytest.raises(ValueError):
        pack_text([[1, 2]], 4, 1)


def test_measure_units_prices_each_layer_alone() -> None:
    teacher = tiny_teacher(**WIDE)
    head = teacher.embed_tokens.weight
    units = ffn_layer_units(teacher)
    prices = measure_units(
        teacher, head, random_blocks(2, length=16), units, Option(4, 128), 1, 8, **QUIET
    )
    assert list(prices) == list(units)
    assert all(price["mean_kl"] > 0 and price["tokens"] == 32 for price in prices.values())
    fresh = tiny_teacher(**WIDE)
    for (_, used), (_, original) in zip(
        teacher.named_parameters(), fresh.named_parameters(), strict=True
    ):
        assert torch.equal(used, original)


def test_concentration_by_price_per_weight() -> None:
    assert concentration({"a": 1.0, "b": 1.0}, {"a": 10, "b": 10}) == pytest.approx(0.5)
    assert concentration({"a": 3.0, "b": 1.0}, {"a": 10, "b": 10}) == pytest.approx(0.75)
    # a (0.2 per weight) whole, then two-thirds of b (0.1 per weight): (2 + 2) / 6.
    shares = concentration({"a": 2.0, "b": 3.0, "c": 1.0}, {"a": 10, "b": 30, "c": 20})
    assert shares == pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        concentration({"a": 1.0}, {"b": 1})


def _checkpoint(directory: Path, tied: bool = True) -> dict[str, torch.Tensor]:
    from safetensors.torch import save_file

    torch.manual_seed(0)
    tensors = {
        TIED_TABLE: torch.randn(16, 128, dtype=torch.bfloat16),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(
            128, 256, dtype=torch.bfloat16
        ),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(
            128, dtype=torch.bfloat16
        ),
        "model.visual.blocks.0.mlp.linear_fc1.weight": torch.randn(256, 128, dtype=torch.bfloat16),
        "mtp.fc.weight": torch.randn(128, 256, dtype=torch.bfloat16),
    }
    if not tied:
        tensors[HEAD] = torch.randn(16, 128, dtype=torch.bfloat16)
    directory.mkdir()
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    return tensors


BODY = "model.language_model.layers.0.mlp.down_proj.weight"
UNTOUCHED = (
    "model.language_model.layers.0.input_layernorm.weight",
    "model.visual.blocks.0.mlp.linear_fc1.weight",
    "mtp.fc.weight",
)


def test_fake_export_ladders_a_tied_table_and_the_body(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    tensors = _checkpoint(tmp_path / "bf16")
    plan = LadderPlan(tuple(range(16)), 4)
    record = write_fake_export(tmp_path / "bf16", tmp_path / "out", plan, Option(6, 128), **QUIET)
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert torch.equal(out[TIED_TABLE], ladder_table(tensors[TIED_TABLE], plan))
    assert torch.equal(out[BODY], fake_quantize(tensors[BODY], Option(6, 128)))
    assert all(torch.equal(out[key], tensors[key]) for key in UNTOUCHED)
    assert (tmp_path / "out" / "tokenizer.json").exists()
    assert record["tied_table"] and record["input_table"] is None
    assert record["body"] == {"option": "w6g128", "linears": 1, "stored_bytes": 25_088}
    assert record["ladder"]["table_bytes"] == plan.table_bytes(128)
    assert json.loads((tmp_path / "out" / MANIFEST).read_text())["fake_quantized"]
    with pytest.raises(FileExistsError):
        write_fake_export(tmp_path / "bf16", tmp_path / "out", plan, None, **QUIET)


def test_fake_export_ladders_an_untied_head(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    tensors = _checkpoint(tmp_path / "bf16", tied=False)
    plan = LadderPlan(tuple(range(16)), 4)
    record = write_fake_export(tmp_path / "bf16", tmp_path / "out", plan, None, **QUIET)
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert torch.equal(out[HEAD], ladder_table(tensors[HEAD], plan))
    assert torch.equal(out[TIED_TABLE], fake_quantize(tensors[TIED_TABLE], Option(4, 128)))
    assert torch.equal(out[BODY], tensors[BODY])
    assert record["input_table"]["option"] == "w4g128" and record["body"] is None


def _drift(nll: float, flips: float, tokens: int = 1000) -> dict:
    scores = {"answers": 10, "tokens": tokens, "flip_rate": flips, "mean_nll": nll}
    return {"scores": {"both": {"all": scores}}}


def test_g1_report_checks_and_decisions() -> None:
    bf16 = _drift(0.128, 0.0035)
    ladder, six, four = _drift(0.1285, 0.004), _drift(0.131, 0.012), _drift(0.168, 0.05)
    report = g1_report(bf16, ladder, six, four, 0.8)
    assert report["checks"] == {"ladder_free": True, "six_bit_pays": True, "menu_enough": True}
    assert report["ladder"]["excess_loss"] == pytest.approx(0.0005)
    assert report["bits_rule"]["measured_factor"] == pytest.approx(0.04 / 0.003)
    assert report["decision"].startswith("go: G2 on the 4/8-bit menu")
    assert g1_report(bf16, ladder, six, four, 0.6)["decision"].startswith("go: G2 needs the 6-bit")
    worse = _drift(0.140, 0.02)
    assert g1_report(bf16, ladder, worse, None, 0.6)["decision"].startswith("stop on the 4B")
    assert g1_report(bf16, _drift(0.130, 0.004), six, None, 0.8)["decision"].startswith("stop:")
    with pytest.raises(ValueError, match="different tokens"):
        excess_loss(_drift(0.13, 0.01, tokens=999), bf16)


def test_cost_in_rupees_and_dollars() -> None:
    assert cost(1.5) == {"hours": 1.5, "rupees": 58.32, "dollars": 0.61}


def _load_driver():
    sys.path.insert(0, "scripts")
    spec = importlib.util.spec_from_file_location("run_blaze_g1", "scripts/run_blaze_g1.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_blaze_g1"] = module
    spec.loader.exec_module(module)
    return module


def test_g1_plan_builds_three_exports_four_scores_and_prices(tmp_path: Path) -> None:
    driver = _load_driver()
    full = driver.plan(tmp_path / "g1", tmp_path / "exports", "SNAPSHOT", None)
    assert [name for name in full if name.startswith("export:")] == [
        "export:blaze_ladder",
        "export:blaze_ladder_w6",
        "export:blaze_ladder_w4",
    ]
    assert [name for name in full if name.startswith("score:")] == [
        "score:bf16",
        "score:blaze_ladder",
        "score:blaze_ladder_w6",
        "score:blaze_ladder_w4",
    ]
    assert full["export:blaze_ladder_w6"][-3:] == ["--ladder", "--body-bits", "6"]
    assert "--limit" not in full["score:bf16"] and "--layers" not in full["prices"]
    pilot = driver.plan(tmp_path / "g1-pilot", tmp_path / "exports", "SNAPSHOT", 10)
    assert pilot["score:blaze_ladder"][-2:] == ["--limit", "10"]
    assert pilot["prices"][-2:] == ["--layers", "0,1"]
    assert pilot["export:blaze_ladder"] == full["export:blaze_ladder"]
