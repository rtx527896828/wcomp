"""Tests for the Phase 2 P0 parametric methods (wcomp.methods.resolve), the +keepcols / +keepsuper helpers in
wcomp.quant.rtn and the f32 Hessian accumulation of wcomp.quant.gptq. CPU only; the 0.8B is used on the meta device."""
import copy
import json
import math
import os

import pytest
import torch
from torch import nn

from wcomp.methods import METHODS, Ctx, family, needs_calib, parse, resolve, select_linears
from wcomp.models import block_linears, load_model, weight_type
import wcomp.quant.gptq as gptq_mod
from wcomp.quant.gptq import add_xtx, hessian_path_check
from wcomp.quant.rtn import (
    is_super_candidate_kind,
    keepcols_mask,
    load_super_candidates,
    quantize_rtn,
    quantize_rtn_keepcols,
    super_candidate_kinds,
    super_weight_bits,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen3.5-0.8B")


def _stub_model() -> nn.Module:
    """Two layers with self_attn.q_proj [64, 256], mlp.up_proj [128, 256], mlp.down_proj [64, 384] (K % 128 == 0)."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(256, 64, bias=False)
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(256, 128, bias=False)
            self.mlp.down_proj = nn.Linear(384, 64, bias=False)

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(), Block()])

    torch.manual_seed(0)
    return Stub().to(torch.bfloat16)


def _weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: lin.weight.detach().clone() for n, lin in block_linears(model)}


# ---------------------------------------------------------------- resolve / parse / family / needs_calib


def test_resolve_round_trips_every_phase1_name():
    for name, method in METHODS.items():
        assert resolve(name) is method
        # Phase 1 behaviour of the helpers is unchanged on Phase 1 names
        assert needs_calib(name) == ("gptq" in name)
        head, _, tail = name.rpartition("-")
        assert family(name) == ((head, int(tail)) if tail.isdigit() and head else (name, None))


@pytest.mark.parametrize("bad", ["foo", "rtn-sym-g128", "rtn-sym-g128-4@only=bogus", "rtn-sym-g128-3@layer=x",
                                 "rtn-sym-g128-4+keepcols=0", "rtn-sym-g128-4+keepcols=1@only=down", "gptq-sym-g128-4@only=down",
                                 "rtn-sym-g96-4", "rtn-sym-g0-4"])
def test_resolve_rejects_malformed_names(bad):
    with pytest.raises(ValueError, match="unknown method"):
        resolve(bad)


def test_parse_parametric_forms():
    s = parse("rtn-sym-g32-4")
    assert (s.bits, s.group, s.sym, s.select, s.keepcols, s.keepsuper) == (4, 32, True, None, None, None)
    assert parse("rtn-sym-g128-4@only=down").select == ("only", "down")
    assert parse("rtn-sym-g128-4@except=in_proj_ab").select == ("except", "in_proj_ab")
    assert parse("rtn-sym-g128-3@layer=5").select == ("layer", "5")
    assert parse("rtn-sym-g128-4+keepcols=0.5").keepcols == 0.5
    s = parse("rtn-sym-g128-3@except=down+keepcols=2")
    assert s.select == ("except", "down") and s.keepcols == 2.0 and s.bits == 3
    s = parse(r"rtn-asym-g128-4+keepsuper=results\probe\qwen3.5-0.8b\superweight.json")
    assert not s.sym and s.keepsuper.endswith("superweight.json")
    for g in (32, 64, 128):
        assert parse(f"rtn-sym-g{g}-3").group == g
    with pytest.raises(ValueError, match="group one of"):  # DESIGN.md P0 names g32 / g64 / g128 only; K=1024 % 96 != 0 would fail late
        parse("rtn-sym-g96-4")


def test_family_and_needs_calib_on_parametric_names():
    assert family("rtn-sym-g32-4") == ("rtn-sym-g32", 4)
    assert family("rtn-sym-g64-3") == ("rtn-sym-g64", 3)
    assert family("rtn-sym-g128-4@only=down") == ("rtn-sym-g128@only=down", 4)
    assert family("rtn-sym-g128-3@layer=17") == ("rtn-sym-g128@layer=17", 3)
    assert family("rtn-sym-g128-4+keepcols=1") == ("rtn-sym-g128+keepcols=1", 4)
    assert family("rtn-sym-g128-3+keepsuper=results/probe/qwen3.5-0.8b/superweight.json") == ("rtn-sym-g128+keepsuper=superweight", 3)
    for n in ("rtn-sym-g32-4", "rtn-sym-g128-4@only=down", "rtn-sym-g128-4+keepsuper=some/gptq_dir/superweight.json"):
        assert not needs_calib(n)


# ---------------------------------------------------------------- selection and the honest bits mix


@pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="needs the 0.8B config")
def test_only_down_selects_the_24_down_proj_of_the_0_8b():
    model, _ = load_model("qwen3.5-0.8b", device="meta")
    linears = block_linears(model)
    selected = select_linears(linears, parse("rtn-sym-g128-4@only=down"))
    assert len(selected) == 24
    assert all(n.endswith(".mlp.down_proj") and weight_type(n) == "down" for n, _ in selected)
    assert sorted({n.split(".")[2] for n, _ in selected}, key=int) == [str(i) for i in range(24)]
    n_down = sum(m.weight.numel() for _, m in selected)
    n_all = sum(m.weight.numel() for _, m in linears)
    expected = (n_down * 4.125 + (n_all - n_down) * 16) / n_all
    print(f"\n0.8B @only=down: {n_down} of {n_all} block params at 4.125 bits -> bits_blocks {expected:.6f}")
    assert 4.125 < expected < 16
    assert len(select_linears(linears, parse("rtn-sym-g128-4@except=down"))) == len(linears) - 24
    layer5 = select_linears(linears, parse("rtn-sym-g128-3@layer=5"))
    assert [n.split(".")[2] for n, _ in layer5] == ["5"] * len(layer5) and len(layer5) == 8  # DeltaNet layer: 5 + 3


def test_only_bits_blocks_is_the_hand_mix_and_other_tensors_are_untouched():
    model = _stub_model()
    before = _weights(model)
    acct = resolve("rtn-sym-g128-4@only=down")(model, Ctx(None, 0))
    after = _weights(model)
    names = [n for n, _ in block_linears(model)]
    assert [t.name for t in acct.tensors] == names  # block order, quantized and 16-bit tensors mixed
    n_down = sum(before[n].numel() for n in names if n.endswith("down_proj"))
    n_all = sum(v.numel() for v in before.values())
    assert math.isclose(acct.bits_blocks(), (n_down * 4.125 + (n_all - n_down) * 16) / n_all, rel_tol=1e-12)
    for n in names:
        if n.endswith("down_proj"):
            assert not torch.equal(before[n], after[n])
            W_hat, _ = quantize_rtn(before[n].float(), 4, 128, True)
            assert torch.equal(after[n], W_hat.to(torch.bfloat16))  # the same grid as the Phase 1 method
        else:
            assert torch.equal(before[n], after[n])
    assert acct.summary()["notes"] == ["bf16, not quantized", "rtn-sym-g128-4: f16 scale"]
    assert acct.bits_all(model) == acct.bits_blocks()  # the stub has no other parameters


def test_except_and_layer_selection():
    model = _stub_model()
    before = _weights(model)
    acct = resolve("rtn-sym-g128-3@except=down")(model, Ctx(None, 0))
    after = _weights(model)
    for n in before:
        assert torch.equal(before[n], after[n]) == n.endswith("down_proj")
    quantized = [t for t in acct.tensors if "not quantized" not in t.note]
    assert len(quantized) == 4 and all(t.index_bits == 3 * t.numel for t in quantized)

    model = _stub_model()
    before = _weights(model)
    resolve("rtn-sym-g128-3@layer=1")(model, Ctx(None, 0))
    after = _weights(model)
    for n in before:
        assert torch.equal(before[n], after[n]) == (".layers.0." in n)

    with pytest.raises(ValueError, match="selects no block linear"):
        resolve("rtn-sym-g128-4@only=in_proj_qkv")(_stub_model(), Ctx(None, 0))


@pytest.mark.parametrize("name,expected", [("rtn-sym-g32-4", 4.5), ("rtn-sym-g64-4", 4.25), ("rtn-sym-g128-4", 4.125),
                                           ("rtn-sym-g32-3", 3.5), ("rtn-sym-g64-3", 3.25)])
def test_group_size_bits(name, expected):
    acct = resolve(name)(_stub_model(), Ctx(None, 0))
    assert math.isclose(acct.bits_blocks(), expected, rel_tol=1e-12)


# ---------------------------------------------------------------- +keepcols


def test_keepcols_keeps_columns_exactly_and_lowers_the_mse_of_the_rest():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(64, 256, generator=g)
    W[:, 7] *= 50.0  # one outlier input column in the first 128-column group
    mask = keepcols_mask(W, 0.5)  # round(256 * 0.005) = 1 -> at least one column
    assert mask.dtype == torch.bool and mask.shape == (256,) and mask.sum() == 1 and mask[7]
    W_hat, info = quantize_rtn_keepcols(W, 4, 128, mask)
    W_rtn, info_rtn = quantize_rtn(W, 4, 128, True)
    assert torch.equal(W_hat[:, 7], W[:, 7])
    rest = ~mask
    mse_keep = ((W_hat - W)[:, rest] ** 2).mean().item()
    mse_rtn = ((W_rtn - W)[:, rest] ** 2).mean().item()
    print(f"\nkeepcols: MSE of the other columns {mse_keep:.4g} vs plain RTN {mse_rtn:.4g} ({mse_rtn / mse_keep:.1f}x)")
    assert mse_keep < 0.1 * mse_rtn
    assert torch.equal(W_hat[:, 128:], W_rtn[:, 128:])  # the second group has no kept column: identical grid
    # bits: 4 per other weight, 16 per kept weight, an f16 scale for every group, a K-bit mask
    assert info["index_bits"] == 4 * 64 * 255
    assert info["aux_bits"] == 16 * 64 * 1 + 64 * 2 * 16 + 256
    assert info_rtn["index_bits"] == 4 * 64 * 256 and info_rtn["aux_bits"] == 64 * 2 * 16


def test_keepcols_mask_count_and_errors():
    W = torch.randn(8, 1024)
    assert keepcols_mask(W, 0.5).sum() == 5 and keepcols_mask(W, 1).sum() == 10 and keepcols_mask(W, 2).sum() == 20
    assert keepcols_mask(torch.randn(8, 16), 1).sum() == 1
    with pytest.raises(ValueError):
        keepcols_mask(W, 0)
    with pytest.raises(ValueError):
        quantize_rtn_keepcols(W, 4, 128, torch.zeros(8, dtype=torch.bool))


def test_keepcols_method_bits_formula():
    model = _stub_model()
    acct = resolve("rtn-sym-g128-4+keepcols=1")(model, Ctx(None, 0))
    for t, (n, lin) in zip(acct.tensors, block_linears(model)):
        N, K = lin.weight.shape
        n_keep = max(1, round(K / 100))  # 256 -> 3, 384 -> 4
        assert t.name == n
        assert t.index_bits == 4 * N * (K - n_keep)
        assert t.aux_bits == 16 * N * n_keep + N * (K // 128) * 16 + K
        assert "keepcols 1%" in t.note
    assert acct.bits_blocks() == acct.total_bits() / acct.numel()


# ---------------------------------------------------------------- +keepsuper


def test_keepsuper_restores_the_listed_entries(tmp_path):
    model = _stub_model()
    before = _weights(model)
    plain = copy.deepcopy(model)
    acct_plain = resolve("rtn-sym-g128-4")(plain, Ctx(None, 0))
    listed = [("model.layers.0.mlp.down_proj", 3, 100), ("model.layers.0.mlp.down_proj", 60, 383),
              ("model.layers.1.self_attn.q_proj", 5, 7)]
    path = tmp_path / "superweight.json"
    path.write_text(json.dumps({"model": "stub", "probe": "P7", "candidates": [
        {"tensor": t + ".weight", "row": r, "col": c, "value": before[t][r, c].item(), "dPPL": 1.0, "KL": 0.1}
        for t, r, c in listed
    ] + [{"tensor": listed[0][0], "row": 3, "col": 100}]}), encoding="utf-8")  # duplicate without .weight
    assert load_super_candidates(str(path)) == listed

    name = f"rtn-sym-g128-4+keepsuper={path}"
    acct = resolve(name)(model, Ctx(None, 0))
    after = _weights(model)
    plain_w = _weights(plain)
    for t, r, c in listed:
        assert after[t][r, c] == before[t][r, c]
        plain_w[t][r, c] = before[t][r, c]
    for n in before:
        assert torch.equal(after[n], plain_w[n])  # everything else is plain RTN
    extra = {t.name: t.aux_bits - p.aux_bits for t, p in zip(acct.tensors, acct_plain.tensors)}
    assert extra["model.layers.0.mlp.down_proj"] == 2 * super_weight_bits(64, 384)
    assert extra["model.layers.1.self_attn.q_proj"] == 1 * super_weight_bits(64, 256)
    assert extra["model.layers.0.self_attn.q_proj"] == 0
    assert super_weight_bits(64, 384) == 16 + 15 and super_weight_bits(1024, 3584) == 16 + 22
    assert family(name) == ("rtn-sym-g128+keepsuper=superweight", 4)

    with pytest.raises(ValueError, match="not a block linear"):
        (tmp_path / "bad.json").write_text(json.dumps([{"tensor": "lm_head", "row": 0, "col": 0}]), encoding="utf-8")
        resolve(f"rtn-sym-g128-4+keepsuper={tmp_path / 'bad.json'}")(_stub_model(), Ctx(None, 0))


# ---------------------------------------------------------------- GPTQ Hessian: f32 window products into f64


def test_add_xtx_f32_matches_f64():
    g = torch.Generator().manual_seed(0)
    K = 512
    acc32 = torch.zeros(K, K, dtype=torch.float64)
    acc64 = torch.zeros(K, K, dtype=torch.float64)
    ref = torch.zeros(K, K, dtype=torch.float64)
    for _ in range(3):
        x = (torch.randn(2048, K, generator=g) * (1 + 20 * (torch.rand(K, generator=g) < 0.01))).to(torch.bfloat16)
        add_xtx(acc32, x, f32=True)
        add_xtx(acc64, x, f32=False)
        ref.addmm_(x.double().T, x.double())
    assert torch.equal(acc64, ref)  # the Phase 1 path, operation for operation
    rel = ((acc32 - acc64).norm() / acc64.norm()).item()
    rel_diag = ((acc32.diagonal() - acc64.diagonal()).abs() / acc64.diagonal()).max().item()
    print(f"\nf32 vs f64 Hessian: relative Frobenius {rel:.2e}, max relative diagonal {rel_diag:.2e}")
    assert rel < 1e-4 and rel_diag < 1e-4


def test_hessian_f64_is_the_default_until_acceptance():
    """Phase 1 names must reproduce their Phase 1 numbers by default: the f32 path is opt-in (runner flag / env)."""
    assert gptq_mod.HESSIAN_F32 is False or os.environ.get("WCOMP_HESSIAN_F32") == "1"
    acc = torch.zeros(4, 4, dtype=torch.float64)
    x = torch.randn(8, 4, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    add_xtx(acc, x)  # the module default
    ref = torch.zeros(4, 4, dtype=torch.float64).addmm_(x.double().T, x.double())
    assert torch.equal(acc, ref) == (not gptq_mod.HESSIAN_F32)


def test_hessian_note_marks_only_the_f32_path():
    """The default (Phase 1) path leaves the GPTQ tensor note byte-identical to Phase 1; only the f32 path is tagged."""
    assert gptq_mod.hessian_note("gptq 3b sym g128 block128 damp0.01", f32=False) == "gptq 3b sym g128 block128 damp0.01"
    assert gptq_mod.hessian_note("gptq 3b sym g128 block128 damp0.01", f32=True) == "gptq 3b sym g128 block128 damp0.01 hessian-f32"
    assert gptq_mod.hessian_note("x") == ("x hessian-f32" if gptq_mod.HESSIAN_F32 else "x")


def test_hessian_path_check_reports_chaos_not_precision():
    """The two Hessians agree to ~1e-7 while GPTQ may still round some weights differently at equal objective."""
    g = torch.Generator().manual_seed(1)
    K, N = 256, 64
    W = torch.randn(N, K, generator=g) * 0.02
    windows = [(torch.randn(512, K, generator=g) * (1 + 20 * (torch.rand(K, generator=g) < 0.02))).to(torch.bfloat16) for _ in range(3)]
    out = hessian_path_check(W, windows, 3, 128, True)
    assert set(out) >= {"rel_frob", "max_rel_diag", "frac_changed", "objective_ratio", "objective_f64", "n_rows"}
    assert out["n_rows"] == 3 * 512 and out["rel_frob"] < 1e-5 and out["max_rel_diag"] < 1e-5
    assert 0.0 <= out["frac_changed"] <= 0.05 and abs(out["objective_ratio"] - 1) < 0.01 and out["objective_f64"] > 0
    # identical Hessians (one window of exact f32-representable values: H32 == H64) give identical quantization
    exact = [torch.randint(-4, 5, (512, K), generator=g).to(torch.bfloat16)]
    same = hessian_path_check(W, exact, 3, 128, True)
    assert same["rel_frob"] == 0.0 and same["frac_changed"] == 0.0 and same["objective_ratio"] == pytest.approx(1.0)


def test_load_super_candidates_reads_the_probe_records_layout(tmp_path):
    """The P7 probe JSON (scripts/probe_acts.py) carries row / col under "stat"; its below-threshold fallback
    record is not a candidate."""
    doc = {"model": "qwen3.5-0.8b", "probe": "P7", "records": [
        {"tensor": "model.layers.23.mlp.down_proj", "layer": 23, "type": "down", "stat": {"row": 0, "col": 254},
         "control": {}, "gain_bits": None, "gain_bits_control": None, "note": "max_spike_below_threshold; zeroed"},
        {"tensor": "model.layers.2.mlp.down_proj", "layer": 2, "type": "down", "stat": {"row": 7, "col": 3000},
         "control": {}, "gain_bits": None, "gain_bits_control": None, "note": "spike+top16; zeroed"},
        {"tensor": "model.layers.20.linear_attn.in_proj_qkv", "layer": 20, "type": "in_proj_qkv", "stat": {"row": 1, "col": 2},
         "control": {}, "gain_bits": None, "gain_bits_control": None, "note": "top16; zeroed"},
        {"tensor": "model.layers.2.mlp.down_proj", "layer": 2, "type": "down", "stat": {"row": 7, "col": 3000},
         "control": {}, "gain_bits": None, "gain_bits_control": None, "note": "top16; duplicate"},
        {"tensor": "model.layers.5.mlp.down_proj", "layer": 5, "type": "down", "stat": {"row": 9, "col": 10},
         "control": {}, "gain_bits": None, "gain_bits_control": None, "note": "max_spike_below_threshold+top16; a top-16 weight"},
        {"tensor": "model.layers.6.mlp.down_proj", "layer": 6, "type": "down", "stat": {"row": 1, "col": 1},
         "control": {}, "gain_bits": None, "gain_bits_control": None, "note": "spike_no_input; output spike without input spike"},
    ]}
    path = tmp_path / "superweight.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert load_super_candidates(str(path)) == [("model.layers.2.mlp.down_proj", 7, 3000), ("model.layers.20.linear_attn.in_proj_qkv", 1, 2),
                                                ("model.layers.5.mlp.down_proj", 9, 10)]
    assert super_candidate_kinds(str(path)) == {"max_spike_below_threshold+top16": 1, "spike+top16": 1, "top16": 1}
    assert is_super_candidate_kind("spike") and is_super_candidate_kind("top16") and is_super_candidate_kind("")
    assert not is_super_candidate_kind("max_spike_below_threshold") and not is_super_candidate_kind("spike_no_input")
    (tmp_path / "bad.json").write_text(json.dumps({"model": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a P7 probe JSON"):
        load_super_candidates(str(tmp_path / "bad.json"))


def test_keepsuper_note_records_kinds_once(tmp_path):
    """The restored weights' P7 kinds are in the tensor note, which is size independent so the row carries it once."""
    model = _stub_model()
    path = tmp_path / "superweight.json"
    path.write_text(json.dumps({"records": [
        {"tensor": "model.layers.0.mlp.down_proj", "stat": {"row": 3, "col": 100}, "note": "top16; zeroed"},
        {"tensor": "model.layers.1.self_attn.q_proj", "stat": {"row": 5, "col": 7}, "note": "top16; zeroed"},
        {"tensor": "model.layers.1.mlp.up_proj", "stat": {"row": 2, "col": 2}, "note": "spike+top16; zeroed"},
        {"tensor": "model.layers.0.self_attn.q_proj", "stat": {"row": 0, "col": 0}, "note": "max_spike_below_threshold; not a candidate"},
    ]}), encoding="utf-8")
    acct = resolve(f"rtn-sym-g128-4+keepsuper={path}")(model, Ctx(None, 0))
    notes = acct.summary()["notes"]
    assert len(notes) == 2  # plain tensors and touched tensors, each once
    clause = [n for n in notes if "keepsuper" in n]
    assert len(clause) == 1 and "3 bf16 weights restored (spike+top16 1, top16 2)" in clause[0] and "16+ceil(log2(N*K)) bits each" in clause[0]
    assert "39 bits" not in clause[0] and "35 bits" not in clause[0]
    by_name = {t.name: t for t in acct.tensors}
    assert by_name["model.layers.0.self_attn.q_proj"].aux_bits == 64 * 2 * 16  # the fallback record was not restored


def test_hessian_path_check_from_accumulated_hessians_matches_windows():
    """scripts/hessian_paths.py keeps (H32 sum, H64 sum, rows) per linear instead of the inputs: same result."""
    from wcomp.quant.gptq import accumulate_hessian_paths

    g = torch.Generator().manual_seed(5)
    K, N = 256, 32
    W = torch.randn(N, K, generator=g) * 0.02
    windows = [(torch.randn(128, K, generator=g) * (1 + 10 * (torch.rand(K, generator=g) < 0.05))).to(torch.bfloat16) for _ in range(3)]
    acc = None
    for x in windows:
        acc = accumulate_hessian_paths(acc, x.reshape(1, 128, K))  # a hook sees [batch, T, K]
    assert acc[2] == [3 * 128]
    a = hessian_path_check(W, windows, 3, 128, True)
    b = hessian_path_check(W, None, 3, 128, True, hessians=acc)
    for k in ("rel_frob", "max_rel_diag", "frac_changed", "objective_ratio", "objective_f64", "n_rows"):
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12), k


def test_keepsuper_clause_says_model_wide(tmp_path):
    """The one clause every touched tensor carries states the model-wide count, since it is not per tensor."""
    model = _stub_model()
    path = tmp_path / "superweight.json"
    path.write_text(json.dumps({"records": [
        {"tensor": "model.layers.0.mlp.down_proj", "stat": {"row": 3, "col": 100}, "note": "spike; zeroed"},
        {"tensor": "model.layers.1.self_attn.q_proj", "stat": {"row": 5, "col": 7}, "note": "top16; zeroed"},
    ]}), encoding="utf-8")
    acct = resolve(f"rtn-sym-g128-4+keepsuper={path}")(model, Ctx(None, 0))
    touched = [t for t in acct.tensors if "keepsuper" in t.note]
    assert len(touched) == 2 and all("2 bf16 weights restored (spike 1, top16 1) model-wide" in t.note for t in touched)
    assert len({t.note for t in touched}) == 1  # one clause text, so the row carries it once
