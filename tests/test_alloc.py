"""Tests for wcomp.alloc (methods A and G of the Phase 3 contract), the per-tensor / per-row widths of the quantizers,
the alloc-* / rowalloc-* / combo-* names of wcomp.methods and scripts/damage_c4.py (the last one on the 0.8B, GPU)."""
from __future__ import annotations

import inspect
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from wcomp import alloc as alloc_mod
from wcomp.alloc import (LEVELS, UNIT_SIDE_BITS, allocate, allocation_objective, mean_bits, row_allocate, shuffle_allocation,
                         shuffle_rows)
from wcomp import methods as methods_mod
from wcomp.methods import METHODS, Ctx, is_phase3_name, resolve
from wcomp.models import block_linears, weight_type
from wcomp.quant.gptq import gptq_model, gptq_quantize
from wcomp.quant.rtn import ROW_SIDE_BITS, apply_rtn, quantize_rtn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3.5-0.8B"

PHASE3_NAMES = [
    "alloc-type-gptq32-avg2.5", "alloc-type-gptq32-avg3", "alloc-type-gptq32-avg3.5", "alloc-type-gptq32-avg4",
    "alloc-tensor-gptq32-avg3", "alloc-tensor-gptq32-avg3.5", "alloc-type-rtn-avg3", "alloc-type-rtn-avg3.5",
    "alloc-type-shuf1-gptq32-avg3", "alloc-type-shuf2-gptq32-avg3",
    "rowalloc-fisher-rtn-avg4", "rowalloc-shuf1-rtn-avg4", "rowalloc-shuf2-rtn-avg4",
    "rowalloc-fisher-gptq32-avg4", "rowalloc-fisher-gptq32-avg3",
    "combo-gptq32-avg2.5", "combo-gptq32-avg3", "combo-gptq32-avg3.5",
]


def _stub_model() -> nn.Module:
    """Two layers with self_attn.q_proj [64, 256], mlp.up_proj [128, 256], mlp.down_proj [64, 384]."""

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


# ---------------------------------------------------------------- allocate / shuffle_allocation


def test_allocate_two_units_closed_form():
    # continuous optimum b_u = target + 0.5 log2(d_u / geomean d) = 4 -/+ 1.66 -> the discrete split (2, 6) at mean 4:
    # objective 16 + 100/16 = 22.25 beats (3, 5) = 29 and (4, 4) = 101
    d, n = {"a": 1.0, "b": 100.0}, {"a": 1000, "b": 1000}
    bits = allocate(d, n, 4.0)
    assert bits == {"a": 2, "b": 6}
    assert mean_bits(bits, n) == pytest.approx(4.0)


def test_allocate_objective_decreases_with_the_target_and_stays_within_budget():
    gen = torch.Generator().manual_seed(0)
    d = {f"u{i}": float(x) for i, x in enumerate(torch.rand(9, generator=gen) * 1e-8)}
    n = {f"u{i}": int(x) for i, x in enumerate(torch.randint(1000, 100000, (9,), generator=gen))}
    prev = None
    for target in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0):
        bits = allocate(d, n, target)
        assert mean_bits(bits, n) <= target + 1e-9
        assert all(b in LEVELS for b in bits.values())
        obj = allocation_objective(bits, d, n)
        if prev is not None:
            assert obj <= prev * (1 + 1e-12)  # a larger budget never gives a worse optimum
        prev = obj
    assert allocate(d, n, 8.0) == {u: 8 for u in d}  # everything at the top level once the budget allows it


def _brute_force(d: dict, n: dict, target: float) -> tuple[float, dict]:
    best = None
    for combo in itertools.product(LEVELS, repeat=len(d)):
        b = dict(zip(d, combo))
        if mean_bits(b, n) <= target + 1e-9:
            obj = allocation_objective(b, d, n)
            if best is None or obj < best[0]:
                best = (obj, b)
    return best


def test_allocate_is_exact_against_brute_force_with_unequal_numel():
    """The greedy marginal-gain ascent (the contract's "exact" claim) is a heuristic for this multiple-choice knapsack;
    the DP must match the brute-force optimum on 3 units with random unequal numel, and on the cases where the greedy
    fails: unequal numel, and equal numel across the 6 -> 8 double step."""
    gen = torch.Generator().manual_seed(0)
    for _ in range(300):
        d = {u: float(x) for u, x in zip("abc", torch.rand(3, generator=gen) * 1000 + 1)}
        n = {u: int(x) for u, x in zip("abc", torch.randint(1, 100, (3,), generator=gen))}
        target = float(torch.rand(1, generator=gen) * 6 + 2)
        got = allocate(d, n, target)
        assert mean_bits(got, n) <= target + 1e-9
        assert allocation_objective(got, d, n) == pytest.approx(_brute_force(d, n, target)[0], rel=1e-12)
    # the reviewer's instances: greedy {8,6,8} (obj 476) vs optimum {4,8,3} (obj 53); equal numel {6,6,6} vs {5,5,8}
    d, n = {"a": 2.9, "b": 122.9, "c": 2.2}, {"a": 2, "b": 62, "c": 2}
    assert allocation_objective(allocate(d, n, 7.73), d, n) == pytest.approx(_brute_force(d, n, 7.73)[0], rel=1e-12)
    d, n = {"a": 12.1, "b": 8.7, "c": 75.4}, {"a": 10, "b": 10, "c": 10}
    got = allocate(d, n, 6.06)
    assert got == {"a": 5, "b": 5, "c": 8} and allocation_objective(got, d, n) < allocation_objective({"a": 6, "b": 6, "c": 6}, d, n)


@pytest.mark.skipif(not (ROOT / "results" / "phase3" / "qwen3.5-0.8b" / "damage_c4.jsonl").exists(), reason="needs the 0.8B damage file")
def test_allocate_on_the_real_0_8b_type_damage_is_the_exact_optimum():
    """The 11 type units of the 0.8B (gcd 196608): the DP never does worse than the greedy marginal-gain ascent and stays
    within the budget at every target (on the pre-exclusion damage file the greedy was 5.8% worse at avg 2.5 and 3.5;
    the exact widths and that gap depend on the measured damage and are not asserted). The file must pass the loader's
    C4-set / window check, so the test never reads a stale damage file silently."""
    rows = alloc_mod.load_damage("qwen3.5-0.8b", "type")
    d = alloc_mod.damage_per_param(rows)
    n = {name: int(r["numel"]) for name, r in rows.items()}
    assert len(d) == 11

    def greedy(target: float) -> dict:
        total = float(sum(n.values()))
        bits = {u: LEVELS[0] for u in d}
        used = LEVELS[0] * total
        while True:
            best = None
            for u in d:
                i = LEVELS.index(bits[u])
                if i + 1 == len(LEVELS) or used + n[u] * (LEVELS[i + 1] - LEVELS[i]) > target * total + 1e-9 * total:
                    continue
                gain = d[u] * (alloc_mod.distortion(LEVELS[i]) - alloc_mod.distortion(LEVELS[i + 1])) / (LEVELS[i + 1] - LEVELS[i])
                if best is None or gain > best[0]:
                    best = (gain, u)
            if best is None:
                return bits
            i = LEVELS.index(bits[best[1]])
            used += n[best[1]] * (LEVELS[i + 1] - LEVELS[i])
            bits[best[1]] = LEVELS[i + 1]

    for target in (2.5, 3.0, 3.5, 4.0):
        exact, g = allocate(d, n, target), greedy(target)
        assert mean_bits(exact, n) <= target + 1e-9
        assert allocation_objective(exact, d, n) <= allocation_objective(g, d, n) * (1 + 1e-12)
        assert all(b in LEVELS for b in exact.values())


def test_row_allocate_clamps_to_the_allocation_levels():
    r = torch.rand(64, generator=torch.Generator().manual_seed(0))
    assert set(row_allocate(r, 2).tolist()) == {2}  # no 1-bit rows: a 1-bit sym grid is {-amax, 0}
    assert set(row_allocate(r, 8).tolist()) == {8}  # no 9-bit rows
    rb = row_allocate(r, 3)
    assert set(rb.tolist()) == {2, 3, 4} and rb.float().mean().item() == pytest.approx(3.0)
    assert set(row_allocate(r, 7).tolist()) == {6, 7, 8}
    with pytest.raises(ValueError):
        row_allocate(r, 9)


def test_allocate_gives_more_bits_to_denser_damage():
    d = {"low": 1e-10, "mid": 1e-9, "high": 1e-8}
    n = {"low": 10_000, "mid": 10_000, "high": 10_000}
    bits = allocate(d, n, 4.0)
    assert bits["low"] <= bits["mid"] <= bits["high"] and bits["low"] < bits["high"]


def test_allocate_rejects_bad_input():
    with pytest.raises(ValueError):
        allocate({"a": 1.0}, {"b": 1}, 4.0)
    with pytest.raises(ValueError):
        allocate({"a": 1.0}, {"a": 1}, 1.0)
    with pytest.raises(ValueError):
        allocate({}, {}, 4.0)


def test_shuffle_allocation_keeps_mean_and_changes_assignment():
    gen = torch.Generator().manual_seed(1)
    d = {f"t{i}": float(x) for i, x in enumerate(torch.rand(11, generator=gen) ** 4 * 1e-8)}
    n = {f"t{i}": int(x) for i, x in enumerate(torch.randint(500_000, 100_000_000, (11,), generator=gen))}
    bits = allocate(d, n, 3.0)
    assert len(set(bits.values())) > 1
    for seed in (1, 2):
        shuf = shuffle_allocation(bits, n, seed)
        assert set(shuf) == set(bits)
        assert shuf != bits
        assert abs(mean_bits(shuf, n) - mean_bits(bits, n)) <= 0.02
        assert all(b in LEVELS for b in shuf.values())
    assert shuffle_allocation(bits, n, 1) != shuffle_allocation(bits, n, 2)


def test_shuffle_allocation_is_a_derangement_and_keeps_the_largest_unit_permuted():
    """The control must not share the real allocation's dominant decision: the permutation has no fixed point, the largest
    unit keeps the width it received (it is frozen during rebalancing) and the info records what the rebalancer touched.
    The 0.8B type shape: 11 units, one of 23% of the parameters at the lowest width."""
    n = {"qkv": 113_246_208, "z": 37_748_736, "ab": 524_288, "out": 37_748_736, "gate": 50_331_648, "up": 50_331_648,
         "down": 50_331_648, "q": 50_331_648, "k": 12_582_912, "v": 12_582_912, "o": 81_854_464}
    bits = {"qkv": 2, "z": 3, "ab": 8, "out": 5, "gate": 3, "up": 3, "down": 3, "q": 3, "k": 4, "v": 5, "o": 5}
    for seed in (1, 2):
        shuf, info = alloc_mod.shuffle_allocation_detail(bits, n, seed)
        assert all(info["permutation"][u] != u for u in bits)  # no unit kept its own slot
        assert info["frozen_unit"] == "qkv" and shuf["qkv"] == info["raw_bits"]["qkv"] != bits["qkv"]
        assert abs(mean_bits(shuf, n) - mean_bits(bits, n)) <= 0.02
        assert all(mv[0] != "qkv" for mv in info["moves"])
        replay = dict(info["raw_bits"])
        for u, frm, to in info["moves"]:
            assert replay[u] == frm
            replay[u] = to
        assert replay == shuf
        assert set(info["units_at_real_width"]) == {u for u in bits if shuf[u] == bits[u]}
        assert info["params_share_at_real_width"] == pytest.approx(sum(n[u] for u in info["units_at_real_width"]) / sum(n.values()))
        assert info["params_share_at_real_width"] < 0.6
    assert shuffle_allocation(bits, n, 1) != shuffle_allocation(bits, n, 2)
    # a single unit has no permutation and comes back unchanged
    assert shuffle_allocation({"a": 4}, {"a": 10}, 1) == {"a": 4}


# ---------------------------------------------------------------- damage file checks


def test_load_damage_refuses_other_c4_sets_and_smoke_windows_in_the_protocol_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    model = _stub_model()
    kl = {"q": 0.5, "up": 0.001, "down": 0.05}
    path = tmp_path / "phase3" / "stub" / "damage_c4.jsonl"
    _write_type_damage(path, model, kl)
    assert set(alloc_mod.load_damage("stub", "type")) == set(kl)
    meta = alloc_mod.damage_meta(path)
    assert meta["c4_set"] == alloc_mod.DAMAGE_C4_SET and meta["windows"] == [32] and len(meta["sha256"]) == 10
    # the pre-exclusion file: rows without "c4_set"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    for r in rows:
        del r["c4_set"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="C4 set None"):
        alloc_mod.load_damage("stub", "type")
    with pytest.raises(ValueError, match="C4 set None"):
        resolve("alloc-type-rtn-avg3")(model, Ctx(None, 0, tag="stub"))
    # another set
    _write_type_damage(path, model, kl, c4_set="seed2")
    with pytest.raises(ValueError, match="move the file aside"):
        alloc_mod.load_damage("stub", "type")
    # a smoke file (2 windows) in the protocol tree is refused, the same rows under a smoke path are accepted
    _write_type_damage(path, model, kl, windows=2)
    with pytest.raises(ValueError, match="smoke damage file cannot feed a protocol allocation"):
        alloc_mod.load_damage("stub", "type")
    smoke_dir = tmp_path / "smoke" / "phase3" / "stub"
    _write_type_damage(smoke_dir / "damage_c4.jsonl", model, kl, windows=2)
    assert alloc_mod.is_smoke_path(smoke_dir / "damage_c4.jsonl") and not alloc_mod.is_smoke_path(path)
    assert set(alloc_mod.load_damage("stub", "type", out_dir=smoke_dir)) == set(kl)
    assert alloc_mod.damage_meta(smoke_dir / "damage_c4.jsonl")["windows"] == [2]
    with pytest.raises(FileNotFoundError):
        alloc_mod.load_damage("nobody", "type")


def test_alloc_and_combo_read_their_inputs_from_the_runner_out_dir_first(tmp_path, monkeypatch):
    """A smoke run with its own damage_c4.jsonl / combo.json under its --results-dir uses those; without them the protocol
    files are read; the alloc JSON and the note record which file (path and sha256)."""
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    model = _stub_model()
    _write_type_damage(tmp_path / "phase3" / "stub" / "damage_c4.jsonl", model, {"q": 0.5, "up": 0.001, "down": 0.05})
    smoke_dir = tmp_path / "smoke" / "phase3" / "stub"
    _write_type_damage(smoke_dir / "damage_c4.jsonl", model, {"q": 0.001, "up": 0.5, "down": 0.05}, windows=2)  # reversed signal
    acct_p = resolve("alloc-type-rtn-avg3")(_stub_model(), Ctx(None, 0, tag="stub", out_dir=tmp_path / "runs"))  # no damage file there
    acct_s = resolve("alloc-type-rtn-avg3")(_stub_model(), Ctx(None, 0, tag="stub", out_dir=smoke_dir))
    out_p = json.load(open(tmp_path / "runs" / "alloc_alloc-type-rtn-avg3_seed0.json", encoding="utf-8"))
    out_s = json.load(open(smoke_dir / "alloc_alloc-type-rtn-avg3_seed0.json", encoding="utf-8"))
    assert out_p["bits"]["q"] > out_p["bits"]["up"] and out_s["bits"]["q"] < out_s["bits"]["up"]
    assert out_p["damage_file"]["path"].endswith("phase3/stub/damage_c4.jsonl") and out_p["damage_file"]["windows"] == [32]
    assert out_s["damage_file"]["path"].endswith("smoke/phase3/stub/damage_c4.jsonl") and out_s["damage_file"]["windows"] == [2]
    assert out_p["damage_file"]["sha256"] != out_s["damage_file"]["sha256"]
    assert any(f"sha {out_p['damage_file']['sha256']}" in t.note and "32 windows" in t.note for t in acct_p.tensors)
    assert any(f"sha {out_s['damage_file']['sha256']}" in t.note and "2 windows" in t.note for t in acct_s.tensors)
    # combo.json: the smoke directory's own decision file, the protocol tree untouched
    import wcomp.methods as methods_mod

    monkeypatch.setattr(methods_mod, "gptq_model",
                        lambda model, calib, bits, group, sym, **kw: apply_rtn(model, bits, group, sym, codebook=kw.get("codebook")))
    (smoke_dir / "combo.json").write_text(json.dumps({"had": False, "lloyd": False, "rowalloc": False}), encoding="utf-8")
    assert methods_mod.combo_path("stub", smoke_dir) == smoke_dir / "combo.json"
    assert methods_mod.combo_path("stub", tmp_path / "runs") == tmp_path / "phase3" / "stub" / "combo.json"
    acct = resolve("combo-gptq32-avg3")(_stub_model(), Ctx(torch.zeros(1, 8, dtype=torch.long), 0, tag="stub", out_dir=smoke_dir))
    payload = json.loads((smoke_dir / "alloc_combo-gptq32-avg3_seed0.json").read_text(encoding="utf-8"))
    assert payload["combo_file"]["path"].endswith("smoke/phase3/stub/combo.json") and payload["combo"] == {"had": False, "lloyd": False, "rowalloc": False}
    assert payload["damage_file"]["windows"] == [2]
    assert any(f"combo.json sha {payload['combo_file']['sha256']}" in t.note for t in acct.tensors)
    assert not (tmp_path / "phase3" / "stub" / "combo.json").exists()
    with pytest.raises(FileNotFoundError):  # a protocol run finds no decision file
        resolve("combo-gptq32-avg3")(_stub_model(), Ctx(torch.zeros(1, 8, dtype=torch.long), 0, tag="stub", out_dir=tmp_path / "runs"))


# ---------------------------------------------------------------- row allocation


def test_row_allocate_hot_row_gets_the_higher_width():
    r = torch.ones(64)
    r[17] = 100.0
    rb = row_allocate(r, 4)
    assert rb.dtype == torch.int64 and rb.shape == (64,)
    assert int(rb[17]) == 5
    assert set(rb.tolist()) <= {3, 4, 5}
    assert rb.float().mean().item() == pytest.approx(4.0)


def test_row_allocate_matches_exact_allocate():
    gen = torch.Generator().manual_seed(3)
    r = torch.rand(16, generator=gen) * 10
    rb = row_allocate(r, 4)
    d = {i: float(r[i]) / 256 for i in range(16)}
    n = {i: 256 for i in range(16)}
    ref = allocate(d, n, 4.0, levels=(3, 4, 5))
    assert rb.tolist() == [ref[i] for i in range(16)]


def test_row_allocate_rejects_width_below_one():
    with pytest.raises(ValueError):
        row_allocate(torch.ones(4), 1)


def test_shuffle_rows_keeps_the_multiset():
    rb = row_allocate(torch.arange(32).float(), 3)
    gen = torch.Generator().manual_seed(0)
    s = shuffle_rows(rb, gen)
    assert sorted(s.tolist()) == sorted(rb.tolist()) and s.tolist() != rb.tolist()


# ---------------------------------------------------------------- quantizers with per-tensor / per-row widths


def test_quantize_rtn_row_bits_uniform_equals_plain_and_mixed_rows_match_their_width():
    torch.manual_seed(0)
    W = torch.randn(64, 256)
    plain, info_p = quantize_rtn(W, 3)
    rows, info_r = quantize_rtn(W, 3, row_bits=torch.full((64,), 3))
    assert torch.equal(plain, rows)
    assert info_p["index_bits"] == info_r["index_bits"] == 3 * 64 * 256
    assert torch.equal(info_p["indices"], info_r["indices"]) and info_p["levels"] == info_r["levels"] == 8
    assert info_p["indices"].dtype == torch.int16 and info_p["indices"].min() >= 0 and info_p["indices"].max() < 8
    rb = torch.tensor([2] * 20 + [3] * 24 + [4] * 20)
    mixed, info_m = quantize_rtn(W, 3, row_bits=rb)
    for b in (2, 3, 4):
        sel = rb == b
        ref, _ = quantize_rtn(W[sel], b)
        assert torch.equal(mixed[sel], ref)
    assert info_m["index_bits"] == 256 * (20 * 2 + 24 * 3 + 20 * 4)
    assert info_m["aux_bits"] == 64 * 2 * 16  # f16 scale per group, no row side info here (the caller charges it)
    assert info_m["levels"] == 16 and info_m["indices"].reshape(64, 256)[:20].max() < 4


@pytest.mark.parametrize("sym", [True, False])
def test_gptq_row_bits_identity_hessian_equals_rtn_rows(sym: bool):
    torch.manual_seed(1)
    W = torch.randn(32, 256)
    rb = torch.tensor([3] * 10 + [4] * 12 + [5] * 10)
    H = torch.eye(256, dtype=torch.float64)
    Q, info_g = gptq_quantize(W, H, 4, 128, sym, row_bits=rb)
    R, info_r = quantize_rtn(W, 4, 128, sym, row_bits=rb)
    assert torch.allclose(Q, R, atol=1e-6)
    assert info_g["index_bits"] == info_r["index_bits"] and info_g["aux_bits"] == info_r["aux_bits"]
    assert torch.equal(info_g["indices"], info_r["indices"]) and info_g["levels"] == 32
    assert "rows{3,4,5}" in info_g["note"]


def test_gptq_plain_indices_match_rtn_with_identity_hessian():
    torch.manual_seed(2)
    W = torch.randn(16, 256)
    Q, info_g = gptq_quantize(W, torch.eye(256, dtype=torch.float64), 3)
    R, info_r = quantize_rtn(W, 3)
    assert torch.allclose(Q, R, atol=1e-6)
    assert torch.equal(info_g["indices"], info_r["indices"]) and info_g["levels"] == 8


def test_apply_rtn_per_tensor_bits_and_row_bits_accounting():
    model = _stub_model()
    names = [n for n, _ in block_linears(model)]
    bits = {n: (3 if weight_type(n) == "q" else 5) for n in names}
    acct = apply_rtn(model, bits, 128, True)
    for t in acct.tensors:
        assert t.index_bits == bits[t.name] * t.numel and t.levels == 2 ** bits[t.name] and t.row_bits is None
        assert t.indices is not None and t.indices.numel() == t.numel
    model = _stub_model()
    rb = {names[0]: torch.tensor([3] * 32 + [5] * 32)}
    acct = apply_rtn(model, 4, 128, True, row_bits=rb)
    t0 = next(t for t in acct.tensors if t.name == names[0])
    assert t0.index_bits == 256 * (32 * 3 + 32 * 5) and t0.row_bits == rb[names[0]].tolist()
    assert t0.aux_bits == 64 * 2 * 16 + ROW_SIDE_BITS * 64 and "bits/row width" in t0.note
    t1 = next(t for t in acct.tensors if t.name == names[1])
    assert t1.index_bits == 4 * t1.numel and t1.row_bits is None


def test_gptq_model_accepts_dict_bits_and_row_bits_signature():
    params = inspect.signature(gptq_model).parameters
    assert "row_bits" in params and "hessian_f32" in params


# ---------------------------------------------------------------- method names


def test_resolve_accepts_every_phase3_name():
    for name in PHASE3_NAMES:
        assert is_phase3_name(name)
        assert callable(resolve(name))
    for name in ("alloc-type-gptq32", "rowalloc-fisher-rtn-avg1", "combo-rtn-avg3", "alloc-row-gptq32-avg3"):
        with pytest.raises(ValueError):
            resolve(name)
    for name, method in METHODS.items():
        assert resolve(name) is method


def _damage_row(unit: str, name: str, numel: int, kl: float, **extra) -> dict:
    """A damage_c4.jsonl row as scripts/damage_c4.py writes it (the C4 set and window count are checked by the loader)."""
    row = {"unit": unit, "name": name, "numel": numel, "kl": kl, "ppl": 20.0, "n_tokens": 4094,
           "c4_set": alloc_mod.DAMAGE_C4_SET, "windows": alloc_mod.DAMAGE_WINDOWS}
    row.update(extra)
    return row


def _write_type_damage(path: Path, model: nn.Module, kl: dict[str, float], **extra) -> None:
    numel: dict[str, int] = {}
    for n, lin in block_linears(model):
        numel[weight_type(n)] = numel.get(weight_type(n), 0) + lin.weight.numel()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t, k in kl.items():
            f.write(json.dumps(_damage_row("type", t, numel[t], k, **extra)) + "\n")


def test_alloc_type_rtn_method_on_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    model = _stub_model()
    kl = {"q": 0.5, "up": 0.001, "down": 0.05}
    _write_type_damage(tmp_path / "phase3" / "stub" / "damage_c4.jsonl", model, kl)
    numel = {t: 0 for t in kl}
    for n, lin in block_linears(model):
        numel[weight_type(n)] += lin.weight.numel()
    expected = allocate({t: kl[t] / numel[t] for t in kl}, numel, 3.5)
    acct = resolve("alloc-type-rtn-avg3.5")(model, Ctx(None, 0, tag="stub"))
    for t in acct.tensors:
        assert t.index_bits == expected[weight_type(t.name)] * t.numel
    assert sum(t.aux_bits for t in acct.tensors) == sum(t.numel // 128 for t in acct.tensors) * 16 + UNIT_SIDE_BITS * 3
    assert acct.bits_blocks() <= 3.5 + 16 / 128 + 1e-3
    out = json.load(open(tmp_path / "phase3" / "stub" / "alloc_alloc-type-rtn-avg3.5_seed0.json", encoding="utf-8"))
    assert out["bits"] == expected and out["unit"] == "type" and out["realized_mean_bits"] <= 3.5
    assert any("alloc-type-rtn-avg3.5" in t.note for t in acct.tensors)
    with pytest.raises(ValueError):
        resolve("alloc-type-rtn-avg3.5")(_stub_model(), Ctx(None, 0))  # no tag


def test_alloc_type_shuffle_control_on_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    model = _stub_model()
    _write_type_damage(tmp_path / "phase3" / "stub" / "damage_c4.jsonl", model, {"q": 0.5, "up": 0.001, "down": 0.05})
    acct = resolve("alloc-type-shuf1-rtn-avg3")(model, Ctx(None, 0, tag="stub"))
    out = json.load(open(tmp_path / "phase3" / "stub" / "alloc_alloc-type-shuf1-rtn-avg3_seed0.json", encoding="utf-8"))
    assert out["bits"] != out["real_bits"] and abs(out["realized_mean_bits"] - out["real_mean_bits"]) <= 0.02
    assert any("shuffled control (seed 1, derangement," in t.note and "units at their real width" in t.note for t in acct.tensors)
    info = out["shuffle"]
    assert set(info) >= {"raw_bits", "permutation", "frozen_unit", "moves", "units_at_real_width", "params_share_at_real_width", "draws"}
    assert all(info["permutation"][u] != u for u in out["bits"])
    assert set(info["units_at_real_width"]) == {u for u in out["bits"] if out["bits"][u] == out["real_bits"][u]}


def test_alloc_tensor_falls_back_to_fisher_rows_totals(tmp_path, monkeypatch):
    """A model without fisher.json and without per-tensor damage rows (the 2B) takes its tensor-level d from the
    "totals" of scripts/fisher_rows.py's fisher_rows.pt."""
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    monkeypatch.setattr(alloc_mod, "PROBE_DIR", tmp_path / "probe")
    model = _stub_model()
    _write_type_damage(tmp_path / "phase3" / "stub" / "damage_c4.jsonl", model, {"q": 0.5, "up": 0.001, "down": 0.05})
    with pytest.raises(FileNotFoundError):
        resolve("alloc-tensor-rtn-avg3")(model, Ctx(None, 0, tag="stub"))
    names = [n for n, _ in block_linears(model)]
    totals = {n: float(i + 1) for i, n in enumerate(names)}
    (tmp_path / "probe" / "stub").mkdir(parents=True)
    torch.save({"rows": {}, "totals": totals}, tmp_path / "probe" / "stub" / "fisher_rows.pt")
    assert alloc_mod.load_fisher_totals("stub") == totals
    acct = resolve("alloc-tensor-rtn-avg3")(model, Ctx(None, 0, tag="stub"))
    out = json.load(open(tmp_path / "phase3" / "stub" / "alloc_alloc-tensor-rtn-avg3_seed0.json", encoding="utf-8"))
    assert out["unit"] == "tensor" and set(out["bits"]) == set(names) and "fisher_rows.pt" in out["source"]
    numel = {n: lin.weight.numel() for n, lin in block_linears(model)}
    assert out["bits"] == allocate({n: totals[n] / numel[n] for n in names}, numel, 3.0)
    assert sum(t.aux_bits for t in acct.tensors) == sum(t.numel // 128 for t in acct.tensors) * 16 + UNIT_SIDE_BITS * len(names)


def test_rowalloc_methods_on_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    monkeypatch.setattr(alloc_mod, "PROBE_DIR", tmp_path / "probe")
    model = _stub_model()
    gen = torch.Generator().manual_seed(0)
    rows = {n: torch.rand(lin.weight.shape[0], generator=gen) for n, lin in block_linears(model)}
    (tmp_path / "probe" / "stub").mkdir(parents=True)
    torch.save({"rows": rows}, tmp_path / "probe" / "stub" / "fisher_rows.pt")
    acct = resolve("rowalloc-fisher-rtn-avg4")(model, Ctx(None, 0, tag="stub"))
    for t in acct.tensors:
        N = len(t.row_bits)
        assert set(t.row_bits) <= {3, 4, 5} and sum(t.row_bits) == 4 * N
        assert t.index_bits == 4 * t.numel and t.aux_bits == (t.numel // 128) * 16 + ROW_SIDE_BITS * N
        expected = row_allocate(rows[t.name], 4).tolist()
        assert t.row_bits == expected
    shuf = resolve("rowalloc-shuf1-rtn-avg4")(_stub_model(), Ctx(None, 0, tag="stub"))
    for t, s in zip(acct.tensors, shuf.tensors):
        assert sorted(t.row_bits) == sorted(s.row_bits) and t.row_bits != s.row_bits
    assert (tmp_path / "phase3" / "stub" / "alloc_rowalloc-shuf1-rtn-avg4_seed0.json").exists()


def test_combo_reads_config_and_composes_the_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    model = _stub_model()
    _write_type_damage(tmp_path / "phase3" / "stub" / "damage_c4.jsonl", model, {"q": 0.5, "up": 0.001, "down": 0.05})
    ctx = Ctx(torch.zeros(1, 8, dtype=torch.long), 0, tag="stub")
    with pytest.raises(FileNotFoundError):
        resolve("combo-gptq32-avg3")(model, ctx)
    cfg = tmp_path / "phase3" / "stub" / "combo.json"
    cfg.write_text(json.dumps({"had": False, "lloyd": True, "rowalloc": "no"}), encoding="utf-8")
    with pytest.raises(ValueError):
        resolve("combo-gptq32-avg3")(model, ctx)
    # lloyd + type allocation, no rotation / row allocation: the stub has no decoder forward, so gptq_model is replaced
    # by a recorder that quantizes with RTN and returns what the combo asked for
    import wcomp.methods as methods_mod

    calls: list[dict] = []

    def fake_gptq_model(model, calib, bits, group, sym, **kwargs):
        calls.append({"bits": bits, **kwargs})
        return apply_rtn(model, bits, group, sym, row_bits=kwargs.get("row_bits"), codebook=kwargs.get("codebook"))

    monkeypatch.setattr(methods_mod, "gptq_model", fake_gptq_model)
    cfg.write_text(json.dumps({"had": False, "lloyd": True, "rowalloc": False}), encoding="utf-8")
    out_dir = tmp_path / "runs"
    acct = resolve("combo-gptq32-avg3")(model, Ctx(torch.zeros(1, 8, dtype=torch.long), 0, tag="stub", out_dir=out_dir))
    assert len(calls) == 1 and calls[0]["hessian_f32"] is True and calls[0]["codebook"] == {"kind": "lloyd"} and calls[0]["row_bits"] is None
    assert set(calls[0]["bits"]) == {n for n, _ in block_linears(model)} and all(b in LEVELS for b in calls[0]["bits"].values())
    assert all("lloyd" in t.note and "combo-gptq32-avg3 = lloyd+alloc-type+gptq32" in t.note for t in acct.tensors)
    assert (out_dir / "alloc_combo-gptq32-avg3_seed0.json").exists()  # next to the runner's rows, not in results/phase3/stub
    assert not (tmp_path / "phase3" / "stub" / "alloc_combo-gptq32-avg3_seed0.json").exists()
    payload = json.loads((out_dir / "alloc_combo-gptq32-avg3_seed0.json").read_text(encoding="utf-8"))
    assert payload["combo"] == {"had": False, "lloyd": True, "rowalloc": False} and payload["unit"] == "type"
    assert payload["realized_mean_bits"] <= 3.0 and all(t.indices is not None for t in acct.tensors)


def test_combo_rowalloc_never_splits_a_type_at_the_lowest_or_highest_level(tmp_path, monkeypatch):
    """A type the allocator puts at 2 bits (or 8) keeps all its rows there: no 1-bit ({-amax, 0}) or 9-bit rows, no
    row side information for those tensors; the other types split b-1 / b / b+1 at a mean of exactly b."""
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    monkeypatch.setattr(alloc_mod, "PROBE_DIR", tmp_path / "probe")
    model = _stub_model()
    # q is hammered, up is negligible: at avg 4 the allocation is q=8, down=4, up=2 (checked below)
    _write_type_damage(tmp_path / "phase3" / "stub" / "damage_c4.jsonl", model, {"q": 500.0, "up": 1e-9, "down": 5.0})
    gen = torch.Generator().manual_seed(0)
    rows = {n: torch.rand(lin.weight.shape[0], generator=gen) for n, lin in block_linears(model)}
    (tmp_path / "probe" / "stub").mkdir(parents=True)
    torch.save({"rows": rows, "calib_seed": 0, "n_windows": 32, "seqlen": 1024}, tmp_path / "probe" / "stub" / "fisher_rows.pt")
    import wcomp.methods as methods_mod

    calls: list[dict] = []

    def fake_gptq_model(model, calib, bits, group, sym, **kwargs):
        calls.append({"bits": bits, **kwargs})
        return apply_rtn(model, bits, group, sym, row_bits=kwargs.get("row_bits"), codebook=kwargs.get("codebook"))

    monkeypatch.setattr(methods_mod, "gptq_model", fake_gptq_model)
    (tmp_path / "phase3" / "stub" / "combo.json").write_text(json.dumps({"had": False, "lloyd": False, "rowalloc": True}), encoding="utf-8")
    acct = resolve("combo-gptq32-avg4")(model, Ctx(torch.zeros(1, 8, dtype=torch.long), 0, tag="stub", out_dir=tmp_path / "runs"))
    bits = calls[0]["bits"]
    types = {weight_type(n): b for n, b in bits.items()}
    assert types == {"q": 8, "down": 4, "up": 2}, types
    row_bits = calls[0]["row_bits"]
    first = {weight_type(n): n for n in reversed([n for n, _ in block_linears(model)])}  # the unit's first tensor carries the side info
    for t in acct.tensors:
        b = bits[t.name]
        side = UNIT_SIDE_BITS if first[weight_type(t.name)] == t.name else 0
        if b in (LEVELS[0], LEVELS[-1]):
            assert t.name not in row_bits and t.row_bits is None and t.index_bits == b * t.numel
            assert t.aux_bits == (t.numel // 128) * 16 + side
        else:
            assert set(t.row_bits) <= {b - 1, b, b + 1} and min(t.row_bits) >= LEVELS[0] and sum(t.row_bits) == b * len(t.row_bits)
            assert t.aux_bits == (t.numel // 128) * 16 + ROW_SIDE_BITS * len(t.row_bits) + side
    assert len(row_bits) == 2 and all(min(rb.tolist()) >= LEVELS[0] and max(rb.tolist()) <= LEVELS[-1] for rb in row_bits.values())
    payload = json.loads((tmp_path / "runs" / "alloc_combo-gptq32-avg4_seed0.json").read_text(encoding="utf-8"))
    assert set(payload["unsplit_tensors"]) == {n for n, b in bits.items() if b in (2, 8)} and "row_levels" in payload
    assert any("not split, no row side info" in t.note for t in acct.tensors)
    # the standalone G rows stay at 3 / 4 / 5 around 4 and refuse a mean whose neighbours leave LEVELS
    with pytest.raises(ValueError):
        resolve("rowalloc-fisher-rtn-avg2")
    with pytest.raises(ValueError):
        resolve("rowalloc-fisher-rtn-avg8")


def test_alloc_tensor_refuses_a_partial_tensor_damage_section(tmp_path, monkeypatch):
    """An interrupted damage_c4.py run leaves 1..185 tensor rows: the method must not silently switch to Fisher totals."""
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    monkeypatch.setattr(alloc_mod, "PROBE_DIR", tmp_path / "probe")
    model = _stub_model()
    path = tmp_path / "phase3" / "stub" / "damage_c4.jsonl"
    _write_type_damage(path, model, {"q": 0.5, "up": 0.001, "down": 0.05})
    names = [n for n, _ in block_linears(model)]
    totals = {n: float(i + 1) for i, n in enumerate(names)}
    (tmp_path / "probe" / "stub").mkdir(parents=True)
    torch.save({"rows": {}, "totals": totals}, tmp_path / "probe" / "stub" / "fisher_rows.pt")
    with open(path, "a", encoding="utf-8") as f:  # 2 of the 6 tensors measured
        for n in names[:2]:
            lin = dict(block_linears(model))[n]
            f.write(json.dumps(_damage_row("tensor", n, lin.weight.numel(), 0.01)) + "\n")
    with pytest.raises(ValueError, match="lacks 4 of the 6 block linears"):
        resolve("alloc-tensor-rtn-avg3")(model, Ctx(None, 0, tag="stub"))
    with open(path, "a", encoding="utf-8") as f:  # the rest: the C4 rows are used, not the Fisher totals
        for n in names[2:]:
            lin = dict(block_linears(model))[n]
            f.write(json.dumps(_damage_row("tensor", n, lin.weight.numel(), 0.01)) + "\n")
    acct = resolve("alloc-tensor-rtn-avg3")(_stub_model(), Ctx(None, 0, tag="stub"))
    out = json.load(open(tmp_path / "phase3" / "stub" / "alloc_alloc-tensor-rtn-avg3_seed0.json", encoding="utf-8"))
    assert out["source"].startswith("d = KL/numel per tensor from") and all("d = KL/numel per tensor from" in t.note for t in acct.tensors)


def test_codebook_method_names_resolve_and_phase1_registry_is_untouched():
    from wcomp.methods import PHASE1_METHODS, needs_calib

    for n in [f"lloyd-gptq32-g128-{b}" for b in (4, 3, 2)] + ["nf-gptq32-g128-4", "nf-gptq32-g128-3", "nfq-gptq32-g128-4", "nfq-gptq32-g128-3"]:
        assert resolve(n) is METHODS[n] and needs_calib(n) and n not in PHASE1_METHODS
    assert "nf-gptq32-g128-2" not in METHODS and "lloyd-gptq32-g128-5" not in METHODS and "nfq-gptq32-g128-2" not in METHODS
    assert PHASE1_METHODS == {"fp16", "had-fp16", "rtn-sym-g128-8", "rtn-sym-g128-4", "rtn-sym-g128-3", "rtn-sym-g128-2",
                              "rtn-asym-g128-4", "rtn-asym-g128-3", "rtn-asym-g128-2", "had-rtn-sym-g128-4", "had-rtn-sym-g128-3",
                              "had-rtn-sym-g128-2", "gptq-sym-g128-4", "gptq-sym-g128-3", "gptq-sym-g128-2", "had-gptq-sym-g128-3"}


# ---------------------------------------------------------------- the damage script on the real 0.8B (GPU)


@pytest.mark.skipif(not torch.cuda.is_available() or not MODEL_DIR.is_dir(), reason="needs CUDA and Qwen3.5-0.8B")
def test_damage_script_smoke_types(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "damage_c4.py"), "--model", "qwen3.5-0.8b", "--units", "type",
           "--windows", "2", "--results-dir", str(tmp_path)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stderr[-3000:]
    out = tmp_path / "smoke" / "qwen3.5-0.8b" / "damage_c4.jsonl"  # --windows 2 is a smoke run
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["unit"] for r in rows] == ["type"] * 11
    assert {r["name"] for r in rows} == {"q", "k", "v", "o", "gate", "up", "down", "in_proj_qkv", "in_proj_z", "in_proj_ab", "out_proj"}
    assert all(r["kl"] >= 0 and r["ppl"] > 0 and r["n_tokens"] == 2 * 2047 for r in rows)
    ref = json.load(open(tmp_path / "smoke" / "qwen3.5-0.8b" / "damage_c4_ref.json", encoding="utf-8"))
    assert ref["fp16"]["kl"] == 0 and ref["rtn-sym-g128-4@all"]["kl"] > max(r["kl"] for r in rows)
    assert sum(r["numel"] for r in rows) == ref["rtn-sym-g128-4@all"]["numel"]
    res2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)  # resume: nothing to add
    assert res2.returncode == 0 and len(out.read_text(encoding="utf-8").splitlines()) == 11


# ---------------------------------------------------------------- review fixes


def test_shuffle_control_never_collapses_to_the_real_allocation_on_the_2b_type_shape():
    """The 2B type shape (7 types, 28 layers, H 2048, I 6144, KV 1024) with the allocation {down: 4, rest 3}: the seed-1
    derangement used to be walked back to the real allocation by the rebalancer (moves down 3->4, k 4->3; share 1.0) and
    was accepted. A rebalanced draw equal to the real allocation, or leaving more than SHUFFLE_MAX_REAL_SHARE of the
    parameters at their real width, is now rejected and the next draw tried; an all-one-width allocation has no control."""
    L, H, I, KV = 28, 2048, 6144, 1024
    n = {"q": L * H * H, "k": L * KV * H, "v": L * KV * H, "o": L * H * H, "gate": L * I * H, "up": L * I * H, "down": L * I * H}
    real = {"q": 3, "k": 3, "v": 3, "o": 3, "gate": 3, "up": 3, "down": 4}
    for seed in (1, 2):
        ctrl, info = alloc_mod.shuffle_allocation_detail(real, n, seed=seed)
        assert ctrl != real
        assert info["params_share_at_real_width"] <= alloc_mod.SHUFFLE_MAX_REAL_SHARE == info["max_real_share"]
        assert abs(mean_bits(ctrl, n) - mean_bits(real, n)) <= 0.02
        assert info["rejected_draws"] >= 0 and info["draws"] >= 1
    assert alloc_mod.shuffle_allocation_detail(real, n, seed=1)[1]["rejected_draws"] >= 1  # the collapsed seed-1 draw was rejected
    with pytest.raises(ValueError, match="no shuffle control possible"):
        alloc_mod.shuffle_allocation_detail({u: 3 for u in n}, n, seed=1)
    with pytest.raises(ValueError, match="no valid shuffle control"):
        alloc_mod.shuffle_allocation_detail(real, n, seed=1, max_real_share=0.0, max_draws=20)  # nothing can pass a zero cap
    # the 0.8B protocol shape is unaffected (shares 0.06-0.20 before and after)
    n8 = {"qkv": 113_246_208, "z": 37_748_736, "ab": 524_288, "out": 37_748_736, "gate": 50_331_648, "up": 50_331_648,
          "down": 50_331_648, "q": 50_331_648, "k": 12_582_912, "v": 12_582_912, "o": 81_854_464}
    bits8 = {"qkv": 2, "z": 3, "ab": 8, "out": 5, "gate": 3, "up": 3, "down": 3, "q": 3, "k": 4, "v": 5, "o": 5}
    for seed in (1, 2):
        ctrl, info = alloc_mod.shuffle_allocation_detail(bits8, n8, seed=seed)
        assert ctrl != bits8 and info["params_share_at_real_width"] <= 0.5


def test_alloc_tensor_requires_damage_rows_for_tensor_damage_models(tmp_path, monkeypatch):
    """For a model in TENSOR_DAMAGE_MODELS (the 0.8B, whose run list measures per-tensor damage) a damage file without
    tensor rows is an error, not a silent switch to the Fisher totals under the same method name."""
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    monkeypatch.setattr(alloc_mod, "PROBE_DIR", tmp_path / "probe")
    monkeypatch.setattr(methods_mod, "TENSOR_DAMAGE_MODELS", ("stub",))
    model = _stub_model()
    _write_type_damage(tmp_path / "phase3" / "stub" / "damage_c4.jsonl", model, {"q": 0.5, "up": 0.001, "down": 0.05})
    names = [n for n, _ in block_linears(model)]
    (tmp_path / "probe" / "stub").mkdir(parents=True)
    torch.save({"rows": {}, "totals": {n: float(i + 1) for i, n in enumerate(names)}}, tmp_path / "probe" / "stub" / "fisher_rows.pt")
    with pytest.raises(ValueError, match="--units tensor"):
        resolve("alloc-tensor-rtn-avg3")(model, Ctx(None, 0, tag="stub"))
    assert "qwen3.5-0.8b" in methods_mod.TENSOR_DAMAGE_MODELS or True  # the monkeypatched tuple; the real one names the 0.8B
    monkeypatch.setattr(methods_mod, "TENSOR_DAMAGE_MODELS", ())
    resolve("alloc-tensor-rtn-avg3")(_stub_model(), Ctx(None, 0, tag="stub"))  # another model: the A2 Fisher fallback
    out = json.load(open(tmp_path / "phase3" / "stub" / "alloc_alloc-tensor-rtn-avg3_seed0.json", encoding="utf-8"))
    assert out["signal"] == "fisher" and methods_mod.alloc_signal(out["source"]) == "fisher"
    assert methods_mod.alloc_signal("d = KL/numel per tensor from damage_c4.jsonl") == "damage"


def test_real_tensor_damage_models_name_the_0_8b():
    assert "qwen3.5-0.8b" in methods_mod.TENSOR_DAMAGE_MODELS


def test_rowalloc_interior_width_keeps_the_row_table_when_rows_come_out_uniform(tmp_path, monkeypatch):
    """Fisher row energies within a factor of 4 make water-filling return every row at the tensor's width. At an interior
    width (3..6) the tensor keeps its row table and its ROW_SIDE_BITS per row (the decoder expects a table for every
    interior-width tensor); only tensors at 2 or 8 bits are left out (no split possible, inferred from the width)."""
    monkeypatch.setattr(alloc_mod, "PHASE3_DIR", tmp_path / "phase3")
    monkeypatch.setattr(alloc_mod, "PROBE_DIR", tmp_path / "probe")
    model = _stub_model()
    rows = {n: torch.linspace(1.0, 3.9, lin.weight.shape[0]) for n, lin in block_linears(model)}
    (tmp_path / "probe" / "stub").mkdir(parents=True)
    torch.save({"rows": rows}, tmp_path / "probe" / "stub" / "fisher_rows.pt")
    names = [n for n, _ in block_linears(model)]
    for b in (3, 4, 5, 6):
        rb = methods_mod.row_allocation(model, "stub", {n: b for n in names}, None)
        assert set(rb) == set(names) and all(bool((t == b).all()) for t in rb.values())
    for b in (2, 8):
        assert methods_mod.row_allocation(model, "stub", {n: b for n in names}, None) == {}
    acct = resolve("rowalloc-fisher-rtn-avg4")(model, Ctx(None, 0, tag="stub"))
    for t in acct.tensors:
        N = len(t.row_bits)
        assert set(t.row_bits) == {4} and t.aux_bits == (t.numel // 128) * 16 + ROW_SIDE_BITS * N
    out = json.load(open(tmp_path / "phase3" / "stub" / "alloc_rowalloc-fisher-rtn-avg4_seed0.json", encoding="utf-8"))
    assert out["unsplit_tensors"] == [] and set(out["rows_per_width"]) == set(names)
