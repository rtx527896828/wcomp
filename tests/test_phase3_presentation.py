"""Phase 3 presentation on synthetic fixtures (no model, no GPU): the extended runs.jsonl row schema, the reference-curve
arithmetic (scripts/phase3_common.py), scripts/plot_rd3.py renders, scripts/phase3_summary.py builds the table with the
expected rows and verdict strings, and the runner's Phase 3 fields (coded bits, alloc, rotation, rd_bound hook)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import phase3_common as p3  # noqa: E402
import phase3_summary as ps  # noqa: E402
import probe_fixtures as fx  # noqa: E402
import run_baselines as rb  # noqa: E402
from wcomp import alloc as alloc_mod  # noqa: E402
from wcomp.bits import BitAccount  # noqa: E402

FP16 = 17.5675
M = "m"


def row(method: str, bits: float, ppl: float, kl: float, seed: int = 0, huff: float | None = "auto", shannon: float | None = "auto",
        note: str = "", rotation: str = "none", alloc=None, calib: bool = True) -> dict:
    r = fx._row(M, method, bits, ppl, kl, seed=seed, calib_seed=seed if calib else None, note=note)
    r["bits_blocks_huff"] = round(bits - 0.42, 4) if huff == "auto" else huff
    r["bits_blocks_shannon"] = round(bits - 0.45, 4) if shannon == "auto" else shannon
    r["bits_blocks_entropy0"] = None if r["bits_blocks_shannon"] is None else round(r["bits_blocks_shannon"] + 0.001, 4)
    r["rotation"], r["alloc"] = rotation, alloc
    assert tuple(r) == rb.ROW_FIELDS
    return r


def phase3_rows() -> list[dict]:
    rows = []
    # reference: gptq32-sym-g128-{4,3,2}, seeds 0 and 1 (seed means: KL 0.068 / 0.295 / 1.925)
    for b, (k0, k1), (p0, p1) in ((4, (0.070, 0.066), (18.06, 18.44)), (3, (0.300, 0.290), (23.3, 23.4)), (2, (1.90, 1.95), (130.0, 135.0))):
        for s, k, p in ((0, k0, p0), (1, k1, p1)):
            rows.append(row(f"gptq32-sym-g128-{b}", b + 0.125, p, k, seed=s, note=f"calib 128x2048 seed {s}; gptq {b}b sym g128 block128 damp0.01 hessian-f32"))
    # D rotations
    for s, k, p in ((0, 0.21, 20.9), (1, 0.205, 20.8)):
        rows.append(row("had-gptq32-sym-g128-3", 3.125, p, k, seed=s, rotation="hadamard"))
    for s, k, p in ((0, 0.066, 18.0), (1, 0.064, 18.0)):  # seed 0 inside the 5% noise: the method does not beat
        rows.append(row("had-gptq32-sym-g128-4", 4.125, p, k, seed=s, rotation="hadamard"))
    for s, k, p in ((0, 1.0, 60.0), (1, 1.05, 62.0)):
        rows.append(row("had-gptq32-sym-g128-2", 2.125, p, k, seed=s, rotation="hadamard"))
    for s, k, p in ((0, 0.215, 21.0), (1, 0.21, 20.9)):
        rows.append(row("rnd-gptq32-sym-g128-3", 3.125, p, k, seed=s, rotation="rnd"))
    rows.append(row("hadsign-gptq32-sym-g128-3", 3.125, 20.95, 0.212, rotation="hadsign"))
    rows.append(row("eig-gptq32-sym-g128-3", 4.4, 19.5, 0.15, rotation="eig"))
    # E codebooks
    rows.append(row("lloyd-gptq32-g128-4", 4.1252, 17.9, 0.058))
    for s, k, p in ((0, 0.25, 22.0), (1, 0.245, 21.9)):
        rows.append(row("lloyd-gptq32-g128-3", 3.1252, p, k, seed=s))
    rows.append(row("lloyd-gptq32-g128-2", 2.1252, 100.0, 1.6))
    rows.append(row("nf-gptq32-g128-4", 4.125, 17.95, 0.062))
    rows.append(row("nf-gptq32-g128-3", 3.125, 22.5, 0.27))
    rows.append(row("nfq-gptq32-g128-4", 4.125, 18.3, 0.075, huff=4.125, shannon=4.12))
    rows.append(row("nfq-gptq32-g128-3", 3.125, 26.0, 0.40, huff=3.125, shannon=3.12))
    # A allocation
    for s, k, p in ((0, 0.24, 22.0), (1, 0.29, 23.2)):  # seed 1 fails the 5% rule
        rows.append(row("alloc-type-gptq32-avg3", 3.13, p, k, seed=s, alloc="alloc type: q=4 k=5 v=5 o=4 gate=3 up=3 down=3 (avg 3)"))
    for s, k, p in ((0, 0.11, 19.2), (1, 0.108, 19.1)):
        rows.append(row("alloc-type-gptq32-avg3.5", 3.63, p, k, seed=s, alloc="alloc type: ... (avg 3.5)"))
    for s, k, p in ((0, 0.7, 40.0), (1, 0.72, 41.0)):
        rows.append(row("alloc-type-gptq32-avg2.5", 2.63, p, k, seed=s, alloc="alloc type: ... (avg 2.5)"))
    for s, k, p in ((0, 0.06, 17.95), (1, 0.058, 17.9)):
        rows.append(row("alloc-type-gptq32-avg4", 4.13, p, k, seed=s, alloc="alloc type: ... (avg 4)"))
    rows.append(row("alloc-tensor-gptq32-avg3", 3.14, 21.8, 0.23, alloc="results/phase3/m/alloc_alloc-tensor-gptq32-avg3_seed0.json"))
    rows.append(row("alloc-tensor-gptq32-avg3.5", 3.64, 19.0, 0.105, alloc="results/phase3/m/alloc_alloc-tensor-gptq32-avg3.5_seed0.json"))
    rows.append(row("alloc-type-rtn-avg3", 3.13, 50.0, 1.1, huff=None, shannon=None, calib=False, alloc="alloc type: ... (avg 3)"))
    rows.append(row("alloc-type-rtn-avg3.5", 3.63, 30.0, 0.5, huff=None, shannon=None, calib=False, alloc="alloc type: ... (avg 3.5)"))
    rows.append(row("alloc-type-shuf1-gptq32-avg3", 3.13, 23.5, 0.30, alloc="alloc type shuffled seed 1: ..."))
    rows.append(row("alloc-type-shuf2-gptq32-avg3", 3.13, 23.6, 0.31, alloc="alloc type shuffled seed 2: ..."))
    # G row allocation
    rows.append(row("rowalloc-fisher-rtn-avg4", 4.14, 21.5, 0.17, huff=None, shannon=None, calib=False, alloc="rowalloc fisher: b-1/b/b+1 rows"))
    rows.append(row("rowalloc-shuf1-rtn-avg4", 4.14, 22.6, 0.20, huff=None, shannon=None, calib=False, alloc="rowalloc shuffled seed 1"))
    rows.append(row("rowalloc-shuf2-rtn-avg4", 4.14, 22.7, 0.205, huff=None, shannon=None, calib=False, alloc="rowalloc shuffled seed 2"))
    for s, k, p in ((0, 0.062, 17.95), (1, 0.060, 17.9)):
        rows.append(row("rowalloc-fisher-gptq32-avg4", 4.14, p, k, seed=s, alloc="rowalloc fisher: ..."))
    for s, k, p in ((0, 0.27, 22.8), (1, 0.265, 22.7)):
        rows.append(row("rowalloc-fisher-gptq32-avg3", 3.14, p, k, seed=s, alloc="rowalloc fisher: ..."))
    # combination
    for s, k, p in ((0, 0.07, 18.1), (1, 0.069, 18.05)):
        rows.append(row("combo-gptq32-avg3", 3.13, p, k, seed=s, huff=2.75 + 0.01 * s, shannon=2.72 + 0.01 * s, rotation="hadamard", alloc="alloc type: ... (avg 3)"))
    for s, k, p in ((0, 0.2, 21.0), (1, 0.21, 21.1)):
        rows.append(row("combo-gptq32-avg2.5", 2.63, p, k, seed=s, huff=2.3, shannon=2.27, rotation="hadamard", alloc="alloc type: ... (avg 2.5)"))
    for s, k, p in ((0, 0.04, 17.8), (1, 0.041, 17.82)):
        rows.append(row("combo-gptq32-avg3.5", 3.63, p, k, seed=s, huff=3.2, shannon=3.17, rotation="hadamard", alloc="alloc type: ... (avg 3.5)"))
    # a smoke row that must be ignored
    smoke = row("gptq32-sym-g128-3", 3.125, 99.0, 9.0, note="limit-windows 8; kl-windows 8; calib-n 16; calib 16x2048 seed 0; gptq 3b")
    rows.append(smoke)
    return rows


def phase1_rows() -> list[dict]:
    r = fx.runs_rows(M) if M in fx.PHASE1 else []
    rows = [fx._row(M, "fp16", 16.0, FP16, 0.0, note="bf16, not quantized"),
            fx._row(M, "rtn-sym-g128-4", 4.125, 22.6658, 0.20514, note="rtn-sym-g128-4: f16 scale"),
            fx._row(M, "rtn-sym-g128-3", 3.125, 60.0, 1.50528, note="rtn-sym-g128-3: f16 scale"),
            fx._row(M, "rtn-sym-g128-2", 2.125, 900.0, 11.06255, note="rtn-sym-g128-2: f16 scale"),
            fx._row(M, "gptq-sym-g128-4", 4.125, 18.0573, 0.07125, calib_seed=0, note="calib 128x2048 seed 0; gptq 4b sym g128 block128 damp0.01"),
            fx._row(M, "gptq-sym-g128-3", 3.125, 24.4482, 0.31305, calib_seed=0, note="calib 128x2048 seed 0; gptq 3b sym g128 block128 damp0.01"),
            fx._row(M, "gptq-sym-g128-3", 3.125, 23.4005, 0.29937, seed=1, calib_seed=1, note="calib 128x2048 seed 1; gptq 3b sym g128 block128 damp0.01"),
            fx._row(M, "gptq-sym-g128-2", 2.125, 132.5722, 1.91236, calib_seed=0, note="calib 128x2048 seed 0; gptq 2b sym g128 block128 damp0.01")]
    return rows + r


TYPES = ["q", "k", "v", "o", "gate", "up", "down", "in_proj_qkv", "in_proj_z", "in_proj_ab", "out_proj"]
WIKI_ONLY_KL = {"q": 0.00637, "k": 0.00308, "v": 0.01017, "o": 0.02524, "gate": 0.01428, "up": 0.02911, "down": 0.03200,
                "in_proj_qkv": 0.01731, "in_proj_z": 0.00886, "in_proj_ab": 0.00191, "out_proj": 0.04739}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("p3")
    rdir, p1dir, probe = root / "results" / "phase3", root / "results", root / "results" / "probe"
    write_jsonl(rdir / M / "runs.jsonl", phase3_rows())
    write_jsonl(p1dir / M / "runs.jsonl", phase1_rows())
    # Phase 2 @only rows (WikiText) and the C4 damage of A1, with the same ranking except one swap
    write_jsonl(probe / M / "ablation.jsonl", [fx._row(M, f"rtn-sym-g128-4@only={t}", 15.0, 17.6, k, note="rtn-sym-g128-4: f16 scale")
                                                for t, k in WIKI_ONLY_KL.items()])
    c4 = {t: k * 1.3 for t, k in WIKI_ONLY_KL.items()}
    c4["gate"], c4["in_proj_qkv"] = c4["in_proj_qkv"], c4["gate"]
    write_jsonl(rdir / M / "damage_c4.jsonl", [{"unit": "type", "name": t, "numel": 1000 * (i + 1), "kl": k, "ppl": 17.6, "n_tokens": 65504,
                                                "c4_set": alloc_mod.DAMAGE_C4_SET, "windows": alloc_mod.DAMAGE_WINDOWS}
                                               for i, (t, k) in enumerate(c4.items())])
    (rdir / M / "combo.json").write_text(json.dumps({"had": True, "lloyd": True, "rowalloc": False}), encoding="utf-8")
    # alloc JSONs next to the rows (wcomp.methods._write_alloc_json): shuffle 1 is a valid control, shuffle 2 collapsed to the
    # real allocation (the defect the alloc.py fix rejects; the summary must still flag an old file), tensor rows say their signal
    real_bits = {"q": 4, "k": 5, "v": 5, "o": 4, "gate": 3, "up": 3, "down": 3}
    shuf1 = {"q": 3, "k": 3, "v": 4, "o": 5, "gate": 4, "up": 3, "down": 3}
    (rdir / M / "alloc_alloc-type-shuf1-gptq32-avg3_seed0.json").write_text(json.dumps(
        {"unit": "type", "bits": shuf1, "real_bits": real_bits, "shuffle": {"params_share_at_real_width": 0.2, "max_real_share": 0.5}}), encoding="utf-8")
    (rdir / M / "alloc_alloc-type-shuf2-gptq32-avg3_seed0.json").write_text(json.dumps(
        {"unit": "type", "bits": real_bits, "real_bits": real_bits, "shuffle": {"params_share_at_real_width": 1.0, "max_real_share": 0.5}}), encoding="utf-8")
    (rdir / M / "alloc_alloc-tensor-gptq32-avg3_seed0.json").write_text(json.dumps(
        {"unit": "tensor", "signal": "damage", "source": "d = KL/numel per tensor from damage_c4.jsonl"}), encoding="utf-8")
    (rdir / M / "alloc_alloc-tensor-gptq32-avg3.5_seed0.json").write_text(json.dumps(
        {"unit": "tensor", "source": "no per-tensor damage rows: d = Fisher total/numel per tensor from fisher.json"}), encoding="utf-8")
    rd = []  # wcomp.rdbound.rd_bound_rows schema: one row per weight type, prefixed with the run's identity
    for method, base in (("gptq32-sym-g128-4", 1.3), ("lloyd-gptq32-g128-4", 1.1)):
        for i, t in enumerate(("q", "down", "out_proj")):
            r = {"model": M, "method": method, "seed": 0, "calib_seed": 0, "bits_blocks": 4.125, "bits_blocks_huff": 3.705,
                 "bits_blocks_shannon": 3.675, "timestamp": fx.TS, "type": t, "D": 0.01 * (1 + i),
                 "bits_fixed": 4.125, "bits_huff": 3.7, "bits_shannon": 3.66, "numel": 1000, "n_tensors": 6}
            if t != "down":  # the down row carries D + bits only: the reader derives the gaps
                r.update(gap_fixed=base + 0.1 * i, gap_huff=base - 0.3 + 0.1 * i, gap_shannon=base - 0.34 + 0.1 * i)
            rd.append(r)
    write_jsonl(rdir / M / "rd_bound.jsonl", rd)
    return {"root": root, "rdir": rdir, "p1dir": p1dir, "probe": probe}


# ---------------------------------------------------------------- row schema / runner fields


def test_row_fields_extend_the_phase1_schema():
    assert rb.ROW_FIELDS[:6] == ("model", "method", "seed", "calib_seed", "bits_blocks", "bits_all")
    assert {"bits_blocks_huff", "bits_blocks_shannon", "alloc", "rotation"} <= set(rb.ROW_FIELDS)
    assert tuple(fx._row(M, "fp16", 16.0, FP16, 0.0)) == rb.ROW_FIELDS


def test_make_row_phase3_fields():
    lin = nn.Linear(256, 8, bias=False)
    acct = BitAccount()
    acct.add("model.layers.0.q", lin.weight.numel(), 4 * lin.weight.numel(), 16 * 8 * 2)
    metrics = {"ppl": 15.0, "nll": 2.7, "n_tokens": 100, "kl": 0.1, "top1_agree": 0.9, "kl_tokens": 50}
    notes = ["calib 128x2048 seed 0", "gptq 3b sym g128 block128 damp0.01 hessian-f32", "input rotation hadamard(1024,2048,3584)"]
    r = rb.make_row(M, "had-gptq32-sym-g128-3", 0, 0, acct, lin, metrics, 1.0, 0.5, notes, "v", "env")
    assert tuple(r) == rb.ROW_FIELDS
    assert r["bits_blocks_huff"] is None and r["bits_blocks_shannon"] is None and r["alloc"] is None and r["rotation"] == "hadamard"
    r = rb.make_row(M, "combo-gptq32-avg3", 0, 0, acct, lin, metrics, 1.0, 0.5, notes + ["alloc type: q=5 k=6 (avg 3)"], "v", "env",
                    coded=(2.75, 2.72), alloc=rb.alloc_of(notes + ["alloc type: q=5 k=6 (avg 3)"]))
    assert r["bits_blocks_huff"] == 2.75 and r["bits_blocks_shannon"] == 2.72 and r["alloc"] == "alloc type: q=5 k=6 (avg 3)"
    json.dumps(r, allow_nan=False)


def test_rotation_and_alloc_from_notes(tmp_path):
    assert rb.rotation_of(["rtn-sym-g128-4: f16 scale"]) == "none"
    assert rb.rotation_of(["gptq 3b; input rotation hadamard(1024,2048,3584)"]) == "hadamard"
    assert rb.rotation_of(["input rotation hadamard(1024) + random_orthogonal(3584, seed=0)"]) == "hadamard"
    assert rb.rotation_of(["input rotation random_orthogonal(1024,2048,3584, seed=1)"]) == "rnd"
    assert rb.rotation_of(["input rotation rnd(1024)"]) == "rnd"
    assert rb.rotation_of(["input rotation hadsign(1024,2048,3584, seed=0)"]) == "hadsign"
    assert rb.rotation_of(["input rotation eigenbasis of the calibration Hessian, bf16 stored"]) == "eig"
    assert rb.rotation_of(["rotation eig(1024) stored bf16; hadamard fallback none"]) == "eig"
    # words are matched exactly, and the fixed token of wcomp.methods.rotated wins over any wording
    assert rb.rotation_of(["eight bf16 matrices stored"]) == "none"
    assert rb.rotation_of(["results/eigx/alloc.json"]) == "none"
    assert rb.rotation_of(["gptq 3b; rotation=eig; some other words hadamard"]) == "eig"
    assert rb.rotation_of(["gptq 3b; rotation=rnd; input rotation random_orthogonal(1024, seed=0)"]) == "rnd"
    assert rb.rotation_of(["rotation=hadsign; input rotation hadsign(1024, seed=0)"]) == "hadsign"
    assert rb.alloc_of(["gptq 3b", "alloc type: q=5 k=6 v=6 (avg 3)"]) == "alloc type: q=5 k=6 v=6 (avg 3)"
    assert rb.alloc_of(["rowalloc fisher: rows at 3/4/5 bits, 2 bits side info per row"]).startswith("rowalloc fisher")
    assert rb.alloc_of(["rtn-sym-g128-4: f16 scale"]) is None
    assert rb.alloc_of(["x"], tmp_path, "alloc-tensor-gptq32-avg3", 0) is None
    p = tmp_path / "alloc_alloc-tensor-gptq32-avg3_seed0.json"
    p.write_text("{}", encoding="utf-8")
    assert rb.alloc_of(["x"], tmp_path, "alloc-tensor-gptq32-avg3", 0) == p.as_posix()


def test_coded_figures_and_rd_bound_hook(monkeypatch, tmp_path):
    acct = BitAccount()
    acct.add("a", 8, 32, 0, "plain")
    assert not rb.has_indices(acct) and rb.coded_figures(acct) == (None, None, None) and rb.coded_detail(acct) is None
    acct.add("b", 8, 32, 16, "coded", indices=torch.tensor([0, 1, 2, 3, 3, 3, 3, 3]), levels=16)
    assert rb.has_indices(acct)
    # the real modules: one detail pass feeds both figures and the rd rows, entropy0 >= shannon
    detail = rb.coded_detail(acct)
    assert [r["name"] for r in detail] == ["a", "b"]
    huff, shannon, entropy0 = rb.coded_figures(acct, detail)
    assert rb.coded_figures(acct) == (huff, shannon, entropy0) and shannon <= entropy0 and huff > 0
    # without wcomp.coding the figures stay None (and nothing raises)
    monkeypatch.setitem(sys.modules, "wcomp.coding", None)
    assert rb.coded_figures(acct) == (None, None, None)
    monkeypatch.setitem(sys.modules, "wcomp.rdbound", None)
    assert rb.rd_bound_rows_for(acct, nn.Linear(2, 2), nn.Linear(2, 2), {"method": "x"}) is None
    # a failure inside either module is logged and swallowed: the row survives with None figures / no rd rows
    def boom(*a, **k):
        raise KeyError("ambiguous checkpoint keys")

    monkeypatch.setitem(sys.modules, "wcomp.coding", types.SimpleNamespace(coded_bits=boom, coded_bits_detail=boom))
    monkeypatch.setitem(sys.modules, "wcomp.rdbound", types.SimpleNamespace(rd_bound_rows=boom))
    assert rb.coded_detail(acct) is None and rb.coded_figures(acct) == (None, None, None)
    assert rb.rd_bound_rows_for(acct, nn.Linear(2, 2), nn.Linear(2, 2), {"method": "x"}) is None
    # with the modules present their results are used (dict or pair)
    seen = {}

    def coded_bits(account, detail=None):
        seen["n"] = len(account.tensors)
        seen["detail"] = detail
        return {"bits_blocks_huff": 3.7, "bits_blocks_shannon": 3.66, "bits_blocks_entropy0": 3.68, "per_tensor": []}

    def rd_bound_rows(account, model, ref, row, detail=None):
        seen["rd_detail"] = detail
        return [{"method": row["method"], "seed": 0, "tensor": t.name, "type": "q", "mse": 0.01, "bits_fixed": 4.0, "bits_huff": 3.7,
                 "bits_shannon": 3.66, "above_fixed": 0.68, "above_huff": 0.38, "above_shannon": 0.34} for t in account.tensors]

    monkeypatch.setitem(sys.modules, "wcomp.coding", types.SimpleNamespace(coded_bits=coded_bits))
    monkeypatch.setitem(sys.modules, "wcomp.rdbound", types.SimpleNamespace(rd_bound_rows=rd_bound_rows))
    assert rb.coded_figures(acct) == (3.7, 3.66, 3.68) and seen["n"] == 2 and seen["detail"] is None
    assert rb.coded_figures(acct, detail) == (3.7, 3.66, 3.68) and seen["detail"] is detail
    monkeypatch.setitem(sys.modules, "wcomp.coding", types.SimpleNamespace(coded_bits=lambda a: (3.5, 3.4)))
    assert rb.coded_figures(acct) == (3.5, 3.4, None)
    rows = rb.rd_bound_rows_for(acct, nn.Linear(2, 2), nn.Linear(2, 2), {"method": "gptq32-sym-g128-4"}, detail)
    assert len(rows) == 2 and rows[0]["method"] == "gptq32-sym-g128-4" and seen["rd_detail"] is detail
    rb.append_jsonl(tmp_path / "rd_bound.jsonl", rows)
    assert len(ps.load_rd_bound(tmp_path / "rd_bound.jsonl")) == 2
    # make_row carries the third figure and accepts the Phase 3 2-tuple
    lin = nn.Linear(256, 8, bias=False)
    metrics = {"ppl": 15.0, "nll": 2.7, "n_tokens": 100, "kl": 0.1, "top1_agree": 0.9, "kl_tokens": 50}
    r = rb.make_row(M, "gptq32-sym-g128-4", 0, 0, acct, lin, metrics, 1.0, 0.5, ["x"], "v", "env", coded=(3.7, 3.66, 3.68))
    assert r["bits_blocks_entropy0"] == 3.68 and tuple(r) == rb.ROW_FIELDS
    r = rb.make_row(M, "gptq32-sym-g128-4", 0, 0, acct, lin, metrics, 1.0, 0.5, ["x"], "v", "env", coded=(3.7, 3.66))
    assert r["bits_blocks_entropy0"] is None


def test_phase1_gptq_names_keep_the_flag_logic_and_new_names_need_none():
    assert rb.hessian_choice_required(["gptq-sym-g128-3"], "results/probe")
    assert not rb.hessian_choice_required(["gptq32-sym-g128-3"], "results/phase3")
    assert not rb.hessian_choice_required(["alloc-type-gptq32-avg3", "combo-gptq32-avg3"], "results/phase3")
    assert rb.planned_results_dir("results/phase3", ["gptq32-sym-g128-3"], None, 32, 128) == "results/phase3"
    assert rb.planned_results_dir("results/phase3", ["gptq32-sym-g128-3"], 8, 8, 128) == str(Path("results/phase3") / "smoke")
    from wcomp.methods import needs_calib

    assert needs_calib("gptq32-sym-g128-3") and needs_calib("alloc-type-gptq32-avg3") and needs_calib("combo-gptq32-avg3")
    assert not needs_calib("rowalloc-fisher-rtn-avg4") and not needs_calib("alloc-type-rtn-avg3")


# ---------------------------------------------------------------- curve arithmetic


XS, YS = [2.125, 3.125, 4.125], [1.925, 0.295, 0.068]


def test_interp_log_returns_endpoints_exactly_and_is_log_linear_between():
    for x, y in zip(XS, YS):
        assert p3.interp_log(XS, YS, x) == y
    mid = p3.interp_log(XS, YS, 2.625)
    assert mid == pytest.approx(math.exp((math.log(1.925) + math.log(0.295)) / 2), rel=1e-12)
    assert p3.interp_log(XS, YS, 3.625) == pytest.approx(math.sqrt(0.295 * 0.068), rel=1e-12)
    # not linear in y: the arithmetic midpoint is not the answer
    assert mid != pytest.approx((1.925 + 0.295) / 2, rel=1e-3)
    # extrapolation extends the end segment
    assert p3.interp_log(XS, YS, 4.625) == pytest.approx(0.068 * math.sqrt(0.068 / 0.295), rel=1e-12)
    assert p3.in_range(XS, 3.0) and not p3.in_range(XS, 4.5)


def test_invert_log_round_trips_and_hits_points_exactly():
    for x, y in zip(XS, YS):
        assert p3.invert_log(XS, YS, y) == x
    for x in (2.3, 3.0, 3.9):
        assert p3.invert_log(XS, YS, p3.interp_log(XS, YS, x)) == pytest.approx(x, abs=1e-12)
    assert p3.invert_log(XS, YS, 0.034) == pytest.approx(4.125 + math.log(0.034 / 0.068) / math.log(0.068 / 0.295), abs=1e-12)


def test_reference_curve_and_bits_saved_on_the_curve_is_zero():
    rows = [r for r in phase3_rows() if r["method"].startswith("gptq32-sym-g128-") and not plot_is_smoke(r)]
    curve = p3.reference_curve(rows, FP16)
    assert curve is not None and curve.usable()
    assert curve.bits == [2.125, 3.125, 4.125]
    assert curve.kls == pytest.approx([1.925, 0.295, 0.068])
    assert [p["n_seeds"] for p in curve.points] == [2, 2, 2]
    assert curve.bits_huff == pytest.approx([1.705, 2.705, 3.705])
    # a row exactly on a curve point: ratio 1, saved 0 exactly, not beating
    on = row("x-gptq32-sym-g128-3", 3.125, FP16 + curve.points[1]["dppl"], 0.295, huff=2.705)
    c = p3.compare(on, curve, FP16)
    assert c["ratio_kl"] == pytest.approx(1.0) and c["saved_fixed"] == 0.0 and c["saved_huff"] == pytest.approx(0.0, abs=1e-12)
    assert c["beats_row"] is False and not c["extrapolated"]
    # a row on the interpolated curve between points
    kl = p3.interp_log(curve.bits, curve.kls, 3.5)
    c = p3.compare(row("x", 3.5, FP16 + p3.interp_log(curve.bits, curve.dppls, 3.5), kl, huff=None), curve, FP16)
    assert c["ratio_kl"] == pytest.approx(1.0) and c["saved_fixed"] == pytest.approx(0.0, abs=1e-9) and c["saved_huff"] is None
    # a row below the curve: saved > 0, beats when both margins clear the noise
    c = p3.compare(row("had-gptq32-sym-g128-3", 3.125, 20.9, 0.21), curve, FP16)
    assert c["ratio_kl"] == pytest.approx(0.21 / 0.295) and c["saved_fixed"] > 0 and c["beats_row"] is True
    assert p3.method_verdict([c, c]) == "yes" and p3.method_verdict([c]) == "yes (1 seed)"
    bad = dict(c, beats_row=False)
    assert p3.method_verdict([c, bad]) == "no (1 of 2 seeds)" and p3.method_verdict([bad]) == "no"
    assert p3.method_verdict([dict(c, beats_row=None)]) == "n/a" and p3.method_verdict([], "reference") == "ref"


def plot_is_smoke(r: dict) -> bool:
    import plot_rd

    return plot_rd.is_smoke(r)


def test_parse_name_labels():
    cases = {
        "gptq32-sym-g128-3": ("reference", "3"), "had-gptq32-sym-g128-2": ("rotation", "had 2"),
        "rnd-gptq32-sym-g128-3": ("rotation", "rnd 3"), "hadsign-gptq32-sym-g128-3": ("rotation", "hadsign 3"),
        "eig-gptq32-sym-g128-3": ("rotation", "eig 3"), "lloyd-gptq32-g128-4": ("codebook", "lloyd 4"),
        "nf-gptq32-g128-3": ("codebook", "nf 3"), "alloc-type-gptq32-avg2.5": ("alloc", "type 2.5"),
        "alloc-tensor-gptq32-avg3.5": ("alloc", "tensor 3.5"), "alloc-type-rtn-avg3": ("alloc", "type rtn 3"),
        "alloc-type-shuf1-gptq32-avg3": ("alloc", "shuf1 3"), "rowalloc-fisher-rtn-avg4": ("rowalloc", "fisher rtn 4"),
        "rowalloc-shuf2-rtn-avg4": ("rowalloc", "shuf2 rtn 4"), "rowalloc-fisher-gptq32-avg3": ("rowalloc", "fisher 3"),
        "combo-gptq32-avg3": ("combo", "avg 3"), "rtn-sym-g128-4": ("other", "rtn-sym-g128-4"),
    }
    for name, (fam, lab) in cases.items():
        p = p3.parse_name(name)
        assert (p["family"], p["label"]) == (fam, lab), name
    assert p3.parse_name("alloc-type-shuf1-gptq32-avg3")["shuffle"] and not p3.parse_name("alloc-type-gptq32-avg3")["shuffle"]
    assert p3.parse_name("alloc-type-gptq32-avg2.5")["avg"] == 2.5 and p3.parse_name("had-gptq32-sym-g128-2")["bits"] == 2
    assert p3.row_label({"method": "combo-gptq32-avg3", "seed": 1}) == "avg 3 s1"
    assert p3.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0) and p3.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


# ---------------------------------------------------------------- summary


def test_summary_rows_and_verdicts(tree):
    text = ps.build_summary([M], tree["rdir"], tree["p1dir"], tree["probe"])
    lines = text.splitlines()
    assert f"## {M}" in lines and "### Rows" in lines and "### Hypotheses" in lines
    assert any(l.startswith("| 4.1250 | 3.7050 |") for l in lines)  # reference curve point, seed mean of the huff bits

    def table_row(method: str, seed: int = 0) -> list[str]:
        for l in lines:
            cells = [c.strip() for c in l.strip("|").split("|")]
            if len(cells) >= 14 and cells[0] == method and cells[1] == str(seed):
                return cells
        raise AssertionError(f"no table row for {method} seed {seed}")

    assert table_row("gptq32-sym-g128-3")[11] == "ref"
    assert table_row("had-gptq32-sym-g128-3")[11] == "yes" and table_row("had-gptq32-sym-g128-3", 1)[11] == "yes"
    assert table_row("had-gptq32-sym-g128-4")[11] == "no (1 of 2 seeds)"
    assert table_row("alloc-type-gptq32-avg3")[11] == "no (1 of 2 seeds)" and table_row("alloc-type-gptq32-avg3.5")[11] == "yes"
    assert table_row("lloyd-gptq32-g128-4")[11] == "yes (1 seed)"
    assert table_row("alloc-type-shuf1-gptq32-avg3")[11] == "no"
    # RTN-quantized rows are not judged against the gptq32 curve (their gates are the RTN lines of H3-A / H3-G)
    assert table_row("alloc-type-rtn-avg3")[11] == "rtn gate" and table_row("rowalloc-fisher-rtn-avg4")[11] == "rtn gate"
    assert table_row("rowalloc-shuf1-rtn-avg4")[11] == "rtn gate" and table_row("rowalloc-fisher-gptq32-avg4")[11] != "rtn gate"
    assert "\"rtn gate\" for RTN-quantized rows" in text and p3.DPPL_RULE_NOTE in text
    # the reference rows' seed spread per width, so the two margins can be read against the spread of what they apply to
    assert "seed spread at 3.1250 bits (seeds 0,1, max / min - 1): KL 3.4%, PPL 0.4%, dPPL 1.7%" in text
    assert "seed spread at 4.1250 bits (seeds 0,1, max / min - 1): KL 6.1%, PPL 2.1%, dPPL 77.2%" in text
    r = table_row("had-gptq32-sym-g128-3")
    assert r[9] == "0.29500" and r[10] == "0.712" and r[12].startswith("+") and r[13].startswith("+")
    r = table_row("alloc-type-rtn-avg3")
    assert r[3] == "—" and r[4] == "—" and r[13] == "—"  # no index stream: no Huffman figures, no Huffman saving
    r = table_row("eig-gptq32-sym-g128-3")
    assert r[9].endswith("*") and r[12].endswith("—") is False  # curve read outside the measured range is flagged
    assert "| 99.00 |" not in text  # the smoke row is ignored
    # hypotheses numbers
    assert "**H3-A**" in text and "verdict avg 3.5: yes" in text and "verdict avg 3: no (1 of 2 seeds)" in text
    assert "real / shuffle KL = 0.800" in text and "Spearman(C4 damage per type, Phase 2 WikiText @only KL at 4 bits) = 0.9" in text
    assert "tensor / type = 0.958" in text
    # shuffle controls: the share of the parameters at their real width is printed, a collapsed control is marked invalid
    shuf = [l for l in lines if l.strip().startswith("- shuffle ")]
    assert len(shuf) == 2
    assert "control share at real width 0.200" in shuf[0] and "CONTROL INVALID" not in shuf[0] and "and a valid control: True" in shuf[0]
    assert "control share at real width 1.000 CONTROL INVALID (identical to the real allocation" in shuf[1] and "and a valid control: False" in shuf[1]
    # alloc-tensor rows say which signal they used
    tv = [l for l in lines if l.strip().startswith("- tensor vs type")]
    assert len(tv) == 2 and "signal: damage" in tv[0] and "signal: fisher" in tv[1]
    # H3-E reads the gap on both definitions: the chosen-coder bound and the contract's order-0 entropy
    assert "huff - shannon gap 0.030, huff - entropy0 gap 0.029 (< 0.05: True)" in text
    assert "nfq (DESIGN.md E3 quantiles) 3:" in text and "nfq (DESIGN.md E3 quantiles) 4:" in text
    assert "**H3-D**" in text and "verdict had 3: yes" in text and "verdict had 4: no (1 of 2 seeds)" in text
    assert "rnd / had (same seed) = 1.024" in text and "stored rotation = 1.275 bits/param over 3.125; > 1: True" in text
    assert "**H3-E**" in text and "saving 0.420 bits/param, huff - shannon gap 0.030" in text and "verdict lloyd 3 (beats: KL and dPPL): yes" in text
    assert "nf share of the lloyd gain =" in text
    assert "RD bound gptq32 4 (numel-weighted over the types): above bound fixed 1.37, huff 1.03, shannon 0.99 bits" in text
    assert "**H3-G**" in text and "gate: **passed**" in text and "ratio 0.829" in text
    assert "**H3-combo**" in text and '"had": true' in text and "reaches it, within 5%: True" in text and "dPPL < +50% on every seed: True" in text
    # RD-bound table: one row per method, the per-type medians, the derived "down" cells
    assert "### RD bound (E4)" in text
    rd_lines = lines[lines.index("### RD bound (E4), bits above the Gaussian bound"):]
    rd_line = next(l for l in rd_lines if l.startswith("| gptq32-sym-g128-4 | 0 |"))
    cells = [c.strip() for c in rd_line.strip("|").split("|")]
    header = next(l for l in rd_lines if l.startswith("| method | seed | q |"))
    hcells = [c.strip() for c in header.strip("|").split("|")]
    assert hcells[2:] == ["q", "down", "out_proj", "all types"]
    assert cells[2] == "1.30 / 1.00 / 0.96"  # the q row's own gaps
    derived = 4.125 - 0.5 * math.log2(1 / 0.02)  # the down row: derived from D and the bit figures
    assert cells[3] == f"{derived:.2f} / {3.7 - 0.5 * math.log2(1 / 0.02):.2f} / {3.66 - 0.5 * math.log2(1 / 0.02):.2f}"
    assert cells[5].split(" / ")[0] == f"{(1.3 + derived + 1.5) / 3:.2f}"  # equal numel: the plain mean over the three types
    # the real runner output of the smoke check (per-type schema) is readable too
    real = ROOT / "results" / "smoke" / "qwen3.5-0.8b" / "rd_bound.jsonl"
    if real.exists():
        rows = ps.load_rd_bound(real)
        assert rows and all(r.get("gap_fixed") is not None and r.get("type") for r in rows)


def test_summary_reference_rows_carry_no_ratio_and_rtn_gate_is_computed(tree):
    text = ps.build_summary([M], tree["rdir"], tree["p1dir"], tree["probe"])
    lines = text.splitlines()
    ref = next(l for l in lines if l.startswith("| gptq32-sym-g128-4 | 0 |"))
    cells = [c.strip() for c in ref.strip("|").split("|")]
    assert cells[10] == "—" and cells[11] == "ref" and cells[12] == "—" and cells[13] == "—"  # seed spread is not a result
    had = [c.strip() for c in next(l for l in lines if l.startswith("| had-gptq32-sym-g128-3 | 0 |")).strip("|").split("|")]
    assert had[10] == "0.712"
    # the RTN gate: alloc-type-rtn-avg{3,3.5} against the uniform rtn-sym-g128-{4,3,2} curve of the Phase 1 file
    rtn = [l for l in lines if l.strip().startswith("- RTN gate avg")]
    assert len(rtn) == 2 and "points at 2.125, 3.125, 4.125 bits" in rtn[0]
    # alloc-type-rtn-avg3: KL 1.1 at 3.13 bits vs the curve through (3.125, 1.50528) and (4.125, 0.20514)
    ref3 = p3.interp_log([2.125, 3.125, 4.125], [11.06255, 1.50528, 0.20514], 3.13)
    assert f"ratio {1.1 / ref3:.3f}" in rtn[0] and ("better than uniform RTN by > 5%: True" in rtn[0]) == (1.1 / ref3 < 0.95)
    # the combination gates are printed as such, next to the two-sided verdicts
    assert "had combo gate (= beats at 3 bits, every seed): **yes**" in text
    assert "lloyd combo gate (KL-only, > 5% below the curve on every seed): **passed**" in text
    assert "verdict lloyd 3 (beats: KL and dPPL): yes" in text
    assert "nf (Gaussian Lloyd-Max optimum) 3:" in text and "nfq = DESIGN.md E3 as written" in text
    assert "not the gap to the order-0 entropy" in text


def test_summary_lloyd_gate_is_kl_only(tmp_path):
    """lloyd-3 with KL 7% below the curve but dPPL only 2% below: the two-sided verdict says no, the contract's combo
    gate (KL-only, both seeds) passes, and the summary prints both."""
    rows = [r for r in phase3_rows() if r["method"].startswith("gptq32-sym-g128-") and not plot_is_smoke(r)]
    curve = p3.reference_curve(rows, FP16)
    kl3, dppl3 = curve.kl_at(3.1252), curve.dppl_at(3.1252)
    for s in (0, 1):
        rows.append(row("lloyd-gptq32-g128-3", 3.1252, FP16 + 0.98 * dppl3, 0.93 * kl3, seed=s))
    rows.append(fx._row(M, "fp16", 16.0, FP16, 0.0, note="bf16, not quantized"))
    write_jsonl(tmp_path / "phase3" / M / "runs.jsonl", rows)
    text = ps.build_summary([M], tmp_path / "phase3", tmp_path / "results", tmp_path / "probe")
    assert "verdict lloyd 3 (beats: KL and dPPL): no" in text
    assert "lloyd combo gate (KL-only, > 5% below the curve on every seed): **passed**" in text
    cmps = [p3.compare(r, curve, FP16) for r in rows if r["method"] == "lloyd-gptq32-g128-3"]
    assert p3.kl_gate(cmps) == "passed" and p3.method_verdict(cmps) == "no"
    assert p3.kl_gate([dict(c, ratio_kl=0.96) for c in cmps]) == "failed" and p3.kl_gate([dict(c, ratio_kl=None) for c in cmps]) == "incomplete"


def test_summary_tolerates_missing_files(tmp_path):
    rdir, p1dir = tmp_path / "phase3", tmp_path / "results"
    text = ps.build_summary(["nothing"], rdir, p1dir, tmp_path / "probe")
    assert "no Phase 3 rows" in text
    # rows but no reference curve and no fp16 in the Phase 3 file: fp16 from Phase 1, hypotheses say what is missing
    write_jsonl(rdir / M / "runs.jsonl", [row("had-gptq32-sym-g128-3", 3.125, 20.9, 0.21)])
    write_jsonl(p1dir / M / "runs.jsonl", [fx._row(M, "fp16", 16.0, FP16, 0.0, note="bf16, not quantized")])
    text = ps.build_summary([M], rdir, p1dir, tmp_path / "probe")
    assert "reference curve: n/a" in text and "| had-gptq32-sym-g128-3 | 0 |" in text and "n/a (missing:" in text
    assert "RD bound: n/a (missing: rd_bound.jsonl)" in text and "components: n/a (missing: combo.json)" in text
    assert "gate: **" not in text  # no gate verdict without the gate rows


def test_summary_script_writes_the_file(tree):
    out = tree["root"] / "summary.md"
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "phase3_summary.py"), "--models", M, "--results-dir", str(tree["rdir"]),
           "--phase1-dir", str(tree["p1dir"]), "--probe-dir", str(tree["probe"]), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert out.exists() and "# Phase 3 summary" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------- plot


def test_plot_rd3_renders(tree):
    out = tree["root"] / "rd3_m.png"
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "plot_rd3.py"), "--model", M, "--results-dir", str(tree["rdir"]),
           "--phase1-dir", str(tree["p1dir"]), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 20_000
    assert "smoke" in r.stderr  # the smoke reference row was reported and ignored
    assert "not on the log axis" not in r.stderr


def test_plot_rd3_x_range_ignores_the_stored_rotation_probe(tmp_path):
    """The eig row at ~17.9 bits must not stretch the axis: the x range comes from the rows at <= 8 bits and the curve, and
    the probe is drawn at the right edge with its bits in the label (reported on stderr)."""
    import plot_rd3

    rows = [r for r in phase3_rows() if not plot_is_smoke(r) and r["method"] != "eig-gptq32-sym-g128-3"]
    eig = row("eig-gptq32-sym-g128-3", 17.8923, 34.7, 0.58, huff=17.4635, shannon=17.4446, rotation="eig")
    curve = p3.reference_curve(rows, FP16)
    methods = [r for r in rows + [eig] if r["method"] not in ("fp16", "had-fp16")]
    lo, hi = plot_rd3.x_range(methods, curve, [])
    assert lo == pytest.approx(min(curve.bits_huff) - plot_rd3.X_PAD)
    assert hi == pytest.approx(4.14 + plot_rd3.X_PAD) and hi < 5
    assert plot_rd3.x_range([eig], None, []) == (1.5, plot_rd3.MAX_PLOT_BITS + plot_rd3.X_PAD)
    write_jsonl(tmp_path / "phase3" / M / "runs.jsonl", rows + [eig])
    write_jsonl(tmp_path / "results" / M / "runs.jsonl", phase1_rows())
    out = tmp_path / "rd3.png"
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "plot_rd3.py"), "--model", M, "--results-dir", str(tmp_path / "phase3"),
           "--phase1-dir", str(tmp_path / "results"), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert out.exists() and "eig-gptq32-sym-g128-3 (seed 0): bits_blocks 17.89 beyond the x range, drawn at the right edge" in r.stderr


def test_plot_rd3_without_reference_or_phase1(tmp_path):
    rdir, p1dir = tmp_path / "phase3", tmp_path / "results"
    write_jsonl(rdir / M / "runs.jsonl", [fx._row(M, "fp16", 16.0, FP16, 0.0, note="bf16, not quantized"),
                                          row("had-gptq32-sym-g128-3", 3.125, 20.9, 0.21),
                                          row("lloyd-gptq32-g128-3", 3.125, FP16 - 0.1, 0.25)])  # dPPL <= 0: reported, skipped
    out = tmp_path / "x.png"
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "plot_rd3.py"), "--model", M, "--results-dir", str(rdir),
           "--phase1-dir", str(p1dir), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert out.exists() and "no gptq32-sym-g128-* reference rows" in r.stderr and "not on the log axis" in r.stderr
    # no fp16 anywhere: a clear error
    write_jsonl(rdir / M / "runs.jsonl", [row("had-gptq32-sym-g128-3", 3.125, 20.9, 0.21)])
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "fp16" in r.stderr


# ---------------------------------------------------------------- review fixes


def test_method_verdict_counts_rows_not_readable_flags():
    yes, no, na = {"beats_row": True}, {"beats_row": False}, {"beats_row": None}
    assert p3.method_verdict([yes, yes]) == "yes" and p3.method_verdict([yes]) == "yes (1 seed)"
    assert p3.method_verdict([yes, na]) == "n/a (1 of 2 seeds readable)"
    assert p3.method_verdict([no, na]) == "n/a (1 of 2 seeds readable)"
    assert p3.method_verdict([na, na]) == "n/a" and p3.method_verdict([yes, no]) == "no (1 of 2 seeds)" and p3.method_verdict([no, no]) == "no"
    # two seed rows, one with kl None: the table says why, not "yes (1 seed)"
    rows = [r for r in phase3_rows() if r["method"].startswith("gptq32-sym-g128-") and not plot_is_smoke(r)]
    curve = p3.reference_curve(rows, FP16)
    cmps = [p3.compare(row("had-gptq32-sym-g128-3", 3.125, 20.9, 0.21), curve, FP16),
            p3.compare(dict(row("had-gptq32-sym-g128-3", 3.125, 20.8, 0.205, seed=1), kl=None), curve, FP16)]
    assert cmps[0]["beats_row"] is True and cmps[1]["beats_row"] is None
    assert p3.method_verdict(cmps) == "n/a (1 of 2 seeds readable)"


def test_compare_flags_extrapolation_only_beyond_the_tolerance():
    """A rowalloc row at 4.1266 bits (2 bits/row side information over 4.125) is 0.0016 bits past the top reference point:
    no star; a row 0.075 bits past it is flagged."""
    rows = [r for r in phase3_rows() if r["method"].startswith("gptq32-sym-g128-") and not plot_is_smoke(r)]
    curve = p3.reference_curve(rows, FP16)
    near = p3.compare(row("rowalloc-fisher-gptq32-avg4", 4.1266, 17.95, 0.062), curve, FP16)
    assert near["extrapolated"] is False and near["far_extrapolated"] is False
    assert near["kl_ref"] == pytest.approx(p3.interp_log(curve.bits, curve.kls, 4.1266))
    far = p3.compare(row("rowalloc-fisher-gptq32-avg4", 4.2, 17.9, 0.05), curve, FP16)
    assert far["extrapolated"] is True and far["far_extrapolated"] is False
    assert p3.in_range([2.125, 4.125], 4.13, p3.EXTRAP_TOL) and not p3.in_range([2.125, 4.125], 4.14, p3.EXTRAP_TOL)
    assert not p3.in_range([2.125, 4.125], 4.13)  # the default is strict (the RTN gate line)


def test_summary_spearman_uses_the_checked_damage_reader(tmp_path):
    """A damage file the allocation methods refuse (no c4_set: a pre-exclusion file) is refused by the H3-A Spearman line too."""
    rdir, p1dir, probe = tmp_path / "phase3", tmp_path / "results", tmp_path / "probe"
    rows = [r for r in phase3_rows() if r["method"].startswith("gptq32-sym-g128-") and not plot_is_smoke(r)]
    write_jsonl(rdir / M / "runs.jsonl", rows)
    write_jsonl(p1dir / M / "runs.jsonl", phase1_rows())
    write_jsonl(probe / M / "ablation.jsonl", [fx._row(M, f"rtn-sym-g128-4@only={t}", 15.0, 17.6, k, note="rtn-sym-g128-4: f16 scale")
                                                for t, k in WIKI_ONLY_KL.items()])
    write_jsonl(rdir / M / "damage_c4.jsonl", [{"unit": "type", "name": t, "numel": 1000, "kl": k, "ppl": 17.6, "n_tokens": 65504}
                                               for t, k in WIKI_ONLY_KL.items()])
    text = ps.build_summary([M], rdir, p1dir, probe)
    assert "Spearman: n/a (damage file refused:" in text and "measured on C4 set None" in text and "Spearman(C4 damage per type" not in text
    # a smoke damage file (2 windows) in the protocol tree is refused as well
    write_jsonl(rdir / M / "damage_c4.jsonl", [{"unit": "type", "name": t, "numel": 1000, "kl": k, "ppl": 17.6, "n_tokens": 65504,
                                                "c4_set": alloc_mod.DAMAGE_C4_SET, "windows": 2} for t, k in WIKI_ONLY_KL.items()])
    text = ps.build_summary([M], rdir, p1dir, probe)
    assert "Spearman: n/a (damage file refused:" in text and "measured on 2 windows" in text


def test_plot_labels_cap_the_push_add_a_leader_line_and_avoid_markers():
    """Six labels at one point: no two label boxes within LABEL_GAP, any label moved more than LABEL_MAX_PUSH steps from its
    slot carries a leader line (so it is never read as the caption of a neighbouring marker); a label whose slot lands on
    another row's marker is moved off it."""
    import itertools

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_rd3

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_yscale("log")
    ax.set_xlim(2, 5)
    ax.set_ylim(0.01, 10)
    labels = [(3.125, 0.3, f"label {i}", "k", (7, -11), "left", False) for i in range(6)]
    out = plot_rd3.place_labels(ax, labels, markers=[(3.125, 0.3)])
    dys = [d for _, d, _ in out]
    assert len(set(dys)) == 6 and all(abs(a - b) >= plot_rd3.LABEL_GAP for a, b in itertools.combinations(dys, 2))
    assert all(leader == (abs(d + 11) > plot_rd3.LABEL_MAX_PUSH * plot_rd3.LABEL_STEP) for _, d, leader in out)
    assert any(leader for *_, leader in out) and not out[0][2]
    assert sum(1 for a in ax.texts if getattr(a, "arrow_patch", None) is not None) == sum(leader for *_, leader in out)
    plt.close(fig)
    # marker avoidance: a marker of another row sitting where the slot would put the label pushes the label elsewhere
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_yscale("log")
    ax.set_xlim(2, 5)
    ax.set_ylim(0.01, 10)
    fig.canvas.draw()
    px = fig.dpi / 72.0
    xd, yd = ax.transData.transform((3.125, 0.3))
    other = tuple(ax.transData.inverted().transform((xd + 20 * px, yd - 11 * px)))  # where the "label 0" box would sit
    alone = plot_rd3.place_labels(ax, [(3.125, 0.3, "label 0", "k", (7, -11), "left", False)], markers=[(3.125, 0.3)])
    assert alone[0][1] == -11
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_yscale("log")
    ax.set_xlim(2, 5)
    ax.set_ylim(0.01, 10)
    moved = plot_rd3.place_labels(ax, [(3.125, 0.3, "label 0", "k", (7, -11), "left", False)], markers=[(3.125, 0.3), other])
    assert moved[0][1] != -11
    plt.close(fig)
