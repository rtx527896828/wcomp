"""Phase 3 quantizer extensions (DESIGN.md "Phase 3 contract"): per-tensor / per-row widths, codebooks, index streams.

CPU: gptq_quantize with H = I equals quantize_rtn exactly on every path (int bits, row_bits, lloyd, nf, asym);
indices round-trip exactly through dequantize_indices on every path; bit accounting (lloyd table, 2 bits per row);
Lloyd-Max reproduces the known 4-level Gaussian optimum and beats the uniform grid on a Laplacian; gptq_model /
apply_rtn with dict bits, row_bits and codebooks on a tiny fake decoder; the env override is gone.
CUDA (skipped without the 0.8B): gptq_model on the real model with a per-type bits dict and the lloyd codebook.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from torch import nn

import wcomp.quant.gptq as gptq_mod
from wcomp.bits import BitAccount
from wcomp.quant.codebook import codebook_mse, gaussian_lloyd_codebook, gaussian_quantile_codebook, lloyd_max, nearest_level
from wcomp.quant.gptq import gptq_model, gptq_quantize
from wcomp.quant.rtn import (ROW_SIDE_BITS, apply_rtn, dequantize_indices, fit_codebook, group_grid, group_grid_rows, quantize_rtn)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3.5-0.8B"


def _W(N: int = 48, K: int = 512, seed: int = 0, scale: float = 0.02) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(N, K, generator=g)


def _rows(N: int, b: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return b - 1 + torch.randint(0, 3, (N,), generator=g)


def _roundtrip(W_hat: torch.Tensor, info: dict) -> None:
    idx = info["indices"]
    assert idx.dtype == torch.int16 and idx.device.type == "cpu" and idx.numel() == W_hat.numel()
    assert int(idx.min()) >= 0 and int(idx.max()) < info["levels"]
    assert torch.equal(dequantize_indices(idx, info["side"]), W_hat)
    assert torch.equal(dequantize_indices(idx, info["side"]).to(torch.bfloat16), W_hat.to(torch.bfloat16))


# ---------------------------------------------------------------- codebook.py


def test_lloyd_max_gaussian_four_levels():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1 << 20, generator=g)
    cb = lloyd_max(x, 4, iters=50)
    expected = torch.tensor([-1.510, -0.4528, 0.4528, 1.510])
    print(f"\nLloyd-Max 4 levels on N(0,1): {cb.tolist()}")
    assert torch.all((cb - expected).abs() / expected.abs() < 0.02)
    assert torch.equal(cb, torch.sort(cb).values)


@pytest.mark.parametrize("levels", [8, 16])
def test_lloyd_max_beats_uniform_grid_on_laplacian(levels: int):
    g = torch.Generator().manual_seed(1)
    u = torch.rand(1 << 18, generator=g) - 0.5
    x = -torch.sign(u) * torch.log1p(-2 * u.abs())  # Laplace(0, 1)
    cb = lloyd_max(x, levels)
    mse_lloyd = codebook_mse(x, cb)
    # the best uniform grid: a symmetric grid of `levels` points, step scanned
    best_uniform = math.inf
    for step in torch.linspace(0.05, 1.5, 300).tolist():
        grid = (torch.arange(levels) - (levels - 1) / 2) * step
        best_uniform = min(best_uniform, codebook_mse(x, grid))
    rtn_grid = group_grid(x.view(1, -1), int(math.log2(levels)), True)
    mse_rtn = float(((rtn_grid[0] * rtn_grid[1]).view(-1) - x).square().mean())
    print(f"\nLaplacian, {levels} levels: MSE lloyd {mse_lloyd:.5f}, best uniform {best_uniform:.5f}, rtn amax grid {mse_rtn:.5f}")
    assert mse_lloyd < best_uniform < mse_rtn


def test_gaussian_lloyd_codebook_is_the_known_optimum():
    cb4 = gaussian_lloyd_codebook(4)
    assert torch.allclose(cb4, torch.tensor([-1.510, -0.4528, 0.4528, 1.510]), atol=5e-4)
    cb16 = gaussian_lloyd_codebook(16)
    assert cb16.shape == (16,) and torch.allclose(cb16, -cb16.flip(0)) and abs(float(cb16[-1]) - 2.733) < 2e-3  # Max 1960
    assert torch.equal(gaussian_lloyd_codebook(8), gaussian_lloyd_codebook(8))  # deterministic: 0 table bits
    # the sample Lloyd-Max on a large Gaussian sample lands on it
    x = torch.randn(1 << 20, generator=torch.Generator().manual_seed(5))
    assert torch.all((lloyd_max(x, 8) - gaussian_lloyd_codebook(8)).abs() < 0.02)


def test_fixed_codebooks_vs_uniform_grid_on_gaussian_weights():
    """Why "nf" is the Gaussian Lloyd-Max optimum and not the equiprobable quantiles: under RMS normalization the
    quantile codebook clips the tails (outer level 1.53 sigma at 16 levels) and loses to the uniform amax grid."""
    W = _W(N=64, K=1024)
    for bits in (3, 4):
        mse = {}
        for kind in ("lloyd", "nf", "nfq"):
            Wr, _ = quantize_rtn(W, bits, 128, codebook={"kind": kind})
            mse[kind] = float((Wr - W).square().mean())
        mse["uniform"] = float((quantize_rtn(W, bits, 128, True)[0] - W).square().mean())
        print(f"\n{bits} bits, Gaussian weights, MSE: " + ", ".join(f"{k} {v:.3e}" for k, v in mse.items()))
        assert mse["lloyd"] <= 1.02 * mse["nf"] < mse["uniform"] < mse["nfq"]


def test_gaussian_codebook_and_nearest_level():
    cb = gaussian_quantile_codebook(16)
    assert cb.shape == (16,) and torch.all(cb[1:] > cb[:-1]) and torch.allclose(cb, -cb.flip(0))
    assert abs(float(cb[-1]) - 1.863) < 2e-3  # Phi^-1(31/32): the tail clip of the equiprobable codebook
    x = torch.tensor([-10.0, 0.0, 10.0, float(cb[3]), float((cb[3] + cb[4]) / 2)])
    idx = nearest_level(x, cb)
    assert idx.tolist() == [0, 7, 15, 3, 3]  # ties go to the lower level (0 is the midpoint of levels 7 and 8)
    # the Lloyd-Max init equals the fixed codebook scaled by the sample RMS
    x = 3.0 * torch.randn(4096, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(lloyd_max(x, 8, iters=0), 3.0 * x.square().mean().sqrt() / 3.0 * gaussian_quantile_codebook(8), rtol=1e-5)


# ---------------------------------------------------------------- GPTQ with H = I equals RTN on every path


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_gptq_identity_equals_rtn_int_bits(sym: bool, bits: int):
    W = _W()
    H = torch.eye(W.shape[1], dtype=torch.float64)
    Wq, ig = gptq_quantize(W, H, bits, 128, sym)
    Wr, ir = quantize_rtn(W, bits, 128, sym)
    assert torch.equal(Wq, Wr)
    assert torch.equal(ig["indices"], ir["indices"]) and ig["levels"] == ir["levels"] == 2**bits
    assert (ig["index_bits"], ig["aux_bits"]) == (ir["index_bits"], ir["aux_bits"])
    _roundtrip(Wq, ig)
    _roundtrip(Wr, ir)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("bits", [3, 4])
def test_gptq_identity_equals_rtn_row_bits(sym: bool, bits: int):
    W = _W(N=64)
    rb = _rows(64, bits)
    assert set(rb.tolist()) == {bits - 1, bits, bits + 1}
    H = torch.eye(W.shape[1], dtype=torch.float64)
    Wq, ig = gptq_quantize(W, H, bits, 128, sym, row_bits=rb)
    Wr, ir = quantize_rtn(W, bits, 128, sym, row_bits=rb)
    assert torch.equal(Wq, Wr)
    assert torch.equal(ig["indices"], ir["indices"])
    assert ig["levels"] == ir["levels"] == 2 ** (bits + 1)
    assert ig["index_bits"] == ir["index_bits"] == int(rb.sum()) * W.shape[1]
    assert ig["aux_bits"] == ir["aux_bits"]
    _roundtrip(Wq, ig)
    _roundtrip(Wr, ir)
    # each row equals the plain RTN of that row at its own width (group_grid_rows is group_grid per row)
    for i in range(64):
        Wi, _ = quantize_rtn(W[i : i + 1], int(rb[i]), 128, sym)
        assert torch.equal(Wr[i : i + 1], Wi)
    # uniform row_bits reproduce the int path exactly
    Wu, iu = quantize_rtn(W, bits, 128, sym, row_bits=torch.full((64,), bits))
    assert torch.equal(Wu, quantize_rtn(W, bits, 128, sym)[0]) and iu["levels"] == 2**bits


@pytest.mark.parametrize("kind", ["lloyd", "nf", "nfq"])
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_gptq_identity_equals_rtn_codebook(kind: str, bits: int):
    W = _W(N=32, K=384)
    H = torch.eye(W.shape[1], dtype=torch.float64)
    spec = {"kind": kind, "bits": bits}
    Wq, ig = gptq_quantize(W, H, bits, 128, codebook=spec)
    Wr, ir = quantize_rtn(W, bits, 128, codebook=spec)
    assert torch.equal(Wq, Wr)
    assert torch.equal(ig["indices"], ir["indices"]) and ig["levels"] == ir["levels"] == 2**bits
    assert (ig["index_bits"], ig["aux_bits"]) == (ir["index_bits"], ir["aux_bits"])
    _roundtrip(Wq, ig)
    _roundtrip(Wr, ir)
    assert kind in ig["note"] and kind in ir["note"]
    if kind != "nfq":  # the fitted and the Gaussian-optimal codebooks beat the uniform sym grid on Gaussian weights
        Wsym, _ = quantize_rtn(W, bits, 128, True)
        assert (Wr - W).square().mean() < (Wsym - W).square().mean()


def test_gptq_codebook_with_row_bits_round_trips():
    W = _W(N=40, K=256)
    rb = _rows(40, 3)
    H = torch.eye(256, dtype=torch.float64)
    for kind in ("lloyd", "nf"):
        Wq, ig = gptq_quantize(W, H, 3, 128, row_bits=rb, codebook={"kind": kind})
        Wr, ir = quantize_rtn(W, 3, 128, row_bits=rb, codebook={"kind": kind})
        assert torch.equal(Wq, Wr) and torch.equal(ig["indices"], ir["indices"])
        assert ig["levels"] == 16 and ig["index_bits"] == int(rb.sum()) * 256
        table = sum(2**b * 16 for b in (2, 3, 4)) if kind == "lloyd" else 0
        assert ig["aux_bits"] == 40 * 2 * 16 + table
        _roundtrip(Wq, ig)
        # indices of a row stay inside the row's own alphabet
        idx = ig["indices"].view(40, 256)
        for i in range(40):
            assert int(idx[i].max()) < 2 ** int(rb[i])


def test_gptq_correlated_hessian_round_trips_every_path():
    """With a real (correlated) Hessian the error feedback changes the weights; the indices must still reproduce W_hat."""
    g = torch.Generator().manual_seed(3)
    N, K, n = 32, 256, 2048
    W = 0.02 * torch.randn(N, K, generator=g)
    A = torch.randn(K, K, generator=g) / math.sqrt(K)
    X = torch.randn(n, K, generator=g) @ A
    H = X.double().T @ X.double() / n
    rb = _rows(N, 3)
    for kw in ({}, {"sym": False}, {"row_bits": rb}, {"codebook": {"kind": "lloyd"}}, {"codebook": {"kind": "nf"}},
               {"row_bits": rb, "codebook": {"kind": "lloyd"}}, {"act_order": True}, {"act_order": True, "row_bits": rb},
               {"act_order": True, "codebook": {"kind": "nf"}}):
        Wq, info = gptq_quantize(W, H, 3, 128, **kw)
        _roundtrip(Wq, info)
        err_q = ((W - Wq) @ X.T).norm()
        err_r = ((W - quantize_rtn(W, 3, 128, kw.get("sym", True), row_bits=kw.get("row_bits"), codebook=kw.get("codebook"))[0]) @ X.T).norm()
        assert err_q < err_r, kw


def test_zero_rms_group_dequantizes_to_zero_on_codebooks():
    W = _W(N=4, K=256)
    W[:, :128] = 0.0
    W[1, 130] = 0.0
    for kind in ("lloyd", "nf"):
        Wr, ir = quantize_rtn(W, 3, 128, codebook={"kind": kind})
        assert torch.all(Wr[:, :128] == 0)
        assert torch.isfinite(Wr).all()
        _roundtrip(Wr, ir)
        Wq, ig = gptq_quantize(W, torch.eye(256, dtype=torch.float64), 3, 128, codebook={"kind": kind})
        assert torch.equal(Wq, Wr)


# ---------------------------------------------------------------- bit accounting


def test_bits_accounting_lloyd_and_rows():
    N, K, b = 64, 512, 3
    numel = N * K
    W = _W(N, K)
    _, il = quantize_rtn(W, b, 128, codebook={"kind": "lloyd", "bits": b})
    assert il["index_bits"] == b * numel
    assert il["aux_bits"] == 16 * numel // 128 + 2**b * 16
    assert (il["index_bits"] + il["aux_bits"]) / numel == pytest.approx(b + 16 / 128 + 2**b * 16 / numel)
    _, inf = quantize_rtn(W, b, 128, codebook={"kind": "nf"})
    assert inf["aux_bits"] == 16 * numel // 128
    with pytest.raises(ValueError):
        quantize_rtn(W, b, 128, codebook={"kind": "lloyd", "bits": b + 1})
    with pytest.raises(ValueError):
        quantize_rtn(W, b, 128, codebook={"kind": "kmeans"})
    rb = _rows(N, b)
    lin = nn.Linear(K, N, bias=False)
    with torch.no_grad():
        lin.weight.copy_(W)
    acc = apply_rtn(None, b, 128, True, linears=[("layers.0.mlp.up_proj", lin)], row_bits={"layers.0.mlp.up_proj": rb})
    t = acc.tensors[0]
    assert t.index_bits == int(rb.sum()) * K
    assert t.aux_bits == 16 * numel // 128 + ROW_SIDE_BITS * N  # f16 scales + 2 bits per row
    assert t.row_bits == rb.tolist() and t.levels == 2 ** (b + 1) and t.indices is not None
    assert "2 bits/row width" in t.note
    assert torch.equal(lin.weight, quantize_rtn(W, b, 128, True, row_bits=rb)[0].to(lin.weight.dtype))


def test_fit_codebook_width_guard_rejects_both_ends():
    """`not 1 <= min and max <= 15` used to parse as `(not 1 <= min) and (max <= 15)`, accepting a 16-bit [N, 65536] table."""
    W = torch.randn(4, 256)
    with pytest.raises(ValueError):
        fit_codebook(W, 128, {"kind": "nf"}, 16)
    with pytest.raises(ValueError):
        fit_codebook(W, 128, {"kind": "nf"}, 0)
    with pytest.raises(ValueError):
        fit_codebook(W, 128, {"kind": "nf"}, 4, row_bits=torch.tensor([4, 4, 16, 4]))
    assert fit_codebook(W, 128, {"kind": "nf"}, 15).levels == 2**15


def test_nf_and_nfq_codebooks_are_the_two_documented_tables():
    """"nf" is the Lloyd-Max optimum of N(0, 1), "nfq" the DESIGN.md E3 quantiles; both 0 table bits, both named in the note."""
    W = torch.randn(8, 256)
    for b in (3, 4):
        nf = fit_codebook(W, 128, {"kind": "nf"}, b)
        nfq = fit_codebook(W, 128, {"kind": "nfq"}, b)
        assert torch.equal(nf.table[0], gaussian_lloyd_codebook(2**b).to(torch.float16).float())
        assert torch.equal(nfq.table[0], gaussian_quantile_codebook(2**b).to(torch.float16).float())
        assert not torch.equal(nf.table, nfq.table) and nf.table_bits == nfq.table_bits == 0
    _, i_nf = quantize_rtn(W, 4, 128, codebook={"kind": "nf"})
    _, i_nfq = quantize_rtn(W, 4, 128, codebook={"kind": "nfq"})
    assert "(nf)" in i_nf["note"] and "(nfq)" in i_nfq["note"] and i_nf["aux_bits"] == i_nfq["aux_bits"]


# ---------------------------------------------------------------- model-level plumbing on a fake decoder


class _Layer(nn.Module):
    def __init__(self, d: int, f: int) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(d, f, bias=False)
        self.mlp.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, hidden_states, **kwargs):
        return (hidden_states + self.mlp.down_proj(torch.relu(self.mlp.up_proj(hidden_states))),)


class _Inner(nn.Module):
    def __init__(self, vocab: int, d: int, f: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Layer(d, f) for _ in range(n_layers)])

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, position_ids=None)[0]
        return x


class _Fake(nn.Module):
    def __init__(self, vocab: int = 64, d: int = 256, f: int = 384, n_layers: int = 2) -> None:
        super().__init__()
        self.config = type("C", (), {"hidden_size": d})()
        self.model = _Inner(vocab, d, f, n_layers)
        torch.manual_seed(0)
        for p in self.parameters():
            p.data.normal_(0, 0.05)


def _fake_and_ids():
    model = _Fake().eval()
    ids = torch.randint(0, 64, (4, 32), generator=torch.Generator().manual_seed(0))
    return model, ids


def test_gptq_model_dict_bits_row_bits_codebook_on_fake_decoder():
    model, ids = _fake_and_ids()
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    assert names == ["model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj", "model.layers.1.mlp.up_proj", "model.layers.1.mlp.down_proj"]
    bits = {n: (3 if "up" in n else 5) for n in names}
    rb = {"model.layers.1.mlp.down_proj": _rows(256, 5)}
    acc = gptq_model(model, ids, bits, 128, True, row_bits=rb, hessian_f32=True)
    assert isinstance(acc, BitAccount) and [t.name for t in acc.tensors] == names
    by = {t.name: t for t in acc.tensors}
    assert by[names[0]].index_bits == 3 * 384 * 256 and by[names[0]].levels == 8
    assert by[names[1]].index_bits == 5 * 256 * 384 and by[names[1]].levels == 32
    t = by[names[3]]
    assert t.index_bits == int(rb[names[3]].sum()) * 384 and t.levels == 64 and t.row_bits == rb[names[3]].tolist()
    assert t.aux_bits == 256 * 3 * 16 + ROW_SIDE_BITS * 256
    for t in acc.tensors:
        assert t.indices is not None and t.indices.dtype == torch.int16 and t.indices.numel() == t.numel
        assert t.note.endswith("hessian-f32") or "hessian-f32" in t.note
    # missing tensor in the bits dict is an error, not a silent default
    with pytest.raises(KeyError):
        gptq_model(_fake_and_ids()[0], ids, {names[0]: 3}, 128, True)
    # codebook on top of a dict allocation: every tensor's codebook has its own width
    model2, _ = _fake_and_ids()
    acc2 = gptq_model(model2, ids, bits, 128, True, codebook={"kind": "lloyd"}, hessian_f32=True)
    by2 = {t.name: t for t in acc2.tensors}
    assert by2[names[0]].aux_bits == 384 * 2 * 16 + 2**3 * 16 and "lloyd" in by2[names[0]].note
    assert by2[names[1]].aux_bits == 256 * 3 * 16 + 2**5 * 16 and by2[names[1]].levels == 32
    with pytest.raises(ValueError):
        gptq_model(_fake_and_ids()[0], ids, 3, 128, True, codebook={"kind": "lloyd", "bits": 4})


def test_hessian_f32_argument_selects_the_path_and_env_is_ignored(monkeypatch):
    monkeypatch.setenv("WCOMP_HESSIAN_F32", "1")
    import importlib

    importlib.reload(gptq_mod)
    try:
        assert gptq_mod.HESSIAN_F32 is False  # the environment override is gone
        model, ids = _fake_and_ids()
        acc64 = gptq_mod.gptq_model(model, ids, 4, 128, True)  # hessian_f32=None -> module flag False -> Phase 1 path
        assert all("hessian-f32" not in t.note for t in acc64.tensors)
        model, ids = _fake_and_ids()
        acc32 = gptq_mod.gptq_model(model, ids, 4, 128, True, hessian_f32=True)
        assert all(t.note.endswith("hessian-f32") for t in acc32.tensors)
        assert acc32.bits_blocks() == acc64.bits_blocks() == pytest.approx(4.125)
    finally:
        monkeypatch.delenv("WCOMP_HESSIAN_F32")
        importlib.reload(gptq_mod)


def test_apply_rtn_dict_bits_and_codebook_on_fake_decoder():
    model, _ = _fake_and_ids()
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    bits = {n: (2 if "up" in n else 4) for n, _ in linears}
    acc = apply_rtn(model, bits, 128, True, linears=linears, codebook={"kind": "nf"})
    for t in acc.tensors:
        b = bits[t.name]
        assert t.index_bits == b * t.numel and t.levels == 2**b and t.aux_bits == 16 * t.numel // 128
        assert t.note.startswith(f"rtn-nf-g128-{b}")
    assert acc.bits_blocks() == pytest.approx(sum(bits[n] * m.weight.numel() for n, m in linears) / acc.numel() + 0.125)


# ---------------------------------------------------------------- the real 0.8B (CUDA)


@pytest.mark.skipif(not torch.cuda.is_available() or not MODEL_DIR.is_dir(), reason="needs CUDA and the 0.8B")
def test_gptq32_real_model_dict_bits_lloyd():
    from wcomp.data import c4_calibration
    from wcomp.models import block_linears, load_model, weight_type

    model, tok = load_model("qwen3.5-0.8b")
    try:
        calib = c4_calibration(tok, "qwen3.5-0.8b", n=4, seed=0)[:, :512]
        linears = block_linears(model)
        bits = {n: (4 if weight_type(n) in ("down", "out_proj", "v") else 3) for n, _ in linears}
        rb = {linears[0][0]: _rows(linears[0][1].out_features, bits[linears[0][0]])}
        acc = gptq_model(model, calib, bits, 128, True, hessian_f32=True, codebook={"kind": "lloyd"}, row_bits=rb)
        assert {t.name for t in acc.tensors} == {n for n, _ in linears}
        for t in acc.tensors:
            assert t.indices is not None and t.indices.numel() == t.numel and t.levels in (8, 16, 32)
            assert "lloyd" in t.note and "hessian-f32" in t.note
        expected = sum(bits[n] * m.weight.numel() for n, m in linears) / acc.numel() + 0.125
        assert abs(acc.bits_blocks() - expected) < 0.01  # codebook tables and the row widths of one tensor
        probe = calib[:1].cuda()
        with torch.no_grad():
            loss = model(probe, labels=probe).loss.item()
        assert math.isfinite(loss) and loss < 8.0
    finally:
        del model
        torch.cuda.empty_cache()


def test_dead_column_is_zero_on_uniform_grids_and_the_nearest_level_on_codebooks():
    """A dead input column (diag(H) == 0) is zeroed in W; the uniform grids reconstruct it as exactly 0, a codebook (even
    level count, no zero level) as the level nearest 0 (+-c_min * scale), as the gptq_quantize docstring states; its error
    reaches no other column (the Hinv row of a dead column is zero off the diagonal) and the index round trip holds."""
    gen = torch.Generator().manual_seed(0)
    N, K = 64, 256
    W = torch.randn(N, K, generator=gen)
    X = torch.randn(1024, K, generator=gen, dtype=torch.float64)
    X[:, 5] = 0.0
    H = X.T @ X / X.shape[0]
    assert H[5, 5] == 0
    Wu, _ = gptq_quantize(W, H, 4, group=128, sym=True)
    assert float(Wu[:, 5].abs().max()) == 0.0
    for kind in ("nf", "nfq", "lloyd"):
        Wc, info = gptq_quantize(W, H, 4, group=128, codebook={"kind": kind})
        cb = info["side"]["codebook"]
        c_min = float(cb.table[:, :16].abs().min())
        scale = info["side"]["scales"][:, 0]
        assert c_min > 0  # no zero level
        assert float(Wc[:, 5].abs().max()) > 0
        assert torch.allclose(Wc[:, 5].abs(), c_min * scale.abs(), rtol=1e-3, atol=1e-6)
        assert torch.equal(Wc, dequantize_indices(info["indices"], info["side"]))
    # the dead column's reconstruction does not depend on the other columns: W without column 5 gives the same values elsewhere
    W2 = W.clone()
    W2[:, 5] = 7.0  # whatever the dead column held is irrelevant
    Wc2, _ = gptq_quantize(W2, H, 4, group=128, codebook={"kind": "nf"})
    Wc1, _ = gptq_quantize(W, H, 4, group=128, codebook={"kind": "nf"})
    assert torch.equal(Wc1, Wc2)
