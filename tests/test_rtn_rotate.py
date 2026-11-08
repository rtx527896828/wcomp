"""Tests for wcomp.quant.rtn and wcomp.quant.rotate (DESIGN.md contracts)."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from wcomp.quant.rotate import HADAMARD_TABLE, hadamard, random_orthogonal, rotate_inputs
from wcomp.quant.rtn import apply_rtn, group_grid, quantize_rtn

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen3.5-0.8B"

HADAMARD_ORDERS = [1, 2, 4, 12, 20, 24, 28, 40, 56, 80, 1024, 2048, 3584, 6144]


# ---------------------------------------------------------------- hadamard / random_orthogonal


@pytest.mark.parametrize("n", HADAMARD_ORDERS)
def test_hadamard_orthonormal(n):
    H = hadamard(n)
    assert H.shape == (n, n) and H.dtype == torch.float32
    # every entry is +-1/sqrt(n)
    assert torch.allclose(H.abs(), torch.full((n, n), n**-0.5), atol=1e-6)
    err = (H.double() @ H.double().T - torch.eye(n, dtype=torch.float64)).abs().max().item()
    print(f"hadamard({n}): max |H H^T - I| = {err:.2e}")
    assert err < 1e-5


@pytest.mark.parametrize("m", sorted(HADAMARD_TABLE))
def test_hadamard_table_entries_are_hadamard(m):
    H = HADAMARD_TABLE[m]
    assert torch.all(H.abs() == 1)
    assert torch.equal(H @ H.T, m * torch.eye(m))


@pytest.mark.parametrize("n", [3, 36, 44, 1000, 48 * 3, 9 * 128])
def test_hadamard_missing_orders_raise(n):
    # 36 = 9*4, 44 = 11*4, 1000 = 125*8, 144 = 9*16, 1152 = 9*128: no table order times a power of 2
    with pytest.raises(ValueError):
        hadamard(n)


def test_random_orthogonal():
    Q = random_orthogonal(96, seed=0)
    assert Q.dtype == torch.float32
    err = (Q @ Q.T - torch.eye(96)).abs().max().item()
    assert err < 1e-5
    assert torch.equal(Q, random_orthogonal(96, seed=0))
    assert not torch.allclose(Q, random_orthogonal(96, seed=1))


# ---------------------------------------------------------------- rotate_inputs on a random linear


@pytest.mark.parametrize("K", [1024, 3584, 36])  # 36 has no Hadamard -> random orthogonal fallback
def test_rotate_inputs_preserves_linear_f32(K):
    torch.manual_seed(0)
    lin = nn.Linear(K, 64, bias=True)
    x = torch.randn(3, 5, K)
    with torch.no_grad():
        y0 = lin(x)
        W0 = lin.weight.clone()
        Rs = rotate_inputs(None, [("lin", lin)], seed=0)
        y1 = lin(x)
    R = Rs["lin"]
    assert R.shape == (K, K) and R.dtype == torch.float32
    assert torch.allclose(lin.weight, W0 @ R, atol=1e-5)  # W (R^T)^-1 == W R up to f32 rounding of R
    rel = (y1 - y0).norm() / y0.norm()
    assert rel < 1e-4, rel.item()
    if K == 36:
        assert torch.allclose(R, random_orthogonal(36, 0))
    else:
        assert torch.allclose(R, hadamard(K))


@pytest.mark.parametrize("K", [1024, 2048, 3584])
def test_rotate_inputs_bf16_no_systematic_rescale(K):
    # bf16(1/sqrt(K)) is not exact for K = 2048, 3584: with W <- W R (R in bf16) the output was rescaled by
    # c^2 K = 0.9998 / 1.0024. Folding (R^T)^-1 of the stored R leaves only unbiased per-element rounding.
    torch.manual_seed(0)
    lin = nn.Linear(K, 64, bias=False).to(torch.bfloat16)
    x = torch.randn(256, K).to(torch.bfloat16)
    with torch.no_grad():
        y0 = lin(x).double()
        Rs = rotate_inputs(None, [("lin", lin)])
        y1 = lin(x).double()
    assert Rs["lin"].dtype == torch.bfloat16
    ls_scale = ((y0 * y1).sum() / (y0 * y0).sum()).item()  # least-squares y1 ~ s * y0
    rel = ((y1 - y0).norm() / y0.norm()).item()
    print(f"K={K}: ls scale {ls_scale:.6f}, rel {rel:.2e}")
    assert abs(ls_scale - 1) < 1e-4
    assert rel < 1e-2


# ---------------------------------------------------------------- quantize_rtn


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_rtn_16bit_sym_reproduces():
    torch.manual_seed(0)
    W = torch.randn(64, 512)
    W_hat, info = quantize_rtn(W, 16, 128, sym=True)
    assert W_hat.dtype == torch.float32 and W_hat.shape == W.shape
    assert _rel(W_hat, W) < 1e-3
    assert info["index_bits"] == 16 * W.numel()


@pytest.mark.parametrize("sym", [True, False])
def test_rtn_error_monotone_in_bits(sym):
    torch.manual_seed(0)
    W = torch.randn(256, 1024)
    errs = [_rel(quantize_rtn(W, b, 128, sym)[0], W) for b in (2, 3, 4, 8)]
    print(f"rtn sym={sym} rel err 2/3/4/8 bits: {errs}")
    assert errs[0] > errs[1] > errs[2] > errs[3]


def _mse_sym_asym(W, bits):
    mse_sym = (quantize_rtn(W, bits, 128, True)[0] - W).square().mean().item()
    mse_asym = (quantize_rtn(W, bits, 128, False)[0] - W).square().mean().item()
    print(f"{bits} bits: mse sym {mse_sym:.4e} asym {mse_asym:.4e} ratio asym/sym {mse_asym / mse_sym:.3f}")
    return mse_sym, mse_asym


@pytest.mark.parametrize("bits", [4, 8])
def test_rtn_asym_not_worse_than_sym_gaussian(bits):
    # On zero-mean Gaussian groups asym wins only from 4 bits up. At 2-3 bits the contract's integer zero
    # (round(-min/scale)) shifts the 2^b-level grid by up to scale/2 off the data range, e.g. the 2-bit grid
    # becomes {-1.7, 0, 1.7, 3.4} sigma, and asym is measurably worse than sym (x1.35 at 2 bits, x1.03 at 3).
    torch.manual_seed(0)
    W = torch.randn(256, 1024)
    mse_sym, mse_asym = _mse_sym_asym(W, bits)
    assert mse_asym <= mse_sym


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_rtn_asym_beats_sym_on_offset_groups(bits):
    # Groups with a nonzero mean: the sym grid wastes half its range, asym must win at every width.
    torch.manual_seed(0)
    W = torch.randn(256, 8, 128) + 1.5 * torch.randn(256, 8, 1)
    mse_sym, mse_asym = _mse_sym_asym(W.reshape(256, 1024), bits)
    assert mse_asym < mse_sym


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_rtngroup_grid(bits):
    torch.manual_seed(1)
    W = torch.randn(16, 4, 128) * torch.rand(16, 4, 1) + torch.randn(16, 4, 1)  # per-group scale/offset
    # sym: q in [-2^(b-1), 2^(b-1)-1], zero = 0, scale = amax / 2^(b-1)
    q, scale, zero = group_grid(W, bits, sym=True)
    assert torch.equal(q, q.round()) and q.min() >= -(2 ** (bits - 1)) and q.max() <= 2 ** (bits - 1) - 1
    assert torch.all(zero == 0)
    amax = W.abs().amax(-1, keepdim=True)
    assert torch.allclose(scale, (amax / 2 ** (bits - 1)).half().float())
    assert torch.equal(scale, scale.half().float())  # f16-representable
    # asym: q in [0, 2^b-1], zero an integer in [0, 2^b-1], scale = (max - min) / (2^b - 1)
    q, scale, zero = group_grid(W, bits, sym=False)
    assert torch.equal(q, q.round()) and q.min() >= 0 and q.max() <= 2**bits - 1
    assert torch.equal(zero, zero.round()) and zero.min() >= 0 and zero.max() <= 2**bits - 1
    rng = W.amax(-1, keepdim=True).clamp(min=0) - W.amin(-1, keepdim=True).clamp(max=0)  # range contains 0
    assert torch.allclose(scale, (rng / (2**bits - 1)).half().float())
    assert torch.equal(scale, scale.half().float())
    W_hat, _ = quantize_rtn(W.reshape(16, 512), bits, 128, sym=False)
    assert torch.allclose(W_hat, ((q - zero) * scale).reshape(16, 512))


def test_rtn_asym_all_positive_group_not_clipped():
    # A group that does not contain 0 must still be covered by the grid (zero point stays in range).
    W = torch.rand(4, 128) * 0.5 + 2.0
    W_hat, _ = quantize_rtn(W, 8, 128, sym=False)
    assert (W_hat - W).abs().max() <= 2.5 / 255  # at most half a step of the [0, max] grid


def test_rtn_info_bits():
    W = torch.randn(64, 1024)
    _, info = quantize_rtn(W, 4, 128, sym=True)
    assert (info["index_bits"] + info["aux_bits"]) / W.numel() == 4.125
    _, info = quantize_rtn(W, 4, 128, sym=False)
    # 4-bit index + per group an f16 scale and a 4-bit zero: 4 + (16 + 4) / 128
    assert (info["index_bits"] + info["aux_bits"]) / W.numel() == 4 + 20 / 128
    _, info = quantize_rtn(W, 3, 0, sym=True)  # per-row
    assert (info["index_bits"] + info["aux_bits"]) / W.numel() == 3 + 16 / 1024


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_rtn_tiny_group_with_exact_zero_is_finite(bits, sym):
    # Groups whose f32 scale rounds to 0 in f16 (amax 2^-123: dead rows of the rotated 0.8B; randn * 1e-7)
    # must not divide by zero: an exact 0 in such a group gave 0/0 = NaN, other entries were silently zeroed.
    torch.manual_seed(0)
    W = torch.randn(3, 128) * 1e-7
    W[0, 0] = 0.0
    W[2] = 2.0**-123
    W[2, 5] = 0.0
    W_hat, _ = quantize_rtn(W, bits, 128, sym)
    assert torch.isfinite(W_hat).all()
    q, scale, zero = group_grid(W, bits, sym)
    assert torch.all(scale > 0) and torch.equal(scale, scale.half().float())
    assert torch.isfinite(q).all() and torch.isfinite(zero).all()
    # a healthy group in the same tensor is unaffected
    W[1] = torch.randn(128)
    W_hat, _ = quantize_rtn(W, bits, 128, sym)
    assert torch.isfinite(W_hat).all()
    assert torch.equal(W_hat[1], quantize_rtn(W[1:2], bits, 128, sym)[0][0])


def test_rtn_rejects_bad_group():
    with pytest.raises(ValueError):
        quantize_rtn(torch.randn(4, 100), 4, 128)


# ---------------------------------------------------------------- apply_rtn on a stub module list


def test_apply_rtn_stub_modules():
    try:
        import wcomp.bits  # noqa: F401
    except ImportError:  # pragma: no cover - only before wcomp/bits.py exists
        stub = types.ModuleType("wcomp.bits")

        class BitAccount:
            def __init__(self):
                self.rows = []

            def add(self, name, numel, index_bits, aux_bits, note=""):
                self.rows.append((name, numel, index_bits, aux_bits, note))

            def bits_blocks(self):
                return sum(r[2] + r[3] for r in self.rows) / sum(r[1] for r in self.rows)

        stub.BitAccount = BitAccount
        sys.modules["wcomp.bits"] = stub
    torch.manual_seed(0)
    lins = [("a.layers.0.x", nn.Linear(1024, 32, bias=False)), ("a.layers.1.y", nn.Linear(256, 16, bias=False))]
    for _, lin in lins:
        lin.to(torch.bfloat16)
    W0 = [lin.weight.detach().clone() for _, lin in lins]
    model = nn.ModuleList([lin for _, lin in lins])
    acct = apply_rtn(model, 4, 128, True, linears=lins)
    assert acct.bits_blocks() == 4.125
    assert {r.name for r in acct.tensors} == {"a.layers.0.x", "a.layers.1.y"}
    for (_, lin), w0 in zip(lins, W0):
        assert lin.weight.dtype == torch.bfloat16
        assert not torch.equal(lin.weight, w0)
        assert _rel(lin.weight.float(), w0.float()) < 0.2  # 4-bit RTN on Gaussian: ~7% error
        # the stored weight is the bf16 cast of the exact RTN reconstruction
        W_hat, _ = quantize_rtn(w0.float(), 4, 128, True)
        assert torch.equal(lin.weight, W_hat.to(torch.bfloat16))


# ---------------------------------------------------------------- had-fp16: rotation-only method


def _stub_model() -> nn.Module:
    """Two 'layers' with mlp.gate_proj / mlp.down_proj so wcomp.models.block_linears finds them."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.gate_proj = nn.Linear(1024, 32, bias=False)
            self.mlp.down_proj = nn.Linear(256, 16, bias=False)

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(), Block()])

    torch.manual_seed(0)
    return Stub().to(torch.bfloat16)


def test_had_fp16_method_records_rotation_only_cost():
    """had-fp16 = rotate_inputs with no quantization: 16 bits, rotation in every note, output within bf16 rounding."""
    from wcomp.methods import METHODS, Ctx, family

    assert "had-fp16" in METHODS
    assert family("had-fp16") == ("had-fp16", None)  # not a rate-distortion point: a reference row
    model = _stub_model()
    lin = model.model.layers[0].mlp.gate_proj
    x = torch.randn(64, 1024).to(torch.bfloat16)
    with torch.no_grad():
        y0 = lin(x).float()
        acct = METHODS["had-fp16"](model, Ctx(None, 0))
        y1 = lin(x).float()
    assert acct.bits_blocks() == 16.0
    assert len(acct.tensors) == 4
    assert all("input rotation hadamard(" in t.note and "not quantized" in t.note for t in acct.tensors)
    rel = ((y1 - y0).norm() / y0.norm()).item()
    print(f"had-fp16 stub: output rel change {rel:.2e}")
    assert 0 < rel < 1e-2  # bf16 rounding of x R and of the folded weight, not an exact identity


# ---------------------------------------------------------------- real model (GPU)


@pytest.mark.skipif(not torch.cuda.is_available() or not MODEL_DIR.exists(), reason="needs CUDA and the 0.8B")
def test_rotate_inputs_real_model_ppl():
    """DESIGN.md rotate.py contract: PPL with rotation and no quantization equals the baseline within 0.1%."""
    from wcomp.data import load_tokenizer, wikitext2_test
    from wcomp.eval import evaluate
    from wcomp.models import block_linears, load_model

    # All windows, not the first 8: the rotation-only PPL shift of the bf16 rounding is systematic, about
    # +0.070% over all 145 windows but +0.096% on the first 8 and +0.089% on the first 32, so a subset check
    # sat at 90-96% of the limit and could flip with the subset. ~1 min on the 0.8B.
    eval_ids = wikitext2_test(load_tokenizer(MODEL_DIR), "qwen3.5-0.8b")
    model, _ = load_model("qwen3.5-0.8b")
    try:
        ppl0 = evaluate(model, None, eval_ids)["ppl"]
        lins = block_linears(model)
        Rs = rotate_inputs(model, lins)
        assert len(Rs) == len(lins) == 18 * 8 + 6 * 7  # 186 block linears
        assert {tuple(R.shape) for R in Rs.values()} == {(1024, 1024), (2048, 2048), (3584, 3584)}
        assert all(R.dtype == torch.bfloat16 and R.is_cuda for R in Rs.values())
        ppl1 = evaluate(model, None, eval_ids)["ppl"]
        rel = abs(ppl1 - ppl0) / ppl0
        print(f"ppl before {ppl0:.5f} after rotation {ppl1:.5f} rel change {rel:.2e} on {eval_ids.shape[0]} windows")
        assert rel < 1e-3, (ppl0, ppl1)
        # the rotated model must quantize without NaN (dead rows of in_proj_qkv have f16 scale 0)
        acct = apply_rtn(model, 4, 128, True, linears=lins)
        assert all(torch.isfinite(lin.weight).all() for _, lin in lins)
        assert acct.bits_blocks() == 4.125
        ppl2 = evaluate(model, None, eval_ids[:2])["ppl"]
        assert torch.isfinite(torch.tensor(ppl2)) and ppl2 < 100
    finally:
        del model
        torch.cuda.empty_cache()


def test_had_notes_summarize_rotations_once():
    """had-* rows carry the inner quantizer's note once with one rotation summary over all input widths, not
    one copy of the inner note per width."""
    from wcomp.methods import METHODS, Ctx

    model = _stub_model()  # input widths 1024 and 256, both Hadamard
    acct = METHODS["had-rtn-sym-g128-4"](model, Ctx(None, 0))
    notes = acct.summary()["notes"]
    assert notes == ["rtn-sym-g128-4: f16 scale; input rotation hadamard(256,1024)"]
    assert acct.bits_blocks() == 4.125
