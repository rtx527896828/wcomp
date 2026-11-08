"""Tests for wcomp.quant.gptq: synthetic GPTQ vs RTN on CPU, exactness with H = I, and the real 0.8B on CUDA."""
import logging
import math
import os
import random
import time

import pytest
import torch

from wcomp.bits import BitAccount
from wcomp.quant.gptq import gptq_model, gptq_quantize, grid_params
from wcomp.quant.rtn import quantize_rtn

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen3.5-0.8B")


def _correlated_problem(N: int = 64, K: int = 256, n: int = 4096, seed: int = 0):
    """Random W and inputs X = Z A with a random mixing matrix A, plus H = X^T X / n in f64."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(N, K, generator=g)
    A = torch.randn(K, K, generator=g) / math.sqrt(K)
    Z = torch.randn(n, K, generator=g)
    X = Z @ A
    H = (X.double().T @ X.double()) / n
    return W, X, H


def _output_error(W: torch.Tensor, W_hat: torch.Tensor, X: torch.Tensor) -> float:
    return ((W - W_hat) @ X.T).norm().item()


def test_gptq_beats_rtn_on_correlated_inputs():
    W, X, H = _correlated_problem()
    W_gptq, info = gptq_quantize(W, H, bits=3, group=128, sym=True)
    W_rtn, _ = quantize_rtn(W, 3, 128, True)
    err_gptq = _output_error(W, W_gptq, X)
    err_rtn = _output_error(W, W_rtn, X)
    print(f"\n3-bit sym g128: output error GPTQ {err_gptq:.2f} vs RTN {err_rtn:.2f} "
          f"({100 * (1 - err_gptq / err_rtn):.1f}% lower)")
    assert err_gptq < 0.8 * err_rtn
    assert W_gptq.shape == W.shape and W_gptq.dtype == torch.float32
    assert info["index_bits"] == 3 * 64 * 256
    assert info["aux_bits"] == 64 * 2 * 16


def test_gptq_block_size_does_not_change_result():
    """The lazy-batch block is an implementation detail: block 64 (groups straddling blocks) ~ block 128."""
    W, X, H = _correlated_problem()
    W_a, _ = gptq_quantize(W, H, bits=3, group=128, block=128)
    W_b, _ = gptq_quantize(W, H, bits=3, group=128, block=64)
    W_c, _ = gptq_quantize(W, H, bits=3, group=128, block=48)
    # Float summation order differs between a rank-1 update and the batched matmul, and a flipped rounding
    # near a grid midpoint cascades along the row through the error feedback, so compare output errors,
    # not entries. The first 48 columns are computed with identical operations in all three runs.
    e_a, e_b, e_c = (_output_error(W, Wq, X) for Wq in (W_a, W_b, W_c))
    assert abs(e_b - e_a) < 0.02 * e_a
    assert abs(e_c - e_a) < 0.02 * e_a
    assert torch.equal(W_a[:, :48], W_c[:, :48])


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("group,block", [(128, 128), (64, 128), (128, 64), (256, 128)])
def test_gptq_identity_hessian_equals_rtn(sym: bool, group: int, block: int):
    g = torch.Generator().manual_seed(1)
    W = torch.randn(64, 256, generator=g)
    H = torch.eye(256, dtype=torch.float64)
    W_gptq, info_g = gptq_quantize(W, H, bits=3, group=group, sym=sym, block=block)
    W_rtn, info_r = quantize_rtn(W, 3, group, sym)  # the real RTN (f16-rounded scale), not a local re-implementation
    assert torch.equal(W_gptq, W_rtn)
    assert info_g["index_bits"] == info_r["index_bits"] and info_g["aux_bits"] == info_r["aux_bits"]


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_gptq_identity_hessian_equals_rtn_all_widths(sym: bool, bits: int):
    g = torch.Generator().manual_seed(3)
    W = 0.02 * torch.randn(64, 1024, generator=g)  # real-weight magnitude: the f16 scale rounding matters here
    H = torch.eye(1024, dtype=torch.float64)
    W_gptq, _ = gptq_quantize(W, H, bits=bits, group=128, sym=sym)
    assert torch.equal(W_gptq, quantize_rtn(W, bits, 128, sym)[0])


def test_gptq_scale_is_f16_realizable():
    g = torch.Generator().manual_seed(4)
    w = torch.randn(64, 128, generator=g)
    for sym in (True, False):
        scale, zero = grid_params(w, 4, sym)
        assert scale.shape == zero.shape == (64,)
        assert torch.equal(scale, scale.half().float())
        assert torch.all(zero >= 0) and torch.all(zero <= 15) and torch.equal(zero, zero.round())


def test_gptq_rejects_ragged_groups_like_rtn():
    for shape in ((16, 1000), (16, 64)):
        W = torch.randn(*shape)
        H = torch.eye(shape[1], dtype=torch.float64)
        with pytest.raises(ValueError):
            gptq_quantize(W, H, bits=4, group=128)
        with pytest.raises(ValueError):
            quantize_rtn(W, 4, 128)


@pytest.mark.parametrize("sym", [True, False])
def test_gptq_tiny_group_with_exact_zero_is_finite(sym: bool):
    # f32 scale below the f16 subnormal floor: the guard must apply after the f16 rounding (0/0 otherwise)
    W = torch.full((4, 256), 2.0**-123)
    W[0, 3] = 0.0
    W[:, 128:] = torch.randn(4, 128)
    H = torch.eye(256, dtype=torch.float64)
    W_gptq, _ = gptq_quantize(W, H, bits=4, group=128, sym=sym)
    assert torch.isfinite(W_gptq).all()
    assert torch.equal(W_gptq, quantize_rtn(W, 4, 128, sym)[0])


def test_gptq_identity_hessian_act_order_per_row():
    """With one group per row the column order is irrelevant: act_order must still reproduce RTN."""
    g = torch.Generator().manual_seed(2)
    W = torch.randn(32, 128, generator=g)
    H = torch.eye(128, dtype=torch.float64)
    W_gptq, info = gptq_quantize(W, H, bits=4, group=128, act_order=True)
    assert torch.equal(W_gptq, quantize_rtn(W, 4, 128, True)[0])
    assert info["aux_bits"] == 32 * 16


def test_gptq_damp_escalation_is_recorded_in_note(caplog):
    """A Hessian with a negative eigenvalue needs the 10x damping escape hatch; the note must say so."""
    caplog.set_level(logging.WARNING, logger="wcomp.quant.gptq")
    g = torch.Generator().manual_seed(5)
    Q, _ = torch.linalg.qr(torch.randn(64, 64, generator=g, dtype=torch.float64))
    eig = torch.ones(64, dtype=torch.float64)
    eig[0] = -0.05  # mean(diag H) ~ 1: damp 0.01 leaves it negative, damp 0.1 makes it positive definite
    H = Q @ torch.diag(eig) @ Q.T
    W = torch.randn(16, 64, generator=g)
    W_hat, info = gptq_quantize(W, H, bits=4, group=64, sym=True)
    assert torch.isfinite(W_hat).all()
    assert info["note"].endswith("damp0.01->0.1"), info["note"]
    assert any("cholesky failed" in r.getMessage() for r in caplog.records)
    _, info_ok = gptq_quantize(W, torch.eye(64, dtype=torch.float64), bits=4, group=64, sym=True)
    assert info_ok["note"].endswith("damp0.01")


def test_gptq_dead_columns_are_zero():
    W, X, H = _correlated_problem(K=256)
    H = H.clone()
    H[5, :] = 0
    H[:, 5] = 0
    W_gptq, _ = gptq_quantize(W, H, bits=4, group=128)
    assert torch.all(W_gptq[:, 5] == 0)
    assert torch.isfinite(W_gptq).all()


_WORDS = (
    "the of and to in a is that for it as was with be by on not he this are or his from at which but have an had "
    "they you were their one all we can her has there been if more when will would who so no she my about out up "
    "time year people way day man thing woman life child world school state family student group country problem "
    "hand part place case week company system program question work government number night point home water room"
).split()


@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(MODEL_DIR), reason="needs CUDA and the 0.8B")
def test_gptq_real_model(caplog):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    caplog.set_level(logging.INFO, logger="wcomp.quant.gptq")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    rng = random.Random(0)
    text = " ".join(rng.choice(_WORDS) for _ in range(8000))
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    assert ids.numel() >= 8 * 512
    calib = ids[: 8 * 512].view(8, 512)

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
    try:
        probe = calib[:1].cuda()
        with torch.no_grad():
            loss0 = model(probe, labels=probe).loss.item()
        w_before = model.model.layers[0].mlp.down_proj.weight.detach().clone()

        t0 = time.time()
        account = gptq_model(model, calib, bits=4, group=128, sym=True)
        dt = time.time() - t0
        for rec in caplog.records:
            print(rec.getMessage())
        print(f"gptq_model total {dt:.1f}s, bits_blocks {account.bits_blocks():.4f}, "
              f"bits_all {account.bits_all(model):.4f}, {len(account.tensors)} tensors")

        assert isinstance(account, BitAccount)
        assert len(account.tensors) == 18 * 8 + 6 * 7  # DeltaNet layers 5+3 linears, attention layers 4+3
        assert abs(account.bits_blocks() - 4.125) < 1e-6
        assert dt < 600
        w_after = model.model.layers[0].mlp.down_proj.weight.detach()
        assert w_after.dtype == torch.bfloat16 and not torch.equal(w_before, w_after)
        with torch.no_grad():
            loss1 = model(probe, labels=probe).loss.item()
        print(f"loss before {loss0:.3f} after 4-bit GPTQ {loss1:.3f}")
        assert math.isfinite(loss1)
        assert loss1 < 1.5 * loss0 + 1.0
    finally:
        del model
        torch.cuda.empty_cache()
