"""GPTQ (Frantar et al. 2022, Algorithm 1) for the block linears of a causal LM.

The quantization grid is literally the one of `wcomp.quant.rtn.quantize_rtn` (`group_grid`: f16-rounded
scale, sym: scale = amax / |qmin| per group; asym: scale / zero from the group's [min(w,0), max(w,0)] range
with a b-bit integer zero), so the bit accounting is identical to RTN and gptq_quantize(W, H=I) reproduces
quantize_rtn exactly. Group parameters are computed from the *current*, error-updated weights when the
group starts (column index % group == 0 in processing order).
"""
from __future__ import annotations

import logging
import math
import os
import time

import torch
import torch.nn as nn
from torch import Tensor

from wcomp.bits import BitAccount
from wcomp.quant.rtn import SCALE_BITS, group_grid

log = logging.getLogger(__name__)

# Hessian accumulation path (DESIGN.md "GPTQ speed"): True forms each window's x^T x in f32 and adds it into
# the f64 accumulator; False is the Phase 1 path (x cast to f64, addmm_ in f64; 1/64 f32 rate on consumer
# Blackwell). The Phase 1 path stays the default until the acceptance row (gptq-sym-g128-3 on the 0.8B, seed 0:
# PPL 24.45 within 0.5%, KL 0.3131 within 2%) is recorded in NOTES.md, so the Phase 1 method names keep producing
# their Phase 1 numbers; the f32 path is selected per run by scripts/run_baselines.py --hessian-f32 (or the env
# override WCOMP_HESSIAN_F32=1). A tensor note of the f32 path ends in " hessian-f32" (hessian_note); the default
# path leaves the Phase 1 note unchanged, so a Phase 1 method name rerun on the default path reproduces its Phase 1
# row note byte-for-byte (the readers scripts/plot_rd.hessian_tag and scripts/probe_io.hessian_tag treat a missing
# token and the older "hessian-f64" token alike).
HESSIAN_F32: bool = os.environ.get("WCOMP_HESSIAN_F32", "0") == "1"


def hessian_note(note: str, f32: bool | None = None) -> str:
    """note + " hessian-f32" when the f32-product path ran; unchanged (the Phase 1 note) on the default f64 path."""
    if f32 is None:
        f32 = HESSIAN_F32
    return note + " hessian-f32" if f32 else note


def add_xtx(acc: Tensor, x: Tensor, f32: bool | None = None) -> None:
    """acc [K, K] f64 += x^T x for x [n, K] (any float dtype). f32 defaults to the module flag HESSIAN_F32."""
    if f32 is None:
        f32 = HESSIAN_F32
    if f32:
        x32 = x.to(torch.float32)
        acc.add_((x32.T @ x32).to(torch.float64))
    else:
        x64 = x.to(torch.float64)
        acc.addmm_(x64.T, x64)


@torch.no_grad()
def accumulate_hessian_paths(acc: tuple[Tensor, Tensor, list[int]] | None, x: Tensor, K: int | None = None) -> tuple[Tensor, Tensor, list[int]]:
    """Add one window x [n, K] into (H32 sum, H64 sum, [rows]) through both paths (f32 products / f64 addmm_); a new
    accumulator is created from x when acc is None. Lets a hook keep two K x K f64 sums per linear instead of x."""
    x = x.reshape(-1, x.shape[-1]) if K is None else x.reshape(-1, K)
    if acc is None:
        K = x.shape[1]
        acc = (torch.zeros(K, K, dtype=torch.float64, device=x.device), torch.zeros(K, K, dtype=torch.float64, device=x.device), [0])
    add_xtx(acc[0], x, f32=True)
    add_xtx(acc[1], x, f32=False)
    acc[2][0] += x.shape[0]
    return acc


@torch.no_grad()
def hessian_path_check(W: Tensor, windows: list[Tensor] | None, bits: int, group: int = 128, sym: bool = True,
                       hessians: tuple[Tensor, Tensor, list[int] | int] | None = None) -> dict:
    """What the acceptance rule cannot separate on its own: the f32-product and f64 Hessians of the same windows
    are numerically indistinguishable (relative Frobenius ~1e-7 .. 1e-6), yet GPTQ's rounding decisions are
    chaotic in H, so the two paths yield different index sets of equal objective value. Returns
    {"rel_frob": ||H32 - H64||_F / ||H64||_F, "max_rel_diag": max_j |H32_jj - H64_jj| / H64_jj,
     "frac_changed": fraction of weights whose quantized value differs between the two paths,
     "objective_ratio": tr(dW32 H64 dW32^T) / tr(dW64 H64 dW64^T), "objective_f64": the latter}.
    A PPL / KL miss of the acceptance row together with frac_changed > 0 and objective_ratio ~ 1 is rounding
    chaos (to be read against the 4.4% seed spread), not an f32 precision defect.
    `hessians` = (H32 sum, H64 sum, rows) from accumulate_hessian_paths replaces `windows` (no x kept)."""
    K = W.shape[1]
    dev = W.device
    if hessians is not None:
        H32, H64, rows = hessians
        rows = rows[0] if isinstance(rows, list) else int(rows)
        H32 = H32.to(dev, torch.float64) / rows
        H64 = H64.to(dev, torch.float64) / rows
    else:
        acc = None
        for x in windows:
            acc = accumulate_hessian_paths(acc, x.to(dev), K)
        H32, H64, rows = acc[0] / acc[2][0], acc[1] / acc[2][0], acc[2][0]
    W32 = W.float()
    Q32, _ = gptq_quantize(W32, H32, bits, group, sym)
    Q64, _ = gptq_quantize(W32, H64, bits, group, sym)
    d32 = (W32 - Q32).double()
    d64 = (W32 - Q64).double()
    obj32 = float(((d32 @ H64) * d32).sum())
    obj64 = float(((d64 @ H64) * d64).sum())
    return {
        "rel_frob": float((H32 - H64).norm() / H64.norm()),
        "max_rel_diag": float(((H32.diagonal() - H64.diagonal()).abs() / H64.diagonal().clamp_min(1e-300)).max()),
        "frac_changed": float((Q32 != Q64).double().mean()),
        "objective_ratio": obj32 / obj64 if obj64 > 0 else 1.0,
        "objective_f64": obj64,
        "n_rows": rows,
    }


def grid_params(w: Tensor, bits: int, sym: bool) -> tuple[Tensor, Tensor]:
    """Per-row (scale, zero) [N] for the group `w` [N, g]: exactly quantize_rtn's grid (wcomp.quant.rtn.group_grid),
    including the f16 rounding of the scale and the clamped integer zero point."""
    _, scale, zero = group_grid(w, bits, sym)
    return scale.squeeze(1), zero.squeeze(1)


def quantize_column(w: Tensor, scale: Tensor, zero: Tensor, bits: int, sym: bool) -> Tensor:
    """Round one column `w` [N] onto the grid and dequantize (same operation order as group_grid)."""
    if sym:
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        qmin, qmax = 0, 2**bits - 1
    q = torch.clamp(torch.round(w / scale) + zero, qmin, qmax)
    return (q - zero) * scale


def _upper_cholesky_of_inverse(H: Tensor, damp: float) -> tuple[Tensor, float]:
    """(U, damp_used) with U^T U = (H + damp_used * mean(diag H) I)^-1, in f64.

    If the damped H is still not positive definite (numerically), the damping is multiplied by 10 and
    retried a few times; this is an escape hatch, not the normal path, and the caller records the damping
    actually used.
    """
    diag_mean = torch.mean(torch.diag(H))
    d = damp
    for _ in range(5):
        Hd = H.clone()
        Hd.diagonal().add_(d * diag_mean)
        try:
            L = torch.linalg.cholesky(Hd)
            Hinv = torch.cholesky_inverse(L)
            return torch.linalg.cholesky(Hinv, upper=True), d
        except RuntimeError:
            log.warning("cholesky failed at damp=%g, retrying with 10x", d)
            d *= 10
    raise RuntimeError("Hessian not positive definite even after extra damping")


def gptq_quantize(
    W: Tensor,
    H: Tensor,
    bits: int,
    group: int = 128,
    sym: bool = True,
    block: int = 128,
    damp: float = 0.01,
    act_order: bool = False,
) -> tuple[Tensor, dict]:
    """Quantize W [N, K] (f32) given the input Hessian H [K, K] (f64, mean of x x^T).

    Columns are processed left to right (or by descending diag(H) with act_order) in lazy batches of
    `block` columns; the quantization error of each column is fed back onto the remaining columns through
    the upper Cholesky factor of H^-1. Dead columns (diag(H) == 0) are set to 0. Returns (W_hat f32 on
    W's device, info) with info = {"index_bits", "aux_bits", "note"}.
    """
    N, K = W.shape
    dev = W.device
    W = W.detach().to(dev, torch.float32).clone()
    H = H.detach().to(dev, torch.float64).clone()
    if group is None or group <= 0:
        group = K
    if K % group != 0:  # same rule as quantize_rtn, so the two methods never report different bits
        raise ValueError(f"K={K} not divisible by group={group}")

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    Hinv, damp_used = _upper_cholesky_of_inverse(H, damp)
    Hinv = Hinv.to(torch.float32)
    del H

    Q = torch.zeros_like(W)
    scale = zero = None
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            col = i1 + i
            if col % group == 0:
                # Group parameters from the current values: the in-block part of the group is up to date in
                # W1; columns past the block end still miss the pending in-block error, Err1[:, :i] @ Hinv.
                gend = min(col + group, K)
                parts = [W1[:, i : min(count, i + group)]]
                if gend > i2:
                    tail = W[:, i2:gend]
                    if i > 0:
                        tail = tail - Err1[:, :i] @ Hinv[i1:col, i2:gend]
                    parts.append(tail)
                scale, zero = grid_params(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0], bits, sym)
            w = W1[:, i]
            d = Hinv1[i, i]
            q = quantize_column(w, scale, zero, bits, sym)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    if act_order:
        Q = Q[:, invperm]
    if not torch.isfinite(Q).all():
        raise RuntimeError("gptq_quantize: non-finite reconstruction")

    n_groups = K // group
    per_group = SCALE_BITS + (0 if sym else bits)  # f16 scale (+ b-bit zero), the same side information as RTN
    aux_bits = N * n_groups * per_group
    note = f"gptq {bits}b {'sym' if sym else 'asym'} g{group} block{block} damp{damp}"
    if damp_used != damp:
        note += f"->{damp_used:g}"  # the Cholesky escape hatch escalated the damping
    if act_order:
        note += " act_order"
        if n_groups > 1:
            # groups are contiguous in the permuted order, so each column needs its group index
            aux_bits += K * math.ceil(math.log2(n_groups))
    info = {"index_bits": bits * N * K, "aux_bits": aux_bits, "note": note}
    return Q, info


class _Stop(Exception):
    """Raised by the layer-0 catcher hook to end the forward after capturing the layer input."""


def _capture_layer_inputs(
    model: nn.Module, layers: nn.ModuleList, calib_ids: Tensor, store_device: torch.device
) -> tuple[list[Tensor], dict[int, dict]]:
    """Inputs of layer 0 for every calibration window, and each layer's keyword arguments.

    The kwargs (position embeddings, masks, position ids, use_cache=False, ...) depend only on the window
    length, so they are recorded once from a full forward of window 0 and reused for every window and
    every pass through the layer. All windows must have the same length and no padding.
    """
    device = next(model.parameters()).device
    layer_kwargs: dict[int, dict] = {}

    def recorder(idx: int):
        def hook(mod, args, kwargs):
            if idx not in layer_kwargs:
                layer_kwargs[idx] = {k: v for k, v in kwargs.items() if k != "hidden_states"}

        return hook

    handles = [layer.register_forward_pre_hook(recorder(i), with_kwargs=True) for i, layer in enumerate(layers)]
    model.model(input_ids=calib_ids[:1].to(device), use_cache=False)
    for h in handles:
        h.remove()
    if len(layer_kwargs) != len(layers):
        raise RuntimeError(f"recorded kwargs for {len(layer_kwargs)} of {len(layers)} layers")

    inps: list[Tensor] = []

    def catcher(mod, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        inps.append(x.to(store_device))
        raise _Stop

    handle = layers[0].register_forward_pre_hook(catcher, with_kwargs=True)
    try:
        for j in range(calib_ids.shape[0]):
            try:
                model.model(input_ids=calib_ids[j : j + 1].to(device), use_cache=False)
            except _Stop:
                pass
    finally:
        handle.remove()
    return inps, layer_kwargs


def _run_layer(layer: nn.Module, x: Tensor, kwargs: dict) -> Tensor:
    out = layer(x, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def gptq_model(
    model: nn.Module,
    calib_ids: Tensor,
    bits: int,
    group: int = 128,
    sym: bool = True,
    block: int = 128,
    damp: float = 0.01,
    act_order: bool = False,
) -> BitAccount:
    """Quantize every nn.Linear under model.model.layers sequentially with GPTQ; weights replaced in bf16.

    For each layer: (1) run the layer on all calibration windows with hooks on each nn.Linear that
    accumulate H = x^T x / n_rows in f64 on the GPU (DeltaNet linears get their own input, so in_proj_*
    and out_proj have different H); (2) quantize each linear and write W_hat back in the weight dtype;
    (3) run the now quantized layer on all windows to produce the next layer's inputs (in place, so the
    stored activations never exceed one copy). Layer-0 inputs come from a catcher hook on layer 0 that
    stops the model forward; layer kwargs are recorded once (see _capture_layer_inputs).
    Calibration activations are kept on the GPU in bf16 when they fit, else on the CPU.
    """
    model.eval()
    device = next(model.parameters()).device
    layers = model.model.layers
    prefix = next(n for n, m in model.named_modules() if m is layers)

    n_win, seqlen = calib_ids.shape
    hidden = model.config.hidden_size
    act_bytes = n_win * seqlen * hidden * 2
    store_device = device
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if act_bytes > 0.5 * free:
            store_device = torch.device("cpu")
    log.info("gptq: %d windows x %d tokens, activations on %s (%.0f MB)", n_win, seqlen, store_device, act_bytes / 2**20)

    t0 = time.time()
    inps, layer_kwargs = _capture_layer_inputs(model, layers, calib_ids, store_device)
    log.info("gptq: captured layer-0 inputs in %.1fs", time.time() - t0)

    account = BitAccount()
    for i, layer in enumerate(layers):
        t_layer = time.time()
        kw = layer_kwargs[i]
        linears = [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]

        hess = {n: [torch.zeros(m.in_features, m.in_features, dtype=torch.float64, device=device), 0] for n, m in linears}

        def make_hook(name: str):
            def hook(mod, args):
                x = args[0].reshape(-1, args[0].shape[-1])
                add_xtx(hess[name][0], x)
                hess[name][1] += x.shape[0]

            return hook

        handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in linears]
        for x in inps:
            _run_layer(layer, x.to(device), kw)
        for h in handles:
            h.remove()
        t_hess = time.time() - t_layer

        t_q = time.time()
        for n, m in linears:
            H, rows = hess[n]
            H /= rows
            W_hat, info = gptq_quantize(m.weight.float(), H, bits, group, sym, block, damp, act_order)
            m.weight.data.copy_(W_hat.to(m.weight.dtype))
            note = hessian_note(info["note"])  # the f32 path is marked; the default path keeps the Phase 1 note
            account.add(f"{prefix}.{i}.{n}", m.weight.numel(), info["index_bits"], info["aux_bits"], note)
        del hess
        t_q = time.time() - t_q

        t_f = time.time()
        for j in range(len(inps)):
            inps[j] = _run_layer(layer, inps[j].to(device), kw).to(store_device)
        t_f = time.time() - t_f
        log.info(
            "gptq layer %2d: %d linears, hessian %.1fs, quantize %.1fs, forward %.1fs, total %.1fs",
            i, len(linears), t_hess, t_q, t_f, time.time() - t_layer,
        )

    del inps
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return account
