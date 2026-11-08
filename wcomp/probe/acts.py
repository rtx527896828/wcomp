"""Activation-based structure probes P5, P6a, P6b, P7 and P2b (DESIGN.md, "Phase 2 contract").

All activation statistics come from C4 calibration windows; WikiText-2 is touched only through wcomp.eval.evaluate
(P7). Nothing here writes a weight back except P7's zero-one-weight ablation, which restores the weight and asserts
the model checksum is unchanged. Gradients are measured in fisher_diag, never applied.
"""
from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import Tensor

from wcomp.eval import evaluate, reference_hidden
from wcomp.models import block_linears, layer_index, weight_type
from wcomp.probe.common import ProbeRecord, shuffle_within_tensor
from wcomp.quant.rtn import quantize_rtn

log = logging.getLogger(__name__)

BITS = 4  # the RTN width every MSE-based sensitivity refers to (Q = RTN-4 in the contract)
GROUP = 128
TAU_FRACTION = 0.05  # active channel: |x_j| > 0.05 * rms of the linear's inputs
RANK_FRACTIONS = (0.01, 0.05, 0.10, 0.25)
RANK_KEYS = ("1pct", "5pct", "10pct", "25pct")
USAGE_TYPES = ("down", "o", "out_proj")  # the P5 primary tensors (intermediate / attention-output inputs)


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _record(name: str, lin: nn.Linear, stat: dict, control: dict, gain: float | None, gain_c: float | None,
            note: str = "") -> ProbeRecord:
    return ProbeRecord(name, layer_index(name), weight_type(name), stat, control, gain, gain_c, note, lin.weight.numel())


def param_checksum(model: nn.Module) -> str:
    """sha256 over the raw bytes of every parameter, in named order (bit-exact unchanged-model check)."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- statistics


def gini(x: Tensor) -> float:
    """Gini coefficient of the non-negative values of x (0 for all equal, 1 - 1/n for one-hot)."""
    x = x.reshape(-1).double()
    n = x.numel()
    s = x.sum()
    if n == 0 or s <= 0:
        return 0.0
    xs = torch.sort(x).values
    i = torch.arange(1, n + 1, dtype=torch.float64, device=x.device)
    return float(2.0 * (i * xs).sum() / (n * s) - (n + 1.0) / n)


def allocation_saving(s: Tensor, n: Tensor, bits: int = BITS, prefix: str = "") -> dict[str, float]:
    """Bit-allocation saving for units with sensitivities s_u and sizes n_u (contract P6).

    d_u = s_u / n_u; with weights w_u = n_u / sum n, saving = 0.5*log2(mean_w(d) / geomean_w(d)) bits/param.
    Interpretation: d_u is floored at mean_w(d) * 2^(-2*bits), since b_u = bits + 0.5*log2(d_u / mean) cannot go
    below 0 bits (an exactly dead column would otherwise make the geomean 0 and the saving infinite); the floored
    weight fraction is reported. Overhead = ceil(log2(#distinct half-bit levels of b_u)) bits per unit / n_u.

    The D ~ 2^(-2b) model does not hold for units the floor assigns 0 bits, and a floored unit contributes
    0.5 * w_u * 2 * bits to the saving regardless of its measured s_u, so two guarded variants are returned too:
    `saving_cap2bit` floors d_u at mean * 2^(-2*(bits-2)) (b_u >= 2 bits) and `saving_unfloored` is the saving
    of the unfloored units alone (their own mean and geomean). Read `saving` next to `frac_floored`.
    """
    s = s.reshape(-1).double()
    n = n.reshape(-1).double().to(s.device)
    w = n / n.sum()
    d = s / n
    mean = float((w * d).sum())
    k = (prefix + "_") if prefix else ""
    if mean <= 0:
        return {k + "saving": 0.0, k + "overhead": 0.0, k + "gain": 0.0, k + "levels": 1, k + "frac_floored": 0.0,
                k + "saving_cap2bit": 0.0, k + "saving_unfloored": 0.0}
    pos = d[d > 0]
    spread = float(torch.log10(pos.max() / pos.min())) if pos.numel() > 0 else 0.0
    floor = mean * 2.0 ** (-2 * bits)
    keep = d >= floor
    frac_floored = float((w * (~keep).double()).sum())
    if keep.any():
        w_k = w[keep] / w[keep].sum()
        mean_k = float((w_k * d[keep]).sum())
        saving_unfloored = 0.5 * (math.log2(mean_k) - float((w_k * torch.log2(d[keep])).sum()))
    else:
        saving_unfloored = 0.0
    floor2 = mean * 2.0 ** (-2 * max(bits - 2, 0))
    saving_cap2bit = 0.5 * (math.log2(mean) - float((w * torch.log2(d.clamp(min=floor2))).sum()))
    d = d.clamp(min=floor)
    saving = 0.5 * (math.log2(mean) - float((w * torch.log2(d)).sum()))
    b_u = bits + 0.5 * torch.log2(d / mean)
    levels = int(torch.unique(torch.round(2 * b_u)).numel())
    overhead = (math.ceil(math.log2(levels)) if levels > 1 else 0) * d.numel() / float(n.sum())
    return {
        k + "saving": saving,
        k + "overhead": overhead,
        k + "gain": saving - overhead,
        k + "levels": levels,
        k + "frac_floored": frac_floored,
        k + "spread_log10": spread,
        k + "saving_cap2bit": saving_cap2bit,
        k + "saving_unfloored": saving_unfloored,
    }


def rank1_energy(M: Tensor, iters: int = 60) -> float:
    """Energy fraction sigma_1^2 / ||M||_F^2 of the best rank-1 fit, by power iteration on M^T M (f32)."""
    M = M.float()
    v = M.sum(0)
    if float(v.norm()) == 0:
        v = torch.randn(M.shape[1], generator=torch.Generator().manual_seed(0)).to(M.device)
    v = v / v.norm()
    for _ in range(iters):
        u = M @ v
        v = M.T @ u
        nv = v.norm()
        if float(nv) == 0:
            return 0.0
        v = v / nv
    sigma1 = float((M @ v).norm())
    total = float(M.pow(2).sum())
    return sigma1 ** 2 / total if total > 0 else 0.0


def rank_profile(sv: Tensor, N: int, K: int, prefix: str = "") -> dict[str, float]:
    """P2 statistics from singular values: energy at the rank grid, ranks for 90/99% energy, stable rank, and the
    best low-rank-plus-quantized-residual gain `0.5*log2(1/e_r) - 16*r*(N+K)/(N*K)` over the grid."""
    e2 = sv.double() ** 2
    tot = e2.sum()
    cum = e2.cumsum(0) / tot
    m = min(N, K)
    k = (prefix + "_") if prefix else ""
    out: dict[str, float] = {k + "stable_rank": float(tot / e2[0])}
    best_gain, best_r = -math.inf, 0
    for frac, key in zip(RANK_FRACTIONS, RANK_KEYS):
        r = max(1, int(round(frac * m)))
        E = float(cum[r - 1])
        e_r = max(1.0 - E, 1e-12)
        gain = 0.5 * math.log2(1.0 / e_r) - 16.0 * r * (N + K) / (N * K)
        out[k + f"energy_r{key}"] = E
        out[k + f"gain_r{key}"] = gain
        if gain > best_gain:
            best_gain, best_r = gain, r
    out[k + "rank_90"] = int((cum < 0.90).sum()) + 1
    out[k + "rank_99"] = int((cum < 0.99).sum()) + 1
    out[k + "best_gain"] = best_gain
    out[k + "best_r"] = best_r
    return out


# --------------------------------------------------------------------------------- input statistics (hooks)


class _Stop(Exception):
    """Raised by a pre-hook on the first decoder layer past the current chunk to end the forward early."""


def _run_until(model: nn.Module, ids: Tensor, stop_layer: int | None) -> None:
    """Forward model.model on ids [1, T]; with stop_layer < n_layers the forward is aborted when that layer starts."""
    layers = model.model.layers
    if stop_layer is None or stop_layer >= len(layers):
        model.model(input_ids=ids, use_cache=False)
        return

    def stop(mod, args):
        raise _Stop

    handle = layers[stop_layer].register_forward_pre_hook(stop)
    try:
        model.model(input_ids=ids, use_cache=False)
    except _Stop:
        pass
    finally:
        handle.remove()


@torch.no_grad()
def collect_input_stats(model: nn.Module, calib_ids: Tensor, full_hessian: bool = False, layer_chunk: int = 6) -> dict[str, dict]:
    """Input statistics of every block linear over the calibration windows, results on the CPU.

    Per linear: "diag_h" [K] f32 = sum_tokens x_j^2 / n, "active" [K] int64 = count(|x_j| > tau), "n_tokens",
    "rms" (over all inputs of that linear), "tau" = 0.05 * rms, "numel" / "shape" of the weight, and with
    full_hessian "H" [K, K] f32 = x^T x / n. Pre-hooks on the linears of `layer_chunk` layers at a time (the full H
    of every linear of a 2B model does not fit next to the model); the forward stops at the chunk's last layer.
    Two passes per chunk: the first gives sum x^2 (so tau is exact), the second counts and accumulates H. Each hook
    sees its own linear's input (in_proj_* share the DeltaNet layer input, out_proj has the core output).
    """
    model.eval()
    dev = _device(model)
    linears = block_linears(model)
    n_layers = len(model.model.layers)
    out: dict[str, dict] = {}
    for start in range(0, n_layers, layer_chunk):
        stop = start + layer_chunk
        chunk = [(n, m) for n, m in linears if start <= layer_index(n) < stop]
        acc = {n: {"sumsq": torch.zeros(m.in_features, dtype=torch.float32, device=dev), "n": 0} for n, m in chunk}

        def hook_sq(name: str):
            def hook(mod, args):
                x = args[0].reshape(-1, args[0].shape[-1]).float()
                acc[name]["sumsq"] += (x * x).sum(0)
                acc[name]["n"] += x.shape[0]

            return hook

        handles = [m.register_forward_pre_hook(hook_sq(n)) for n, m in chunk]
        for j in range(calib_ids.shape[0]):
            _run_until(model, calib_ids[j : j + 1].to(dev), stop)
        for h in handles:
            h.remove()

        for n, m in chunk:
            a = acc[n]
            a["rms"] = math.sqrt(float(a["sumsq"].sum()) / (a["n"] * m.in_features))
            a["tau"] = TAU_FRACTION * a["rms"]
            a["active"] = torch.zeros(m.in_features, dtype=torch.int64, device=dev)
            if full_hessian:
                a["H"] = torch.zeros(m.in_features, m.in_features, dtype=torch.float32, device=dev)

        def hook_cnt(name: str):
            tau = acc[name]["tau"]

            def hook(mod, args):
                x = args[0].reshape(-1, args[0].shape[-1])
                acc[name]["active"] += (x.abs() > tau).sum(0)
                if full_hessian:
                    x32 = x.float()
                    acc[name]["H"].addmm_(x32.T, x32)

            return hook

        handles = [m.register_forward_pre_hook(hook_cnt(n)) for n, m in chunk]
        for j in range(calib_ids.shape[0]):
            _run_until(model, calib_ids[j : j + 1].to(dev), stop)
        for h in handles:
            h.remove()

        for n, m in chunk:
            a = acc[n]
            rec = {
                "diag_h": (a["sumsq"] / a["n"]).cpu(),
                "active": a["active"].cpu(),
                "n_tokens": int(a["n"]),
                "rms": a["rms"],
                "tau": a["tau"],
                "numel": m.weight.numel(),
                "shape": tuple(m.weight.shape),
            }
            if full_hessian:
                rec["H"] = (a["H"] / a["n"]).cpu()
            out[n] = rec
        del acc
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        log.info("input stats: layers %d-%d done (%d linears)", start, min(stop, n_layers) - 1, len(chunk))
    return out


# ------------------------------------------------------------------------------------------------------ P5


def usage_stats(stats: dict[str, dict], stats_half: dict[str, dict] | None = None) -> list[ProbeRecord]:
    """P5: neuron usage from per-channel input energy diag(H), for every block linear (primary: down/o/out_proj).

    stat: gini of channel energy, fraction of channels carrying 50% / 90% of the energy, energy share of the top 20%
    channels, fraction of channels active on < 1% of tokens / never, max/median energy. With `stats_half` (the
    seed-2 C4 half) the Spearman of channel energy between the halves; Gini is permutation invariant, so the control
    reported is that split-half agreement (gini of the second half included). No gain here (see P6, columns).
    """
    recs = []
    for name, st in stats.items():
        e = st["diag_h"].double()
        K = e.numel()
        tot = float(e.sum())
        cum = torch.sort(e, descending=True).values.cumsum(0) / max(tot, 1e-30)
        rate = st["active"].double() / st["n_tokens"]
        stat = {
            "gini": gini(e),
            "frac_channels_50": (int((cum < 0.5).sum()) + 1) / K,
            "frac_channels_90": (int((cum < 0.9).sum()) + 1) / K,
            "energy_share_top20pct": float(cum[max(1, int(round(0.2 * K))) - 1]),
            "frac_active_lt_1pct": float((rate < 0.01).double().mean()),
            "frac_never_active": float((rate == 0).double().mean()),
            "mean_active_rate": float(rate.mean()),
            "energy_max_over_median": float(e.max() / e.median()) if float(e.median()) > 0 else None,
            "rms": st["rms"],
            "tau": st["tau"],
            "n_tokens": st["n_tokens"],
        }
        control: dict = {}
        if stats_half is not None and name in stats_half:
            e2 = stats_half[name]["diag_h"].double()
            rho = spearmanr(e.numpy(), e2.numpy()).correlation
            stat["spearman_split_half"] = float(rho)
            control["spearman_split_half"] = float(rho)
            control["gini_half2"] = gini(e2)
            control["n_tokens_half2"] = stats_half[name]["n_tokens"]
        wt = weight_type(name)
        note = ("primary (P5 target type); " if wt in USAGE_TYPES else "") + \
            "Gini is permutation invariant: control = split-half Spearman of channel energy (C4 seeds 0 vs 2)"
        recs.append(ProbeRecord(name, layer_index(name), wt, stat, control, None, None, note, st["numel"]))
    return recs


# ----------------------------------------------------------------------------------------------------- P6a


@torch.no_grad()
def hessian_sensitivity(model: nn.Module, stats: dict[str, dict], bits: int = BITS, group: int = GROUP) -> tuple[list[ProbeRecord], dict]:
    """P6a: s_t = sum_j H_jj * sum_i (W - RTN_b(W))_ij^2 per block linear, and the allocation savings.

    Per record (within the tensor, fixed tensor average): allocation at 128-column-group and column granularity;
    gain_bits = column-granularity gain. The returned global dict holds the model-level allocation across all
    tensors / groups / columns (fixed model average), which is what the contract's saving at each granularity means;
    the within-tensor numbers decompose it (global column saving = tensor saving + numel-weighted within-tensor
    column saving, up to the 2^(-2b) floor). The saving is permutation invariant, so no shuffle control; the
    prediction check against the P0 @only KLs is done by `prediction_check`.

    s_t is in absolute output units (||dW X||^2), so across weight types it tracks each linear's input scale
    (post-norm inputs of q/k/v/gate/up/in_proj vs the small inputs of down/o/out_proj) as much as loss
    sensitivity; the model-wide allocation therefore floors whole types at 0 bits. Each record also carries
    `out_energy` = sum_j H_jj sum_i W_ij^2 (||W X||^2) and `s_rel` = s_t / out_energy (relative output error),
    and the global dict carries `alloc_tensor_within_type_*` (tensor granularity restricted to units of one type,
    numel-weighted over types) and `alloc_tensor_rel_*` (tensor granularity on s_rel) next to the contract's
    absolute-units numbers, plus the floored fractions; the headline is to be read with those.
    """
    dev = _device(model)
    recs: list[ProbeRecord] = []
    s_t, n_t, s_g, n_g, s_c, n_c = [], [], [], [], [], []
    s_rel_t: list[float] = []
    by_type: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for name, lin in block_linears(model):
        W = lin.weight.float()
        N, K = W.shape
        g = group if K % group == 0 else K  # every real block linear has K % 128 == 0; the fallback is for tiny test models
        W_hat, _ = quantize_rtn(W, bits, g, True)
        col_err = ((W - W_hat) ** 2).sum(0).double()  # [K]
        h = stats[name]["diag_h"].to(dev).double()
        s_col = h * col_err
        s_grp = s_col.view(K // g, g).sum(1)
        s_tensor = float(s_col.sum())
        out_energy = float((h * (W.double() ** 2).sum(0)).sum())
        stat = {
            "s": s_tensor,
            "s_per_param": s_tensor / (N * K),
            "mse_rtn": float(col_err.sum()) / (N * K),
            "mean_diag_h": float(h.mean()),
            "frac_dead_cols": float((h == 0).double().mean()),
            "out_energy": out_energy,
            "s_rel": s_tensor / out_energy if out_energy > 0 else None,
        }
        stat.update(allocation_saving(s_grp, torch.full((K // g,), float(N * g)), bits, "alloc_group"))
        stat.update(allocation_saving(s_col, torch.full((K,), float(N)), bits, "alloc_col"))
        note = ("permutation-invariant statistic: shuffle control replaced by the P0 prediction check "
                "(see global.prediction_check); within-tensor allocation at fixed tensor average")
        recs.append(_record(name, lin, stat, {}, stat["alloc_col_gain"], None, note))
        s_t.append(s_tensor)
        n_t.append(float(N * K))
        s_rel_t.append(s_tensor / out_energy if out_energy > 0 else 0.0)
        by_type[weight_type(name)][0].append(s_tensor)
        by_type[weight_type(name)][1].append(float(N * K))
        s_g.append(s_grp.cpu())
        n_g.append(torch.full((K // g,), float(N * g)))
        s_c.append(s_col.cpu())
        n_c.append(torch.full((K,), float(N)))
    glob: dict = {"bits": bits, "group": group}
    glob.update(allocation_saving(torch.tensor(s_t), torch.tensor(n_t), bits, "alloc_tensor"))
    glob.update(allocation_saving(torch.cat(s_g), torch.cat(n_g), bits, "alloc_group"))
    glob.update(allocation_saving(torch.cat(s_c), torch.cat(n_c), bits, "alloc_col"))
    # Guarded cross-tensor numbers: within-type allocation (no across-type comparison of absolute units) and the
    # allocation on the relative output error s_t / ||W X||^2.
    within = {"saving": 0.0, "gain": 0.0, "frac_floored": 0.0}
    total_n = sum(n_t)
    for s_list, n_list in by_type.values():
        a = allocation_saving(torch.tensor(s_list), torch.tensor(n_list), bits)
        wt = sum(n_list) / total_n
        for key in within:
            within[key] += wt * a[key]
    glob.update({f"alloc_tensor_within_type_{key}": v for key, v in within.items()})
    glob.update(allocation_saving(torch.tensor(s_rel_t), torch.tensor(n_t), bits, "alloc_tensor_rel"))
    glob["caveat"] = ("alloc_*_saving uses s_t in absolute output units and floors units at 0 bits (D ~ 2^-2b invalid "
                      "there): read it with alloc_*_frac_floored, alloc_*_saving_cap2bit / _unfloored, "
                      "alloc_tensor_within_type_saving, alloc_tensor_rel_saving and the prediction_check Spearman")
    return recs, glob


def prediction_check(records: list[ProbeRecord], ablation_rows: list[dict], stat_key: str = "s") -> dict:
    """Spearman of the per-type (sum over layers) and per-layer (sum over types) sensitivity with the measured P0
    `@only=<type>` KL (4 and 3 bits) and `@layer=<i>` KL (3 bits) rows, where those rows exist.

    The sensitivity is the RTN-4 proxy for every row; the 3-bit comparisons are rank correlations across the same
    tensors at a different width, stated as such.
    """
    s_type: dict[str, float] = defaultdict(float)
    s_layer: dict[int, float] = defaultdict(float)
    for r in records:
        s_type[r.type] += float(r.stat[stat_key])
        s_layer[r.layer] += float(r.stat[stat_key])
    out: dict = {}
    for bits in (4, 3):
        kl: dict[str, float] = {}
        for row in ablation_rows:
            m = row.get("method", "")
            if m.startswith(f"rtn-sym-g128-{bits}@only=") and row.get("kl") is not None:
                kl[m.split("=", 1)[1]] = row["kl"]
        common = sorted(set(kl) & set(s_type))
        out[f"n_types_only_{bits}bit"] = len(common)
        if len(common) >= 3:
            out[f"spearman_type_only_{bits}bit"] = float(spearmanr([s_type[t] for t in common], [kl[t] for t in common]).correlation)
    kl_layer: dict[int, float] = {}
    for row in ablation_rows:
        m = row.get("method", "")
        if m.startswith("rtn-sym-g128-3@layer=") and row.get("kl") is not None:
            kl_layer[int(m.split("=", 1)[1])] = row["kl"]
    common_l = sorted(set(kl_layer) & set(s_layer))
    out["n_layers_3bit"] = len(common_l)
    if len(common_l) >= 3:
        out["spearman_layer_3bit"] = float(spearmanr([s_layer[i] for i in common_l], [kl_layer[i] for i in common_l]).correlation)
    out["s_per_type"] = dict(s_type)
    out["s_per_layer"] = {str(k): v for k, v in sorted(s_layer.items())}
    return out


# ----------------------------------------------------------------------------------------------------- P6b


def fisher_accumulate(model: nn.Module, calib_ids: Tensor, n_windows: int = 32, seqlen: int = 1024,
                      batched: bool = True) -> tuple[dict[str, Tensor], dict[str, Tensor] | None, int, int]:
    """Position-wise empirical-Fisher approximation of every block-linear weight, from hooks, one forward/backward per window.

    For a linear y = W x the hooks give, per position t, the input x_t and the output gradient d_t = d(sum_s nll_s)/d y_t
    of the window's summed NLL, and the statistic is F = (1/T_total) sum_t (d_t^2)(x_t^2)^T, accumulated in f32 from
    each linear's input x [T, K] (forward hook) and output gradient d [T, N] (tensor hook on the output). This is the
    K-FAC-style independent-positions approximation: it equals the per-token empirical Fisher mean_t (d nll_t/dW)^2
    only in a model without token mixing (then d_t = d nll_t/d y_t and d nll_t/dW = d_t x_t^T); with attention or a
    recurrent state position t feeds later positions, the cross-position terms of d nll_t/dW are dropped and the two
    differ (a true per-token Fisher needs one backward per target position and is not computed here). No weight
    gradient is formed: every parameter stays requires_grad=False and only the embedding output is made a grad leaf so
    the backward reaches every linear (measurement, not training; nothing is written back).
    With `batched` the round-1 statistic is accumulated too: the per-window squared mean-NLL gradient
    (sum_t d_t x_t^T / T)^2 averaged over windows (kept on the CPU). It is not the empirical Fisher: cross-token terms
    survive the per-window sum, which lowers its rank-1 energy and raises its Gini.
    Returns (F per linear [N, K] f32 on the model device, batched F per linear or None, n_tokens, n_windows)."""
    dev = _device(model)
    linears = block_linears(model)
    before = param_checksum(model)
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    acc = {n: torch.zeros(lin.weight.shape, dtype=torch.float32, device=dev) for n, lin in linears}
    acc_b = {n: torch.zeros(lin.weight.shape, dtype=torch.float32) for n, lin in linears} if batched else None
    n = min(n_windows, calib_ids.shape[0])
    cur = {"targets": 1}

    def make_fwd(name: str):
        def fwd(mod, args, out):
            x = args[0]

            def on_grad(d: Tensor) -> None:
                xx = x.reshape(-1, x.shape[-1]).float()
                dd = d.reshape(-1, d.shape[-1]).float()
                acc[name].addmm_((dd * dd).T, xx * xx)
                if acc_b is not None:
                    g = dd.T @ xx  # the window's summed-NLL weight gradient; /targets^2 makes it the mean-NLL gradient squared
                    acc_b[name].add_((g * g).mul_(1.0 / cur["targets"] ** 2).cpu())

            if out.requires_grad:
                out.register_hook(on_grad)

        return fwd

    def leaf(mod, args, out):
        return out.detach().requires_grad_(True)

    handles = [lin.register_forward_hook(make_fwd(nm)) for nm, lin in linears]
    embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else model.model.embed_tokens
    handles.append(embed.register_forward_hook(leaf))
    n_tokens = 0
    try:
        for w in range(n):
            ids = calib_ids[w : w + 1, :seqlen].to(dev)
            cur["targets"] = int(ids.shape[1]) - 1
            with torch.enable_grad():
                h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
                logits = model.lm_head(h[:-1]).float()
                loss = F.cross_entropy(logits, ids[0, 1:], reduction="sum")  # sum: d_t = d(sum_s nll_s)/d y_t, the position-wise approximation (docstring)
                loss.backward()
            n_tokens += cur["targets"]
            del h, logits, loss
    finally:
        for hd in handles:
            hd.remove()
        for p in model.parameters():
            p.requires_grad_(False)
            p.grad = None
        model.train(was_training)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if param_checksum(model) != before:
        raise AssertionError("fisher_accumulate changed a parameter")
    for name in acc:
        acc[name] /= max(n_tokens, 1)
        if acc_b is not None:
            acc_b[name] /= max(n, 1)
    return acc, acc_b, n_tokens, n


def fisher_diag(model: nn.Module, calib_ids: Tensor, n_windows: int = 32, seqlen: int = 1024, bits: int = BITS,
                batched: bool = True) -> tuple[list[ProbeRecord], dict]:
    """P6b: position-wise empirical-Fisher approximation of the block-linear weights (fisher_accumulate), per tensor.

    stat (on the position-wise F): total, Gini over elements, rank-1 fit energy of F ~ r c^T, allocation saving at
    element / row / column granularity (within the tensor); gain_bits = best of the three. Control: F shuffled
    within the tensor (seed 0). With `batched` the round-1 per-window statistic is reported next to it under
    stat.batched_* (total, gini, rank1_energy, and the ratio of the totals) with its own shuffled rank-1 floor in
    control.batched_rank1_energy; the headline numbers and the H-P6 "rank-1 fit > 70 %" test use the position-wise F
    (not the per-token Fisher: the two coincide only without token mixing, see fisher_accumulate).
    The parameter checksum must be unchanged. The global dict carries the tensor-granularity allocation.
    """
    dev = _device(model)
    linears = block_linears(model)
    acc, acc_b, n_tokens, n = fisher_accumulate(model, calib_ids, n_windows, seqlen, batched)
    recs: list[ProbeRecord] = []
    totals, numels = [], []
    with torch.no_grad():
        for name, lin in linears:
            Fm = acc[name]
            N, K = Fm.shape
            W = lin.weight.float()
            W_hat, _ = quantize_rtn(W, bits, GROUP if K % GROUP == 0 else K, True)
            total = float(Fm.sum())
            stat = {
                "total": total,
                "per_param": total / (N * K),
                "gini": gini(Fm),
                "frac_zero": float((Fm == 0).double().mean()),
                "rank1_energy": rank1_energy(Fm),
                "s_rtn_fisher": float((Fm.double() * (W - W_hat).double() ** 2).sum()),
            }
            stat.update(allocation_saving(Fm.reshape(-1), torch.ones(N * K), bits, "alloc_elem"))
            stat.update(allocation_saving(Fm.sum(1), torch.full((N,), float(K)), bits, "alloc_row"))
            stat.update(allocation_saving(Fm.sum(0), torch.full((K,), float(N)), bits, "alloc_col"))
            Fc = shuffle_within_tensor(Fm, 0)
            control = {"rank1_energy": rank1_energy(Fc)}
            control.update(allocation_saving(Fc.reshape(-1), torch.ones(N * K), bits, "alloc_elem"))
            control.update(allocation_saving(Fc.sum(1), torch.full((N,), float(K)), bits, "alloc_row"))
            control.update(allocation_saving(Fc.sum(0), torch.full((K,), float(N)), bits, "alloc_col"))
            if acc_b is not None:
                Fb = acc_b[name].to(dev)
                tb = float(Fb.sum())
                stat.update({"batched_total": tb, "batched_gini": gini(Fb), "batched_rank1_energy": rank1_energy(Fb),
                             "batched_over_positionwise_total": tb / total if total > 0 else None})
                control["batched_rank1_energy"] = rank1_energy(shuffle_within_tensor(Fb, 0))
                acc_b[name] = None
                del Fb
            gain = max(stat["alloc_elem_gain"], stat["alloc_row_gain"], stat["alloc_col_gain"])
            gain_c = max(control["alloc_elem_gain"], control["alloc_row_gain"], control["alloc_col_gain"])
            note = (f"position-wise empirical-Fisher approximation over {n} C4 windows x {seqlen} tokens ({n_tokens} tokens): "
                    "F = mean_t (d_t^2)(x_t^2)^T with d_t = d(sum_s nll_s)/d y_t from hooks (K-FAC independent-positions "
                    "approximation; equals the per-token Fisher mean_t (d nll_t/dW)^2 only without token mixing), no weight gradient formed; "
                    "control = F shuffled within tensor (seed 0); gain = best within-tensor allocation of element/row/column granularity"
                    + ("; stat.batched_* = the per-window mean-NLL gradient squared (not a Fisher: cross-token terms survive), "
                       "for comparison only" if acc_b is not None else ""))
            recs.append(_record(name, lin, stat, control, gain, gain_c, note))
            totals.append(total)
            numels.append(float(N * K))
            del Fm, Fc, W, W_hat
            acc[name] = None
    glob = {"n_windows": n, "seqlen": seqlen, "n_tokens": n_tokens, "bits": bits, "checksum_unchanged": True,
            "definition": "position-wise empirical-Fisher approximation F = mean_t (d_t^2)(x_t^2)^T, d_t = d(sum_s nll_s)/d y_t "
                          "(K-FAC independent positions; equals the per-token Fisher only without token mixing) (headline, rank-1 test); "
                          "batched per-window gradient squared under stat.batched_*"}
    glob.update(allocation_saving(torch.tensor(totals), torch.tensor(numels), bits, "alloc_tensor"))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return recs, glob


# ----------------------------------------------------------------------------------------------------- P2b


def _cholesky_damped(H: Tensor, damp: float) -> tuple[Tensor, float]:
    """Lower Cholesky factor of H + d*mean(diag H)*I in f64, escalating d by 10x if H is not positive definite."""
    diag_mean = float(torch.mean(torch.diag(H)))
    d = damp
    for _ in range(5):
        Hd = H.clone()
        Hd.diagonal().add_(d * diag_mean)
        try:
            return torch.linalg.cholesky(Hd), d
        except RuntimeError:
            log.warning("cholesky failed at damp=%g, retrying with 10x", d)
            d *= 10
    raise RuntimeError("Hessian not positive definite even after extra damping")


def _rtn_output_error(X: Tensor, Lf: Tensor, bits: int, group: int) -> float:
    """||(X - RTN_b(X)) L||_F^2 = tr(dX H dX^T): the output-error energy of RTN on X under H = L L^T."""
    K = X.shape[1]
    X_hat, _ = quantize_rtn(X, bits, group if K % group == 0 else K, True)
    return float(((X - X_hat) @ Lf).double().pow(2).sum())


@torch.no_grad()
def weighted_rank_profile(W: Tensor, Lf: Tensor, bits: int = BITS, group: int = GROUP, prefix: str = "") -> dict[str, float]:
    """P2b for one matrix: the energy / rank statistics of W @ L (rank_profile) and, per r of the grid, the *measured*
    gain of a low-rank-plus-RTN-residual code.

    A_r = rank-r truncation of W L (f32 SVD), E_r = W - A_r L^-1 (the L-metric residual, stored quantized), and
      gain_r = 0.5 * log2(||dW L||^2 / ||dE_r L||^2) - 16 r (N + K) / (N K),  dX = X - RTN_b(X),
    i.e. the output-error ratio the quantizer actually reaches, minus bf16 U, V. The energy formula of P2,
    0.5*log2(1/e_r), is kept under formula_gain_r* / formula_best_gain as a statistic only: RTN noise scales with
    the weight-space amplitude of E_r, which stays near that of W (residual_weight_energy_r*), so the L-weighted
    residual energy e_r does not predict the output error (it overstated the gain 5-10x on the 0.8B).
    rtn_err_ratio_r* is the measured ||dW L||^2 / ||dE_r L||^2."""
    N, K = W.shape
    U, S, Vh = torch.linalg.svd(W @ Lf, full_matrices=False)
    prof = rank_profile(S, N, K)
    k = (prefix + "_") if prefix else ""
    out: dict[str, float] = {}
    for key, v in prof.items():
        out[k + ("formula_" + key if key.startswith("gain_r") or key in ("best_gain", "best_r") else key)] = v
    eW = _rtn_output_error(W, Lf, bits, group)
    w_energy = float(W.double().pow(2).sum())
    m = min(N, K)
    best_gain, best_r = -math.inf, 0
    for frac, key in zip(RANK_FRACTIONS, RANK_KEYS):
        r = max(1, int(round(frac * m)))
        A = (U[:, :r] * S[:r]) @ Vh[:r]
        E = W - torch.linalg.solve_triangular(Lf, A, upper=False, left=False)  # A L^-1
        eE = _rtn_output_error(E, Lf, bits, group)
        ratio = eW / max(eE, 1e-12 * eW) if eW > 0 else 1.0
        gain = 0.5 * math.log2(ratio) - 16.0 * r * (N + K) / (N * K)
        out[k + f"rtn_err_ratio_r{key}"] = ratio
        out[k + f"residual_weight_energy_r{key}"] = float(E.double().pow(2).sum()) / w_energy if w_energy > 0 else None
        out[k + f"gain_r{key}"] = gain
        if gain > best_gain:
            best_gain, best_r = gain, r
        del A, E
    out[k + "best_gain"] = best_gain
    out[k + "best_r"] = best_r
    del U, S, Vh
    return out


@torch.no_grad()
def weighted_low_rank(model: nn.Module, stats: dict[str, dict], damp: float = 0.01, control_seeds: tuple[int, ...] = (0, 1),
                      bits: int = BITS, group: int = GROUP) -> list[ProbeRecord]:
    """P2b: rank profile of W @ L with H = L L^T the (damped) calibration Hessian of each block linear, and the
    measured low-rank-plus-RTN-residual gain per r (weighted_rank_profile).

    ||dW X||^2 = ||dW L||^2, so the gain is the output-error ratio RTN-b reaches on the L-metric residual E_r versus
    on W, minus the bf16 U, V overhead. Control: W shuffled within the tensor (seeds 0 and 1), the same L, same
    measurement. Tensors without a full H in `stats` are skipped. Damping as GPTQ (0.01 x mean diag), recorded per
    record ("damp").
    """
    dev = _device(model)
    recs: list[ProbeRecord] = []
    for name, lin in block_linears(model):
        if "H" not in stats[name]:
            continue
        H = stats[name]["H"].to(dev, torch.float64)
        L, damp_used = _cholesky_damped(H, damp)
        Lf = L.float()
        del H, L
        W = lin.weight.float()
        N, K = W.shape
        stat = weighted_rank_profile(W, Lf, bits, group)
        stat["damp"] = damp_used
        stat["plain_stable_rank"] = rank_profile(torch.linalg.svdvals(W), N, K)["stable_rank"]
        control: dict = {}
        for seed in control_seeds:
            Wc = shuffle_within_tensor(W, seed)
            control.update(weighted_rank_profile(Wc, Lf, bits, group, f"seed{seed}"))
            del Wc
        gains_c = [control[f"seed{seed}_best_gain"] for seed in control_seeds]
        control["gain_bits_best"] = sum(gains_c) / len(gains_c)  # the P2 / P3 convention: mean over the control seeds
        control["gain_bits_best_spread"] = max(gains_c) - min(gains_c)
        formula_c = [control[f"seed{seed}_formula_best_gain"] for seed in control_seeds]
        control["formula_gain_bits_best"] = sum(formula_c) / len(formula_c)  # the contract's e_r formula on the control, next to stat.formula_best_gain
        note = ("activation-weighted rank of W @ chol(H + damp*mean(diag H) I); gain_r = measured "
                f"0.5*log2(||dW L||^2 / ||dE_r L||^2) - 16 r (N+K)/(N K) with E_r = W - lowrank_r(W L) L^-1 and dX = X - RTN-{bits} g{group}(X) "
                "(formula_gain_r* / formula_best_gain = the contract's P2 energy formula 0.5*log2(1/e_r) - overhead, a statistic, not bits/param; "
                "its control mean is control.formula_gain_bits_best); control = W shuffled "
                f"within tensor (seeds {list(control_seeds)}), same L, same measurement; gain_bits_control = mean over the control seeds")
        recs.append(_record(name, lin, stat, control, stat["best_gain"], control["gain_bits_best"], note))
        del Lf, W
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return recs


# ------------------------------------------------------------------------------------------------------ P7


@torch.no_grad()
def down_proj_spikes(model: nn.Module, calib_window: Tensor) -> list[dict]:
    """Per layer: max |input| and max |output| of mlp.down_proj on one window, with argmax channel and token."""
    dev = _device(model)
    downs = [(n, m) for n, m in block_linears(model) if weight_type(n) == "down"]
    found: dict[str, dict] = {}

    def make_hook(name: str):
        def hook(mod, args, out):
            x = args[0].reshape(-1, args[0].shape[-1]).float().abs()
            y = out.reshape(-1, out.shape[-1]).float().abs()
            xv, xi = x.reshape(-1).max(0)
            yv, yi = y.reshape(-1).max(0)
            found[name] = {
                "tensor": name,
                "layer": layer_index(name),
                "in_max": float(xv),
                "in_token": int(xi) // x.shape[1],
                "in_channel": int(xi) % x.shape[1],
                "out_max": float(yv),
                "out_token": int(yi) // y.shape[1],
                "out_channel": int(yi) % y.shape[1],
            }

        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in downs]
    try:
        model.model(input_ids=calib_window.reshape(1, -1).to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return [found[n] for n, _ in downs]


def super_weights(model: nn.Module, calib_window: Tensor, eval_ids: Tensor, spike_factor: float = 20.0, top_k: int = 16,
                  kl_windows: int = 16) -> tuple[list[ProbeRecord], dict]:
    """P7 (Yu et al. 2024 recipe): candidates from down_proj activation spikes plus the global top-k |w|.

    A layer whose down_proj output max exceeds spike_factor x the median layer max *and* whose input max exceeds
    spike_factor x the median input max (the contract's "co-occurs with an input spike") gives the candidate
    W[argmax output channel, argmax input channel] (kind "spike"); an output spike without an input spike is
    measured too but recorded under kind "spike_no_input", which +keepsuper does not restore. "same_token" says
    whether both maxima are at the same position. When no layer qualifies, the strongest output spike is measured
    under kind "max_spike_below_threshold" (not a candidate) unless it is one of the global top-k weights, in which
    case it is simply a "top<k>" candidate. Each distinct candidate is zeroed and evaluated on eval_ids[:kl_windows] (kl_windows for
    the KL) against the untouched model: the reference's final hidden states are computed before any edit
    (wcomp.eval.reference_hidden) and the reference logits use lm_head, which is never a block linear, so the
    reference is the untouched model without a second copy. Controls: the largest other |w| of the same tensor and
    one random weight of the same tensor (seed 0). Every weight is restored and the parameter checksum asserted.
    """
    dev = _device(model)
    model.eval()
    before = param_checksum(model)
    linears = dict(block_linears(model))
    layer_stats = down_proj_spikes(model, calib_window)
    out_med = float(torch.tensor([s["out_max"] for s in layer_stats]).median())
    in_med = float(torch.tensor([s["in_max"] for s in layer_stats]).median())
    cands: dict[tuple[str, int, int], dict] = {}
    for s in layer_stats:
        s["out_spike"] = s["out_max"] > spike_factor * out_med
        s["in_spike"] = s["in_max"] > spike_factor * in_med
        if s["out_spike"]:
            key = (s["tensor"], s["out_channel"], s["in_channel"])
            cands[key] = {"kind": "spike" if s["in_spike"] else "spike_no_input", "co_occurs": bool(s["in_spike"]),
                          "same_token": s["in_token"] == s["out_token"]}
    pool: list[tuple[float, str, int, int]] = []
    for name, lin in linears.items():
        W = lin.weight
        v, i = W.abs().reshape(-1).float().topk(min(top_k, W.numel()))
        for vv, ii in zip(v.tolist(), i.tolist()):
            pool.append((vv, name, ii // W.shape[1], ii % W.shape[1]))
    pool.sort(reverse=True)
    top_keys = [(name, r, c) for _, name, r, c in pool[:top_k]]
    if not cands:
        # No layer crosses the threshold: still measure the strongest output spike so the record says how far
        # the recipe's best candidate is from mattering (marked below threshold; not a contract candidate),
        # unless that weight is a top-k candidate anyway (then top-k membership is its kind).
        s = max(layer_stats, key=lambda d: d["out_max"])
        key = (s["tensor"], s["out_channel"], s["in_channel"])
        if key not in top_keys:
            cands[key] = {"kind": "max_spike_below_threshold", "co_occurs": bool(s["in_spike"]), "same_token": s["in_token"] == s["out_token"]}
    for rank, key in enumerate(top_keys):
        if key in cands:
            cands[key]["kind"] += f"+top{top_k}"
        else:
            cands[key] = {"kind": f"top{top_k}", "co_occurs": None, "same_token": None}
        cands[key]["magnitude_rank"] = rank

    eval16 = eval_ids[:kl_windows]
    ref_hidden = reference_hidden(model, eval16, kl_windows)
    ref = evaluate(model, None, eval16)
    cache: dict[tuple[str, int, int], dict] = {}

    def measure(name: str, r: int, c: int) -> dict:
        key = (name, r, c)
        if key not in cache:
            lin = linears[name]
            with torch.no_grad():
                old = lin.weight[r, c].clone()
                lin.weight[r, c] = 0
                try:
                    m = evaluate(model, model, eval16, kl_windows=kl_windows, ref_hidden=ref_hidden)
                finally:
                    lin.weight[r, c] = old
            cache[key] = {"row": r, "col": c, "value": float(old), "ppl": m["ppl"],
                          "dppl_pct": 100.0 * (m["ppl"] / ref["ppl"] - 1.0), "kl": m["kl"], "top1_agree": m["top1_agree"]}
        return cache[key]

    recs: list[ProbeRecord] = []
    for (name, r, c), meta in cands.items():
        lin = linears[name]
        N, K = lin.weight.shape
        stat = dict(measure(name, r, c))
        stat["abs_value"] = abs(stat["value"])
        stat["co_occurs"] = meta["co_occurs"]
        stat["same_token"] = meta["same_token"]
        stat["magnitude_rank"] = meta.get("magnitude_rank")
        with torch.no_grad():
            A = lin.weight.abs().float()
            A[r, c] = -1.0
            j = int(A.reshape(-1).argmax())
        g = torch.Generator().manual_seed(0)
        rr, cc = int(torch.randint(N, (1,), generator=g)), int(torch.randint(K, (1,), generator=g))
        while (rr, cc) == (r, c):
            rr, cc = int(torch.randint(N, (1,), generator=g)), int(torch.randint(K, (1,), generator=g))
        mag = measure(name, j // K, j % K)
        rnd = measure(name, rr, cc)
        control = {f"magnitude_{k}": v for k, v in mag.items()}
        control.update({f"random_{k}": v for k, v in rnd.items()})
        note = (f"{meta['kind']}; zeroed on {eval16.shape[0]} WikiText-2 windows, kl_windows {kl_windows}; controls: largest "
                "other |w| of the tensor, one random weight of the tensor (seed 0); measured dPPL/KL, no bits estimate")
        recs.append(_record(name, lin, stat, control, None, None, note))
        log.info("P7 %s[%d,%d]=%.4g (%s): dPPL %+.2f%% KL %.4f | magnitude ctrl dPPL %+.2f%% | random ctrl dPPL %+.2f%%",
                 name, r, c, stat["value"], meta["kind"], stat["dppl_pct"], stat["kl"], mag["dppl_pct"], rnd["dppl_pct"])
    after = param_checksum(model)
    if after != before:
        raise AssertionError("super_weights left the model changed")
    extra = {
        "layer_stats": layer_stats,
        "out_max_median": out_med,
        "in_max_median": in_med,
        "spike_factor": spike_factor,
        "top_k": top_k,
        "reference": {"ppl": ref["ppl"], "n_tokens": ref["n_tokens"], "windows": int(eval16.shape[0]), "kl_windows": kl_windows},
        "n_candidates": len(cands),
        "n_spike_candidates": sum(1 for m in cands.values() if "spike" in m["kind"].split("+")),
        "kinds": sorted({m["kind"] for m in cands.values()}),
        "n_evaluations": len(cache),
        "checksum_unchanged": True,
    }
    return recs, extra
