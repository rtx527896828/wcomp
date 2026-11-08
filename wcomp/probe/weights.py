"""Weight-only structure probes P1-P4 (DESIGN.md, Phase 2 contract). No data, no model forward.

Every function returns ProbeRecord(s) with the identity fields (tensor, layer, type) left for the caller to fill.
Inputs are 2-D weights [N, K] (N outputs, K inputs), bf16 or f32; the math runs in f32 (f64 where it is cheap).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from wcomp.probe.common import (
    ProbeRecord,
    bits_from_mse_ratio,
    entropy_bits,
    freq_table_overhead_bits,
    gaussian_control,
    log2_factorial,
    n_top_columns,
    shuffle_within_tensor,
)
from wcomp.quant.rtn import group_grid, quantize_rtn, quantize_rtn_keepcols

GROUP = 128
BITS = (4, 3, 2)
R_FRACS = (0.01, 0.05, 0.10, 0.25)
PCTS = (0.5, 1.0, 2.0)
STUDENT_NU_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)
STUDENT_EM_ITERS = 12
GAUSS_FRAC_GT4 = 6.334e-5  # P(|z| > 4) and P(|z| > 6) for an iid standard Gaussian (analytic references for the P1 control)
GAUSS_FRAC_GT6 = 1.973e-9
ENTROPY_GAIN_CONTROL_FLOOR = 0.005  # bits/param: |gain - gain_control| of an iid Gaussian W at b = 4/3/2 (std-estimation bias; 0.002-0.005 measured)
# Block linears whose input is the residual stream (K = hidden size): P3 cross-layer outlier consistency.
HIDDEN_INPUT_TYPES = frozenset({"q", "k", "v", "gate", "up", "in_proj_qkv", "in_proj_z", "in_proj_ab"})
# Axis of the intermediate (neuron) dimension for the P4 permutation alignment: rows of gate/up, columns of down.
INTERMEDIATE_AXIS = {"gate": 0, "up": 0, "down": 1}
EPS_ENERGY = 1e-12  # floor on a residual energy fraction (an exactly low-rank tensor would give log(0))
MEDIAN_MAX_NUMEL = 1 << 26  # above this the Laplace location is the mean (torch.median sorts a copy on the GPU)


def _group_size(K: int, group: int) -> int:
    """The Phase 1 grid uses `group` consecutive inputs; a K that is not divisible falls back to one group per row."""
    return group if group and group > 0 and K % group == 0 else K


def _f(x) -> float:
    return float(x.item() if isinstance(x, Tensor) else x)


# ------------------------------------------------------------------------------------------------ P1


def _moments(x: Tensor) -> tuple[float, float, float, float, float]:
    """(mean, std, excess kurtosis, frac beyond 4 sigma, frac beyond 6 sigma) of a flat f32 tensor."""
    mu = x.mean()
    xc = x - mu
    var = (xc * xc).mean()
    std = var.sqrt()
    m4 = (xc * xc).square().mean()
    kurt = m4 / (var * var) - 3.0 if var > 0 else torch.zeros(())
    a = xc.abs()
    frac4 = (a > 4 * std).float().mean()
    frac6 = (a > 6 * std).float().mean()
    return _f(mu), _f(std), _f(kurt), _f(frac4), _f(frac6)


def _family_nll(x: Tensor, nu_grid: Sequence[float] = STUDENT_NU_GRID) -> dict:
    """Per-element NLL (nats) of the MLE Gaussian, Laplace and Student-t (nu on a grid, scale by EM) fits."""
    n = x.numel()
    mu = x.mean()
    xc = x - mu
    var = (xc * xc).mean()
    nll_gauss = 0.5 * math.log(2 * math.pi * _f(var)) + 0.5
    loc = x.median() if n <= MEDIAN_MAX_NUMEL else mu
    b = (x - loc).abs().mean()
    nll_laplace = math.log(2 * _f(b)) + 1.0
    # Student-t with location mu: EM on the scale for each nu (w_i = (nu+1)/(nu + x_i^2/s2), s2 <- mean(w_i x_i^2)).
    x2 = xc * xc
    best_nll, best_nu = math.inf, None
    for nu in nu_grid:
        s2 = var * ((nu - 2) / nu) if nu > 2 else var * 0.5
        for _ in range(STUDENT_EM_ITERS):
            w = (nu + 1.0) / (nu + x2 / s2)
            s2 = (w * x2).mean()
        s2f = _f(s2)
        const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi * s2f)
        nll = -const + (nu + 1) / 2 * _f(torch.log1p(x2 / (nu * s2)).mean())
        if nll < best_nll:
            best_nll, best_nu = nll, nu
    nlls = {"gauss": nll_gauss, "laplace": nll_laplace, "student": best_nll}
    return {
        "nll_gauss": nll_gauss,
        "nll_laplace": nll_laplace,
        "nll_student": best_nll,
        "student_nu": best_nu,
        "best_family": min(nlls, key=nlls.get),
    }


def index_entropies(W: Tensor, bits: Sequence[int] = BITS, group: int = GROUP) -> dict[int, float]:
    """H_b of the pooled index distribution of the Phase 1 grid (sym, group g) at each b, using group_grid's q."""
    N, K = W.shape
    g = _group_size(K, group)
    Wg = W.float().reshape(N, K // g, g)
    out = {}
    for b in bits:
        q, _, _ = group_grid(Wg, b, True)
        out[b] = entropy_bits(q)
    return out


def bf16_word_entropies(W: Tensor) -> dict[str, float]:
    """Entropy (bits) of the bf16 word of W (65536 symbols) and of its sign / exponent / mantissa fields.

    W is cast to bf16 first (lossless for a weight that came from bf16).
    """
    words = W.to(torch.bfloat16).reshape(-1).view(torch.int16)
    counts = torch.zeros(65536, dtype=torch.int64, device=W.device)
    step = 1 << 26
    for i in range(0, words.numel(), step):
        idx = words[i : i + step].to(torch.int32) & 0xFFFF
        counts += torch.bincount(idx, minlength=65536)
    total = float(words.numel())

    def h(c: Tensor) -> float:
        p = c[c > 0].to(torch.float64) / total
        return float(-(p * torch.log2(p)).sum().item())

    codes = torch.arange(65536, device=W.device)
    sign = torch.bincount(codes >> 15, weights=counts.double(), minlength=2)
    expo = torch.bincount((codes >> 7) & 0xFF, weights=counts.double(), minlength=256)
    mant = torch.bincount(codes & 0x7F, weights=counts.double(), minlength=128)
    return {"H_word": h(counts), "H_sign": h(sign), "H_exp": h(expo), "H_mant": h(mant)}


def marginal_entropy(W: Tensor, group: int = GROUP, bits: Sequence[int] = BITS, seed: int = 0) -> ProbeRecord:
    """P1: marginal statistics, best-fit family, grid index entropies and the bf16 lossless bound.

    Control: iid Gaussian with the same per-group std (gaussian_control), same grid; the shuffle control is
    vacuous for a marginal. gain_bits of the record is the b = 4 entropy-coding gain (the Phase 1 width);
    the 3- and 2-bit gains and the lossless saving are in stat / control.
    """
    W32 = W.float()
    N, K = W32.shape
    numel = N * K
    x = W32.reshape(-1)
    mu, std, kurt, frac4, frac6 = _moments(x)
    stat: dict = {"numel": numel, "mean": mu, "std": std, "kurtosis_excess": kurt, "frac_gt4sigma": frac4, "frac_gt6sigma": frac6}
    stat.update(_family_nll(x))
    for b, H in index_entropies(W32, bits, group).items():
        stat[f"H_idx_b{b}"] = H
        stat[f"gain_bits_b{b}"] = b - H - freq_table_overhead_bits(b, numel)
    words = bf16_word_entropies(W)
    stat.update(words)
    stat["lossless_bits_saved"] = 16.0 - words["H_word"]

    G = gaussian_control(W32, group, seed)
    _, _, kurt_c, frac4_c, frac6_c = _moments(G.reshape(-1))
    # The control's own shape statistics include the per-group std estimation: pooled over groups it is a Gaussian
    # scale mixture with s^2 ~ sigma^2 chi2_(g-1)/g, excess kurtosis 3*Var(s^2)/E[s^2]^2 = 6/(g-1) (~0.047 at g = 128)
    # rather than 0, and slightly heavier tails; the analytic iid-Gaussian values are given next to it.
    g_eff = _group_size(K, group)
    control: dict = {"kurtosis_excess": kurt_c, "frac_gt4sigma": frac4_c, "frac_gt6sigma": frac6_c,
                     "kurtosis_excess_expected_from_std_estimation": 6.0 / (g_eff - 1) if g_eff > 1 else 0.0,
                     "gauss_kurtosis_excess": 0.0, "gauss_frac_gt4sigma": GAUSS_FRAC_GT4, "gauss_frac_gt6sigma": GAUSS_FRAC_GT6}
    for b, H in index_entropies(G, bits, group).items():
        control[f"H_idx_b{b}"] = H
        control[f"gain_bits_b{b}"] = b - H - freq_table_overhead_bits(b, numel)
    words_c = bf16_word_entropies(G)
    control.update(words_c)
    control["lossless_bits_saved"] = 16.0 - words_c["H_word"]
    del G
    b0 = bits[0]
    return ProbeRecord(
        tensor="", layer=-1, type="", stat=stat, control=control,
        gain_bits=stat[f"gain_bits_b{b0}"], gain_bits_control=control[f"gain_bits_b{b0}"],
        note=f"control: iid Gaussian, matched per-group std (g{g_eff}, seed {seed}); "
             f"gain_bits = b={b0} entropy-coding gain minus a 2^b x 16-bit frequency table; the control's own kurtosis / tail "
             f"fractions include the {g_eff}-sample std estimation (excess kurtosis ~ +{6.0 / max(g_eff - 1, 1):.3f} for a Gaussian W, "
             f"analytic Gaussian 0 / {GAUSS_FRAC_GT4:.2g} / {GAUSS_FRAC_GT6:.2g} under gauss_*), and |gain - gain_control| below about "
             f"{ENTROPY_GAIN_CONTROL_FLOOR} bits is within the control's floor",
        numel=numel,
    )


# ------------------------------------------------------------------------------------------------ P2


def singular_values(W: Tensor) -> Tensor:
    """Singular values of W [N, K], descending, f64. Direct f32 svdvals for tensors up to 2^26 elements;
    larger ones (the embedding) go through the f64 Gram matrix of the short side, accumulated in row chunks."""
    N, K = W.shape
    if W.numel() <= (1 << 26):
        return torch.linalg.svdvals(W.float()).double()
    A = W if N >= K else W.T  # tall
    m = A.shape[1]
    G = torch.zeros(m, m, dtype=torch.float64, device=W.device)
    step = max(1, (1 << 25) // m)
    for i in range(0, A.shape[0], step):
        c = A[i : i + step].double()
        G += c.T @ c
    ev = torch.linalg.eigvalsh(G).clamp_min(0)
    return ev.sqrt().flip(0)


def low_rank_stats(S: Tensor, N: int, K: int, r_fracs: Sequence[float] = R_FRACS, r_extra: Sequence[int] = ()) -> dict:
    """Energy fractions, ranks for 90/99% energy, stable rank and the best low-rank + quantized-residual gain.

    At rank r the residual energy fraction e_r saves 0.5*log2(1/e_r) bits on the residual; U, V in bf16 cost
    16*r*(N+K)/(N*K). The maximum is taken over the r grid (fractions of min(N, K), plus any r_extra).
    """
    S = S.double()
    m = S.numel()
    E = S * S
    total = E.sum()
    cum = E.cumsum(0) / total
    stat: dict = {
        "stable_rank": _f(total / E[0]) if E[0] > 0 else 0.0,
        "rank90": int(torch.searchsorted(cum, torch.tensor(0.9, dtype=cum.dtype, device=cum.device)).item()) + 1,
        "rank99": int(torch.searchsorted(cum, torch.tensor(0.99, dtype=cum.dtype, device=cum.device)).item()) + 1,
    }
    stat["rank90_frac"] = stat["rank90"] / m
    stat["rank99_frac"] = stat["rank99"] / m
    grid: list[tuple[str, int]] = []
    for f in r_fracs:
        r = min(m, max(1, int(round(f * m))))
        grid.append((f"f{f:g}", r))
    for r in r_extra:
        grid.append((f"r{int(r)}", min(m, max(1, int(r)))))
    best_gain, best_r, best_key = -math.inf, None, None
    for key, r in grid:
        energy = _f(cum[r - 1])
        e = max(1.0 - energy, EPS_ENERGY)
        saved = 0.5 * math.log2(1.0 / e)
        overhead = 16.0 * r * (N + K) / (N * K)
        gain = saved - overhead
        stat[f"energy_{key}"] = energy
        stat[f"gain_{key}"] = gain
        if gain > best_gain:
            best_gain, best_r, best_key = gain, r, key
    stat["gain_bits_best"] = best_gain
    stat["best_r"] = best_r
    stat["best_r_frac"] = best_r / m
    return stat


def low_rank(W: Tensor, controls: Sequence[Tensor], r_fracs: Sequence[float] = R_FRACS, r_extra: Sequence[int] = ()) -> ProbeRecord:
    """P2: spectrum statistics and the low-rank + residual gain on W and on each shuffled control."""
    N, K = W.shape
    stat = low_rank_stats(singular_values(W), N, K, r_fracs, r_extra)
    stat["numel"] = N * K
    control = _mean_over_controls([low_rank_stats(singular_values(C), N, K, r_fracs, r_extra) for C in controls], "gain_bits_best")
    return ProbeRecord(
        tensor="", layer=-1, type="", stat=stat, control=control,
        gain_bits=stat["gain_bits_best"], gain_bits_control=control.get("gain_bits_best"),
        note=f"control: {len(controls)} within-tensor shuffles (mean reported, per-seed gains in control); "
             "gain = 0.5*log2(1/e_r) - 16 r (N+K)/(N K), max over the r grid",
        numel=N * K,
    )


def _mean_over_controls(stats: Sequence[dict], gain_key: str) -> dict:
    """Mean of each numeric key over the control runs; the per-control gains and their spread are kept."""
    out: dict = {}
    if not stats:
        return out
    for k in stats[0]:
        vals = [s[k] for s in stats if isinstance(s.get(k), (int, float)) and not isinstance(s.get(k), bool)]
        if len(vals) == len(stats):
            out[k] = sum(vals) / len(vals)
    for i, s in enumerate(stats):
        out[f"{gain_key}_seed{i}"] = s[gain_key]
    gains = [s[gain_key] for s in stats]
    out[f"{gain_key}_spread"] = max(gains) - min(gains)
    return out


# ------------------------------------------------------------------------------------------------ P3


def _norm_stats(W: Tensor) -> dict:
    """max/median, counts above 5x and 10x the median and excess kurtosis of the column and row L2 norms."""
    out: dict = {}
    for name, v in (("col", W.norm(dim=0)), ("row", W.norm(dim=1))):
        med = v.median()
        _, _, kurt, _, _ = _moments(v)
        out[f"{name}_max_over_median"] = _f(v.max() / med) if med > 0 else 0.0
        out[f"{name}_n_gt5x_median"] = int((v > 5 * med).sum().item())
        out[f"{name}_n_gt10x_median"] = int((v > 10 * med).sum().item())
        out[f"{name}_kurtosis_excess"] = kurt
    return out


def rtn_mse(W: Tensor, bits: int = 4, group: int = GROUP) -> float:
    W_hat, _ = quantize_rtn(W, bits, _group_size(W.shape[1], group), True)
    return _f(((W - W_hat) ** 2).mean())


def rtn_mse_keepcols(W: Tensor, cols: Tensor, bits: int = 4, group: int = GROUP) -> float:
    """MSE of RTN over the full tensor when the columns `cols` (index or bool [K] tensor) are kept exactly and
    excluded from the grid: the P0 +keepcols code path (wcomp.quant.rtn.quantize_rtn_keepcols), so the P3
    estimate and the direct ablation quantize identically."""
    K = W.shape[1]
    mask = cols.to(W.device) if cols.dtype == torch.bool else torch.zeros(K, dtype=torch.bool, device=W.device).index_fill_(0, cols.to(W.device), True)
    W_hat, _ = quantize_rtn_keepcols(W, bits, _group_size(K, group), mask, True)
    return _f(((W - W_hat) ** 2).mean())


def _outlier_gains(W: Tensor, pcts: Sequence[float], bits: int, group: int) -> dict:
    N, K = W.shape
    col = W.norm(dim=0)
    order = torch.argsort(col, descending=True)
    mse = rtn_mse(W, bits, group)
    out: dict = {f"mse_rtn{bits}": mse}
    best, best_p = -math.inf, None
    for p in pcts:
        n_keep = n_top_columns(K, p)
        rho = mse / rtn_mse_keepcols(W, order[:n_keep], bits, group)
        # bf16 instead of b-bit on the n_keep kept columns + a K-bit mask: exactly the P0 +keepcols charge
        # (wcomp.quant.rtn.quantize_rtn_keepcols); the contract's p/100 * 12 is this at bits = 4 and K*p/100 integer.
        overhead = (16.0 - bits) * n_keep / K + 1.0 / N
        gain = bits_from_mse_ratio(rho) - overhead
        out[f"rho_p{p:g}"] = rho
        out[f"gain_p{p:g}"] = gain
        out[f"kept_frac_p{p:g}"] = n_keep / K
        if gain > best:
            best, best_p = gain, p
    out["gain_bits_best"] = best
    out["best_pct"] = best_p
    return out


def outliers(W: Tensor, controls: Sequence[Tensor], pcts: Sequence[float] = PCTS, bits: int = 4, group: int = GROUP) -> ProbeRecord:
    """P3 per tensor: column / row norm statistics and the keep-top-p%-columns RTN gain, on W and the controls."""
    W = W.float()
    N, K = W.shape
    stat = {"numel": N * K, **_norm_stats(W), **_outlier_gains(W, pcts, bits, group)}
    control = _mean_over_controls([{**_norm_stats(C.float()), **_outlier_gains(C.float(), pcts, bits, group)} for C in controls], "gain_bits_best")
    return ProbeRecord(
        tensor="", layer=-1, type="", stat=stat, control=control,
        gain_bits=stat["gain_bits_best"], gain_bits_control=control.get("gain_bits_best"),
        note=f"rho = MSE(RTN-{bits}) / MSE(RTN-{bits}, top-p% columns kept in bf16, excluded from the grid); "
             f"overhead (16-{bits})*n_keep/K + 1/N = the P0 +keepcols charge (contract: p/100*12 + 1/N); "
             f"top-p% = n_keep = round(p/100*K) columns (>= 1); control: {len(controls)} shuffles",
        numel=N * K,
    )


def outlier_consistency(col_norms: Sequence[tuple[int, Tensor]], top: int = 16, seed: int = 0) -> ProbeRecord:
    """P3 cross-layer consistency of the top-`top` input channels (by column norm) over the given layers.

    stat: mean pairwise Jaccard of the top sets; control: the same with random channel ids, one seeded draw per
    layer shared by that layer's entries (a pooled series holds several linears of one layer, which share their
    input; one draw per layer gives the control the same number of effectively independent pairs as W, so its
    variance is matched, not only its mean). Also the number of channels in the top set of at least half the layers.
    """
    if len(col_norms) < 2:
        raise ValueError("need at least two layers")
    K = col_norms[0][1].numel()
    tops = [set(torch.topk(v, min(top, K)).indices.tolist()) for _, v in col_norms]
    ctrl_by_layer: dict[int, set] = {}  # one draw per layer (for a one-entry-per-layer series this is the per-entry draw)
    for layer, _ in col_norms:
        if layer not in ctrl_by_layer:
            g = torch.Generator().manual_seed(seed + 1000003 * (layer + 1))
            ctrl_by_layer[layer] = set(torch.randperm(K, generator=g)[: min(top, K)].tolist())
    ctrl = [ctrl_by_layer[layer] for layer, _ in col_norms]

    layers = [layer for layer, _ in col_norms]
    # Only pairs from different layers count (a pooled series holds q/k/v/gate/up/in_proj_* of one layer, which share
    # their input and would inflate W against the independent control); the same pairs are used for the control.
    pairs = [(i, j) for i in range(len(tops)) for j in range(i + 1, len(tops)) if layers[i] != layers[j]]
    n_skipped = len(tops) * (len(tops) - 1) // 2 - len(pairs)
    if not pairs:
        raise ValueError("need at least two different layers")

    def mean_jaccard(sets: list[set]) -> float:
        return sum(len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i, j in pairs) / len(pairs)

    counts: dict[int, int] = {}
    for s in tops:
        for c in s:
            counts[c] = counts.get(c, 0) + 1
    half = math.ceil(len(tops) / 2)
    stat = {
        "n_layers": len(tops), "top": top, "K": K,
        "jaccard_mean": mean_jaccard(tops),
        "n_pairs": len(pairs),
        "n_distinct_layer_pairs": len({(layers[i], layers[j]) for i, j in pairs}),
        "n_same_layer_pairs_skipped": n_skipped,
        "n_channels_in_half_layers": sum(1 for c in counts.values() if c >= half),
        "n_distinct_channels": len(counts),
    }
    control = {"jaccard_mean": mean_jaccard(ctrl)}
    return ProbeRecord(
        tensor="", layer=-1, type="", stat=stat, control=control, gain_bits=None, gain_bits_control=None,
        note=f"top-{top} input channels by column L2 norm per layer, mean pairwise Jaccard over cross-layer pairs only "
             f"({n_skipped} same-layer pairs skipped); control: random channel ids, one draw per layer shared by its entries (seed {seed}), same pairs",
        numel=0,
    )


# ------------------------------------------------------------------------------------------------ P4


def gram_factor(X: Tensor) -> tuple[Tensor, Tensor] | None:
    """For a tall X [n, k] (n > k): (G, L) with G = X^T X and L L^T = G, in f64 (Cholesky, or the symmetric
    square root if G is numerically singular). None for n <= k, where the direct N x N product is used instead.
    Precomputed once per tensor, it makes every pair's Procrustes residual a k^3 problem (see procrustes_residual)."""
    n, k = X.shape
    if n <= k:
        return None
    X = X.double()
    G = X.T @ X
    try:
        L = torch.linalg.cholesky(G + torch.eye(k, dtype=G.dtype, device=G.device) * (G.diagonal().mean() * 1e-12))
    except RuntimeError:
        w, Q = torch.linalg.eigh(G)
        L = Q * w.clamp_min(0).sqrt()
    return G, L


def _nuclear_cross(A: Tensor, B: Tensor, fa, fb) -> Tensor:
    """||B A^T||_* for A, B [n, k] f64. With n > k: B A^T = Q_B L_B^T L_A Q_A^T for X = Q_X L_X^T (Q_X orthonormal
    columns), so its singular values are those of L_B^T L_A, whose squares are the eigenvalues of L_A^T G_B L_A."""
    if fa is None:
        M = B @ A.T
        return torch.linalg.eigvalsh(M @ M.T).clamp_min(0).sqrt().sum()
    _, LA = fa
    GB, _ = fb
    S = LA.T @ GB @ LA
    return torch.linalg.eigvalsh(S).clamp_min(0).sqrt().sum()


def procrustes_residual(A: Tensor, B: Tensor, fa=None, fb=None) -> float:
    """min over orthogonal R of ||B - R A||^2 / ||B||^2 for A, B [N, K] (R acts on the output axis),
    = (||A||^2 + ||B||^2 - 2 ||B A^T||_*) / ||B||^2, in f64. fa / fb: gram_factor(A) / gram_factor(B) if cached."""
    A = A.double()
    B = B.double()
    N, K = A.shape
    if N > K:
        fa = gram_factor(A) if fa is None else fa
        fb = gram_factor(B) if fb is None else fb
    else:
        fa = fb = None
    nuc = _nuclear_cross(A, B, fa, fb)
    nb = (B * B).sum()
    return _f(((A * A).sum() + nb - 2 * nuc) / nb)


def permutation_residual(A: Tensor, B: Tensor, axis: int) -> tuple[float, float]:
    """Best one-to-one matching of A's slices along `axis` to B's by cosine (linear_sum_assignment), then
    (||B - P A||^2 / ||B||^2, mean matched cosine). axis 0 permutes rows (neurons of gate/up), 1 columns (down)."""
    if axis == 1:
        A, B = A.T, B.T
    A = A.float()
    B = B.float()
    An = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Bn = B / B.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C = Bn @ An.T  # C[i, j] = cos(B_i, A_j)
    rows, cols = linear_sum_assignment(C.double().cpu().numpy(), maximize=True)
    cols_t = torch.as_tensor(cols, device=A.device)
    Ap = A[cols_t]  # Ap[i] is the A slice matched to B[i] (rows come back sorted 0..M-1)
    e = _f(((B - Ap) ** 2).sum() / (B * B).sum())
    cos_mean = _f(C[torch.as_tensor(rows, device=A.device), cols_t].mean())
    return e, cos_mean


def _pair_core(A: Tensor, B: Tensor, fo_a=None, fo_b=None, fi_a=None, fi_b=None) -> dict:
    """Raw cosine, raw residual and the output- / input-side Procrustes residuals of B against A (f32 in, f64
    math). fo_* / fi_*: cached gram_factor of the tensor / of its transpose (None: computed here)."""
    A64, B64 = A.double(), B.double()
    nb = (B64 * B64).sum()
    na = (A64 * A64).sum()
    N, K = A64.shape
    if N > K:
        fo_a = gram_factor(A64) if fo_a is None else fo_a
        fo_b = gram_factor(B64) if fo_b is None else fo_b
    else:
        fo_a = fo_b = None
    if K > N:
        fi_a = gram_factor(A64.T) if fi_a is None else fi_a
        fi_b = gram_factor(B64.T) if fi_b is None else fi_b
    else:
        fi_a = fi_b = None
    nuc_out = _nuclear_cross(A64, B64, fo_a, fo_b)
    nuc_in = _nuclear_cross(A64.T, B64.T, fi_a, fi_b)
    return {
        "cos_raw": _f((A64 * B64).sum() / (na.sqrt() * nb.sqrt())),
        "e_raw": _f(((B64 - A64) ** 2).sum() / nb),
        "e_out": _f((na + nb - 2 * nuc_out) / nb),
        "e_in": _f((na + nb - 2 * nuc_in) / nb),
    }


def _pair_stats(A: Tensor, B: Tensor, perm_axis: int | None) -> dict:
    """Raw and aligned residual energies of B against A plus the differential-coding gains after overhead."""
    A32, B32 = A.float(), B.float()
    N, K = A32.shape
    out = _pair_core(A32, B32)
    gains = {
        "gain_raw": 0.5 * math.log2(1.0 / max(out["e_raw"], EPS_ENERGY)),
        "gain_out": 0.5 * math.log2(1.0 / max(out["e_out"], EPS_ENERGY)) - 16.0 * N * N / (N * K),
        "gain_in": 0.5 * math.log2(1.0 / max(out["e_in"], EPS_ENERGY)) - 16.0 * K * K / (N * K),
    }
    if perm_axis is not None:
        M = A32.shape[perm_axis]
        e_perm, cos_perm = permutation_residual(A32, B32, perm_axis)
        out["e_perm"] = e_perm
        out["cos_perm_mean"] = cos_perm
        gains["gain_perm"] = 0.5 * math.log2(1.0 / max(e_perm, EPS_ENERGY)) - log2_factorial(M) / (N * K)
    out.update(gains)
    out["gain_bits_best"] = max(gains.values())
    return out


def interlayer(series: Sequence[tuple[str, int, Tensor]], perm_axis: int | None = None, seed: int = 0) -> list[ProbeRecord]:
    """P4 for one weight type: `series` = (name, layer, W) over layers, same shape. Raw cosine and the two
    Procrustes residuals are computed for every pair; records are written for the adjacent pairs and for the
    best pair, which also carries the all-pairs summary. The best pair is the one with the largest
    overhead-corrected gain over the raw / output-side / input-side alignments (the quantity gain_bits_best
    reports), not the smallest residual: the long-side Procrustes of a non-square tensor fits almost anything
    (control residuals near 0) and never pays for its 16*max(N,K)/min(N,K)-bit rotation. The permutation
    alignment (perm_axis 0 / 1, gate-up / down) runs on the recorded pairs only. Control: W_i and W_j
    independently shuffled (seeds seed, seed + 1), same alignments; the best-pair selection itself is made on W
    and is not controlled (the control of that record is one shuffled pair), which its note states.
    """
    entries = sorted(series, key=lambda e: e[1])
    if len(entries) < 2:
        return []
    N, K = entries[0][2].shape
    # Gram factors once per tensor (the long side only, min(N,K)^2 f64 each), so every pair costs O(min(N,K)^3).
    fo = [gram_factor(e[2].float()) if N > K else None for e in entries]
    fi = [gram_factor(e[2].float().T) if K > N else None for e in entries]
    table: dict[tuple[int, int], dict] = {}
    for a in range(len(entries)):
        A = entries[a][2].float()
        for b in range(a + 1, len(entries)):
            B = entries[b][2].float()
            table[(a, b)] = _pair_core(A, B, fo[a], fo[b], fi[a], fi[b])
            del B
        del A
    del fo, fi

    def corrected_gain(core: dict) -> float:
        return max(0.5 * math.log2(1.0 / max(core["e_raw"], EPS_ENERGY)),
                   0.5 * math.log2(1.0 / max(core["e_out"], EPS_ENERGY)) - 16.0 * N / K,
                   0.5 * math.log2(1.0 / max(core["e_in"], EPS_ENERGY)) - 16.0 * K / N)

    best_pair = max(table, key=lambda p: corrected_gain(table[p]))
    selected = [(a, a + 1) for a in range(len(entries) - 1)]
    if best_pair not in selected:
        selected.append(best_pair)
    n_pairs = len(table)
    summary = {
        "n_pairs": n_pairs,
        "allpairs_cos_raw_mean": sum(t["cos_raw"] for t in table.values()) / n_pairs,
        "allpairs_cos_raw_max": max(t["cos_raw"] for t in table.values()),
        "allpairs_e_raw_min": min(t["e_raw"] for t in table.values()),
        "allpairs_e_out_min": min(t["e_out"] for t in table.values()),
        "allpairs_e_out_mean": sum(t["e_out"] for t in table.values()) / n_pairs,
        "allpairs_e_in_min": min(t["e_in"] for t in table.values()),
        "allpairs_e_in_mean": sum(t["e_in"] for t in table.values()) / n_pairs,
    }
    records = []
    for a, b in selected:
        name_a, layer_a, Wa = entries[a]
        name_b, layer_b, Wb = entries[b]
        stat = {"layer_i": layer_a, "layer_j": layer_b, "is_adjacent": int(b == a + 1), "is_best": int((a, b) == best_pair)}
        stat.update(_pair_stats(Wa, Wb, perm_axis))
        if (a, b) == best_pair:
            stat.update(summary)
        Ca = shuffle_within_tensor(Wa, seed)
        Cb = shuffle_within_tensor(Wb, seed + 1)
        control = _pair_stats(Ca, Cb, perm_axis)
        del Ca, Cb
        stat["numel"] = N * K
        records.append(ProbeRecord(
            tensor=f"{name_a} -> {name_b}", layer=layer_b, type="", stat=stat, control=control,
            gain_bits=stat["gain_bits_best"], gain_bits_control=control["gain_bits_best"],
            note="e = ||W_j - align(W_i)||^2/||W_j||^2; gain = 0.5*log2(1/e) - overhead (R bf16: 16 N/K or 16 K/N; "
                 "permutation: log2(M!)/(N K); raw: 0); control: both tensors shuffled independently"
                 + ("; best pair = max overhead-corrected gain over all pairs on W (selection uncontrolled: the control "
                    "is one shuffled pair, control spread across pairs is ~0.003 in e); carries the all-pairs summary"
                    if (a, b) == best_pair else ""),
            numel=N * K,
        ))
    return records
