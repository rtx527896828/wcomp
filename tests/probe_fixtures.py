"""Synthetic Phase 2 probe fixtures in the schema the probe modules write (DESIGN.md "Output"; stat names as in
wcomp/probe/weights.py and wcomp/probe/acts.py, checked against their smoke outputs on 2026-11-08). No model, no GPU.

Used by tests/test_plots_phase2.py and, as a script, to render the Phase 2 figures before the real probes exist:
  uv run python tests/probe_fixtures.py --out results/fixture --model qwen3.5-0.8b
writes results/fixture/probe/<model>/{marginal,lowrank,lowrank_weighted,outliers,interlayer,acts,sensitivity,
fisher,superweight}.json + ablation.jsonl and results/fixture/<model>/runs.jsonl (Phase 1 row schema).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from pathlib import Path

QWEN_HIDDEN = {"qwen3.5-0.8b": (1024, 3584, 24), "qwen3.5-2b": (2048, 6144, 24)}
MINICPM = {"minicpm5-2b": (2048, 6144, 42)}
R_FRACS = [0.01, 0.05, 0.1, 0.25]
R_PCT = {0.01: 1, 0.05: 5, 0.1: 10, 0.25: 25}
PCTS = [0.5, 1.0, 2.0]
HIDDEN_INPUT_TYPES = ["gate", "in_proj_ab", "in_proj_qkv", "in_proj_z", "k", "q", "up", "v"]
TS = dt.datetime(2026, 11, 8, 12, 0, tzinfo=dt.timezone.utc).isoformat(timespec="seconds")


def layer_kinds(model: str) -> list[str]:
    """'attn' / 'delta' per decoder layer: Qwen3.5 has full attention every 4th layer, MiniCPM5 everywhere."""
    if model in MINICPM:
        return ["attn"] * MINICPM[model][2]
    return ["attn" if i % 4 == 3 else "delta" for i in range(QWEN_HIDDEN[model][2])]


def layer_linears(model: str, i: int) -> list[tuple[str, str, int, int]]:
    """(full module name, weight type, N, K) of every block linear of decoder layer i, in forward order."""
    if model in MINICPM:
        h, inter, _ = MINICPM[model]
        heads = [("self_attn.q_proj", "q", h, h), ("self_attn.k_proj", "k", h // 4, h), ("self_attn.v_proj", "v", h // 4, h),
                 ("self_attn.o_proj", "o", h, h)]
    else:
        h, inter, _ = QWEN_HIDDEN[model]
        if layer_kinds(model)[i] == "attn":
            heads = [("self_attn.q_proj", "q", 4 * h, h), ("self_attn.k_proj", "k", h // 2, h), ("self_attn.v_proj", "v", h // 2, h),
                     ("self_attn.o_proj", "o", h, 2 * h)]
        else:
            heads = [("linear_attn.in_proj_qkv", "in_proj_qkv", 6 * h, h), ("linear_attn.in_proj_z", "in_proj_z", 2 * h, h),
                     ("linear_attn.in_proj_b", "in_proj_ab", 16, h), ("linear_attn.in_proj_a", "in_proj_ab", 16, h),
                     ("linear_attn.out_proj", "out_proj", h, 2 * h)]
    mlp = [("mlp.gate_proj", "gate", inter, h), ("mlp.up_proj", "up", inter, h), ("mlp.down_proj", "down", h, inter)]
    return [(f"model.layers.{i}.{n}", t, N, K) for n, t, N, K in heads + mlp]


def all_linears(model: str) -> list[tuple[str, int, str, int, int]]:
    """(name, layer, type, N, K) over all layers in forward order."""
    out = []
    for i in range(len(layer_kinds(model))):
        out += [(n, i, t, N, K) for n, t, N, K in layer_linears(model, i)]
    return out


def embedding(model: str) -> tuple[str, int, int]:
    emb = 248320 if "qwen" in model else 130560
    return "model.embed_tokens", emb, (QWEN_HIDDEN.get(model) or MINICPM[model])[0]


def _record(name: str, layer: int, typ: str, numel: int, stat: dict, control: dict, gain, gain_c, note: str = "") -> dict:
    return {"tensor": name, "layer": layer, "type": typ, "stat": stat, "control": control, "gain_bits": gain,
            "gain_bits_control": gain_c, "note": note, "numel": numel}


def _isnum(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _aggregate(records: list[dict]) -> dict:
    """wcomp.probe.common.aggregate: per type, numel-weighted mean of every numeric stat / control entry and gain."""
    out: dict[str, dict] = {}
    for typ in sorted({r["type"] for r in records}):
        recs = [r for r in records if r["type"] == typ]
        agg = {"n_records": len(recs), "numel": sum(r["numel"] for r in recs), "stat": {}, "control": {}}
        for section in ("stat", "control"):
            keys = []
            for r in recs:
                keys += [k for k in r[section] if k not in keys]
            for k in keys:
                num = den = 0.0
                for r in recs:
                    v = r[section].get(k)
                    if _isnum(v):
                        num += v * r["numel"]
                        den += r["numel"]
                agg[section][k] = num / den if den > 0 else None
        for k in ("gain_bits", "gain_bits_control"):
            num = den = 0.0
            for r in recs:
                if _isnum(r[k]):
                    num += r[k] * r["numel"]
                    den += r["numel"]
            agg[k] = num / den if den > 0 else None
        out[typ] = agg
    return out


def _doc(model: str, probe: str, params: dict, records: list[dict], glob: dict | None = None, extra_agg: dict | None = None) -> dict:
    doc = {"model": model, "probe": probe, "code_version": "fixture/0000000000", "timestamp": TS, "params": params,
           "records": records, "aggregate": _aggregate(records)}
    if extra_agg:
        doc["aggregate"].update(extra_agg)
    if glob is not None:
        doc["global"] = glob
    return doc


# Per-type "personalities": (excess kurtosis, column outlier ratio, rank deficiency, sensitivity per param).
PERSONA = {
    "q": (3.0, 4.0, 0.0, 1.0), "k": (2.0, 3.0, 0.3, 0.6), "v": (1.5, 2.5, 0.25, 2.0), "o": (4.0, 6.0, 0.0, 3.0),
    "gate": (2.5, 3.0, 0.0, 0.8), "up": (2.0, 3.0, 0.0, 0.7), "down": (8.0, 12.0, 0.0, 10.0),
    "in_proj_qkv": (2.5, 4.0, 0.05, 1.2), "in_proj_z": (2.0, 3.0, 0.0, 0.9), "in_proj_ab": (1.0, 2.0, 0.0, 30.0),
    "out_proj": (5.0, 8.0, 0.0, 4.0), "embed": (6.0, 3.0, 0.2, 0.0),
}


# ------------------------------------------------------------------------------------------------ P1


def _marginal_record(name: str, layer: int, typ: str, N: int, K: int, rng: random.Random) -> dict:
    kurt = max(0.1, PERSONA[typ][0] * (1 + 0.3 * rng.uniform(-1, 1)) * (1 + 0.5 * max(layer, 0) / 24))
    numel = N * K
    table = lambda b: (2 ** b) * 16 / numel  # noqa: E731
    stat = {"numel": numel, "mean": 1e-5 * rng.uniform(-1, 1), "std": 0.02 * (1 + 0.2 * rng.random()), "kurtosis_excess": kurt,
            "frac_gt4sigma": 1e-3 * (1 + kurt / 3), "frac_gt6sigma": 1e-5 * (1 + kurt), "nll_gauss": -2.68, "nll_laplace": -2.70,
            "nll_student": -2.72, "student_nu": 4.0 + 10 / (1 + kurt), "best_family": "student" if kurt > 2 else "laplace"}
    control = {"kurtosis_excess": 0.0 + 0.01 * rng.random(), "frac_gt4sigma": 6.3e-5, "frac_gt6sigma": 2e-9}
    for b in (4, 3, 2):
        H = max(0.5, b - 0.25 - 0.04 * kurt - 0.02 * rng.random())
        Hg = b - 0.22 - 0.01 * rng.random()
        stat[f"H_idx_b{b}"] = H
        stat[f"gain_bits_b{b}"] = b - H - table(b)
        control[f"H_idx_b{b}"] = Hg
        control[f"gain_bits_b{b}"] = b - Hg - table(b)
    stat.update({"H_word": 11.5 + 0.3 * rng.random(), "H_sign": 1.0, "H_exp": 2.7, "H_mant": 7.0})
    stat["lossless_bits_saved"] = 16 - stat["H_word"]
    control.update({"H_word": 12.6 + 0.1 * rng.random(), "H_sign": 1.0, "H_exp": 2.65, "H_mant": 6.97})
    control["lossless_bits_saved"] = 16 - control["H_word"]
    return _record(name, layer, typ, numel, stat, control, stat["gain_bits_b4"], control["gain_bits_b4"],
                   "control: iid Gaussian, matched per-group std (g128, seed 0); gain_bits = b=4 entropy-coding gain minus a 2^b x 16-bit table")


def marginal(model: str, rng: random.Random) -> dict:
    records = [_marginal_record(n, l, t, N, K, rng) for n, l, t, N, K in all_linears(model)]
    name, V, h = embedding(model)
    records.append(_marginal_record(name, -1, "embed", V, h, rng))
    return _doc(model, "marginal", {"group": 128, "bits": [4, 3, 2], "control": "gaussian_control seed 0", "include_embedding": True}, records)


# ------------------------------------------------------------------------------------------------ P2 / P2b


def _rank_profile(N: int, K: int, deficiency: float, rng: random.Random, weighted: bool) -> dict:
    """energy / gain at the r grid plus ranks, keyed by r fraction."""
    m = min(N, K)
    out: dict = {}
    for f in R_FRACS:
        base = f ** 0.75 * (1 + 0.4 * (max(N, K) / m) ** -0.5)
        captured = min(0.999, base + deficiency * (1 - f) ** 0.5 + (0.05 if weighted else 0) + 0.01 * rng.random())
        gain = 0.5 * math.log2(1 / (1 - captured)) - 16 * int(f * m) * (N + K) / (N * K)
        out[f] = (captured, gain)
    out["stable_rank"] = m * (0.5 - 0.4 * deficiency) * (1 + 0.05 * rng.random())
    out["rank90"] = int(m * (0.6 - 0.5 * deficiency))
    out["rank99"] = int(m * (0.95 - 0.4 * deficiency))
    best_f = max(R_FRACS, key=lambda f: out[f][1])
    out["best_gain"], out["best_r"] = out[best_f][1], int(best_f * m)
    return out


def _lowrank_record(name: str, layer: int, typ: str, N: int, K: int, rng: random.Random) -> dict:
    p = _rank_profile(N, K, PERSONA[typ][2], rng, False)
    c0 = _rank_profile(N, K, 0.0, rng, False)
    c1 = _rank_profile(N, K, 0.0, rng, False)
    stat = {"stable_rank": p["stable_rank"], "rank90": p["rank90"], "rank99": p["rank99"], "rank90_frac": p["rank90"] / min(N, K),
            "rank99_frac": p["rank99"] / min(N, K)}
    control = {"stable_rank": (c0["stable_rank"] + c1["stable_rank"]) / 2, "rank90": (c0["rank90"] + c1["rank90"]) / 2,
               "rank99": (c0["rank99"] + c1["rank99"]) / 2}
    for f in R_FRACS:
        stat[f"energy_f{f}"], stat[f"gain_f{f}"] = p[f]
        control[f"energy_f{f}"] = (c0[f][0] + c1[f][0]) / 2
        control[f"gain_f{f}"] = (c0[f][1] + c1[f][1]) / 2
    stat.update({"gain_bits_best": p["best_gain"], "best_r": p["best_r"], "best_r_frac": p["best_r"] / min(N, K), "numel": N * K})
    control.update({"gain_bits_best": (c0["best_gain"] + c1["best_gain"]) / 2, "best_r": c0["best_r"],
                    "gain_bits_best_seed0": c0["best_gain"], "gain_bits_best_seed1": c1["best_gain"],
                    "gain_bits_best_spread": abs(c0["best_gain"] - c1["best_gain"])})
    return _record(name, layer, typ, N * K, stat, control, p["best_gain"], control["gain_bits_best"],
                   "control: 2 within-tensor shuffles (mean reported, per-seed gains in control); gain = 0.5*log2(1/e_r) - 16 r (N+K)/(N K), max over the r grid")


def lowrank(model: str, rng: random.Random) -> dict:
    records = [_lowrank_record(n, l, t, N, K, rng) for n, l, t, N, K in all_linears(model)]
    name, V, h = embedding(model)
    records.append(_lowrank_record(name, -1, "embed", V, h, rng))
    return _doc(model, "lowrank", {"include_embedding": True, "r_fracs": R_FRACS, "control_seeds": [0, 1]}, records)


def lowrank_weighted(model: str, rng: random.Random) -> dict:
    records = []
    for name, layer, typ, N, K in all_linears(model):
        p = _rank_profile(N, K, PERSONA[typ][2] + 0.1, rng, True)
        stat = {"stable_rank": p["stable_rank"] * 0.3}
        control: dict = {}
        for f in R_FRACS:
            stat[f"energy_r{R_PCT[f]}pct"], stat[f"gain_r{R_PCT[f]}pct"] = p[f]
        stat.update({"rank_90": p["rank90"], "rank_99": p["rank99"], "best_gain": p["best_gain"], "best_r": p["best_r"], "damp": 0.01,
                     "plain_stable_rank": p["stable_rank"]})
        cs = {}
        for seed in (0, 1):
            c = _rank_profile(N, K, 0.0, rng, True)
            cs[seed] = c
            control[f"seed{seed}_stable_rank"] = c["stable_rank"]
            for f in R_FRACS:
                control[f"seed{seed}_energy_r{R_PCT[f]}pct"], control[f"seed{seed}_gain_r{R_PCT[f]}pct"] = c[f]
            control[f"seed{seed}_rank_90"], control[f"seed{seed}_rank_99"] = c["rank90"], c["rank99"]
            control[f"seed{seed}_best_gain"], control[f"seed{seed}_best_r"] = c["best_gain"], c["best_r"]
        records.append(_record(name, layer, typ, N * K, stat, control, p["best_gain"], cs[0]["best_gain"],
                               "W @ L, L = chol(H + damp): activation-weighted rank; control: shuffled W (seeds [0, 1]), same L; gain_bits_control = seed-0 control"))
    return _doc(model, "lowrank_weighted", {"windows": 128, "seqlen": 2048, "calib_seed": 0, "damp": 0.01, "control_seeds": [0, 1],
                                            "rank_fractions": R_FRACS}, records)


# ------------------------------------------------------------------------------------------------ P3


def _outlier_stats(N: int, K: int, ratio: float, rng: random.Random, shuffled: bool) -> dict:
    r_col = (1.07 + 0.05 * rng.random()) if shuffled else ratio * (1 + 0.2 * rng.uniform(-1, 1))
    st = {"col_max_over_median": r_col, "col_n_gt5x_median": 0 if shuffled or ratio < 5 else int(K * 0.004 * ratio / 5),
          "col_n_gt10x_median": 0 if shuffled or ratio < 10 else int(K * 0.001), "col_kurtosis_excess": (0.4 if shuffled else 2 + ratio) * (1 + 0.1 * rng.random()),
          "row_max_over_median": (1.25 if shuffled else 1.5 + 0.5 * rng.random()), "row_n_gt5x_median": 0, "row_n_gt10x_median": 0,
          "row_kurtosis_excess": 1.3 + 0.2 * rng.random(), "mse_rtn4": 3.5e-6 * (2 if shuffled else 1)}
    best = -1e9
    for p in PCTS:
        f = p / 100
        rho = 1 + (0.02 if shuffled else 0.08 * ratio * f ** 0.5) + 0.005 * rng.random()
        gain = 0.5 * math.log2(rho) - (f * 12 + 1 / N)
        st[f"rho_p{p:g}"], st[f"gain_p{p:g}"], st[f"kept_frac_p{p:g}"] = rho, gain, round(f * K) / K
        if gain > best:
            best, st["best_pct"] = gain, p
    st["gain_bits_best"] = best
    return st


def outliers(model: str, rng: random.Random) -> dict:
    records = []
    for name, layer, typ, N, K in all_linears(model):
        stat = {"numel": N * K, **_outlier_stats(N, K, PERSONA[typ][1], rng, False)}
        c0, c1 = _outlier_stats(N, K, PERSONA[typ][1], rng, True), _outlier_stats(N, K, PERSONA[typ][1], rng, True)
        control = {k: (c0[k] + c1[k]) / 2 for k in c0}
        control.update({"gain_bits_best_seed0": c0["gain_bits_best"], "gain_bits_best_seed1": c1["gain_bits_best"],
                        "gain_bits_best_spread": abs(c0["gain_bits_best"] - c1["gain_bits_best"])})
        records.append(_record(name, layer, typ, N * K, stat, control, stat["gain_bits_best"], control["gain_bits_best"],
                               "rho = MSE(RTN-4) / MSE(RTN-4, top-p% columns kept in bf16, excluded from the grid); overhead p/100*12 + 1/N; control: 2 shuffles"))
    hidden = (QWEN_HIDDEN.get(model) or MINICPM[model])[0]
    n_layers = len(layer_kinds(model))
    jac = {"all_hidden_input": {"n_records": 1, "numel": 0,
                                "stat": {"n_layers": n_layers, "top": 16, "K": hidden, "jaccard_mean": 0.35 + 0.1 * rng.random(),
                                         "n_channels_in_half_layers": 9, "n_distinct_channels": 40},
                                "control": {"jaccard_mean": 0.01 + 0.02 * rng.random()}, "gain_bits": None, "gain_bits_control": None}}
    return _doc(model, "outliers", {"group": 128, "bits": 4, "pcts": PCTS, "control_seeds": [0, 1], "consistency_top": 16,
                                    "hidden_input_types": HIDDEN_INPUT_TYPES}, records, extra_agg=jac)


# ------------------------------------------------------------------------------------------------ P4


def interlayer(model: str, rng: random.Random) -> dict:
    records = []
    by_type: dict[str, list] = {}
    for x in all_linears(model):
        by_type.setdefault(x[2], []).append(x)
    for typ, items in by_type.items():
        items.sort(key=lambda x: x[1])
        for idx in range(1, len(items)):
            for is_best, src in ((0, idx - 1), (1, max(0, idx - 1 - rng.randint(0, 3)))):
                if is_best and src == idx - 1:
                    continue  # the best pair is the adjacent one: a single record would carry is_best = 1
                name_i, layer_i, _, N, K = items[src]
                name_j, layer_j, _, _, _ = items[idx]

                def stats(shuffled: bool) -> dict:
                    e_raw = (1.98 + 0.02 * rng.random()) if shuffled else (1.90 + 0.08 * rng.random())
                    st = {"layer_i": layer_i, "layer_j": layer_j, "is_adjacent": int(src == idx - 1), "is_best": is_best,
                          "cos_raw": (1e-3 if shuffled else 2e-2) * rng.uniform(-1, 1), "e_raw": e_raw,
                          "e_out": e_raw - (0.85 if shuffled else 0.9) - 0.02 * rng.random(), "e_in": (0.14 if shuffled else 0.12) + 0.02 * rng.random()}
                    if typ in ("gate", "up", "down"):
                        st["e_perm"] = e_raw - (0.2 if shuffled else 0.25) - 0.02 * rng.random()
                        st["cos_perm_mean"] = 0.1 + 0.05 * rng.random()
                    over = {"raw": 0.0, "out": 16 * N / K, "in": 16 * K / N, "perm": math.lgamma(max(N, K) + 1) / math.log(2) / (N * K)}
                    gains = {}
                    for a in ("raw", "out", "in", "perm"):
                        if f"e_{a}" in st:
                            gains[a] = 0.5 * math.log2(1 / st[f"e_{a}"]) - over[a]
                            st[f"gain_{a}"] = gains[a]
                    st["gain_bits_best"] = max(gains.values())
                    return st

                stat, control = stats(False), stats(True)
                stat["numel"] = N * K
                records.append(_record(f"{name_i} -> {name_j}", layer_j, typ, N * K, stat, control, stat["gain_bits_best"], control["gain_bits_best"],
                                       "e = ||W_j - align(W_i)||^2/||W_j||^2; gain = 0.5*log2(1/e) - overhead (R bf16: 16 N/K or 16 K/N; permutation: log2(M!)/(N K); raw: 0); control: both tensors shuffled independently"))
    return _doc(model, "interlayer", {"include_embedding": False, "control": "one shuffled pair per record (W_i seed 0, W_j seed 1); no second control replicate", "perm_axis": {"gate": 0, "up": 0, "down": 1},
                                      "pairs_recorded": "adjacent + best (lowest Procrustes residual) pair"}, records)


# ------------------------------------------------------------------------------------------------ P5


def acts(model: str, rng: random.Random) -> dict:
    records = []
    for name, layer, typ, N, K in all_linears(model):
        conc = (0.55 if typ == "down" else 0.35 if typ in ("o", "out_proj") else 0.2) * (1 + 0.3 * rng.uniform(-1, 1))
        a = 1 - conc  # Lorenz curve y = x^a
        stat = {"gini": (1 - a) / (1 + a), "frac_channels_50": 0.5 ** (1 / a), "frac_channels_90": 0.9 ** (1 / a),
                "energy_share_top20pct": 0.2 ** a, "frac_active_lt_1pct": 0.02 * conc * rng.random(), "frac_never_active": 0.0,
                "mean_active_rate": 0.95 - 0.1 * conc, "energy_max_over_median": 5 + 200 * conc * rng.random(), "rms": 1.2,
                "tau": 0.06, "n_tokens": 65536, "spearman_split_half": (0.97 - 0.1 * rng.random()) if typ == "down" else 0.85 + 0.12 * rng.random()}
        control = {"spearman_split_half": stat["spearman_split_half"], "gini_half2": stat["gini"] * (1 + 0.02 * rng.uniform(-1, 1)), "n_tokens_half2": 65536}
        records.append(_record(name, layer, typ, N * K, stat, control, None, None,
                               "Gini is permutation invariant: control = split-half Spearman of channel energy (C4 seeds 0 vs 2)"))
    return _doc(model, "acts", {"windows": 32, "seqlen": 2048, "calib_seed": 0, "split_half_seed": 2, "tau_fraction": 0.05,
                                "primary_types": ["down", "o", "out_proj"]}, records,
                glob={"n_primary": sum(1 for r in records if r["type"] in ("down", "o", "out_proj")), "min_split_half_spearman_primary": 0.85})


# ------------------------------------------------------------------------------------------------ P6a / P6b


def _alloc(prefix: str, saving: float, overhead: float, levels: int) -> dict:
    return {f"{prefix}_saving": saving, f"{prefix}_overhead": overhead, f"{prefix}_gain": saving - overhead, f"{prefix}_levels": levels,
            f"{prefix}_frac_floored": 0.0, f"{prefix}_spread_log10": 1 + saving}


def sensitivity(model: str, rng: random.Random) -> dict:
    records = []
    for name, layer, typ, N, K in all_linears(model):
        n_t = N * K
        s_per = 1e-6 * PERSONA[typ][3] * (1 + 0.5 * rng.random()) * (1 + (layer / 8) ** 2 * 0.1)
        stat = {"s": s_per * n_t, "s_per_param": s_per, "mse_rtn": 3.5e-6, "mean_diag_h": 1.5 + rng.random(), "frac_dead_cols": 0.0,
                **_alloc("alloc_group", 0.03 + 0.2 * rng.random(), 3 / (128 * 8), 3), **_alloc("alloc_col", 0.08 + 0.3 * rng.random(), 3 / 1024, 5)}
        records.append(_record(name, layer, typ, n_t, stat, {}, stat["alloc_col_gain"], None,
                               "permutation-invariant statistic: shuffle control replaced by the P0 prediction check (see global.prediction_check)"))
    glob = {"bits": 4, "group": 128, **_alloc("alloc_tensor", 0.42, 1.5e-6, 8), **_alloc("alloc_group", 0.58, 3 / 128, 8), **_alloc("alloc_col", 0.66, 3 / 1024, 8)}
    return _doc(model, "sensitivity", {"windows": 128, "seqlen": 2048, "calib_seed": 0, "bits": 4, "group": 128}, records, glob=glob)


def fisher(model: str, rng: random.Random) -> dict:
    records = []
    for name, layer, typ, N, K in all_linears(model):
        r1 = 0.75 + 0.2 * rng.random() if typ in ("down", "o", "out_proj") else 0.5 + 0.3 * rng.random()
        stat = {"total": 1e-3 * PERSONA[typ][3] * (1 + rng.random()), "per_param": 1e-9 * PERSONA[typ][3], "gini": 0.6 + 0.3 * rng.random(),
                "frac_zero": 0.001, "rank1_energy": r1, "s_rtn_fisher": 3e-5,
                **_alloc("alloc_elem", 2.0, 5.0, 20), **_alloc("alloc_row", 1.5 + 0.3 * rng.random(), 4 / N, 15), **_alloc("alloc_col", 0.3 + 0.3 * rng.random(), 3 / K, 5)}
        control = {"gini": 0.6 + 0.3 * rng.random(), "rank1_energy": 1 / min(N, K) * (1 + rng.random()),
                   **_alloc("alloc_elem", 2.0, 5.0, 20), **_alloc("alloc_row", 0.02, 2 / N, 4), **_alloc("alloc_col", 0.004, 2 / K, 3)}
        gain = max(stat["alloc_elem_gain"], stat["alloc_row_gain"], stat["alloc_col_gain"])
        gain_c = max(control["alloc_elem_gain"], control["alloc_row_gain"], control["alloc_col_gain"])
        records.append(_record(name, layer, typ, N * K, stat, control, gain, gain_c,
                               "empirical Fisher diagonal over 32 C4 windows of 1024 tokens; control: F shuffled within the tensor (seed 0)"))
    glob = {"n_windows": 32, "seqlen": 1024, "bits": 4, "checksum_unchanged": True, **_alloc("alloc_tensor", 0.46, 1.5e-6, 10)}
    return _doc(model, "fisher", {"windows": 32, "seqlen": 1024, "fisher_windows": 32, "fisher_tokens": 1024, "bits": 4}, records, glob=glob)


# ------------------------------------------------------------------------------------------------ P7


def superweight(model: str, rng: random.Random) -> dict:
    fp = PHASE1[model][0]
    hidden, inter = (QWEN_HIDDEN.get(model) or MINICPM[model])[:2]
    cands = [(2, 1.9, 180.0, 0.9, 0.8, "spike"), (3, 1.2, 4.0, 0.3, 0.1, "top16")] if model not in MINICPM else [(2, 2.3, 450.0, 1.1, 0.5, "spike")]
    records = []
    for layer, val, dppl, dppl_mag, dppl_rand, kind in cands:
        row, col = rng.randint(0, hidden - 1), rng.randint(0, inter - 1)
        stat = {"row": row, "col": col, "value": val, "ppl": fp * (1 + dppl / 100), "dppl_pct": dppl, "kl": dppl / 150, "top1_agree": 0.99 - dppl / 1000,
                "abs_value": abs(val), "co_occurs": kind == "spike", "same_token": kind == "spike", "magnitude_rank": 0}
        control = {"magnitude_row": rng.randint(0, hidden - 1), "magnitude_col": rng.randint(0, inter - 1), "magnitude_value": val * 0.8,
                   "magnitude_ppl": fp * (1 + dppl_mag / 100), "magnitude_dppl_pct": dppl_mag, "magnitude_kl": dppl_mag / 150, "magnitude_top1_agree": 0.99,
                   "random_row": rng.randint(0, hidden - 1), "random_col": rng.randint(0, inter - 1), "random_value": 0.02,
                   "random_ppl": fp * (1 + dppl_rand / 100), "random_dppl_pct": dppl_rand, "random_kl": dppl_rand / 150, "random_top1_agree": 0.99}
        records.append(_record(f"model.layers.{layer}.mlp.down_proj", layer, "down", 1, stat, control, None, None,
                               f"{kind}; zeroed on 16 WikiText-2 windows, kl_windows 16; controls: largest other |w| of the tensor, one random weight of the tensor (seed 0)"))
    layer_stats = []
    for i in range(len(layer_kinds(model))):
        spike = i == 2
        layer_stats.append({"tensor": f"model.layers.{i}.mlp.down_proj", "layer": i, "in_max": 10 * (1 + rng.random()) * (20 if spike else 1), "in_token": rng.randint(0, 2047),
                            "in_channel": rng.randint(0, inter - 1), "out_max": 50 * (1 + rng.random()) * (40 if spike else 1), "out_token": rng.randint(0, 2047),
                            "out_channel": rng.randint(0, hidden - 1), "out_spike": spike, "in_spike": spike})
    return _doc(model, "superweight", {"calib_window": 0, "eval_windows": 16, "spike_factor": 20.0, "top_k": 16}, records,
                glob={"layer_stats": layer_stats, "n_candidates": len(records)})


# ------------------------------------------------------------------------------------------------ ablation rows

PHASE1 = {  # fp16 ppl, rtn-sym-g128-4 (ppl, kl), rtn-sym-g128-3 (ppl, kl), gptq-4 (ppl, kl), gptq-3 (ppl, kl) per model (NOTES.md Phase 1)
    "qwen3.5-0.8b": (17.57, (22.67, 0.2051), (84.77, 1.5053), (18.06, 0.0712), (24.45, 0.3131)),
    "qwen3.5-2b": (11.81, (13.13, 0.1331), (34.80, 1.0302), (12.47, 0.0591), (15.67, 0.2719)),
    "minicpm5-2b": (13.78, (17.45, 0.2682), (97.96, 2.0744), (14.58, 0.0771), (18.34, 0.3426)),
}


def _row(model: str, method: str, bits: float, ppl: float, kl: float, seed: int = 0, calib_seed=None, note: str = "", ts: str = TS) -> dict:
    return {"model": model, "method": method, "seed": seed, "calib_seed": calib_seed, "bits_blocks": bits, "bits_all": bits * 0.5 + 8,
            "ppl": ppl, "kl": kl, "top1_agree": max(0.0, 1 - kl), "n_tokens": 296815, "kl_tokens": 65504, "time_s": 30.0,
            "time_method_s": 0.1, "note": note, "git_or_code_version": "fixture/0000000000",
            "env": "torch 2.11.0+cu128; transformers 5.22.2; kernels none", "timestamp": ts}


def runs_rows(model: str) -> list[dict]:
    fp, r4, r3, g4, g3 = PHASE1[model]
    return [_row(model, "fp16", 16.0, fp, 0.0, note="bf16, not quantized"),
            _row(model, "rtn-sym-g128-4", 4.125, *r4, note="rtn-sym-g128-4: f16 scale"),
            _row(model, "rtn-sym-g128-3", 3.125, *r3, note="rtn-sym-g128-3: f16 scale"),
            _row(model, "gptq-sym-g128-4", 4.125, *g4, calib_seed=0, note="calib 128x2048 seed 0; gptq 4b sym g128 block128 damp0.01"),
            _row(model, "gptq-sym-g128-3", 3.125, *g3, calib_seed=0, note="calib 128x2048 seed 0; gptq 3b sym g128 block128 damp0.01")]


def ablation_rows(model: str, rng: random.Random) -> list[dict]:
    fp, r4, r3, g4, g3 = PHASE1[model]
    lins = all_linears(model)
    types = []
    for x in lins:
        if x[2] not in types:
            types.append(x[2])
    numel_total = sum(N * K for _, _, _, N, K in lins)
    numel_type = {t: sum(N * K for _, _, tt, N, K in lins if tt == t) for t in types}
    sens = {t: PERSONA[t][3] * numel_type[t] for t in types}
    tot = sum(sens.values())
    rows = []
    for bits, (ppl_j, kl_j) in ((4, r4), (3, r3)):
        frac_bits = {4: 4.125, 3: 3.125}[bits]
        for t in types:
            share = sens[t] / tot * (1 + 0.2 * rng.uniform(-1, 1))
            b = (frac_bits * numel_type[t] + 16 * (numel_total - numel_type[t])) / numel_total
            rows.append(_row(model, f"rtn-sym-g128-{bits}@only={t}", b, fp * math.exp(math.log(ppl_j / fp) * share), kl_j * share * 0.9,
                             note=f"rtn-sym-g128-{bits}: f16 scale; only {t}"))
            if bits == 4:
                b_ex = (frac_bits * (numel_total - numel_type[t]) + 16 * numel_type[t]) / numel_total
                rows.append(_row(model, f"rtn-sym-g128-4@except={t}", b_ex, fp + (ppl_j - fp) * (1 - share), kl_j * (1 - share),
                                 note=f"rtn-sym-g128-4: f16 scale; except {t}"))
    n_layers = len(layer_kinds(model))
    for i in range(n_layers):
        share = (1 + (2.0 if i < 3 else 0.0) + 0.5 * rng.random()) / n_layers
        numel_layer = sum(N * K for _, l, _, N, K in lins if l == i)
        b = (3.125 * numel_layer + 16 * (numel_total - numel_layer)) / numel_total
        rows.append(_row(model, f"rtn-sym-g128-3@layer={i}", b, fp * math.exp(math.log(r3[0] / fp) * share), r3[1] * share,
                         note=f"rtn-sym-g128-3: f16 scale; layer {i}"))
    for g, extra_bits, factor in ((32, 0.5, 0.45), (64, 0.25, 0.7)):
        for bits, (ppl_j, kl_j) in ((4, r4), (3, r3)):
            rows.append(_row(model, f"rtn-sym-g{g}-{bits}", bits + extra_bits, fp + (ppl_j - fp) * factor, kl_j * factor, note=f"rtn-sym-g{g}-{bits}: f16 scale"))
    for p, rec in (("0.5", 0.75), ("1", 0.65), ("2", 0.6)):
        for bits, (ppl_j, kl_j) in ((4, r4), (3, r3)):
            b = {4: 4.125, 3: 3.125}[bits] + float(p) / 100 * 12 + 1 / 1024
            rows.append(_row(model, f"rtn-sym-g128-{bits}+keepcols={p}", b, fp + (ppl_j - fp) * rec, kl_j * rec,
                             note=f"rtn-sym-g128-{bits}: f16 scale; keepcols {p}% bf16 + K-bit mask"))
    rows.append(_row(model, f"rtn-sym-g128-4+keepsuper=results/probe/{model}/superweight.json", 4.125 + 1e-6, fp + (r4[0] - fp) * 0.95, r4[1] * 0.95,
                     note="rtn-sym-g128-4: f16 scale; keepsuper 2 weights"))
    rows.append(_row(model, "gptq-sym-g128-4", 4.125, g4[0] * 1.01, g4[1] * 1.03, seed=1, calib_seed=1, note="calib 128x2048 seed 1; gptq 4b sym g128 block128 damp0.01"))
    rows.append(_row(model, "had-fp16", 16.0, fp * 1.0007, 0.0012, note="bf16, not quantized; input rotation hadamard(1024,2048,3584)"))
    rows.append(_row(model, "had-gptq-sym-g128-3", 3.125, g3[0] * 0.9, g3[1] * 0.85, calib_seed=0,
                     note="calib 128x2048 seed 0; gptq 3b sym g128 block128 damp0.01; input rotation hadamard(1024,2048,3584)"))
    return rows


PROBE_BUILDERS = {"marginal": marginal, "lowrank": lowrank, "lowrank_weighted": lowrank_weighted, "outliers": outliers, "interlayer": interlayer,
                  "acts": acts, "sensitivity": sensitivity, "fisher": fisher, "superweight": superweight}


def write_fixtures(results_dir: Path, model: str, probes=None, seed: int = 0, with_runs: bool = True, with_ablation: bool = True) -> dict[str, Path]:
    """Write every probe JSON (or the given subset), ablation.jsonl and results/<model>/runs.jsonl; returns the paths."""
    rng = random.Random(f"{model}-{seed}")
    pdir = results_dir / "probe" / model
    pdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for probe in probes or PROBE_BUILDERS:
        path = pdir / f"{probe}.json"
        path.write_text(json.dumps(PROBE_BUILDERS[probe](model, rng), indent=1), encoding="utf-8")
        out[probe] = path
    if with_ablation:
        path = pdir / "ablation.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in ablation_rows(model, rng):
                f.write(json.dumps(r) + "\n")
        out["ablation"] = path
    if with_runs:
        rdir = results_dir / model
        rdir.mkdir(parents=True, exist_ok=True)
        path = rdir / "runs.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in runs_rows(model):
                f.write(json.dumps(r) + "\n")
        out["runs"] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/fixture")
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=sorted(PHASE1))
    args = ap.parse_args()
    for probe, path in write_fixtures(Path(args.out), args.model).items():
        print(probe, path)


if __name__ == "__main__":
    main()
