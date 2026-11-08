"""Tests for wcomp.probe.common and wcomp.probe.weights (DESIGN.md, Phase 2 contract) on synthetic tensors."""
from __future__ import annotations

import json
import math

import pytest
import torch

from wcomp.probe import weights as pw
from wcomp.probe.common import (
    ProbeRecord,
    aggregate,
    bits_from_mse_ratio,
    entropy_bits,
    gaussian_control,
    log2_factorial,
    n_top_columns,
    shuffle_within_tensor,
    write_probe_json,
)
from wcomp.quant.rotate import random_orthogonal
from wcomp.quant.rtn import group_grid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _gauss(N, K, seed=0, std=0.02):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(N, K, generator=g) * std).to(DEVICE)


# ---------------------------------------------------------------- common


def test_shuffle_keeps_sorted_values_and_is_seeded():
    W = _gauss(64, 256)
    S = shuffle_within_tensor(W, 0)
    assert S.shape == W.shape and S.dtype == W.dtype
    assert torch.equal(S.reshape(-1).sort().values, W.reshape(-1).sort().values)
    assert not torch.equal(S, W)
    assert torch.equal(S, shuffle_within_tensor(W, 0))
    assert not torch.equal(S, shuffle_within_tensor(W, 1))


def test_gaussian_control_matches_group_std():
    W = _gauss(64, 512)
    W[:, :128] *= 5  # first group has 5x the std
    G = gaussian_control(W, group=128, seed=0)
    assert G.shape == W.shape
    sw = W.reshape(64, 4, 128).std(-1, unbiased=False)
    sg = G.reshape(64, 4, 128).std(-1, unbiased=False)
    assert (sg / sw).mean().item() == pytest.approx(1.0, abs=0.02)
    assert torch.equal(G, gaussian_control(W, 128, 0))


def test_uniform_index_entropy_is_exactly_b():
    for b in (2, 3, 4):
        q = torch.arange(2**b).repeat(1000)
        assert entropy_bits(q) == pytest.approx(b, abs=1e-12)
        qs = q - 2 ** (b - 1)  # signed grid indices
        assert entropy_bits(qs.float()) == pytest.approx(b, abs=1e-12)
    assert entropy_bits(torch.zeros(100)) == 0.0


def test_gain_helpers():
    assert bits_from_mse_ratio(4.0) == 1.0
    assert bits_from_mse_ratio(1.0) == 0.0
    assert log2_factorial(5) == pytest.approx(math.log2(120), rel=1e-12)
    assert n_top_columns(1024, 0.5) == 5 and n_top_columns(1024, 1) == 10 and n_top_columns(16, 0.5) == 1


def test_aggregate_is_numel_weighted_and_skips_non_numbers():
    r1 = ProbeRecord("a", 0, "q", {"x": 1.0, "s": "gauss"}, {"x": 0.0}, 1.0, 0.0, "", numel=100)
    r2 = ProbeRecord("b", 1, "q", {"x": 3.0, "s": "laplace", "y": 5.0}, {"x": 1.0}, 3.0, None, "", numel=300)
    r3 = ProbeRecord("c", 0, "k", {"x": 7.0}, {}, None, None, "", numel=10)
    agg = aggregate([r1, r2, r3])
    assert agg["q"]["stat"]["x"] == pytest.approx(2.5)
    assert agg["q"]["stat"]["y"] == pytest.approx(5.0)
    assert "s" in agg["q"]["stat"] and agg["q"]["stat"]["s"] is None
    assert agg["q"]["control"]["x"] == pytest.approx(0.75)
    assert agg["q"]["gain_bits"] == pytest.approx(2.5) and agg["q"]["gain_bits_control"] == pytest.approx(0.0)
    assert agg["k"]["stat"]["x"] == 7.0 and agg["k"]["gain_bits"] is None and agg["k"]["numel"] == 10
    assert agg["q"]["n_records"] == 2 and agg["q"]["n_excluded"] == 0
    # summary records (numel 0) and the P4 best-of-all-pairs record (is_best 1, is_adjacent 0) stay out of the aggregate
    r4 = ProbeRecord("q:cross-layer", -1, "q", {"jaccard_mean": 0.5, "n_layers": 24}, {"jaccard_mean": 0.01}, None, None, "", numel=0)
    r5 = ProbeRecord("best", 7, "q", {"x": 100.0, "is_best": 1, "is_adjacent": 0}, {"x": 9.0}, 50.0, 0.0, "", numel=400)
    r6 = ProbeRecord("adj", 8, "q", {"x": 1.0, "is_best": 1, "is_adjacent": 1}, {"x": 0.0}, 1.0, 0.0, "", numel=100)  # best AND adjacent stays
    agg2 = aggregate([r1, r2, r3, r4, r5, r6])
    assert agg2["q"]["n_records"] == 3 and agg2["q"]["n_excluded"] == 2 and agg2["q"]["numel"] == 500
    assert agg2["q"]["stat"]["x"] == pytest.approx((100 + 900 + 100) / 500) and "jaccard_mean" not in agg2["q"]["stat"]
    assert agg2["q"]["gain_bits"] == pytest.approx((100 + 900 + 100) / 500)
    only = aggregate([r4])  # a type with nothing but a summary keeps it (no weight, so its numbers are None)
    assert only["q"]["n_records"] == 1 and only["q"]["n_excluded"] == 0 and only["q"]["stat"]["jaccard_mean"] is None


def test_write_probe_json_schema(tmp_path):
    rec = ProbeRecord("model.layers.0.mlp.up_proj", 0, "up", {"x": 1.0, "nan": float("nan")}, {"x": 0.5}, 0.1, 0.0, "n", numel=8)
    path = tmp_path / "p.json"
    doc = write_probe_json(path, "qwen3.5-0.8b", "P-test", {"group": 128}, [rec])
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert set(on_disk) == {"model", "probe", "code_version", "timestamp", "params", "records", "aggregate"}
    assert on_disk["model"] == "qwen3.5-0.8b" and on_disk["probe"] == "P-test" and on_disk["params"] == {"group": 128}
    assert "/" in on_disk["code_version"] and len(on_disk["code_version"].split("/")[1]) == 10
    r = on_disk["records"][0]
    assert {"tensor", "layer", "type", "stat", "control", "gain_bits", "gain_bits_control", "note"} <= set(r)
    assert r["stat"]["nan"] is None  # non-finite values are written as null, never NaN
    assert on_disk["aggregate"]["up"]["gain_bits"] == pytest.approx(0.1)
    assert doc["records"][0]["tensor"] == rec.tensor


# ---------------------------------------------------------------- P1


def test_marginal_iid_gaussian_matches_control():
    W = _gauss(512, 1024, seed=3)
    rec = pw.marginal_entropy(W)
    s, c = rec.stat, rec.control
    assert abs(s["H_idx_b4"] - c["H_idx_b4"]) < 0.02
    assert abs(s["kurtosis_excess"]) < 0.05 and abs(c["kurtosis_excess"]) < 0.05
    assert s["best_family"] in ("gauss", "student") and s["nll_gauss"] < s["nll_laplace"]
    assert s["nll_student"] <= s["nll_gauss"] + 1e-3  # a large nu is Gaussian
    assert rec.gain_bits == pytest.approx(s["gain_bits_b4"]) and abs(rec.gain_bits - rec.gain_bits_control) < 0.02
    assert 0 < s["H_word"] <= 16 and s["H_sign"] == pytest.approx(1.0, abs=0.01) and s["H_mant"] < 7.0 + 1e-9
    assert s["lossless_bits_saved"] == pytest.approx(16 - s["H_word"])


def test_marginal_heavy_tail_beats_gaussian_and_uses_grid_indices():
    g = torch.Generator().manual_seed(0)
    W = (torch.distributions.StudentT(3.0).sample((256, 1024), ) * 0.01).to(DEVICE)
    rec = pw.marginal_entropy(W)
    s = rec.stat
    assert s["kurtosis_excess"] > 1.0 and s["best_family"] == "student" and 2.0 <= s["student_nu"] <= 5.0
    # heavy tails widen the groups' amax, so the indices concentrate: more coding gain than the Gaussian control
    assert s["H_idx_b4"] < rec.control["H_idx_b4"] - 0.1
    # the entropy is of exactly the Phase 1 grid's indices
    q, _, _ = group_grid(W.float().reshape(256, 8, 128), 4, True)
    assert s["H_idx_b4"] == pytest.approx(entropy_bits(q), abs=1e-9)
    assert s["gain_bits_b4"] == pytest.approx(4 - s["H_idx_b4"] - 16 * 16 / W.numel())


def test_laplace_family_detected():
    W = (torch.distributions.Laplace(0.0, 0.01).sample((256, 512))).to(DEVICE)
    s = pw.marginal_entropy(W).stat
    assert s["nll_laplace"] < s["nll_gauss"] and s["best_family"] in ("laplace", "student")


# ---------------------------------------------------------------- P2


def test_low_rank_exact_rank8():
    g = torch.Generator().manual_seed(1)
    U = torch.randn(256, 8, generator=g)
    V = torch.randn(8, 512, generator=g)
    W = (U @ V * 0.01).to(DEVICE)
    controls = [shuffle_within_tensor(W, 0), shuffle_within_tensor(W, 1)]
    rec = pw.low_rank(W, controls, r_extra=(8,))
    s, c = rec.stat, rec.control
    assert s["energy_r8"] == pytest.approx(1.0, abs=1e-6)
    assert s["rank99"] <= 8 and s["stable_rank"] <= 8.0 + 1e-6
    assert rec.gain_bits > 5.0 and s["best_r"] == 8
    assert c["energy_r8"] < 0.2 and rec.gain_bits_control < 0.1
    assert "gain_bits_best_seed0" in c and "gain_bits_best_seed1" in c and c["gain_bits_best_spread"] < 0.05


def test_low_rank_gaussian_has_no_gain_and_contract_keys():
    W = _gauss(256, 512)
    rec = pw.low_rank(W, [shuffle_within_tensor(W, 0)])
    for f in pw.R_FRACS:
        assert f"energy_f{f:g}" in rec.stat and f"gain_f{f:g}" in rec.stat
    assert rec.gain_bits < 0 and abs(rec.gain_bits - rec.gain_bits_control) < 0.05
    assert rec.stat["rank90"] <= rec.stat["rank99"] <= 256


def test_singular_values_gram_path_matches_svd():
    W = _gauss(300, 64)
    direct = torch.linalg.svdvals(W).double()
    N, K = W.shape
    A = W
    G = (A.double().T @ A.double())
    via_gram = torch.linalg.eigvalsh(G).clamp_min(0).sqrt().flip(0)
    assert torch.allclose(direct, via_gram, rtol=1e-4, atol=1e-7)
    assert torch.allclose(pw.singular_values(W), direct, rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------- P3


def test_outliers_iid_gaussian_rho_matches_control():
    W = _gauss(512, 1024, seed=5)
    controls = [shuffle_within_tensor(W, 0), shuffle_within_tensor(W, 1)]
    rec = pw.outliers(W, controls)
    s, c = rec.stat, rec.control
    for p in pw.PCTS:
        assert abs(s[f"rho_p{p:g}"] / c[f"rho_p{p:g}"] - 1) < 0.03
        assert s[f"rho_p{p:g}"] >= 1.0
    assert s["col_n_gt5x_median"] == 0 and s["row_n_gt5x_median"] == 0
    assert s["col_max_over_median"] < 1.5
    assert abs(rec.gain_bits - rec.gain_bits_control) < 0.03 and rec.gain_bits < 0


def test_outliers_planted_columns_give_gain():
    W = _gauss(512, 1024, seed=6)
    W[:, [3, 500, 777, 1000, 1023]] *= 20  # five outlier input channels (0.5% of 1024)
    rec = pw.outliers(W, [shuffle_within_tensor(W, 0)])
    s = rec.stat
    assert s["col_n_gt10x_median"] == 5 and s["col_max_over_median"] > 10
    assert s["rho_p0.5"] > 10 and rec.gain_bits > 1.0 and rec.gain_bits_control < 0.1
    assert s["best_pct"] == 0.5
    # kept columns are restored exactly and excluded from the grid
    cols = torch.tensor([3, 500, 777, 1000, 1023], device=DEVICE)
    assert pw.rtn_mse_keepcols(W, cols) < pw.rtn_mse(W) / 10


def test_outlier_consistency_jaccard():
    K = 1024
    g = torch.Generator().manual_seed(0)
    base = torch.rand(K, generator=g)
    hot = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    norms = []
    for layer in range(6):
        v = base + torch.rand(K, generator=g) * 0.1
        v[hot] += 10
        norms.append((layer, v))
    rec = pw.outlier_consistency(norms, top=16, seed=0)
    assert rec.stat["jaccard_mean"] == pytest.approx(1.0) and rec.stat["n_channels_in_half_layers"] == 16
    assert rec.control["jaccard_mean"] < 0.05
    rnd = [(layer, torch.rand(K, generator=g)) for layer in range(6)]
    assert pw.outlier_consistency(rnd, 16, 0).stat["jaccard_mean"] < 0.05


# ---------------------------------------------------------------- P4


def test_procrustes_known_rotation():
    A = _gauss(512, 1024, seed=7)
    R = random_orthogonal(512, seed=1).to(DEVICE)
    assert pw.procrustes_residual(A, R @ A) < 1e-4  # output side, N < K: direct path
    Rin = random_orthogonal(1024, seed=2).to(DEVICE)
    assert pw.procrustes_residual(A.T, (A @ Rin).T) < 1e-4  # input side, via the QR path (K > N)
    A2 = _gauss(1024, 512, seed=8)  # N > K: output side through the QR path
    R2 = random_orthogonal(1024, seed=3).to(DEVICE)
    assert pw.procrustes_residual(A2, R2 @ A2) < 1e-4
    # two independent Gaussians: alignment fits little; residual stays near 1
    B = _gauss(512, 1024, seed=9)
    e = pw.procrustes_residual(A, B)
    assert 0.5 < e < 1.0


def test_permutation_alignment_recovers_row_permutation():
    A = _gauss(512, 256, seed=10)
    g = torch.Generator().manual_seed(4)
    perm = torch.randperm(512, generator=g).to(DEVICE)
    B = A[perm]
    e, cos = pw.permutation_residual(A, B, axis=0)
    assert e < 1e-10 and cos == pytest.approx(1.0, abs=1e-6)
    e_col, _ = pw.permutation_residual(A.T, B.T, axis=1)
    assert e_col < 1e-10


def test_interlayer_records_and_gains():
    g = torch.Generator().manual_seed(11)
    base = torch.randn(256, 512, generator=g) * 0.02
    series = []
    for layer in range(4):
        W = base + torch.randn(256, 512, generator=g) * 0.002  # layers share a common component
        series.append((f"model.layers.{layer}.mlp.up_proj", layer, W.to(DEVICE).to(torch.bfloat16)))
    recs = pw.interlayer(series, perm_axis=0, seed=0)
    assert len(recs) >= 3 and sum(r.stat["is_adjacent"] for r in recs) == 3 and sum(r.stat["is_best"] for r in recs) == 1
    best = next(r for r in recs if r.stat["is_best"])
    assert best.stat["n_pairs"] == 6 and "allpairs_e_out_min" in best.stat
    for r in recs:
        assert r.stat["e_raw"] < 0.05 and r.stat["cos_raw"] > 0.95  # strongly related tensors
        assert r.stat["e_out"] <= r.stat["e_raw"] + 1e-6 and r.stat["e_in"] <= r.stat["e_raw"] + 1e-6
        assert "e_perm" in r.stat and r.stat["e_perm"] <= r.stat["e_raw"] + 1e-6
        assert r.gain_bits > 1.0 and r.gain_bits_control < 0.5
        assert r.control["e_raw"] > 1.5  # independent shuffles: ||B - A||^2 ~ 2 ||B||^2
        assert r.tensor.endswith(f"model.layers.{r.layer}.mlp.up_proj")
    assert pw.interlayer(series[:1], perm_axis=None) == []


def test_p3_keepcols_mse_is_the_p0_code_path():
    """P3's estimate quantizes exactly as P0 +keepcols does (one definition of the kept count and of the grid)."""
    from wcomp.quant.rtn import keepcols_mask, n_keep_columns, quantize_rtn_keepcols

    W = _gauss(256, 1024, seed=11)
    W[:, [5, 600]] *= 15
    assert n_top_columns(1024, 2) == n_keep_columns(1024, 2) == 20
    mask = keepcols_mask(W, 1)
    W_hat, info = quantize_rtn_keepcols(W, 4, 128, mask, True)
    direct = float(((W - W_hat) ** 2).mean())
    assert pw.rtn_mse_keepcols(W, mask) == pytest.approx(direct, rel=1e-6)
    assert pw.rtn_mse_keepcols(W, mask.nonzero().squeeze(1)) == pytest.approx(direct, rel=1e-6)
    assert info["n_keep"] == 10


def test_p3_overhead_is_the_p0_keepcols_charge():
    """The P3 estimate's overhead equals the extra bits/param the P0 +keepcols row pays: (16-b)*n_keep/K + 1/N with
    the same n_keep the grid used, not the nominal p/100 (round(1024*0.5/100) = 5 columns, not 5.12)."""
    from wcomp.quant.rtn import keepcols_mask, quantize_rtn, quantize_rtn_keepcols

    W = _gauss(256, 1024, seed=12)
    for bits in (4, 3):
        rec = pw.outliers(W, [shuffle_within_tensor(W, 0)], pw.PCTS, bits, pw.GROUP)
        for p in pw.PCTS:
            n_keep = n_top_columns(1024, p)
            _, plain = quantize_rtn(W, bits, 128, True)
            _, kept = quantize_rtn_keepcols(W, bits, 128, keepcols_mask(W, p), True)
            charge = ((kept["index_bits"] + kept["aux_bits"]) - (plain["index_bits"] + plain["aux_bits"])) / W.numel()
            overhead = bits_from_mse_ratio(rec.stat[f"rho_p{p:g}"]) - rec.stat[f"gain_p{p:g}"]
            assert overhead == pytest.approx(charge, rel=1e-9) == pytest.approx((16 - bits) * n_keep / 1024 + 1 / 256, rel=1e-9)
            assert rec.stat[f"kept_frac_p{p:g}"] == n_keep / 1024
        assert "the P0 +keepcols charge" in rec.note


def test_interlayer_best_pair_is_selected_by_the_reported_gain():
    """The best pair maximizes the overhead-corrected gain that gain_bits_best reports; the long-side Procrustes of a
    non-square tensor (residual near 0 even on shuffled controls) does not select it."""
    g = torch.Generator().manual_seed(21)
    series = []
    for layer in range(4):
        W = torch.randn(16, 1024, generator=g) * 0.02  # in_proj_b-like: 16 x 1024, independent layers
        series.append((f"model.layers.{layer}.linear_attn.in_proj_b", layer, W.to(DEVICE)))
    with torch.no_grad():
        series[2] = (series[2][0], 2, series[0][2] * 0.999 + torch.randn(16, 1024, generator=g).to(DEVICE) * 1e-4)  # pair (0, 2) is nearly identical
    recs = pw.interlayer(series, perm_axis=None, seed=0)
    best = next(r for r in recs if r.stat["is_best"])
    assert (best.stat["layer_i"], best.stat["layer_j"]) == (0, 2)
    for r in recs:
        assert r.control["e_in"] < 0.05  # the 1024 x 1024 input-side rotation fits the shuffled control too
        assert r.gain_bits == pytest.approx(max(r.stat["gain_raw"], r.stat["gain_out"], r.stat["gain_in"]))
        assert r.stat["gain_in"] < -500  # never the alignment that attains gain_bits_best
    assert "selection uncontrolled" in best.note and best.gain_bits == pytest.approx(best.stat["gain_raw"])
    assert all("selection uncontrolled" not in r.note for r in recs if not r.stat["is_best"])


def test_outlier_consistency_skips_same_layer_pairs():
    """A pooled series (several linears of one layer) counts only cross-layer pairs, for W and for the control."""
    K = 512
    g = torch.Generator().manual_seed(2)
    hot_a = torch.arange(16)
    hot_b = torch.arange(100, 116)
    norms = []
    for layer in range(3):  # two entries per layer sharing the layer's hot set, different sets across layers
        hot = hot_a if layer % 2 == 0 else hot_b
        for _ in range(2):
            v = torch.rand(K, generator=g)
            v[hot] += 10
            norms.append((layer, v))
    rec = pw.outlier_consistency(norms, top=16, seed=0)
    assert rec.stat["n_layers"] == 6 and rec.stat["n_pairs"] == 12 and rec.stat["n_same_layer_pairs_skipped"] == 3
    # cross-layer pairs: layers 0 and 2 share hot_a (4 pairs of 12 agree fully), the rest are disjoint
    assert rec.stat["jaccard_mean"] == pytest.approx(4 / 12)
    assert rec.control["jaccard_mean"] < 0.05
    assert "same-layer pairs skipped" in rec.note and rec.stat["n_distinct_layer_pairs"] == 3
    # variance-matched control: one random set per layer, shared by the layer's entries, so the pooled control equals
    # the control of a one-entry-per-layer series over the same layers (every cross-layer pair repeats a layer pair)
    one_per_layer = [norms[0], norms[2], norms[4]]
    assert rec.control["jaccard_mean"] == pytest.approx(pw.outlier_consistency(one_per_layer, 16, 0).control["jaccard_mean"])
    with pytest.raises(ValueError, match="different layers"):
        pw.outlier_consistency([(0, torch.rand(K, generator=g)), (0, torch.rand(K, generator=g))], 16, 0)
    # a plain one-entry-per-layer series is unchanged by the rule
    plain = [(layer, torch.rand(K, generator=g)) for layer in range(5)]
    assert pw.outlier_consistency(plain, 16, 0).stat["n_same_layer_pairs_skipped"] == 0


def test_marginal_control_states_its_std_estimation_bias():
    """The Gaussian control is a scale mixture (per-group sampled std): its stated expected excess kurtosis is 6/(g-1)
    and the analytic iid-Gaussian references sit next to the measured control moments."""
    W = _gauss(256, 1024, seed=4)
    rec = pw.marginal_entropy(W)
    c = rec.control
    assert c["kurtosis_excess_expected_from_std_estimation"] == pytest.approx(6 / 127)
    assert c["gauss_kurtosis_excess"] == 0.0 and c["gauss_frac_gt4sigma"] == pytest.approx(6.33e-5, rel=1e-2)
    assert 0.0 < c["kurtosis_excess"] < 0.15  # positive for an iid Gaussian W: the mixture, not the data
    assert "std estimation" in rec.note and "gauss_*" in rec.note
    assert abs(rec.gain_bits - rec.gain_bits_control) < 2 * pw.ENTROPY_GAIN_CONTROL_FLOOR  # an iid Gaussian W sits inside the stated floor
