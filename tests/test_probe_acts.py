"""Tests for wcomp.probe.acts: synthetic HF-shaped stack (known input energies), Gini, allocation closed forms,
Fisher and super-weight passes leaving the model bit-identical."""
import copy
import math

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from wcomp.probe import acts
from wcomp.probe.acts import (
    allocation_saving,
    collect_input_stats,
    fisher_diag,
    gini,
    hessian_sensitivity,
    param_checksum,
    rank_profile,
    super_weights,
    usage_stats,
    weighted_low_rank,
)


class _Layer(nn.Module):
    def __init__(self, hidden: int, inter: int, mix: bool = False):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)
        self.mix = mix

    def forward(self, h):
        h = h + self.mlp.down_proj(torch.relu(self.mlp.up_proj(h)))
        if self.mix:  # causal running mean over positions: position t feeds every later position, as attention does
            h = torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        return h


class _Inner(nn.Module):
    def __init__(self, vocab: int, hidden: int, inter: int, n_layers: int, mix: bool = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_Layer(hidden, inter, mix) for _ in range(n_layers)])

    def forward(self, input_ids, use_cache=False):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return BaseModelOutput(last_hidden_state=h)


class TinyLM(nn.Module):
    """HF-shaped causal LM: model.embed_tokens, model.layers[i].mlp.{up,down}_proj, lm_head. Position-wise by default
    (no token mixing); mix=True adds a causal running mean after every MLP."""

    def __init__(self, vocab=50, hidden=8, inter=16, n_layers=2, seed=0, mix=False):
        super().__init__()
        torch.manual_seed(seed)
        self.model = _Inner(vocab, hidden, inter, n_layers, mix)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():  # known, unequal per-channel input energies for layer 0
            self.model.embed_tokens.weight *= torch.arange(1, hidden + 1).float()

    def forward(self, input_ids, use_cache=False):
        return CausalLMOutput(logits=self.lm_head(self.model(input_ids).last_hidden_state))


def _ids(n=3, T=12, vocab=50, seed=1):
    return torch.randint(0, vocab, (n, T), generator=torch.Generator().manual_seed(seed))


def test_input_stats_match_known_inputs():
    model = TinyLM().eval()
    ids = _ids()
    stats = collect_input_stats(model, ids, full_hessian=True, layer_chunk=2)
    with torch.no_grad():
        x = model.model.embed_tokens(ids).reshape(-1, 8)
        a = torch.relu(model.model.layers[0].mlp.up_proj(x))
    n = x.shape[0]
    up, down = stats["model.layers.0.mlp.up_proj"], stats["model.layers.0.mlp.down_proj"]
    assert up["n_tokens"] == n and down["n_tokens"] == n
    assert torch.allclose(up["diag_h"], (x * x).mean(0), atol=1e-4)
    assert torch.allclose(up["H"], x.T @ x / n, atol=1e-4)
    assert torch.allclose(torch.diag(up["H"]), up["diag_h"], atol=1e-5)
    assert torch.allclose(down["diag_h"], (a * a).mean(0), atol=1e-4)
    assert torch.allclose(down["H"], a.T @ a / n, atol=1e-4)
    # tau = 0.05 * rms over all inputs of that linear; the count uses exactly that tau
    rms = math.sqrt(float((x * x).sum()) / x.numel())
    assert up["rms"] == pytest.approx(rms, rel=1e-6) and up["tau"] == pytest.approx(0.05 * rms, rel=1e-6)
    assert torch.equal(up["active"], (x.abs() > 0.05 * rms).sum(0))
    assert up["shape"] == (16, 8) and up["numel"] == 128
    # chunking (with the early-stop hook) does not change anything
    stats1 = collect_input_stats(model, ids, full_hessian=True, layer_chunk=1)
    for k in stats:
        assert torch.equal(stats1[k]["H"], stats[k]["H"]) and torch.equal(stats1[k]["active"], stats[k]["active"])
    assert set(stats) == {f"model.layers.{i}.mlp.{p}_proj" for i in range(2) for p in ("up", "down")}


def test_gini():
    assert gini(torch.ones(100)) == pytest.approx(0.0, abs=1e-12)
    one_hot = torch.zeros(1000)
    one_hot[7] = 3.0
    assert gini(one_hot) == pytest.approx(1 - 1 / 1000, abs=1e-9)
    assert 0.3 < gini(torch.arange(1, 101).float()) < 0.4  # uniform ramp: (n-1)/(3n) = 0.33


def test_allocation_saving_closed_forms():
    equal = allocation_saving(torch.full((10,), 3.0), torch.full((10,), 7.0))
    assert equal["saving"] == pytest.approx(0.0, abs=1e-12) and equal["overhead"] == 0.0 and equal["levels"] == 1
    # two equally sized unit classes with sensitivity density 1 and 100: 0.5*log2(mean/geomean) = 0.5*log2(50.5/10)
    s = torch.tensor([1.0] * 4 + [100.0] * 4)
    n = torch.ones(8)
    out = allocation_saving(s, n)
    assert out["saving"] == pytest.approx(0.5 * math.log2(50.5 / 10.0), rel=1e-9)
    assert out["levels"] == 2 and out["overhead"] == pytest.approx(1.0 * 8 / 8)
    assert out["gain"] == pytest.approx(out["saving"] - 1.0)
    # numel weighting: the same density pattern with unit sizes 3 and 1 changes the weights
    out_w = allocation_saving(torch.tensor([3.0, 100.0]), torch.tensor([3.0, 1.0]))
    mean, geo = (3 * 1 + 1 * 100) / 4, 2 ** ((3 * math.log2(1) + math.log2(100)) / 4)
    assert out_w["saving"] == pytest.approx(0.5 * math.log2(mean / geo), rel=1e-9)
    # a dead unit is floored at mean * 2^-8 instead of making the saving infinite
    dead = allocation_saving(torch.tensor([0.0, 1.0, 1.0, 1.0]), torch.ones(4))
    assert math.isfinite(dead["saving"]) and dead["frac_floored"] == pytest.approx(0.25)
    assert dead["saving"] == pytest.approx(0.5 * (math.log2(0.75) - 0.25 * math.log2(0.75 * 2 ** -8)), rel=1e-9)
    prefixed = allocation_saving(s, n, prefix="alloc_col")
    assert set(prefixed) == {"alloc_col_saving", "alloc_col_overhead", "alloc_col_gain", "alloc_col_levels",
                             "alloc_col_frac_floored", "alloc_col_spread_log10", "alloc_col_saving_cap2bit", "alloc_col_saving_unfloored"}


def test_allocation_saving_guarded_variants_expose_the_floor():
    """A unit far below the 2^-8 floor contributes 0.5 * w * 8 bits to `saving` regardless of its s_u; the guarded
    variants do not: cap2bit floors at mean 2^-4 (b_u >= 2) and unfloored drops the unit."""
    s = torch.tensor([1e-9, 1.0, 1.0, 1.0])
    n = torch.ones(4)
    out = allocation_saving(s, n)
    mean = 0.75 + 2.5e-10
    assert out["frac_floored"] == pytest.approx(0.25)
    assert out["saving"] == pytest.approx(0.5 * (math.log2(mean) - 0.25 * math.log2(mean * 2 ** -8)), rel=1e-6)
    assert out["saving"] > 0.8  # the floored unit alone contributes 0.5 * 0.25 * 8 = 1 bit of the 0.84
    assert out["saving_cap2bit"] == pytest.approx(0.5 * (math.log2(mean) - 0.25 * math.log2(mean * 2 ** -4)), rel=1e-6)
    assert out["saving_cap2bit"] < out["saving"]
    assert out["saving_unfloored"] == pytest.approx(0.0, abs=1e-12)  # the three remaining units are equal
    # without floored units the three savings coincide
    even = allocation_saving(torch.tensor([1.0, 4.0]), torch.ones(2))
    assert even["frac_floored"] == 0.0
    assert even["saving"] == pytest.approx(even["saving_cap2bit"]) == pytest.approx(even["saving_unfloored"])


def test_rank_profile_and_rank1_energy():
    u, v = torch.randn(64, 1), torch.randn(1, 32)
    sv = torch.linalg.svdvals(u @ v)
    out = rank_profile(sv, 64, 32)
    assert out["energy_r1pct"] == pytest.approx(1.0, abs=1e-6) and out["rank_99"] == 1 and out["stable_rank"] == pytest.approx(1.0, abs=1e-6)
    assert out["best_r"] == 1 and out["best_gain"] > 10  # e_r floored at 1e-12 -> 0.5*log2(1e12) - 16*96/2048
    assert acts.rank1_energy(u @ v) == pytest.approx(1.0, abs=1e-5)
    F = torch.rand(40, 30)
    assert abs(acts.rank1_energy(F) - float(torch.linalg.svdvals(F)[0] ** 2 / F.pow(2).sum())) < 1e-5


def test_usage_and_sensitivity_on_tiny():
    model = TinyLM().eval()
    stats = collect_input_stats(model, _ids(), full_hessian=True)
    stats2 = collect_input_stats(model, _ids(seed=5), full_hessian=False)
    recs = usage_stats(stats, stats2)
    assert len(recs) == 4 and all(0 <= r.stat["gini"] <= 1 for r in recs)
    down = next(r for r in recs if r.type == "down")
    assert "spearman_split_half" in down.stat and down.note.startswith("primary")
    recs6, glob = hessian_sensitivity(model, stats)
    assert all(r.stat["s"] >= 0 for r in recs6) and glob["alloc_tensor_saving"] >= 0
    # the headline travels with its guards: floored fraction, capped / unfloored savings, within-type and relative numbers
    for g in ("tensor", "group", "col"):
        assert {f"alloc_{g}_frac_floored", f"alloc_{g}_saving_cap2bit", f"alloc_{g}_saving_unfloored"} <= set(glob)
    assert {"alloc_tensor_within_type_saving", "alloc_tensor_within_type_frac_floored", "alloc_tensor_rel_saving", "caveat"} <= set(glob)
    for r in recs6:
        assert r.stat["out_energy"] > 0 and r.stat["s_rel"] == pytest.approx(r.stat["s"] / r.stat["out_energy"])
    # with diag(H) = 1 everywhere, s_t is the plain RTN-4 MSE sum, and s_rel = MSE sum / ||W||^2
    ones = {k: dict(v, diag_h=torch.ones_like(v["diag_h"])) for k, v in stats.items()}
    recs1, glob1 = hessian_sensitivity(model, ones)
    for r in recs1:
        assert r.stat["s"] == pytest.approx(r.stat["mse_rtn"] * r.numel, rel=1e-6)
    # within-type allocation: every type has one tensor per layer here, so it is the numel-weighted mean over the
    # types of the two-layer allocation of that type, never more than the all-tensors saving plus the floor effects
    assert 0 <= glob1["alloc_tensor_within_type_saving"] <= glob1["alloc_tensor_saving"] + 1e-9 or glob1["alloc_tensor_frac_floored"] > 0
    recs2b = weighted_low_rank(model, stats)
    assert len(recs2b) == 4 and all("seed0_best_gain" in r.control and "seed1_best_gain" in r.control for r in recs2b)
    for r in recs2b:  # P2 / P3 convention: the control gain is the mean over the two shuffle seeds
        assert r.gain_bits_control == pytest.approx((r.control["seed0_best_gain"] + r.control["seed1_best_gain"]) / 2)
        assert r.control["gain_bits_best"] == r.gain_bits_control and "mean over the control seeds" in r.note
        # the contract's e_r formula travels next to the measured gain, on W (stat) and as the control mean
        assert r.control["formula_gain_bits_best"] == pytest.approx((r.control["seed0_formula_best_gain"] + r.control["seed1_formula_best_gain"]) / 2)
        assert "formula_best_gain" in r.stat and "control.formula_gain_bits_best" in r.note
    check = acts.prediction_check(recs6, [{"method": "rtn-sym-g128-4@only=up", "kl": 0.1},
                                          {"method": "rtn-sym-g128-4@only=down", "kl": 0.2}])
    assert check["n_types_only_4bit"] == 2 and "spearman_type_only_4bit" not in check


def test_fisher_diag_leaves_parameters_bit_identical():
    model = TinyLM().eval()
    for p in model.parameters():
        p.requires_grad_(True)  # from_pretrained models come with requires_grad=True
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cs = param_checksum(model)
    recs, glob = fisher_diag(model, _ids(n=4, T=16), n_windows=4, seqlen=8)
    assert param_checksum(model) == cs
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert len(recs) == 4 and all(r.stat["total"] > 0 for r in recs)
    assert all(0 <= r.stat["rank1_energy"] <= 1 + 1e-6 and 0 <= r.control["rank1_energy"] <= 1 + 1e-6 for r in recs)
    assert glob["n_windows"] == 4 and glob["seqlen"] == 8 and glob["checksum_unchanged"]


def test_super_weights_restores_weights():
    model = TinyLM(n_layers=3).eval()
    before = copy.deepcopy(model.state_dict())
    eval_ids = _ids(n=6, T=16, seed=9)
    recs, extra = super_weights(model, _ids(n=1, T=16, seed=3)[0], eval_ids, spike_factor=1.5, top_k=2, kl_windows=4)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    assert extra["checksum_unchanged"] and len(extra["layer_stats"]) == 3
    assert len(recs) >= 2
    # kinds follow the contract: "spike" only with a co-occurring input spike, otherwise "spike_no_input"; the
    # fallback never coincides with a top-k weight; the extra dict counts the spike candidates
    assert set(extra["kinds"]) == {r.note.split(";")[0] for r in recs} and "n_spike_candidates" in extra
    for r in recs:
        kind = r.note.split(";")[0].split("+")
        if "spike" in kind:
            assert r.stat["co_occurs"] is True
        if "spike_no_input" in kind:
            assert r.stat["co_occurs"] is False
        if "max_spike_below_threshold" in kind:
            assert r.stat["magnitude_rank"] is None and len(kind) == 1
    assert extra["n_spike_candidates"] == sum(1 for r in recs if "spike" in r.note.split(";")[0].split("+"))
    for r in recs:
        for key in ("row", "col", "value", "ppl", "dppl_pct", "kl", "top1_agree"):
            assert key in r.stat and f"magnitude_{key}" in r.control and f"random_{key}" in r.control
        assert (r.control["magnitude_row"], r.control["magnitude_col"]) != (r.stat["row"], r.stat["col"])
        assert (r.control["random_row"], r.control["random_col"]) != (r.stat["row"], r.stat["col"])
    assert extra["reference"]["windows"] == 4


def test_super_weight_fallback_yields_to_top_k():
    """When no layer crosses the spike threshold and the strongest spike's weight is a global top-k weight, the
    record is a plain top-k candidate (not tagged as the fallback, so +keepsuper keeps it)."""
    model = TinyLM(n_layers=1).eval()  # one layer: it is the strongest spike by definition
    window = _ids(n=1, T=16, seed=3)[0]
    s = acts.down_proj_spikes(model, window)[0]
    lin = dict(acts.block_linears(model))[s["tensor"]]
    with torch.no_grad():
        lin.weight[s["out_channel"], s["in_channel"]] = 8.0  # the global |w| maximum (others < 0.5), keeps the argmax channels
    recs, extra = super_weights(model, window, _ids(n=4, T=16, seed=9), spike_factor=1e9, top_k=2, kl_windows=2)
    kinds = {(r.tensor, r.stat["row"], r.stat["col"]): r.note.split(";")[0] for r in recs}
    assert kinds[(s["tensor"], s["out_channel"], s["in_channel"])] == "top2"
    assert "max_spike_below_threshold" not in extra["kinds"] and extra["n_spike_candidates"] == 0
    assert extra["reference"]["windows"] == 2 and extra["n_candidates"] == 2  # eval on kl_windows windows


def _fisher_references(model: nn.Module, ids: torch.Tensor) -> tuple[dict, dict, dict]:
    """Three reference statistics per block linear, by autograd on the weights / outputs: the position-wise statistic
    sum_t (d_t x_t^T)^2 / T_total with d_t = d(sum_s nll_s)/d y_t (what the hooks accumulate), the batched per-window
    mean-NLL weight gradient squared averaged over windows, and the brute-force per-token Fisher
    sum_t (d nll_t/dW)^2 / T_total (one backward per target position)."""
    linears = acts.block_linears(model)
    weights = [lin.weight for _, lin in linears]
    pos = {name: torch.zeros_like(lin.weight, dtype=torch.float32) for name, lin in linears}
    bat = {name: torch.zeros_like(lin.weight, dtype=torch.float32) for name, lin in linears}
    tok = {name: torch.zeros_like(lin.weight, dtype=torch.float32) for name, lin in linears}
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def make(name):
        def hook(mod, args, out):
            cache[name] = (args[0].detach(), out)

        return hook

    handles = [lin.register_forward_hook(make(n)) for n, lin in linears]
    for w in weights:
        w.requires_grad_(True)
    n_tokens = 0
    for w in range(ids.shape[0]):
        x = ids[w : w + 1]
        logits = model(x).logits[0, :-1]
        nll = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none")
        n_tokens += nll.numel()
        for t in range(nll.numel()):
            for (name, _), g in zip(linears, torch.autograd.grad(nll[t], weights, retain_graph=True)):
                tok[name] += g.float() ** 2
        for (name, _), g in zip(linears, torch.autograd.grad(nll.mean(), weights, retain_graph=True)):
            bat[name] += g.float() ** 2
        names = [n for n, _ in linears]
        d_all = torch.autograd.grad(nll.sum(), [cache[n][1] for n in names])  # d_t = d(sum_s nll_s)/d y_t per linear
        for name, d in zip(names, d_all):
            xin = cache[name][0]
            dd = d.reshape(-1, d.shape[-1]).float()
            xx = xin.reshape(-1, xin.shape[-1]).float()
            pos[name] += (dd * dd).T @ (xx * xx)
    for h in handles:
        h.remove()
    for p in model.parameters():
        p.requires_grad_(False)
    n = ids.shape[0]
    return ({k: v / n_tokens for k, v in pos.items()}, {k: v / n for k, v in bat.items()}, {k: v / n_tokens for k, v in tok.items()})


@pytest.mark.parametrize("mix", [False, True])
def test_fisher_accumulate_is_the_positionwise_statistic_and_batched_the_mean_gradient_squared(mix):
    """The hook accumulation equals sum_t (d_t x_t^T)^2 / T with d_t = d(sum_s nll_s)/d y_t (the position-wise
    empirical-Fisher approximation) whether or not positions mix; it is the brute-force per-token Fisher
    mean_t (d nll_t/dW)^2 only in the position-wise model (mix=False) and differs from it once a causal running mean
    mixes positions (mix=True), which is why the labels say "position-wise", not "per-token". The batched statistic is
    the per-window mean-NLL weight gradient squared, averaged over windows (round-1 definition)."""
    model = TinyLM(mix=mix).eval()
    ids = _ids(n=3, T=10)
    acc, acc_b, n_tokens, n = acts.fisher_accumulate(model, ids, n_windows=3, seqlen=10, batched=True)
    assert n == 3 and n_tokens == 3 * 9
    pos, bat, tok = _fisher_references(model, ids)
    for name in acc:
        assert torch.allclose(acc[name], pos[name], rtol=1e-4, atol=1e-7), name
        assert torch.allclose(acc_b[name], bat[name], rtol=1e-4, atol=1e-7), name
        # the two definitions differ (cross-token terms): the test would be vacuous if they coincided
        assert not torch.allclose(acc[name], acc_b[name], rtol=1e-2)
    per_token = {name: torch.allclose(acc[name], tok[name], rtol=1e-4, atol=1e-7) for name in acc}
    if mix:
        assert not per_token["model.layers.0.mlp.up_proj"] and not per_token["model.layers.0.mlp.down_proj"]
        rel = {name: float((acc[name] - tok[name]).norm() / tok[name].norm()) for name in acc}
        assert max(rel.values()) > 0.1, rel  # a label "per-token Fisher" would be wrong by more than noise here
    else:
        assert all(per_token.values()), per_token
    recs, glob = fisher_diag(model, ids, n_windows=3, seqlen=10)
    for r in recs:
        assert {"batched_total", "batched_gini", "batched_rank1_energy", "batched_over_positionwise_total"} <= set(r.stat)
        assert "batched_rank1_energy" in r.control
        assert "position-wise" in r.note and "d_t = d(sum_s nll_s)/d y_t" in r.note and "only without token mixing" in r.note
        assert "mean_t (d nll_t / dW)^2 =" not in r.note and "per-token empirical Fisher" not in r.note
    assert glob["n_tokens"] == 27 and "position-wise" in glob["definition"] and "per-token empirical Fisher" not in glob["definition"]
    recs_nb, _ = fisher_diag(model, ids, n_windows=3, seqlen=10, batched=False)
    assert all("batched_total" not in r.stat for r in recs_nb)


def test_weighted_rank_profile_gain_is_the_measured_rtn_ratio():
    """P2b gain_r = 0.5*log2(||dW L||^2 / ||dE_r L||^2) - 16 r (N+K)/(N K) with E_r = W - lowrank_r(W L) L^-1 and
    dX = X - RTN-4(X), recomputed independently here; the e_r energy formula is kept only as formula_gain_r*."""
    from wcomp.quant.rtn import quantize_rtn

    g = torch.Generator().manual_seed(3)
    N, K = 48, 32
    W = torch.randn(N, K, generator=g) * 0.02
    X = torch.randn(256, K, generator=g) * (1 + 30 * (torch.rand(K, generator=g) < 0.1))  # anisotropic inputs
    H = (X.T @ X / 256).double()
    L, _ = acts._cholesky_damped(H, 0.01)
    Lf = L.float()
    out = acts.weighted_rank_profile(W, Lf, 4, 128)
    U, S, Vh = torch.linalg.svd(W @ Lf, full_matrices=False)
    eW = float(((W - quantize_rtn(W, 4, K, True)[0]) @ Lf).double().pow(2).sum())
    for frac, key in zip(acts.RANK_FRACTIONS, acts.RANK_KEYS):
        r = max(1, round(frac * min(N, K)))
        E = W - ((U[:, :r] * S[:r]) @ Vh[:r]) @ torch.linalg.inv(Lf)
        eE = float(((E - quantize_rtn(E, 4, K, True)[0]) @ Lf).double().pow(2).sum())
        gain = 0.5 * math.log2(eW / eE) - 16 * r * (N + K) / (N * K)
        assert out[f"gain_r{key}"] == pytest.approx(gain, rel=1e-4, abs=1e-5)
        assert out[f"rtn_err_ratio_r{key}"] == pytest.approx(eW / eE, rel=1e-4)
        assert 0 < out[f"residual_weight_energy_r{key}"] <= 1.0 + 1e-6
        # the energy formula is still there as a statistic, and here it is far above the measured gain
        e_r = 1 - out[f"energy_r{key}"]
        assert out[f"formula_gain_r{key}"] == pytest.approx(0.5 * math.log2(1 / max(e_r, 1e-12)) - 16 * r * (N + K) / (N * K), rel=1e-6)
    assert out["best_gain"] == max(out[f"gain_r{k}"] for k in acts.RANK_KEYS)
    assert out["formula_best_gain"] == max(out[f"formula_gain_r{k}"] for k in acts.RANK_KEYS)
    # an exactly low-rank W L (rank 2 = 5 % of 32) quantizes its residual to nothing: the measured gain is large
    A = (torch.randn(N, 2, generator=g) @ torch.randn(2, K, generator=g)) * 0.02
    W2 = A @ torch.linalg.inv(Lf)
    out2 = acts.weighted_rank_profile(W2, Lf, 4, 128)
    assert out2["gain_r5pct"] > 5 and out2["best_r"] == 2
    prefixed = acts.weighted_rank_profile(W, Lf, 4, 128, prefix="seed0")
    assert set(prefixed) == {"seed0_" + k for k in out}
