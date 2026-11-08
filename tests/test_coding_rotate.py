"""Tests for the Phase 3 coding / bound tools (wcomp.coding, wcomp.rdbound) and the rotation variants (method D)."""
from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from wcomp.bits import BitAccount
from wcomp.coding import (CHOICE_BITS, canonical_codes, coded_bits, coded_bits_detail, huffman_code_lengths, huffman_decode,
                          huffman_encode, tensor_coded_bits)
from wcomp.quant.rotate import (ROTATION_KINDS, collect_input_hessians, eig_rotation, hadamard, hadsign, random_orthogonal,
                                random_signs, rotate_inputs, rotation_bits, shared_input_groups)
from wcomp.quant.rtn import group_grid, quantize_rtn
from wcomp.rdbound import RD_FIELDS, normalized_sq_error, rd_bound_bits, rd_bound_rows


def _kraft(lengths):
    return sum(2.0 ** -l for l in lengths if l > 0)


def _entropy(counts) -> float:
    c = torch.tensor(counts, dtype=torch.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * torch.log2(p)).sum())


# ---------------------------------------------------------------- Huffman


def test_huffman_textbook_lengths_and_kraft():
    # dyadic distribution 1/2, 1/4, 1/8, 1/16, 1/16: the unique optimal lengths are 1, 2, 3, 4, 4 and the code is complete
    lengths = huffman_code_lengths([8, 4, 2, 1, 1])
    assert lengths == [1, 2, 3, 4, 4]
    assert _kraft(lengths) == pytest.approx(1.0)
    counts = [8, 4, 2, 1, 1]
    assert sum(l * c for l, c in zip(lengths, counts)) == pytest.approx(_entropy(counts) * sum(counts))
    # 1/2, 1/4, 1/8, 1/8 -> 1, 2, 3, 3; zero-count symbols get length 0 and do not take code space
    assert huffman_code_lengths([4, 2, 1, 1]) == [1, 2, 3, 3]
    assert huffman_code_lengths([0, 4, 0, 2, 1, 1, 0]) == [0, 1, 0, 2, 3, 3, 0]
    # non-dyadic: average length within 1 bit of the entropy, Kraft <= 1
    counts = [40, 20, 20, 10, 10]
    lengths = huffman_code_lengths(counts)
    avg = sum(l * c for l, c in zip(lengths, counts)) / sum(counts)
    H = _entropy(counts)
    assert H <= avg + 1e-12 and avg < H + 1
    assert avg == pytest.approx(2.2)
    assert _kraft(lengths) <= 1.0
    # lone symbol: 1 bit per symbol; empty: nothing
    assert huffman_code_lengths([0, 7, 0]) == [0, 1, 0]
    assert huffman_code_lengths([0, 0]) == [0, 0]


def test_canonical_codes_are_prefix_free():
    lengths = huffman_code_lengths([40, 20, 20, 10, 10, 3, 1])
    codes = canonical_codes(lengths)
    words = [format(code, f"0{l}b") for code, l in codes.values()]
    assert len(set(words)) == len(words)
    for a in words:
        for b in words:
            assert a == b or not b.startswith(a)


def test_coded_bits_uniform_indices_equal_b_per_symbol_plus_table():
    for b in (2, 3, 4):
        levels = 2**b
        p = torch.arange(levels * levels)  # every pair once per repeat, so singles and pairs are both uniform
        idx = torch.stack([p // levels, p % levels], dim=1).reshape(-1).repeat(50).to(torch.int16)
        n = idx.numel()
        c = tensor_coded_bits(idx, levels)
        assert c["huff1"] == b * n + b * levels
        assert c["shannon"] == pytest.approx(b * n)
        # pairs are uniform too: 2b per pair plus the larger table, never better
        assert c["huff2"] == 2 * b * (n // 2) + 2 * b * levels**2
        # on a uniform stream both Huffman tables cost more than they save: the plain b-bit code wins (+ 2 choice bits)
        assert c["plain"] == b * n and c["best"] == c["plain"] + CHOICE_BITS and c["choice"] == "plain"
        assert c["entropy"] == pytest.approx(b)


def test_coded_bits_skewed_distribution_pair_code_within_0p1_of_entropy():
    # discretized Gaussian over 16 levels (what a 4-bit RTN index stream looks like)
    g = torch.Generator().manual_seed(0)
    n = 1_000_000
    z = torch.randn(n, generator=g)
    idx = torch.clamp(torch.round(z / 0.35) + 8, 0, 15).to(torch.int16)
    c = tensor_coded_bits(idx, 16)
    H = c["entropy"]
    assert 2.5 < H < 3.7
    per_symbol_pair = (c["huff2"] - 8 * 256) / n
    assert per_symbol_pair < H + 0.1
    assert c["best"] / n < H + 0.1
    assert c["shannon"] / n == pytest.approx(H if c["choice"] == "huff1" else c["pair_entropy"])  # the chosen coder's bound
    # a very peaked 3-bit stream: the single-symbol code is far from H (p_max ~ 0.6), the pair code recovers most of it
    idx3 = torch.clamp(torch.round(z / 1.5) + 4, 0, 7).to(torch.int16)
    c3 = tensor_coded_bits(idx3, 8)
    assert c3["huff2"] < c3["huff1"] and c3["choice"] == "huff2"
    assert (c3["huff2"] - 6 * 64) / n < c3["entropy"] + 0.1
    assert c3["shannon"] <= c3["huff2"] - 6 * 64
    # odd length: the last element is counted at b bits in the pair code
    c_odd = tensor_coded_bits(idx[:1001], 16)
    c_even = tensor_coded_bits(idx[:1000], 16)
    assert c_odd["n"] == 1001 and c_odd["huff2"] >= c_even["huff2"]


def test_huffman_decode_round_trip():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(5000, generator=g)
    idx = torch.clamp(torch.round(z / 0.35) + 8, 0, 15).to(torch.int64)
    counts = torch.bincount(idx, minlength=16)
    lengths = huffman_code_lengths(counts)
    stream, n_bits = huffman_encode(idx, lengths)
    assert n_bits == int(sum(l * c for l, c in zip(lengths, counts.tolist())))
    assert huffman_decode(stream, n_bits, lengths, idx.numel()) == idx.tolist()
    # the pair-symbol stream decodes back to the index pairs
    pairs = idx[: 2 * (idx.numel() // 2)].reshape(-1, 2)
    sym = pairs[:, 0] * 16 + pairs[:, 1]
    lengths2 = huffman_code_lengths(torch.bincount(sym, minlength=256))
    stream2, n_bits2 = huffman_encode(sym, lengths2)
    back = torch.tensor(huffman_decode(stream2, n_bits2, lengths2, sym.numel()))
    rebuilt = torch.stack([back // 16, back % 16], dim=1).reshape(-1)
    assert torch.equal(rebuilt, pairs.reshape(-1))


def test_tensor_coded_bits_rejects_out_of_range():
    with pytest.raises(ValueError):
        tensor_coded_bits(torch.tensor([0, 16], dtype=torch.int16), 16)


# ---------------------------------------------------------------- BitAccount indices and coded_bits


def _rtn_indices(W: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    N, K = W.shape
    q, _, _ = group_grid(W.float().reshape(N, K // group, group), bits, True)
    return (q.reshape(-1) + 2 ** (bits - 1)).to(torch.int16)


def test_bit_account_indices_and_coded_bits():
    acc = BitAccount()
    acc.add("plain", 1024, 4 * 1024, 16 * 8, "fixed")  # Phase 1 call, unchanged
    assert acc.tensors[0].indices is None and acc.tensors[0].levels is None
    p = torch.arange(256)  # uniform 4-bit stream whose pairs are uniform too, 1024 symbols
    idx = torch.stack([p // 16, p % 16], dim=1).reshape(-1).repeat(2)
    acc.add("coded", 1024, 4 * 1024, 16 * 8, "with indices", indices=idx, levels=16)
    t = acc.tensors[1]
    assert t.indices.dtype == torch.int16 and t.indices.device.type == "cpu" and t.levels == 16
    assert acc.bits_blocks() == 4.125
    s = acc.summary()
    assert s["tensors"][1]["has_indices"] is True and "indices" not in s["tensors"][1]
    out = coded_bits(acc)
    # uniform indices: the plain 4-bit code wins (a Huffman table would only add 64 bits), huff = 4 * 1024 + CHOICE_BITS;
    # shannon = 4 * 1024; the index-less tensor stays fixed
    expected_huff = (4 * 1024 + 128 + 4 * 1024 + CHOICE_BITS + 128) / 2048
    assert out["bits_blocks_huff"] == pytest.approx(expected_huff)
    assert out["bits_blocks_shannon"] == pytest.approx(4.125)
    detail = coded_bits_detail(acc)
    assert detail[0]["coded"] is None and detail[1]["coded"]["choice"] == "plain" and detail[1]["coded"]["huff1"] == 4 * 1024 + 64
    with pytest.raises(ValueError):
        acc.add("bad1", 4, 16, 0, indices=torch.zeros(4, dtype=torch.int16))  # levels missing
    with pytest.raises(ValueError):
        acc.add("bad2", 4, 16, 0, indices=torch.zeros(3, dtype=torch.int16), levels=16)  # wrong count


def test_coded_bits_gaussian_rtn4_saves_bits():
    torch.manual_seed(0)
    W = torch.randn(256, 1024)
    acc = BitAccount()
    acc.add("w", W.numel(), 4 * W.numel(), 16 * W.numel() // 128, "rtn", indices=_rtn_indices(W, 4), levels=16)
    out = coded_bits(acc)
    # H-P1: a Gaussian 4-bit RTN index stream has about 3.6-3.7 bits of entropy
    assert 3.4 < out["bits_blocks_shannon"] - 0.125 < 3.8
    assert out["bits_blocks_shannon"] <= out["bits_blocks_huff"] < 4.125


def test_row_allocated_tensor_is_coded_per_width_class():
    """Method G streams (rows at 3 / 5 bits, levels 2^5): coded per width class the Huffman figure stays at or below
    the fixed sum of row widths and the Shannon figure below the Huffman one; the pooled 32-level coding would exceed
    the fixed figure (the width mixture is already paid for by the 2 bits/row side information)."""
    from wcomp.coding import tensor_coded_bits, tensor_coded_bits_rows

    gen = torch.Generator().manual_seed(0)
    N, K = 64, 1024
    row_bits = [3] * 32 + [5] * 32
    idx = torch.cat([torch.randint(0, 8, (32, K), generator=gen), torch.randint(0, 32, (32, K), generator=gen)]).to(torch.int16)
    fixed = sum(row_bits) * K
    per_class = tensor_coded_bits_rows(idx.reshape(-1), row_bits)
    pooled = tensor_coded_bits(idx.reshape(-1), 32)
    assert per_class["n"] == N * K and per_class["widths"] == {3: 32 * K, 5: 32 * K} and per_class["levels"] == 32
    assert per_class["shannon"] <= per_class["best"] <= fixed + 2 * (3 * 8 + 5 * 32 + 1)  # within the two tables + choice bits
    assert per_class["shannon"] == pytest.approx(fixed, rel=2e-3)  # uniform within each class: entropy = the row width
    assert pooled["best"] > fixed and pooled["shannon"] > fixed  # what the pooled coding would have charged
    acc = BitAccount()
    acc.add("rows", N * K, fixed, N * (K // 128) * 16 + 2 * N, "rows{3,5}", indices=idx.reshape(-1), levels=32, row_bits=row_bits)
    out = coded_bits(acc)
    tables = (3 * 8 + 5 * 32 + 2) / (N * K)  # uniform classes: the Huffman code is the fixed width plus its tables and choice bits
    assert out["bits_blocks_shannon"] <= out["bits_blocks_huff"] <= acc.bits_blocks() + tables + 1e-9
    with pytest.raises(ValueError):
        tensor_coded_bits_rows(idx.reshape(-1)[:-1], row_bits)


def test_shannon_figure_is_the_better_of_the_two_coder_bounds():
    """A stream whose symbols alternate deterministically has H1 = 1 bit but H2 / 2 ~ 0: the pair code codes it in
    ~0 bits per symbol and the Shannon figure follows the chosen (pair) bound, so it never exceeds the coder's
    table-free part; on an i.i.d. stream the single-symbol code wins and the figure is H1."""
    from wcomp.coding import tensor_coded_bits

    idx = torch.tensor([0, 3] * 2048, dtype=torch.int16)
    c = tensor_coded_bits(idx, 4)
    assert c["entropy"] == pytest.approx(1.0) and c["pair_entropy"] == pytest.approx(0.0)
    assert c["choice"] == "huff2" and c["shannon"] == pytest.approx(0.0)
    assert c["shannon"] <= c["best"] - 1 - 2 * 2 * 16  # the pair table and choice bit are the only cost
    iid = torch.randint(0, 4, (4096,), generator=torch.Generator().manual_seed(0)).to(torch.int16)
    c = tensor_coded_bits(iid, 4)
    # (near-uniform: the plain code or the single-symbol Huffman code wins, both bound by the order-0 entropy)
    assert c["choice"] in ("huff1", "plain") and c["shannon"] == pytest.approx(c["entropy"] * 4096) and c["shannon"] <= c["best"]


# ---------------------------------------------------------------- rotations: hadsign, rnd, eig


def test_random_signs_and_hadsign_orthonormal():
    s = random_signs(64, 0)
    assert set(s.tolist()) <= {-1.0, 1.0} and torch.equal(s, random_signs(64, 0))
    assert not torch.equal(s, random_signs(64, 1))
    R = hadsign(1024, 0)
    assert torch.allclose(R, hadamard(1024) * random_signs(1024, 0)[None, :])
    err = (R.double() @ R.double().T - torch.eye(1024, dtype=torch.float64)).abs().max().item()
    assert err < 1e-5
    with pytest.raises(ValueError):
        hadsign(36, 0)


@pytest.mark.parametrize("kind", ["hadsign", "rnd"])
@pytest.mark.parametrize("K", [1024, 36])
def test_hadsign_and_rnd_rotations_are_exact(kind, K):
    torch.manual_seed(0)
    lin = nn.Linear(K, 64, bias=True)
    x = torch.randn(3, 5, K)
    with torch.no_grad():
        y0 = lin(x)
        W0 = lin.weight.clone()
        Rs = rotate_inputs(None, [("lin", lin)], seed=3, kind=kind)
        y1 = lin(x)
    R = Rs["lin"]
    assert R.shape == (K, K) and R.dtype == torch.float32
    assert torch.allclose(lin.weight, W0 @ R, atol=1e-5)
    assert ((y1 - y0).norm() / y0.norm()).item() < 1e-4
    if kind == "rnd" or K == 36:
        assert torch.allclose(R, random_orthogonal(K, 3))
    else:
        assert torch.allclose(R, hadsign(K, 3))
    assert rotation_bits(Rs, [("lin", lin)], kind) == {"lin": 0}


def test_rotate_inputs_default_kind_is_the_phase1_hadamard():
    torch.manual_seed(0)
    lin_a, lin_b = nn.Linear(1024, 8, bias=False), nn.Linear(1024, 8, bias=False)
    lin_b.load_state_dict(lin_a.state_dict())
    with torch.no_grad():
        Ra = rotate_inputs(None, [("lin", lin_a)], seed=0)
        Rb = rotate_inputs(None, [("lin", lin_b)], seed=0, kind="hadamard")
    assert torch.equal(Ra["lin"], Rb["lin"]) and torch.equal(lin_a.weight, lin_b.weight)
    with pytest.raises(ValueError):
        rotate_inputs(None, [("lin", lin_a)], kind="learned")


def _correlated_inputs(K: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(K, K, generator=g) / math.sqrt(K)
    A[:, 0] *= 5  # one strong direction, so the Hessian is far from a multiple of I
    return torch.randn(n, K, generator=g) @ A


def test_eig_rotation_diagonalizes_the_hessian_and_preserves_the_output():
    K, N, n = 96, 32, 8192
    x = _correlated_inputs(K, n)
    H = (x.double().T @ x.double()) / n
    off_before = (H - torch.diag(torch.diag(H))).norm() / torch.diag(H).norm()
    assert off_before > 0.1
    R = eig_rotation(H)
    assert R.dtype == torch.float32 and R.shape == (K, K)
    assert (R.double() @ R.double().T - torch.eye(K, dtype=torch.float64)).abs().max() < 1e-5
    assert torch.equal(R, eig_rotation(H))  # deterministic sign convention
    torch.manual_seed(0)
    lin = nn.Linear(K, N, bias=False)
    with torch.no_grad():
        y0 = lin(x)
        Rs = rotate_inputs(None, [("lin", lin)], kind="eig", hessians={"lin": H})
        y1 = lin(x)
    assert torch.allclose(Rs["lin"], R)
    assert ((y1 - y0).norm() / y0.norm()).item() < 1e-4
    xr = x @ Rs["lin"]  # what the hook feeds the rotated linear
    Hr = (xr.double().T @ xr.double()) / n
    off_after = (Hr - torch.diag(torch.diag(Hr))).norm() / torch.diag(Hr).norm()
    print(f"eig rotation: relative off-diagonal Frobenius before {off_before:.3f} after {off_after:.2e}")
    assert off_after < 1e-3
    # the eigenvalues come out in descending order on the diagonal
    d = torch.diag(Hr)
    assert torch.all(d[:-1] >= d[1:] - 1e-6)
    # bits: the stored bf16 matrix, 16 K K in total, 16 K / N per parameter for a lone linear
    bits = rotation_bits(Rs, [("lin", lin)], "eig")
    assert bits == {"lin": 16 * K * K}
    assert bits["lin"] / (N * K) == pytest.approx(16 * K / N)


def test_eig_rotation_is_shared_within_an_input_group_and_charged_once():
    K = 64
    lins = [("model.layers.0.mlp.gate_proj", nn.Linear(K, 48, bias=False)),
            ("model.layers.0.mlp.up_proj", nn.Linear(K, 16, bias=False)),
            ("model.layers.0.mlp.down_proj", nn.Linear(K, 8, bias=False)),
            ("model.layers.1.mlp.gate_proj", nn.Linear(K, 8, bias=False))]
    assert shared_input_groups(lins) == [["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
                                         ["model.layers.0.mlp.down_proj"], ["model.layers.1.mlp.gate_proj"]]
    x = _correlated_inputs(K, 4096)
    H = (x.double().T @ x.double()) / 4096
    H2 = (x.double().T @ x.double()) / 4096 + 0.3 * torch.eye(K, dtype=torch.float64)
    hess = {n: (H if "layers.0" in n else H2) for n, _ in lins}
    Rs = rotate_inputs(None, lins, kind="eig", hessians=hess)
    assert Rs["model.layers.0.mlp.gate_proj"] is Rs["model.layers.0.mlp.up_proj"]
    assert Rs["model.layers.0.mlp.down_proj"] is not Rs["model.layers.0.mlp.gate_proj"]
    bits = rotation_bits(Rs, lins, "eig")
    total = 16 * K * K
    assert bits["model.layers.0.mlp.gate_proj"] + bits["model.layers.0.mlp.up_proj"] == total
    assert bits["model.layers.0.mlp.gate_proj"] == total * 48 // 64  # by numel: 48 K vs 16 K
    assert bits["model.layers.0.mlp.down_proj"] == total and bits["model.layers.1.mlp.gate_proj"] == total
    with pytest.raises(ValueError):
        rotate_inputs(None, lins, kind="eig")  # neither calib_ids nor hessians


# ---------------------------------------------------------------- collect_input_hessians on a stub decoder


class _Block(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.up_proj = nn.Linear(hidden, inter, bias=False)
        self.mlp.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return x + self.mlp.down_proj(torch.nn.functional.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))


class _Inner(nn.Module):
    def __init__(self, hidden: int, inter: int, n_layers: int, vocab: int = 50):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_Block(hidden, inter) for _ in range(n_layers)])

    def forward(self, input_ids, use_cache=False):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class _StubLM(nn.Module):
    def __init__(self, hidden: int = 64, inter: int = 96, n_layers: int = 3):
        super().__init__()
        self.model = _Inner(hidden, inter, n_layers)

    def forward(self, input_ids):
        return self.model(input_ids)


def test_collect_input_hessians_and_eig_rotation_on_a_stub_lm():
    from wcomp.models import block_linears

    torch.manual_seed(0)
    model = _StubLM()
    ids = torch.randint(0, 50, (4, 32))
    lins = block_linears(model)
    assert len(lins) == 9
    H = collect_input_hessians(model, lins, ids)
    assert H["model.layers.0.mlp.gate_proj"] is H["model.layers.0.mlp.up_proj"]
    assert H["model.layers.1.mlp.down_proj"].shape == (96, 96) and H["model.layers.1.mlp.down_proj"].dtype == torch.float64
    # the layer-0 gate/up input is the embedding: check H against the direct computation
    x = model.model.embed(ids).reshape(-1, 64).double()
    assert torch.allclose(H["model.layers.0.mlp.gate_proj"], x.T @ x / x.shape[0], rtol=1e-4, atol=1e-6)
    # chunked (layer_range) collection matches the all-at-once one for the layers inside the chunk
    H01 = collect_input_hessians(model, lins, ids, layer_range=(0, 2))
    assert set(H01) == {n for n, _ in lins if ".layers.2." not in n}
    assert torch.allclose(H01["model.layers.1.mlp.down_proj"], H["model.layers.1.mlp.down_proj"])
    with torch.no_grad():
        y0 = model(ids)
        Rs = rotate_inputs(model, kind="eig", calib_ids=ids, layer_chunk=2)
        y1 = model(ids)
    assert set(Rs) == {n for n, _ in lins}
    assert ((y1 - y0).norm() / y0.norm()).item() < 1e-4
    bits = rotation_bits(Rs, lins, "eig")
    assert sum(bits.values()) == 3 * (16 * 64 * 64 + 16 * 96 * 96)  # per layer one 64x64 (gate/up) and one 96x96 (down)


# ---------------------------------------------------------------- methods: names, needs_calib, notes


def test_phase3_rotation_method_names_resolve():
    from wcomp.methods import METHODS, family, needs_calib, resolve

    names = [f"gptq32-sym-g128-{b}" for b in (4, 3, 2)] + [f"had-gptq32-sym-g128-{b}" for b in (4, 3, 2)]
    names += ["rnd-gptq32-sym-g128-3", "hadsign-gptq32-sym-g128-3", "eig-gptq32-sym-g128-3"]
    names += [f"lloyd-gptq32-g128-{b}" for b in (4, 3, 2)] + ["nf-gptq32-g128-4", "nf-gptq32-g128-3"]
    for n in names:
        assert resolve(n) is METHODS[n]
        assert needs_calib(n)
    assert family("had-gptq32-sym-g128-3") == ("had-gptq32-sym-g128", 3)
    assert family("eig-gptq32-sym-g128-3") == ("eig-gptq32-sym-g128", 3)
    assert needs_calib("eig-rtn-sym-g128-3") and not needs_calib("had-rtn-sym-g128-3")
    # Phase 1 names are untouched objects
    assert resolve("gptq-sym-g128-3") is METHODS["gptq-sym-g128-3"]
    assert resolve("had-gptq-sym-g128-3") is METHODS["had-gptq-sym-g128-3"]
    assert "hadamard" in ROTATION_KINDS


def test_rotated_method_notes_and_bits_on_a_stub():
    from wcomp.methods import GROUP, Ctx, rotated, rtn

    model = _StubLM(hidden=1024, inter=256)
    acct = rotated("hadsign", rtn(4, True))(model, Ctx(None, 5))
    assert acct.bits_blocks() == 4.125
    assert acct.summary()["notes"] == ["rtn-sym-g128-4: f16 scale; rotation=hadsign; input rotation hadsign(256,1024, seed=5)"]
    model = _StubLM(hidden=1024, inter=256)
    acct = rotated("rnd", rtn(4, True))(model, Ctx(None, 5))
    assert acct.summary()["notes"] == ["rtn-sym-g128-4: f16 scale; rotation=rnd; input rotation random_orthogonal(256,1024, seed=5)"]
    model = _StubLM(hidden=128, inter=256)
    ids = torch.randint(0, 50, (2, 32))
    acct = rotated("eig", rtn(4, True))(model, Ctx(ids, 0))
    stored = 3 * (16 * 128 * 128 + 16 * 256 * 256)
    assert acct.total_bits() == 4.125 * acct.numel() + stored
    assert all("input rotation eig(128,256)" in t.note and "6 bf16 matrices stored" in t.note for t in acct.tensors)
    with pytest.raises(ValueError):
        rotated("eig", rtn(4, True))(_StubLM(), Ctx(None, 0))
    assert GROUP == 128


# ---------------------------------------------------------------- RD bound


def test_normalized_sq_error_and_rd_bound_bits():
    W = torch.randn(4, 256)
    err, n = normalized_sq_error(W, W, 128)
    assert err == 0 and n == 1024
    # an error of exactly one group RMS on every weight gives D = 1 -> bound 0 bits
    rms = W.reshape(4, 2, 128).square().mean(-1, keepdim=True).sqrt().expand(4, 2, 128).reshape(4, 256)
    err, n = normalized_sq_error(W, W + rms, 128)
    assert err / n == pytest.approx(1.0, rel=1e-6)
    assert rd_bound_bits(1.0) == 0 and rd_bound_bits(0.25) == 1.0 and rd_bound_bits(0.0) == rd_bound_bits(1e-12)
    # an all-zero group contributes nothing instead of NaN
    Wz = W.clone()
    Wz[:, :128] = 0
    err, _ = normalized_sq_error(Wz, Wz + 1.0, 128)
    assert math.isfinite(err)


def test_rd_bound_rows_on_a_stub_with_reference():
    from wcomp.models import block_linears

    torch.manual_seed(0)
    model = _StubLM(hidden=1024, inter=256, n_layers=2)
    ref = copy.deepcopy(model)
    lins = block_linears(model)
    acct = BitAccount()
    with torch.no_grad():
        for name, lin in lins:
            W = lin.weight.data.clone()
            idx = _rtn_indices(W, 4)
            W_hat, _ = quantize_rtn(W, 4, 128, True)
            lin.weight.data.copy_(W_hat)
            acct.add(name, W.numel(), 4 * W.numel(), 16 * W.numel() // 128, "rtn", indices=idx, levels=16)
    rows = rd_bound_rows(model, acct, ref=ref)
    assert {r["type"] for r in rows} == {"gate", "up", "down"}
    for r in rows:
        assert tuple(r) == RD_FIELDS
        assert r["bits_fixed"] == pytest.approx(4.125)
        assert r["bits_shannon"] <= r["bits_huff"] < r["bits_fixed"]
        bound = 0.5 * math.log2(1 / r["D"])
        assert r["gap_fixed"] == pytest.approx(4.125 - bound)
        assert r["gap_huff"] == pytest.approx(r["bits_huff"] - bound)
        assert r["gap_shannon"] == pytest.approx(r["bits_shannon"] - bound)
        # nn.Linear's uniform init, 4-bit RTN g128: D ~ 0.005 (bound ~3.9 bits), the fixed-rate code sits ~0.3 above
        assert 0.003 < r["D"] < 0.02 and 0.2 < r["gap_fixed"] < 1.2
    assert sum(r["numel"] for r in rows) == acct.numel()
    # the runner passes its runs.jsonl row: its identifying fields prefix every rd_bound row
    tagged = rd_bound_rows(model, acct, ref=ref, row={"model": "stub", "method": "rtn-sym-g128-4", "seed": 0, "ppl": 1.0})
    assert tuple(tagged[0])[:3] == ("model", "method", "seed") and "ppl" not in tagged[0]
    assert tuple(tagged[0])[3:] == RD_FIELDS
    # without a reference the original weights would have to come from disk: the stub has no files
    with pytest.raises((FileNotFoundError, AttributeError)):
        rd_bound_rows(model, acct)


@pytest.mark.parametrize("kind", ["hadamard", "rnd"])
def test_rd_bound_rows_compare_in_the_rotated_basis(kind):
    """A rotated model's W_hat lives in the rotated input basis: rd_bound folds the reference weight the same way
    (the module's wcomp_input_fold), so an unquantized rotated model has D at the bf16 rounding floor, not O(1)."""
    from wcomp.models import block_linears
    from wcomp.quant.rotate import rotate_inputs

    torch.manual_seed(0)
    model = _StubLM(hidden=1024, inter=256, n_layers=1).to(torch.bfloat16)
    ref = copy.deepcopy(model)
    rotate_inputs(model, seed=3, kind=kind)
    acct = BitAccount()
    for name, lin in block_linears(model):
        assert getattr(lin, "wcomp_input_fold", None) is not None
        acct.add(name, lin.weight.numel(), 16 * lin.weight.numel(), 0, "rotated, not quantized")
    rows = rd_bound_rows(model, acct, ref=ref)
    assert all(r["D"] < 1e-4 for r in rows), [(r["type"], r["D"]) for r in rows]  # bf16 rounding of the fold only
    for lin in dict(block_linears(model)).values():
        del lin.wcomp_input_fold
    unfolded = rd_bound_rows(model, acct, ref=ref)
    assert all(r["D"] > 0.5 for r in unfolded)  # what the comparison across bases would have said


def test_plain_code_fallback_bounds_huff_by_fixed_plus_the_choice_bits():
    """The smoke charged small 8-bit tensors (16384 params: the 8 * 256 = 2048-bit table is 0.125 b/param) and near-uniform
    3-bit tensors above their fixed width because the two-way choice had no table-free option. With the plain b-bit code as
    the third option, huff <= fixed + CHOICE_BITS per tensor; the Shannon figure (order-0 entropy, as for huff1) stays below."""
    gen = torch.Generator().manual_seed(0)
    for b, n in ((8, 16384), (3, 16384), (4, 4096)):
        idx = torch.randint(0, 2**b, (n,), generator=gen).to(torch.int16)
        c = tensor_coded_bits(idx, 2**b)
        assert c["best"] <= n * b + CHOICE_BITS
        assert c["shannon"] <= c["best"] and c["shannon"] <= n * b
        if c["choice"] == "plain":
            assert c["shannon"] == pytest.approx(c["entropy"] * n)
    # account level: a 36-tensor 8-bit type of 16384 params each, uniform indices
    acc = BitAccount()
    for i in range(36):
        idx = torch.randint(0, 256, (16384,), generator=gen).to(torch.int16)
        acc.add(f"t{i}", 16384, 8 * 16384, 16 * 128, "8b", indices=idx, levels=256)
    out = coded_bits(acc)
    assert out["bits_blocks_huff"] <= acc.bits_blocks() + CHOICE_BITS / 16384 + 1e-12
    assert out["bits_blocks_shannon"] <= out["bits_blocks_huff"]
    # a skewed stream still takes a Huffman code and the figure is unchanged apart from the extra choice bit
    z = torch.randn(100_000, generator=gen)
    idx = torch.clamp(torch.round(z / 0.35) + 8, 0, 15).to(torch.int16)
    c = tensor_coded_bits(idx, 16)
    assert c["choice"] in ("huff1", "huff2") and c["best"] == min(c["huff1"], c["huff2"]) + CHOICE_BITS < c["plain"]
