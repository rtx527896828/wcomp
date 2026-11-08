"""Round-to-nearest group quantization (DESIGN.md, wcomp/quant/rtn.py contract)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from wcomp.quant.codebook import gaussian_lloyd_codebook, gaussian_quantile_codebook, lloyd_max, midpoints

SCALE_BITS = 16  # scales are stored as f16
# Phase 3 method E2 / E3 (DESIGN.md): "lloyd" = per-tensor Lloyd-Max fit (2^b * 16 table bits); the fixed codebooks
# (0 bits): "nf" = the Lloyd-Max optimum of N(0, 1) (wcomp.quant.codebook.gaussian_lloyd_codebook), "nfq" = the
# equiprobable Gaussian quantiles (gaussian_quantile_codebook; clips the tails under RMS normalization, see codebook.py).
CODEBOOK_KINDS = ("lloyd", "nf", "nfq")


def group_grid(Wg: Tensor, bits: int, sym: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize Wg [..., g] along its last dim. Returns (q, scale, zero), scale/zero shaped [..., 1].

    sym:  scale = amax / 2^(b-1), q in [-2^(b-1), 2^(b-1)-1], zero = 0 (the literal DESIGN.md grid). amax is
          unsigned, so in a group whose largest-magnitude entry is positive (about half of them) that entry is
          clipped to (2^(b-1)-1)/2^(b-1) * amax; at 2 bits the grid is {-amax, -amax/2, 0, +amax/2}. A sign-aware
          scale (negate the group so the extreme lands on qmin) would cost the same bits and about 4% less MSE;
          it is deliberately not done here so the numbers stay on the contract grid (to be stated in NOTES.md).
    asym: scale = (max - min) / (2^b - 1), zero = round(-min / scale) as a b-bit integer, q in [0, 2^b-1],
          with the range [min, max] first extended to contain 0 (as in the GPTQ quantizer) so that the
          integer zero point is representable; otherwise an all-positive group would be clipped. This is a
          deviation from the contract's literal "min/max", to be stated in NOTES.md (Phase 1).
    The scale is rounded to f16 before use so the reconstruction is realizable from the stored side info.
    The zero-scale guard (scale 1, so the group dequantizes to 0) is applied *after* that rounding: a group
    whose f32 scale is below the smallest f16 subnormal (~6e-8) would otherwise divide by 0 (NaN for exact
    zeros in the group); such groups exist in the real 0.8B after the Hadamard rotation.
    This is also the grid of wcomp.quant.gptq (same bits, same side information).
    """
    if sym:
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        amax = Wg.abs().amax(dim=-1, keepdim=True)
        scale = (amax / float(-qmin)).to(torch.float16).to(Wg.dtype)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.zeros_like(scale)
    else:
        qmin, qmax = 0, 2**bits - 1
        wmin = Wg.amin(dim=-1, keepdim=True).clamp(max=0)
        wmax = Wg.amax(dim=-1, keepdim=True).clamp(min=0)
        scale = ((wmax - wmin) / float(qmax)).to(torch.float16).to(Wg.dtype)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.clamp(torch.round(-wmin / scale), qmin, qmax)
    q = torch.clamp(torch.round(Wg / scale) + zero, qmin, qmax)
    return q, scale, zero


_group_grid = group_grid  # old private name


def quantize_rtn(W: Tensor, bits: int, group: int = 128, sym: bool = True, row_bits: Tensor | None = None,
                 codebook: dict | None = None) -> tuple[Tensor, dict]:
    """RTN-quantize W [N, K] (f32) with groups of `group` consecutive inputs along K.

    group <= 0 means one group per output row. Returns the dequantized f32 weight and
    info = {"index_bits", "aux_bits", "note", "indices", "levels", "side"}; aux counts an f16 scale per group plus a
    b-bit zero if asym. "indices" (Phase 3) are the grid indices q - qmin as an int16 CPU tensor [N*K] (row-major) and
    "levels" = 2^bits; the reconstruction is (indices + qmin - zero) * scale. "side" is the side information
    (scales, zeros, qmin, codebook, group) that dequantize_indices(indices, side) turns back into W_hat; every path
    asserts that this round trip is exact (torch.equal in f32).
    `row_bits` (Phase 3, method G): an int tensor [N] giving each output row its own width; row i is quantized on
    the `row_bits[i]`-bit grid of group_grid (same scale rule, so uniform row_bits reproduce `bits` exactly), `bits`
    is then ignored for the grid, index_bits = K * sum(row_bits), levels = 2^max(row_bits) and the indices of row i
    are q - qmin_i in 0..2^row_bits[i]-1. The per-row width side information (ROW_SIDE_BITS per row) is charged by
    the caller (apply_rtn / gptq_model), not here.
    `codebook` (Phase 3, method E2 / E3): {"kind": "lloyd" | "nf" | "nfq", "bits": b (optional, must equal `bits`),
    "iters": 50} replaces the uniform grid (`sym` is then irrelevant): each group is normalized by its RMS (f16
    scale, a zero RMS stores scale 0 and the group dequantizes to 0) and every weight goes to the nearest level of a
    per-tensor codebook with 2^bits entries, fitted by Lloyd-Max on the tensor's normalized weights (f16-rounded,
    2^bits * 16 bits of table) or a fixed Gaussian codebook (0 bits; "nf" the Lloyd-Max optimum of N(0, 1), "nfq" the
    equiprobable quantiles, see CODEBOOK_KINDS); with `row_bits` one codebook per distinct width, fitted on the rows
    of that width.
    """
    if W.dim() != 2:
        raise ValueError(f"expected [N, K], got {tuple(W.shape)}")
    N, K = W.shape
    g = K if group is None or group <= 0 else group
    if K % g != 0:
        raise ValueError(f"K={K} not divisible by group={g}")
    W = W.float()
    if codebook is not None:
        return _quantize_rtn_codebook(W, g, bits, codebook, row_bits)
    if row_bits is not None:
        return _quantize_rtn_rows(W, g, sym, row_bits)
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in 1..16, got {bits}")
    q, scale, zero = group_grid(W.reshape(N, K // g, g), bits, sym)
    W_hat = ((q - zero) * scale).reshape(N, K)
    n_groups = N * (K // g)
    qmin = -(2 ** (bits - 1)) if sym else 0
    side = {"scales": scale.reshape(N, K // g), "zeros": None if sym else zero.reshape(N, K // g), "qmin": qmin,
            "codebook": None, "group": g, "perm": None}
    info = {
        "index_bits": bits * N * K,
        "aux_bits": n_groups * (SCALE_BITS + (0 if sym else bits)),
        "note": f"rtn-{'sym' if sym else 'asym'}-g{g}-{bits}: f16 scale" + ("" if sym else f", {bits}-bit zero"),
        "indices": (q - qmin).reshape(-1).to("cpu", torch.int16),
        "levels": 2**bits,
        "side": side,
    }
    if bits > 15:
        # 2^16 levels do not fit the int16 index stream (and the 16-bit grid has nothing to entropy-code): no index record
        info["indices"] = info["levels"] = info["side"] = None
        return W_hat, info
    check_roundtrip(W_hat, info)
    return W_hat, info


ROW_SIDE_BITS = 2  # Phase 3 method G: the stored width of one row's bit-width choice (3 levels b-1 / b / b+1)


def check_row_bits(row_bits: Tensor, N: int) -> Tensor:
    """row_bits as a CPU int64 tensor [N] with every width in 1..16."""
    rb = row_bits.detach().to("cpu", torch.int64).reshape(-1)
    if rb.shape != (N,):
        raise ValueError(f"row_bits must have one entry per output row ({N}), got {tuple(row_bits.shape)}")
    if not bool(((rb >= 1) & (rb <= 16)).all()):
        raise ValueError(f"row_bits must be in 1..16, got {int(rb.min())}..{int(rb.max())}")
    return rb


def row_bits_note(row_bits: Tensor) -> str:
    """"rows{3,4,5}" for the distinct widths of a row_bits tensor (a stable spelling for notes / names)."""
    return "rows{" + ",".join(str(int(b)) for b in torch.unique(row_bits.to("cpu").long())) + "}"


def row_qrange(row_bits: Tensor, sym: bool, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    """(qmin, qmax) per row, float tensors [N] on row_bits' device: sym [-2^(b-1), 2^(b-1)-1], asym [0, 2^b-1]."""
    rb = row_bits.to(dtype)
    if sym:
        half = torch.pow(2.0, rb - 1)
        return -half, half - 1
    return torch.zeros_like(rb), torch.pow(2.0, rb) - 1


def group_grid_rows(Wg: Tensor, row_bits: Tensor, sym: bool) -> tuple[Tensor, Tensor, Tensor]:
    """group_grid with a per-row width: Wg [N, ..., g] (groups along the last dim, rows along the first), row_bits
    [N] ints. Same rule and operation order as group_grid with the row's own qmin / qmax, so uniform row_bits
    reproduce group_grid(Wg, b, sym) exactly; vectorized over the rows (no per-width loop). Returns (q, scale, zero)
    with scale / zero shaped [N, ..., 1]."""
    N = Wg.shape[0]
    shape = (N,) + (1,) * (Wg.dim() - 1)
    qmin, qmax = row_qrange(row_bits.to(Wg.device), sym, Wg.dtype)
    qmin, qmax = qmin.view(shape), qmax.view(shape)
    if sym:
        amax = Wg.abs().amax(dim=-1, keepdim=True)
        scale = (amax / -qmin).to(torch.float16).to(Wg.dtype)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.zeros_like(scale)
    else:
        wmin = Wg.amin(dim=-1, keepdim=True).clamp(max=0)
        wmax = Wg.amax(dim=-1, keepdim=True).clamp(min=0)
        scale = ((wmax - wmin) / qmax).to(torch.float16).to(Wg.dtype)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.clamp(torch.round(-wmin / scale), qmin, qmax)
    q = torch.clamp(torch.round(Wg / scale) + zero, qmin, qmax)
    return q, scale, zero


def _quantize_rtn_rows(W: Tensor, g: int, sym: bool, row_bits: Tensor) -> tuple[Tensor, dict]:
    """quantize_rtn with a per-row width (group_grid_rows, vectorized over the rows)."""
    N, K = W.shape
    G = K // g
    rb = check_row_bits(row_bits, N).to(W.device)
    q, scale, zero = group_grid_rows(W.reshape(N, G, g), rb, sym)
    W_hat = ((q - zero) * scale).reshape(N, K)
    qmin = row_qrange(rb, sym)[0]
    idx = (q - qmin.view(N, 1, 1)).reshape(N, K)
    zero_bits = 0 if sym else int(rb.sum()) * G
    side = {"scales": scale.reshape(N, G), "zeros": None if sym else zero.reshape(N, G), "qmin": qmin, "codebook": None,
            "group": g, "perm": None}
    info = {
        "index_bits": int(rb.sum()) * K,
        "aux_bits": N * G * SCALE_BITS + zero_bits,
        "note": f"rtn-{'sym' if sym else 'asym'}-g{g}-{row_bits_note(rb)}: f16 scale" + ("" if sym else ", per-row-width zero"),
        "indices": idx.reshape(-1).to("cpu", torch.int16),
        "levels": 2 ** int(rb.max()),
        "side": side,
    }
    check_roundtrip(W_hat, info)
    return W_hat, info


# ---------------------------------------------------------------- Phase 3 (DESIGN.md E2 / E3): codebooks, index round trip


@dataclass
class Codebook:
    """Per-row codebook table of one tensor: `table` [N, levels] f32 (f16-valued) and the decision thresholds `mids`
    [N, levels-1] (+inf past a row's own 2^b_i levels, so a row never selects a padded entry). Rows of the same width
    share one codebook ("lloyd": fitted; "nf": the fixed Lloyd-Max optimum of N(0, 1); "nfq": the fixed Gaussian
    quantiles, DESIGN.md E3 as written). `table_bits` is the stored side information (lloyd: 16 bits per entry of every
    distinct codebook; nf / nfq: 0)."""

    kind: str
    table: Tensor
    mids: Tensor
    levels: int
    table_bits: int
    widths: tuple[int, ...]

    def quantize(self, xn: Tensor) -> Tensor:
        """Nearest-level indices (int64, [N, M]) of the normalized weights xn [N, M] (ties go to the lower level)."""
        return torch.searchsorted(self.mids, xn.contiguous())

    def dequant(self, idx: Tensor) -> Tensor:
        """Codebook values [N, M] of the indices idx [N, M] (normalized, before the group scale)."""
        return self.table.gather(1, idx)

    def quantize_column(self, w: Tensor, scale: Tensor, safe_scale: Tensor) -> tuple[Tensor, Tensor]:
        """One column w [N] with its group scales [N]: (dequantized [N], indices int64 [N]). `safe_scale` is `scale`
        with zeros replaced by 1 (a zero-RMS group dequantizes to 0 through the stored scale 0)."""
        idx = self.quantize((w / safe_scale).unsqueeze(1))
        return self.dequant(idx)[:, 0] * scale, idx[:, 0]


def codebook_spec(codebook: dict, bits: int) -> tuple[str, int]:
    """(kind, iters) of a codebook spec {"kind", "bits"?, "iters"?}; a "bits" entry must equal the tensor's width."""
    kind = codebook.get("kind")
    if kind not in CODEBOOK_KINDS:
        raise ValueError(f"codebook kind must be one of {CODEBOOK_KINDS}, got {kind!r}")
    if codebook.get("bits") is not None and int(codebook["bits"]) != int(bits):
        raise ValueError(f"codebook bits {codebook['bits']} differ from the tensor's width {bits}")
    return kind, int(codebook.get("iters", 50))


def rms_scale(Wg: Tensor) -> Tensor:
    """f16-rounded RMS of each group, Wg [..., g] -> [..., 1] (no zero guard: an all-zero group keeps scale 0)."""
    return Wg.square().mean(dim=-1, keepdim=True).sqrt().to(torch.float16).to(Wg.dtype)


def safe_scale(scale: Tensor) -> Tensor:
    """`scale` with zeros replaced by 1, the divisor for normalizing a group (its stored scale stays 0)."""
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def fit_codebook(W: Tensor, g: int, codebook: dict, bits: int, row_bits: Tensor | None = None) -> Codebook:
    """The Codebook of W [N, K] f32 for groups of g columns: per distinct width b (one, or the widths of row_bits)
    the 2^b-level codebook on the group-normalized weights w / rms_g of the rows of that width, Lloyd-Max from the
    Gaussian-quantile initialisation ("lloyd", f16-rounded, 2^b * 16 bits) or a fixed Gaussian codebook (0 bits,
    f16-rounded too): "nf" the Lloyd-Max optimum of N(0, 1), "nfq" the equiprobable quantiles. Fitted on W as given
    (the original weights, before any GPTQ update)."""
    N, K = W.shape
    kind, iters = codebook_spec(codebook, bits)
    rb = torch.full((N,), int(bits), dtype=torch.int64) if row_bits is None else check_row_bits(row_bits, N)
    rb = rb.to(W.device)
    widths = tuple(int(b) for b in torch.unique(rb).tolist())
    if not (1 <= min(widths) and max(widths) <= 15):
        raise ValueError(f"codebook widths must be in 1..15 (int16 indices), got {widths}")
    levels = 2 ** max(widths)
    table = torch.zeros(N, levels, dtype=torch.float32, device=W.device)
    mids = torch.full((N, levels - 1), float("inf"), dtype=torch.float32, device=W.device)
    xn = None
    if kind == "lloyd":
        Wg = W.reshape(N, K // g, g)
        xn = (Wg / safe_scale(rms_scale(Wg))).reshape(N, K)
    table_bits = 0
    for b in widths:
        rows = rb == b
        L = 2**b
        if kind == "lloyd":
            cb = lloyd_max(xn[rows], L, iters=iters, init="gaussian")
            table_bits += L * 16
        elif kind == "nf":
            cb = gaussian_lloyd_codebook(L, W.device)
        else:
            cb = gaussian_quantile_codebook(L, W.device)
        cb = cb.to(torch.float16).to(torch.float32)  # the stored (f16) codebook is what the decoder has
        table[rows, :L] = cb
        mids[rows, : L - 1] = midpoints(cb)
    return Codebook(kind, table, mids, levels, table_bits, widths)


def codebook_note(cb: Codebook) -> str:
    if cb.kind == "lloyd":
        return f"f16 rms scale, {'+'.join(f'2^{b}' for b in cb.widths)} x f16 codebook"
    return "f16 rms scale, fixed " + ("gaussian lloyd-max codebook (nf)" if cb.kind == "nf" else "gaussian quantile codebook (nfq)")


def _quantize_rtn_codebook(W: Tensor, g: int, bits: int, codebook: dict, row_bits: Tensor | None) -> tuple[Tensor, dict]:
    """quantize_rtn on a codebook: RMS-normalized groups, nearest codebook level (fit_codebook)."""
    N, K = W.shape
    G = K // g
    rb = None if row_bits is None else check_row_bits(row_bits, N).to(W.device)
    if rb is None and not 1 <= bits <= 15:
        raise ValueError(f"bits must be in 1..15 for a codebook, got {bits}")
    cb = fit_codebook(W, g, codebook, bits, rb)
    Wg = W.reshape(N, G, g)
    scale = rms_scale(Wg)
    idx = cb.quantize((Wg / safe_scale(scale)).reshape(N, K))
    W_hat = (cb.dequant(idx).reshape(N, G, g) * scale).reshape(N, K)
    width = f"{bits}" if rb is None else row_bits_note(rb)
    side = {"scales": scale.reshape(N, G), "zeros": None, "qmin": None, "codebook": cb, "group": g, "perm": None}
    info = {
        "index_bits": (bits * N if rb is None else int(rb.sum())) * K,
        "aux_bits": N * G * SCALE_BITS + cb.table_bits,
        "note": f"rtn-{cb.kind}-g{g}-{width}: {codebook_note(cb)}",
        "indices": idx.reshape(-1).to("cpu", torch.int16),
        "levels": cb.levels,
        "side": side,
    }
    check_roundtrip(W_hat, info)
    return W_hat, info


def expand_groups(x: Tensor, g: int) -> Tensor:
    """Per-group values x [N, G] repeated over the g columns of each group -> [N, G*g]."""
    N, G = x.shape
    return x.unsqueeze(2).expand(N, G, g).reshape(N, G * g)


def grid_indices(Q: Tensor, scales: Tensor, zeros: Tensor | None, qmin, g: int) -> Tensor:
    """Recover the grid indices q - qmin (int16 [N, K]) from a uniform-grid reconstruction Q = (q - zero) * scale with
    per-group scales [N, G] and zeros [N, G] (None for sym). Exact: (q - zero) is an integer of at most 2^b and the
    scale has 11 significant bits, so the f32 product is exact for b <= 13 and Q / scale is that integer; round()
    absorbs the rounding of wider grids. `qmin` is a Python int or a per-row tensor [N]."""
    N, K = Q.shape
    q = torch.round(Q / expand_groups(scales, g))
    if zeros is not None:
        q = q + expand_groups(zeros, g)
    if isinstance(qmin, Tensor):
        q = q - qmin.to(Q.device, Q.dtype).view(N, 1)
    else:
        q = q - qmin
    return q.to(torch.int16)


def dequantize_indices(indices: Tensor, side: dict) -> Tensor:
    """W_hat [N, K] f32 from the stored indices and side information {"scales" [N, G], "zeros" [N, G] | None, "qmin"
    (int | Tensor [N] | None), "codebook" (Codebook | None), "group", "perm" (column permutation of act_order | None)}.
    Uses the quantizers' own operation order, so the result equals their W_hat bit for bit."""
    scales = side["scales"]
    N, G = scales.shape
    g = side["group"]
    K = G * g
    idx = indices.to(scales.device, torch.int64).reshape(N, K)
    perm = side.get("perm")
    if perm is not None:
        idx = idx[:, perm]  # into processing order, where the groups are contiguous
    s = expand_groups(scales, g)
    cb = side.get("codebook")
    if cb is not None:
        W = cb.dequant(idx) * s
    else:
        q = idx.to(scales.dtype)
        qmin = side["qmin"]
        q = q + (qmin.to(scales.device, scales.dtype).view(N, 1) if isinstance(qmin, Tensor) else qmin)
        if side.get("zeros") is not None:
            q = q - expand_groups(side["zeros"], g)
        W = q * s
    if perm is not None:
        W = W[:, torch.argsort(perm)]
    return W


def check_roundtrip(W_hat: Tensor, info: dict) -> None:
    """Assert that the stored indices and side information reproduce W_hat exactly (f32, torch.equal)."""
    back = dequantize_indices(info["indices"], info["side"])
    if not torch.equal(back, W_hat):
        bad = (back != W_hat).sum().item()
        raise AssertionError(f"index round trip is not exact: {bad} of {W_hat.numel()} weights differ "
                             f"(max |diff| {(back - W_hat).abs().max().item():.3e}); note {info.get('note')!r}")


def bits_for(bits, name: str) -> int:
    """The width of tensor `name` from an int (every tensor) or a dict[name -> int] (Phase 3 allocation)."""
    if isinstance(bits, dict):
        if name not in bits:
            raise KeyError(f"no bit width for {name!r} in the per-tensor bits mapping")
        return int(bits[name])
    return int(bits)


def apply_rtn(model: nn.Module, bits, group: int = 128, sym: bool = True, linears=None, row_bits: dict | None = None,
              codebook: dict | None = None):
    """Quantize every block linear of `model` in place (weight <- W_hat cast to the weight's dtype, bf16).

    `linears` defaults to wcomp.models.block_linears(model); a list of (name, nn.Linear) overrides it.
    Returns a wcomp.bits.BitAccount with one entry per quantized weight. Raises if a reconstruction is
    not finite (a NaN weight would silently poison every downstream metric).
    Phase 3: `bits` may be a dict[name -> int] (per-tensor widths, method A) and `row_bits` a dict[name -> int
    Tensor [N]] (per-row widths, method G; a tensor absent from the dict keeps `bits`); a row-allocated tensor is
    charged ROW_SIDE_BITS per row on top of the grid's side information and says so in its note. `codebook`
    {"kind": "lloyd" | "nf", "bits": optional} (method E2 / E3) replaces the uniform grid for every tensor (its "bits",
    when given, must equal an int `bits`; with a dict `bits` each tensor's codebook has the tensor's own width); the
    codebook table bits are in the tensor's aux_bits (quantize_rtn).
    The bf16 store is the contract; it puts the 8-bit (and 16-bit) rows on a bf16 floor: q * f16 scale needs
    ~19 mantissa bits and bf16 keeps 8, so at 8 bits ~5-6% of the stored weight MSE is the cast, not the code
    (negligible at <= 4 bits); to be stated in NOTES.md (Phase 1).
    """
    from wcomp.bits import BitAccount

    if linears is None:
        from wcomp.models import block_linears

        linears = block_linears(model)
    if codebook is not None and codebook.get("bits") is not None and not isinstance(bits, dict) and int(codebook["bits"]) != int(bits):
        raise ValueError(f"codebook bits {codebook['bits']} differ from bits {bits}")
    account = BitAccount()
    with torch.no_grad():
        for name, lin in linears:
            W = lin.weight.data
            rb = row_bits.get(name) if row_bits else None
            b = bits_for(bits, name)
            cb = None if codebook is None else {**codebook, "bits": b}
            W_hat, info = quantize_rtn(W.float(), b, group, sym, row_bits=rb, codebook=cb)
            if not torch.isfinite(W_hat).all():
                raise RuntimeError(f"apply_rtn: non-finite reconstruction for {name}")
            W.copy_(W_hat.to(W.dtype))
            aux, note, rows = info["aux_bits"], info["note"], None
            if rb is not None:
                aux += ROW_SIDE_BITS * W.shape[0]
                note += f"; {ROW_SIDE_BITS} bits/row width"
                rows = check_row_bits(rb, W.shape[0]).tolist()
            account.add(name, W.numel(), info["index_bits"], aux, note, indices=info["indices"], levels=info["levels"], row_bits=rows)
    return account


# ---------------------------------------------------------------- Phase 2 (DESIGN.md P0): +keepcols and +keepsuper.
# Additive helpers; nothing above this line changed.


def n_keep_columns(K: int, pct: float) -> int:
    """How many columns 'the top pct percent' of K are: max(1, round(K * pct / 100)); 1024 columns at
    0.5 / 1 / 2 % keep 5 / 10 / 20. The one definition shared by P0 +keepcols and the P3 estimate."""
    if not 0 < pct <= 100:
        raise ValueError(f"pct must be in (0, 100], got {pct}")
    return max(1, int(round(K * pct / 100.0)))


def keepcols_mask(W: Tensor, pct: float) -> Tensor:
    """Bool [K] marking the input columns of W [N, K] whose L2 norm is in the top `pct` percent of the tensor."""
    if W.dim() != 2:
        raise ValueError(f"expected [N, K], got {tuple(W.shape)}")
    K = W.shape[1]
    n_keep = n_keep_columns(K, pct)
    idx = torch.topk(W.float().norm(dim=0), n_keep).indices
    mask = torch.zeros(K, dtype=torch.bool, device=W.device)
    mask[idx] = True
    return mask


def quantize_rtn_keepcols(W: Tensor, bits: int, group: int, keep_mask: Tensor, sym: bool = True) -> tuple[Tensor, dict]:
    """quantize_rtn with the columns of `keep_mask` [K] kept exactly (bf16 columns of P0 +keepcols).

    The group is still `group` consecutive columns; kept columns are zeroed before the grid is computed, so they
    do not take part in the group amax (sym) or range (asym: the range already contains 0), and are restored
    afterwards. Bits: b per quantized weight, 16 per kept weight, one f16 scale (+ b-bit zero if asym) for every
    group, and a K-bit column mask per tensor. A group whose columns are all kept gets the zero-scale guard
    (scale 1) and dequantizes to the restored values anyway.
    """
    if W.dim() != 2:
        raise ValueError(f"expected [N, K], got {tuple(W.shape)}")
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in 1..16, got {bits}")
    N, K = W.shape
    g = K if group is None or group <= 0 else group
    if K % g != 0:
        raise ValueError(f"K={K} not divisible by group={g}")
    keep = keep_mask.to(W.device, torch.bool)
    if keep.shape != (K,):
        raise ValueError(f"keep_mask must be [K]={K}, got {tuple(keep.shape)}")
    W = W.float()
    Wm = W.masked_fill(keep.unsqueeze(0), 0.0)
    q, scale, zero = group_grid(Wm.reshape(N, K // g, g), bits, sym)
    W_hat = ((q - zero) * scale).reshape(N, K)
    W_hat[:, keep] = W[:, keep]
    n_keep = int(keep.sum().item())
    n_groups = N * (K // g)
    info = {
        "index_bits": bits * N * (K - n_keep),
        "aux_bits": 16 * N * n_keep + n_groups * (SCALE_BITS + (0 if sym else bits)) + K,
        "note": f"rtn-{'sym' if sym else 'asym'}-g{g}-{bits}: f16 scale" + ("" if sym else f", {bits}-bit zero")
        + "; keepcols: bf16 columns + K-bit mask",
        "n_keep": n_keep,
    }
    return W_hat, info


def apply_rtn_keepcols(model: nn.Module, bits: int, group: int, pct: float, sym: bool = True, linears=None):
    """apply_rtn with the top `pct` percent input columns (L2 norm, per tensor) of each linear kept in bf16."""
    from wcomp.bits import BitAccount

    if linears is None:
        from wcomp.models import block_linears

        linears = block_linears(model)
    account = BitAccount()
    with torch.no_grad():
        for name, lin in linears:
            W = lin.weight.data
            W32 = W.float()
            W_hat, info = quantize_rtn_keepcols(W32, bits, group, keepcols_mask(W32, pct), sym)
            if not torch.isfinite(W_hat).all():
                raise RuntimeError(f"apply_rtn_keepcols: non-finite reconstruction for {name}")
            W.copy_(W_hat.to(W.dtype))
            note = info["note"].replace("keepcols:", f"keepcols {pct:g}%:")
            account.add(name, W.numel(), info["index_bits"], info["aux_bits"], note)
    return account


def super_weight_bits(N: int, K: int) -> int:
    """Side information for one restored weight of an [N, K] tensor: a bf16 value plus its flat address."""
    return 16 + math.ceil(math.log2(N * K))


def _candidate_kind(note) -> str:
    """The kind token(s) of a P7 record note ("spike", "top16", "spike+top16", "max_spike_below_threshold", ...)."""
    return str(note or "").split(";")[0].strip()


def is_super_candidate_kind(kind: str) -> bool:
    """A P7 record is a +keepsuper candidate iff one of its '+'-joined kind tokens is "spike" (a co-occurring
    down_proj activation spike) or "top<k>" (global top-k |w|); the recipe's fallback measurement
    ("max_spike_below_threshold") and an output spike without input spike ("spike_no_input") are not. A record
    with no kind (a bare "candidates" list) is a candidate."""
    tokens = [t for t in kind.split("+") if t]
    if not tokens:
        return True
    return any(t == "spike" or t.startswith("top") for t in tokens)


def super_candidate_records(path: str) -> list[dict]:
    """The +keepsuper candidates of a P7 JSON as {"tensor", "row", "col", "kind"}, deduplicated, order kept.

    Accepted layouts: the probe output written by scripts/probe_acts.py (a dict with "records", each {"tensor",
    "stat": {"row", "col"}, "note"}; the note's first clause is the kind and selects via is_super_candidate_kind),
    a dict with a "candidates" list, or a bare list; entries carry "tensor" (module name, a trailing ".weight" is
    dropped) and "row" / "col" at the top level or under "stat". At most 64 (DESIGN.md P0 +keepsuper)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        cands = [(c, "") for c in data]
    elif isinstance(data, dict) and isinstance(data.get("candidates"), list):
        cands = [(c, "") for c in data["candidates"]]
    elif isinstance(data, dict) and isinstance(data.get("records"), list):
        cands = [(r, _candidate_kind(r.get("note"))) for r in data["records"]]
        cands = [(r, k) for r, k in cands if is_super_candidate_kind(k)]
    else:
        raise ValueError(f"{path}: expected a P7 probe JSON ('records'), a dict with a 'candidates' list, or a list")
    out: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for c, kind in cands:
        tensor = str(c["tensor"])
        if tensor.endswith(".weight"):
            tensor = tensor[: -len(".weight")]
        stat = c.get("stat") if isinstance(c.get("stat"), dict) else {}
        row = c["row"] if "row" in c else stat["row"]
        col = c["col"] if "col" in c else stat["col"]
        triple = (tensor, int(row), int(col))
        if triple not in seen:
            seen.add(triple)
            out.append({"tensor": tensor, "row": int(row), "col": int(col), "kind": kind or str(c.get("kind", "")) or "listed"})
    if len(out) > 64:
        raise ValueError(f"{path}: {len(out)} candidates, the contract allows at most 64")
    return out


def load_super_candidates(path: str) -> list[tuple[str, int, int]]:
    """(tensor, row, col) triples of super_candidate_records(path)."""
    return [(c["tensor"], c["row"], c["col"]) for c in super_candidate_records(path)]


def super_candidate_kinds(path: str) -> dict[str, int]:
    """{kind: count} of the candidates +keepsuper would restore from a P7 JSON ("spike", "top16", "spike+top16", "listed")."""
    counts: dict[str, int] = {}
    for c in super_candidate_records(path):
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    return dict(sorted(counts.items()))
