"""Round-to-nearest group quantization (DESIGN.md, wcomp/quant/rtn.py contract)."""
from __future__ import annotations

import json
import math

import torch
from torch import Tensor, nn

SCALE_BITS = 16  # scales are stored as f16


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


def quantize_rtn(W: Tensor, bits: int, group: int = 128, sym: bool = True) -> tuple[Tensor, dict]:
    """RTN-quantize W [N, K] (f32) with groups of `group` consecutive inputs along K.

    group <= 0 means one group per output row. Returns the dequantized f32 weight and
    info = {"index_bits", "aux_bits", "note"}; aux counts an f16 scale per group plus a b-bit zero if asym.
    """
    if W.dim() != 2:
        raise ValueError(f"expected [N, K], got {tuple(W.shape)}")
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in 1..16, got {bits}")
    N, K = W.shape
    g = K if group is None or group <= 0 else group
    if K % g != 0:
        raise ValueError(f"K={K} not divisible by group={g}")
    W = W.float()
    q, scale, zero = group_grid(W.reshape(N, K // g, g), bits, sym)
    W_hat = ((q - zero) * scale).reshape(N, K)
    n_groups = N * (K // g)
    info = {
        "index_bits": bits * N * K,
        "aux_bits": n_groups * (SCALE_BITS + (0 if sym else bits)),
        "note": f"rtn-{'sym' if sym else 'asym'}-g{g}-{bits}: f16 scale" + ("" if sym else f", {bits}-bit zero"),
    }
    return W_hat, info


def apply_rtn(model: nn.Module, bits: int, group: int = 128, sym: bool = True, linears=None):
    """Quantize every block linear of `model` in place (weight <- W_hat cast to the weight's dtype, bf16).

    `linears` defaults to wcomp.models.block_linears(model); a list of (name, nn.Linear) overrides it.
    Returns a wcomp.bits.BitAccount with one entry per quantized weight. Raises if a reconstruction is
    not finite (a NaN weight would silently poison every downstream metric).
    The bf16 store is the contract; it puts the 8-bit (and 16-bit) rows on a bf16 floor: q * f16 scale needs
    ~19 mantissa bits and bf16 keeps 8, so at 8 bits ~5-6% of the stored weight MSE is the cast, not the code
    (negligible at <= 4 bits); to be stated in NOTES.md (Phase 1).
    """
    from wcomp.bits import BitAccount

    if linears is None:
        from wcomp.models import block_linears

        linears = block_linears(model)
    account = BitAccount()
    with torch.no_grad():
        for name, lin in linears:
            W = lin.weight.data
            W_hat, info = quantize_rtn(W.float(), bits, group, sym)
            if not torch.isfinite(W_hat).all():
                raise RuntimeError(f"apply_rtn: non-finite reconstruction for {name}")
            W.copy_(W_hat.to(W.dtype))
            account.add(name, W.numel(), info["index_bits"], info["aux_bits"], info["note"])
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
