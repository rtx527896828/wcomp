"""Round-to-nearest group quantization (DESIGN.md, wcomp/quant/rtn.py contract)."""
from __future__ import annotations

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
