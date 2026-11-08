"""Scalar codebooks for non-uniform quantization (DESIGN.md Phase 3, method E2 / E3).

lloyd_max fits 2^b centroids to a sample by Lloyd-Max iterations (1-D k-means: nearest-centroid assignment,
then each centroid becomes the mean of its cell). gaussian_quantile_codebook is the equiprobable codebook
(quantiles of N(0, 1) at the cell midpoints (i + 0.5) / L), the Lloyd-Max initialisation the contract names.
gaussian_lloyd_codebook is the fixed codebook of the "nf" rows: the Lloyd-Max optimum of N(0, 1) itself, computed
from the density (data independent, so 0 table bits; for 4 levels +-0.4528 / +-1.510, Max 1960). The equiprobable
codebook is deliberately not the fixed codebook: with RMS-normalized groups its outer level sits at 1.86 sigma for
16 levels, it clips the tails and its MSE on Gaussian weights is about twice the uniform amax grid's at 4 bits
(measured in tests/test_quant_phase3.py), whereas the Gaussian optimum reaches 2.73 sigma and beats the grid.
nearest_level maps samples to the index of the nearest centroid (ties go to the lower level).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

_SQRT_2PI = math.sqrt(2 * math.pi)


def gaussian_lloyd_codebook(levels: int, device=None, dtype: torch.dtype = torch.float32, iters: int = 5000) -> Tensor:
    """Sorted [levels] MSE-optimal codebook for N(0, 1), by Lloyd-Max iterations on the density in f64 (the centroid
    of a cell [a, b] is (phi(a) - phi(b)) / (Phi(b) - Phi(a))), from the equiprobable quantiles; stops when the
    codebook is stationary to 1e-13. Deterministic, so a decoder recomputes it: 0 bits of table."""
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    cb = gaussian_quantile_codebook(levels, dtype=torch.float64)
    inf = torch.tensor([float("inf")], dtype=torch.float64)
    for _ in range(iters):
        edges = torch.cat([-inf, midpoints(cb), inf])
        phi = torch.exp(-0.5 * edges.square()) / _SQRT_2PI  # phi(+-inf) = 0
        Phi = 0.5 * (1 + torch.erf(edges / math.sqrt(2.0)))
        new = (phi[:-1] - phi[1:]) / (Phi[1:] - Phi[:-1])
        if torch.all((new - cb).abs() < 1e-13):
            cb = new
            break
        cb = new
    return cb.to(device=device, dtype=dtype)


def gaussian_quantile_codebook(levels: int, device=None, dtype: torch.dtype = torch.float32) -> Tensor:
    """Sorted [levels] codebook c_i = Phi^-1((i + 0.5) / levels) for a unit-variance Gaussian (0 bits of table)."""
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    p = (torch.arange(levels, dtype=torch.float64, device=device) + 0.5) / levels
    return (math.sqrt(2.0) * torch.erfinv(2 * p - 1)).to(dtype)


def midpoints(codebook: Tensor) -> Tensor:
    """Decision thresholds between consecutive (sorted) centroids, [..., L-1]."""
    return 0.5 * (codebook[..., :-1] + codebook[..., 1:])


def nearest_level(xn: Tensor, codebook: Tensor) -> Tensor:
    """Index (int64, same shape as xn) of the nearest entry of the sorted 1-D `codebook` for every sample."""
    if codebook.dim() != 1:
        raise ValueError(f"codebook must be 1-D, got {tuple(codebook.shape)}")
    mids = midpoints(codebook.to(xn.dtype))
    return torch.bucketize(xn, mids, right=False)


def lloyd_max(x: Tensor, levels: int, iters: int = 50, init: str | Tensor = "gaussian") -> Tensor:
    """Sorted [levels] f32 centroids minimising the sample MSE of `x` (any shape, flattened) after `iters` Lloyd-Max
    iterations from the Gaussian-quantile codebook scaled by the sample RMS (init="gaussian") or from a given
    sorted codebook. Iterations stop early when the codebook no longer changes (the fixed point of the remaining
    iterations). An empty cell keeps its centroid. Means are accumulated in f64."""
    x = x.detach().reshape(-1).to(torch.float32)
    if x.numel() == 0:
        raise ValueError("lloyd_max needs a non-empty sample")
    if isinstance(init, Tensor):
        cb = init.detach().to(x.device, torch.float32).reshape(-1).clone()
        if cb.numel() != levels:
            raise ValueError(f"init codebook has {cb.numel()} entries, levels={levels}")
    elif init == "gaussian":
        rms = x.double().square().mean().sqrt().float()
        cb = gaussian_quantile_codebook(levels, x.device) * (rms if rms > 0 else 1.0)
    else:
        raise ValueError(f"unknown init {init!r}")
    for _ in range(iters):
        idx = nearest_level(x, cb)
        counts = torch.bincount(idx, minlength=levels)
        sums = torch.bincount(idx, weights=x.double(), minlength=levels)
        new = torch.where(counts > 0, sums / counts.clamp_min(1).double(), cb.double()).float()
        new, _ = torch.sort(new)
        if torch.equal(new, cb):
            break
        cb = new
    return cb


def codebook_mse(x: Tensor, codebook: Tensor) -> float:
    """Mean squared error of quantizing `x` to the nearest entry of `codebook`."""
    xf = x.reshape(-1).float()
    q = codebook.to(xf.device, torch.float32)[nearest_level(xf, codebook)]
    return float((xf.double() - q.double()).square().mean())
