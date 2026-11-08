"""Orthogonal input rotations for block linears (DESIGN.md, wcomp/quant/rotate.py contract)."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _quadratic_character(q: int) -> Tensor:
    """chi(a) for a in 0..q-1 over the prime field GF(q): 0 at 0, +1 on squares, -1 otherwise."""
    squares = {(a * a) % q for a in range(1, q)}
    return torch.tensor([0 if a == 0 else (1 if a in squares else -1) for a in range(q)], dtype=torch.float32)


def _jacobsthal(q: int) -> Tensor:
    """Q[a, b] = chi(b - a) (q x q). Skew-symmetric for q = 3 mod 4, symmetric for q = 1 mod 4."""
    chi = _quadratic_character(q)
    idx = (torch.arange(q)[None, :] - torch.arange(q)[:, None]) % q
    return chi[idx]


def _paley_i(q: int) -> Tensor:
    """Paley construction I: Hadamard matrix of order q + 1 for a prime q = 3 mod 4."""
    if q % 4 != 3:
        raise ValueError(f"Paley I needs q = 3 mod 4, got {q}")
    S = torch.zeros(q + 1, q + 1)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = _jacobsthal(q)
    return torch.eye(q + 1) + S


def _paley_ii(q: int) -> Tensor:
    """Paley construction II: Hadamard matrix of order 2(q + 1) for a prime q = 1 mod 4."""
    if q % 4 != 1:
        raise ValueError(f"Paley II needs q = 1 mod 4, got {q}")
    S = torch.zeros(q + 1, q + 1)
    S[0, 1:] = 1
    S[1:, 0] = 1
    S[1:, 1:] = _jacobsthal(q)
    A = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    B = torch.tensor([[1.0, -1.0], [-1.0, -1.0]])
    return torch.kron(S, A) + torch.kron(torch.eye(q + 1), B)


_H2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])

# Unnormalized (+-1) Hadamard matrices of non-power-of-two order. Order 40 has no Paley construction
# (39 is not a prime power; Paley II needs q = 19 = 1 mod 4) and is the Sylvester doubling of H_20.
HADAMARD_TABLE: dict[int, Tensor] = {
    1: torch.ones(1, 1),
    12: _paley_i(11),
    20: _paley_i(19),
    28: _paley_ii(13),
    40: torch.kron(_H2, _paley_i(19)),
}


def _sylvester(k: int) -> Tensor:
    """Unnormalized Sylvester Hadamard matrix of order 2^k."""
    H = torch.ones(1, 1)
    for _ in range(k):
        H = torch.kron(H, _H2)
    return H


def hadamard(n: int) -> Tensor:
    """Orthonormal Hadamard matrix of order n = m * 2^k with m in HADAMARD_TABLE (f32).

    Raises ValueError if n has no such factorization; the caller then falls back to random_orthogonal.
    """
    for m in sorted(HADAMARD_TABLE):
        if n % m == 0:
            p = n // m
            if p & (p - 1) == 0:
                H = torch.kron(HADAMARD_TABLE[m], _sylvester(p.bit_length() - 1))
                return H / math.sqrt(n)
    raise ValueError(f"no Hadamard matrix of order {n} from table {sorted(HADAMARD_TABLE)} x powers of 2")


def random_orthogonal(n: int, seed: int) -> Tensor:
    """Haar-random orthogonal n x n matrix (f32) from a seeded Gaussian QR, signs fixed by diag(R)."""
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))[None, :]
    return Q.to(torch.float32)


def _rotation(K: int, seed: int) -> Tensor:
    try:
        return hadamard(K)
    except ValueError:
        return random_orthogonal(K, seed)


def rotate_inputs(model: nn.Module, linears=None, seed: int = 0) -> dict[str, Tensor]:
    """Rotate the input space of each linear, exactly function preserving.

    For y = x W^T set W <- W R and feed x <- x R via a forward_pre_hook, so x R R^T W^T = x W^T.
    R is hadamard(K) when it exists, else random_orthogonal(K, seed) (the caller records that case).
    `linears` defaults to wcomp.models.block_linears(model). Returns {name: R} with R in the weight's
    dtype and device (bf16 on the GPU for the real models).

    The hook applies the *stored* (rounded) R, which is only approximately orthogonal: for the Hadamard in
    bf16, R R^T = c^2 n I with c = bf16(1 / sqrt(n)) (c^2 n = 0.9998 for n = 2048, 1.0024 for 3584). So the
    weight is folded with the exact inverse of the stored matrix, W <- W (R^T)^-1 (computed in f64; for an
    exactly orthogonal R this is W R), and x R (W (R^T)^-1)^T = x W^T up to per-element bf16 rounding only.

    That rounding (of x R in the hook and of the folded weight) is not free: on Qwen3.5-0.8B it perturbs each
    linear's output by ~0.3% and raises WikiText-2 PPL by about +0.07% (all 145 windows, +0.10% on the first 8),
    within the contract's 0.1% but systematic. The method "had-fp16" (wcomp.methods) records this rotation-only
    cost as a row of its own so that had-* results are read against it.
    """
    if linears is None:
        from wcomp.models import block_linears

        linears = block_linears(model)
    cache: dict[tuple[int, torch.dtype, torch.device], tuple[Tensor, Tensor]] = {}
    out: dict[str, Tensor] = {}
    with torch.no_grad():
        for name, lin in linears:
            W = lin.weight.data
            key = (W.shape[1], W.dtype, W.device)
            if key not in cache:
                R = _rotation(W.shape[1], seed).to(device=W.device, dtype=W.dtype)
                fold = torch.linalg.inv(R.double()).T.float()  # (R^T)^-1 of the stored R
                cache[key] = (R, fold)
            R, fold = cache[key]
            W.copy_((W.float() @ fold).to(W.dtype))
            lin.register_forward_pre_hook(_make_hook(R))
            out[name] = R
    return out


def _make_hook(R: Tensor):
    def hook(module: nn.Module, args: tuple):
        x = args[0]
        return (x @ R.to(x.dtype),) + tuple(args[1:])

    return hook
