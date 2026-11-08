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


# ---------------------------------------------------------------- Phase 3 (DESIGN.md method D): rotation variants

ROTATION_KINDS = ("hadamard", "hadsign", "rnd", "eig")


def random_signs(n: int, seed: int) -> Tensor:
    """Seeded +-1 vector of length n (f32); 0 bits of side information like every seeded draw."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=gen).to(torch.float32) * 2 - 1


def hadsign(n: int, seed: int) -> Tensor:
    """hadamard(n) @ diag(random_signs(n, seed)): still orthonormal, rows of the Hadamard with random column
    signs, so the rotation is not the same fixed matrix for every tensor of the same width across seeds.
    Raises ValueError like hadamard(n) when no Hadamard of order n exists."""
    return hadamard(n) * random_signs(n, seed)[None, :]


def eig_rotation(H: Tensor) -> Tensor:
    """Eigenvectors of the symmetric input Hessian H [K, K] as the rotation R (f32, columns = eigenvectors,
    descending eigenvalue, each column's largest-|.| entry made positive so the matrix is deterministic).
    x R has the diagonal Hessian R^T H R = diag(eigenvalues): the input coordinates are decorrelated."""
    if H.dim() != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got {tuple(H.shape)}")
    Hs = H.to(torch.float64)
    Hs = 0.5 * (Hs + Hs.T)
    evals, evecs = torch.linalg.eigh(Hs)
    order = torch.argsort(evals, descending=True)
    V = evecs[:, order]
    pivot = V.abs().argmax(dim=0)
    sign = torch.sign(V[pivot, torch.arange(V.shape[1], device=V.device)])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return (V * sign[None, :]).to(torch.float32)


# Leaf names of block linears that read the same input tensor inside one decoder layer (one Hessian, one R).
_SHARED_INPUT: dict[str, str] = {
    "in_proj_qkv": "in_proj", "in_proj_z": "in_proj", "in_proj_a": "in_proj", "in_proj_b": "in_proj",
    "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv",
    "gate_proj": "gateup", "up_proj": "gateup",
}


def shared_input_groups(linears) -> list[list[str]]:
    """Names of `linears` [(name, nn.Linear)] grouped by shared input: per decoder layer the DeltaNet in_proj_*
    family, q/k/v, gate/up; every other linear (down, o, out_proj, or anything outside model.layers) is its own
    group. Order follows `linears`."""
    from wcomp.models import layer_index

    groups: dict[tuple, list[str]] = {}
    for name, _ in linears:
        leaf = name.split(".")[-1]
        if leaf in _SHARED_INPUT:
            try:
                key = ("layer", layer_index(name), _SHARED_INPUT[leaf])
            except ValueError:
                key = ("name", name)
        else:
            key = ("name", name)
        groups.setdefault(key, []).append(name)
    return list(groups.values())


class _Stop(Exception):
    """Raised by a pre-hook on the first decoder layer past the current chunk to end the forward early."""


def _run_until(model: nn.Module, ids: Tensor, stop_layer: int | None) -> None:
    layers = model.model.layers
    if stop_layer is None or stop_layer >= len(layers):
        model.model(input_ids=ids, use_cache=False)
        return

    def stop(mod, args):
        raise _Stop

    handle = layers[stop_layer].register_forward_pre_hook(stop)
    try:
        model.model(input_ids=ids, use_cache=False)
    except _Stop:
        pass
    finally:
        handle.remove()


@torch.no_grad()
def collect_input_hessians(model: nn.Module, linears, calib_ids: Tensor, layer_range: tuple[int, int] | None = None) -> dict[str, Tensor]:
    """H = x^T x / n (f64, on the model device) of the input of every shared-input group of `linears` over the
    calibration windows (f32 per-window products into an f64 sum, the Phase 3 GPTQ path). Returned keyed by every
    member name (members of a group share one tensor object). With `layer_range` = (start, stop) only the linears
    of decoder layers start..stop-1 are hooked and the forward stops at layer `stop` (chunking for memory)."""
    from wcomp.models import layer_index

    dev = next(model.parameters()).device
    modules = dict(linears)
    groups = shared_input_groups(linears)
    if layer_range is not None:
        start, stop = layer_range
        groups = [g for g in groups if start <= layer_index(g[0]) < stop]
    else:
        stop = None
    acc: dict[int, list] = {}  # group index -> [H sum, rows]
    handles = []
    for gi, names in enumerate(groups):
        lin = modules[names[0]]
        acc[gi] = [torch.zeros(lin.in_features, lin.in_features, dtype=torch.float64, device=dev), 0]

        def make_hook(gi: int):
            def hook(mod, args):
                x = args[0].reshape(-1, args[0].shape[-1]).to(torch.float32)
                acc[gi][0].add_((x.T @ x).to(torch.float64))
                acc[gi][1] += x.shape[0]

            return hook

        handles.append(lin.register_forward_pre_hook(make_hook(gi)))
    try:
        for j in range(calib_ids.shape[0]):
            _run_until(model, calib_ids[j : j + 1].to(dev), stop)
    finally:
        for h in handles:
            h.remove()
    out: dict[str, Tensor] = {}
    for gi, names in enumerate(groups):
        Hsum, rows = acc[gi]
        if rows == 0:
            raise RuntimeError(f"no input seen for {names[0]} over the calibration windows")
        H = Hsum / rows
        for n in names:
            out[n] = H
    return out


def rotation_bits(rotations: dict[str, Tensor], linears, kind: str) -> dict[str, int]:
    """Side information charged per linear for the rotations: 0 for the seeded / deterministic kinds, and for
    "eig" the stored bf16 matrix, 16 * K * K bits per distinct R, split over the linears that share that very
    tensor object in proportion to their numel (the per-param charge is then uniform within the group and
    16 K / N bits/param for a lone linear)."""
    modules = dict(linears)
    out = {name: 0 for name in rotations}
    if kind != "eig":
        return out
    by_id: dict[int, list[str]] = {}
    for name, R in rotations.items():
        by_id.setdefault(id(R), []).append(name)
    for names in by_id.values():
        K = rotations[names[0]].shape[0]
        total = 16 * K * K
        numels = [modules[n].weight.numel() for n in names]
        n_all = sum(numels)
        charged = 0
        for n, k in zip(names[:-1], numels[:-1]):
            share = total * k // n_all
            out[n] = share
            charged += share
        out[names[-1]] = total - charged  # the remainder, so the group's sum is exactly 16 K K
    return out


def rotate_inputs(model: nn.Module, linears=None, seed: int = 0, kind: str = "hadamard", calib_ids: Tensor | None = None,
                  hessians: dict[str, Tensor] | None = None, layer_chunk: int = 6) -> dict[str, Tensor]:
    """Rotate the input space of each linear, exactly function preserving.

    For y = x W^T set W <- W R and feed x <- x R via a forward_pre_hook, so x R R^T W^T = x W^T.
    R is hadamard(K) when it exists, else random_orthogonal(K, seed) (the caller records that case).
    `linears` defaults to wcomp.models.block_linears(model). Returns {name: R} with R in the weight's
    dtype and device (bf16 on the GPU for the real models).

    Phase 3 kinds (DESIGN.md method D), all applied through the same fold / hook path:
      "hadamard" (default, the Phase 1 behaviour), "hadsign" (hadamard(K) times a seeded random +-1 diagonal, the
      random_orthogonal fallback where no Hadamard exists), "rnd" (random_orthogonal(K, seed) for every width),
      "eig" (per shared-input group the eigenvectors of its calibration Hessian, from `hessians` {name: H [K, K]}
      or collected over `calib_ids` with hooks, `layer_chunk` decoder layers at a time; members of a group get the
      same R object, so rotation_bits can charge the stored bf16 matrix once). The seeded kinds cost 0 bits; "eig"
      is the one stored matrix, charged by rotation_bits().

    The hook applies the *stored* (rounded) R, which is only approximately orthogonal: for the Hadamard in
    bf16, R R^T = c^2 n I with c = bf16(1 / sqrt(n)) (c^2 n = 0.9998 for n = 2048, 1.0024 for 3584). So the
    weight is folded with the exact inverse of the stored matrix, W <- W (R^T)^-1 (computed in f64; for an
    exactly orthogonal R this is W R), and x R (W (R^T)^-1)^T = x W^T up to per-element bf16 rounding only.

    That rounding (of x R in the hook and of the folded weight) is not free: on Qwen3.5-0.8B it perturbs each
    linear's output by ~0.3% and raises WikiText-2 PPL by about +0.07% (all 145 windows, +0.10% on the first 8),
    within the contract's 0.1% but systematic. The method "had-fp16" (wcomp.methods) records this rotation-only
    cost as a row of its own so that had-* results are read against it.
    """
    if kind not in ROTATION_KINDS:
        raise ValueError(f"unknown rotation kind {kind!r}; known: {ROTATION_KINDS}")
    if linears is None:
        from wcomp.models import block_linears

        linears = block_linears(model)
    if kind == "eig":
        return _rotate_inputs_eig(model, linears, calib_ids, hessians, layer_chunk)

    def make(K: int) -> Tensor:
        if kind == "hadamard":
            return _rotation(K, seed)
        if kind == "rnd":
            return random_orthogonal(K, seed)
        try:
            return hadsign(K, seed)
        except ValueError:
            return random_orthogonal(K, seed)

    cache: dict[tuple[int, torch.dtype, torch.device], tuple[Tensor, Tensor]] = {}
    out: dict[str, Tensor] = {}
    with torch.no_grad():
        for name, lin in linears:
            W = lin.weight.data
            key = (W.shape[1], W.dtype, W.device)
            if key not in cache:
                R = make(W.shape[1]).to(device=W.device, dtype=W.dtype)
                fold = torch.linalg.inv(R.double()).T.float()  # (R^T)^-1 of the stored R
                cache[key] = (R, fold)
            R, fold = cache[key]
            W.copy_((W.float() @ fold).to(W.dtype))
            lin.register_forward_pre_hook(_make_hook(R))
            lin.wcomp_input_fold = fold  # the basis change the quantizer sees (wcomp.rdbound folds the original the same way)
            out[name] = R
    return out


def _apply_rotation(lin: nn.Linear, R32: Tensor) -> Tensor:
    """Store R in the weight's dtype / device, fold W <- W (R^T)^-1 of the stored R, hook x <- x R; returns the stored R."""
    W = lin.weight.data
    R = R32.to(device=W.device, dtype=W.dtype)
    fold = torch.linalg.inv(R.double()).T.float()
    W.copy_((W.float() @ fold).to(W.dtype))
    lin.register_forward_pre_hook(_make_hook(R))
    lin.wcomp_input_fold = fold
    return R


@torch.no_grad()
def _rotate_inputs_eig(model: nn.Module, linears, calib_ids: Tensor | None, hessians: dict[str, Tensor] | None,
                       layer_chunk: int) -> dict[str, Tensor]:
    """The "eig" kind: one R per shared-input group from its Hessian's eigenvectors. With `hessians` given, every
    linear of a group is rotated with the eigenvectors of the first member's H; otherwise the Hessians are collected
    over `calib_ids` for `layer_chunk` decoder layers at a time (the forward stops after the chunk), that chunk is
    rotated, and the next chunk's inputs are collected through the already rotated layers (identical up to bf16
    rounding, the same inputs GPTQ will later see)."""
    from wcomp.models import layer_index

    modules = dict(linears)
    out: dict[str, Tensor] = {}

    def rotate_groups(groups: list[list[str]], H_of: dict[str, Tensor]) -> None:
        for names in groups:
            R32 = eig_rotation(H_of[names[0]])
            stored = None
            for n in names:
                K = modules[n].in_features
                if R32.shape[0] != K:
                    raise ValueError(f"{n}: Hessian is {tuple(R32.shape)} but the linear has {K} inputs")
                if stored is None:
                    stored = _apply_rotation(modules[n], R32)
                else:
                    _apply_rotation(modules[n], stored.float())
                out[n] = stored  # one object per group, so rotation_bits charges the matrix once

    if hessians is not None:
        rotate_groups(shared_input_groups(linears), hessians)
        return out
    if calib_ids is None or model is None:
        raise ValueError('rotate_inputs(kind="eig") needs calib_ids (C4 windows) or precomputed hessians')
    n_layers = len(model.model.layers)
    layered = [(n, m) for n, m in linears]
    for start in range(0, n_layers, layer_chunk):
        stop = min(start + layer_chunk, n_layers)
        chunk = [(n, m) for n, m in layered if start <= layer_index(n) < stop]
        if not chunk:
            continue
        H_of = collect_input_hessians(model, chunk, calib_ids, layer_range=(start, stop))
        rotate_groups(shared_input_groups(chunk), H_of)
        del H_of
        if next(model.parameters()).device.type == "cuda":
            torch.cuda.empty_cache()
    missing = [n for n, _ in linears if n not in out]
    if missing:
        raise RuntimeError(f"eig rotation: {len(missing)} linears outside model.layers were not rotated, e.g. {missing[0]}")
    return out


def _make_hook(R: Tensor):
    def hook(module: nn.Module, args: tuple):
        x = args[0]
        return (x @ R.to(x.dtype),) + tuple(args[1:])

    return hook
