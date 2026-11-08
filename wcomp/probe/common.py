"""Shared pieces of the Phase 2 probes (DESIGN.md, "Phase 2 contract", common definitions).

Controls, the bits/param conversions, the per-record container and the JSON writer.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from wcomp.quant.rtn import n_keep_columns
from wcomp.version import code_version

FREQ_TABLE_BITS = 16  # one 16-bit frequency per index value for the entropy-coder side information (P1)


def _generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def shuffle_within_tensor(W: Tensor, seed: int) -> Tensor:
    """Seeded random permutation of all elements of W (same shape, dtype, device): the shuffled control."""
    g = _generator(seed, W.device)
    perm = torch.randperm(W.numel(), generator=g, device=W.device)
    return W.reshape(-1)[perm].reshape(W.shape)


def gaussian_control(W: Tensor, group: int = 128, seed: int = 0) -> Tensor:
    """iid Gaussian with the same shape and the same per-group std along the input (last) dim as W [N, K].

    Groups are `group` consecutive inputs (the Phase 1 grid); if K is not divisible by `group`, one group per row.
    Population std (unbiased=False) so the control's expected energy per group matches W's exactly.
    """
    if W.dim() != 2:
        raise ValueError(f"expected [N, K], got {tuple(W.shape)}")
    N, K = W.shape
    g = group if group and group > 0 and K % group == 0 else K
    Wg = W.float().reshape(N, K // g, g)
    std = Wg.std(dim=-1, keepdim=True, unbiased=False)
    noise = torch.randn(Wg.shape, generator=_generator(seed, W.device), device=W.device, dtype=torch.float32)
    return (noise * std).reshape(N, K).to(W.dtype)


def bits_from_mse_ratio(rho: float) -> float:
    """Bits/param worth of reaching the same distortion with a factor rho less MSE: D ~ 2^(-2b)."""
    return 0.5 * math.log2(rho)


def entropy_bits(q: Tensor) -> float:
    """Empirical entropy in bits of the values of an integer-valued tensor (any dtype with integral values)."""
    q = q.reshape(-1)
    if q.numel() == 0:
        return 0.0
    qi = q.to(torch.int64) if q.dtype.is_floating_point else q.to(torch.int64)
    qi = qi - qi.min()
    counts = torch.bincount(qi).to(torch.float64)
    p = counts[counts > 0] / float(q.numel())
    return float(-(p * torch.log2(p)).sum().item())


def freq_table_overhead_bits(bits: int, numel: int) -> float:
    """Side information of an entropy coder: one 16-bit frequency per symbol of a b-bit alphabet, per tensor."""
    return (2**bits) * FREQ_TABLE_BITS / numel


def log2_factorial(m: int) -> float:
    """log2(m!) via lgamma: the bits of a stored permutation of m items (P4)."""
    return math.lgamma(m + 1) / math.log(2)


def n_top_columns(K: int, pct: float) -> int:
    """How many columns 'the top pct percent' of K are (P3): the P0 +keepcols definition, wcomp.quant.rtn.n_keep_columns."""
    return n_keep_columns(K, pct)


@dataclass
class ProbeRecord:
    """One probe record (DESIGN.md output schema) plus `numel`, the weight used by the per-type aggregate."""

    tensor: str
    layer: int
    type: str
    stat: dict[str, Any] = field(default_factory=dict)
    control: dict[str, Any] = field(default_factory=dict)
    gain_bits: float | None = None
    gain_bits_control: float | None = None
    note: str = ""
    numel: int = 0


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def aggregate(records: Sequence[ProbeRecord]) -> dict[str, dict]:
    """Per weight type: numel-weighted mean of every numeric entry of stat / control and of the two gains.

    A key is averaged over the records of that type that carry a finite number for it (None and strings are
    skipped, with their numel left out of that key's denominator). Summary records (numel 0: the P3 cross-layer
    Jaccard) and the P4 best-of-all-pairs record (stat.is_best 1, is_adjacent 0: selected on W over all pairs) are not
    tensor measurements of the type and are left out ("n_excluded" counts them); a type with nothing else keeps them.
    """
    out: dict[str, dict] = {}
    by_type: dict[str, list[ProbeRecord]] = {}
    for r in records:
        by_type.setdefault(r.type, []).append(r)
    for t, all_recs in by_type.items():
        recs = [r for r in all_recs if r.numel > 0 and not (r.stat.get("is_best") == 1 and r.stat.get("is_adjacent") == 0)]
        if not recs:
            recs = all_recs
        agg: dict[str, Any] = {"n_records": len(recs), "n_excluded": len(all_recs) - len(recs),
                               "numel": sum(r.numel for r in recs), "stat": {}, "control": {}}
        for section in ("stat", "control"):
            keys: list[str] = []
            for r in recs:
                for k in getattr(r, section):
                    if k not in keys:
                        keys.append(k)
            for k in keys:
                num = den = 0.0
                for r in recs:
                    v = getattr(r, section).get(k)
                    if _is_number(v):
                        num += v * r.numel
                        den += r.numel
                agg[section][k] = num / den if den > 0 else None
        for k in ("gain_bits", "gain_bits_control"):
            num = den = 0.0
            for r in recs:
                v = getattr(r, k)
                if _is_number(v):
                    num += v * r.numel
                    den += r.numel
            agg[k] = num / den if den > 0 else None
        out[t] = agg
    return out


def _jsonable(v: Any) -> Any:
    if isinstance(v, Tensor):
        v = v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def write_probe_json(path: str | Path, model: str, probe: str, params: dict, records: Sequence[ProbeRecord]) -> dict:
    """Write {"model", "probe", "code_version", "timestamp", "params", "records", "aggregate"} to path; returns it."""
    doc = {
        "model": model,
        "probe": probe,
        "code_version": code_version(),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "params": _jsonable(params),
        "records": [_jsonable(asdict(r)) for r in records],
        "aggregate": _jsonable(aggregate(records)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, allow_nan=False)
    return doc
