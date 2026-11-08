"""Bit allocation across units (DESIGN.md Phase 3, method A) and across rows within a tensor (method G).

Objective (A2): minimize sum_u n_u d_u 2^(-2 (b_u - 4)) subject to the numel-weighted mean of b_u <= target, b_u in
LEVELS, where d_u is the measured damage per parameter of unit u (KL_u / n_u from results/phase3/<model>/damage_c4.jsonl,
or a Fisher total / n_u) and 2^(-2b) is the high-resolution distortion law. This is a multiple-choice knapsack (one
level per unit, weight n_u b_u, hard budget), solved exactly by dynamic programming over the integer budget in units of
gcd(numel) (allocate). The contract's greedy marginal-gain ascent is not exact for it: with unequal numel, and even with
equal numel across the 6 -> 8 double step, it can return a worse allocation at the same realized mean (on the 0.8B type
damage the greedy objective is 5.8% above the optimum at avg 2.5 and 3.5; tests/test_alloc.py checks the DP against a
brute force). Side information: UNIT_SIDE_BITS per unit (3 bits index into LEVELS), ROW_SIDE_BITS per row for the row
allocation (wcomp.quant.rtn). Row widths never leave [LEVELS[0], LEVELS[-1]]: a tensor at 2 or 8 bits keeps all its rows
there (a 1-bit sym grid is {-amax, 0}, i.e. pruning).
Loaders for the allocation inputs (damage rows, Fisher totals and row sums) live here too; every file is read at
call time so a method resolves without the file existing. A damage file is accepted only when every row records the
damage C4 set this code defines (DAMAGE_C4_SET: seed 2 with the seed-0 / seed-1 calibration documents excluded) and, in
the protocol tree, the protocol window count, so a pre-exclusion or smoke file can never feed a protocol allocation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
PHASE3_DIR = ROOT / "results" / "phase3"
PROBE_DIR = ROOT / "results" / "probe"

LEVELS: tuple[int, ...] = (2, 3, 4, 5, 6, 8)
UNIT_SIDE_BITS = 3  # ceil(log2(len(LEVELS))) per allocated unit
ROW_SPAN = 1  # rows get b - ROW_SPAN .. b + ROW_SPAN
# A shuffle control is a control only if it differs from the real allocation: a rebalanced draw that equals it, or
# leaves more than this share of the parameters at their real width, is rejected and the next draw is tried.
SHUFFLE_MAX_REAL_SHARE = 0.5

# The damage set (A1, scripts/damage_c4.py): C4 seed 2 with the documents of the protocol calibration sets of the run
# seeds excluded, DAMAGE_WINDOWS windows. Every damage row records DAMAGE_C4_SET; load_damage refuses any other value.
DAMAGE_C4_SEED = 2
DAMAGE_CALIB_SEEDS: tuple[int, ...] = (0, 1)
DAMAGE_CALIB_N, DAMAGE_CALIB_SEQLEN = 128, 2048
DAMAGE_C4_SET = (f"seed{DAMAGE_C4_SEED} excluding the documents of the {DAMAGE_CALIB_N}x{DAMAGE_CALIB_SEQLEN} calibration sets "
                 f"of seeds {','.join(map(str, DAMAGE_CALIB_SEEDS))}")
DAMAGE_WINDOWS = 32


def distortion(bits: float) -> float:
    """Relative MSE of a b-bit uniform quantizer under the 6.02 dB/bit law, normalized to 1 at 4 bits."""
    return 2.0 ** (-2.0 * (bits - 4.0))


def mean_bits(bits: dict, n: dict) -> float:
    """numel-weighted mean width of an allocation."""
    total = sum(n[u] for u in bits)
    return sum(n[u] * bits[u] for u in bits) / total if total else 0.0


def allocation_objective(bits: dict, d: dict, n: dict) -> float:
    """sum_u n_u d_u D(b_u), the A2 objective of an allocation."""
    return sum(n[u] * d[u] * distortion(bits[u]) for u in bits)


def allocate(d: dict, n: dict, target_mean_bits: float, levels: tuple[int, ...] = LEVELS) -> dict:
    """Exact allocation (A2): {unit: bits} minimizing sum_u n_u d_u D(b_u) with the numel-weighted mean <= target_mean_bits.

    d: damage per parameter of each unit (>= 0), n: numel of each unit (same keys). Multiple-choice knapsack DP: with
    g = gcd(numel) and w_u = n_u / g, the budget of extra bits above levels[0] is B = floor((target - levels[0]) * sum w_u);
    dp[j] = least objective over the units seen so far with extra weight <= j, updated per unit with
    dp'[j] = min_k dp[j - w_u (L_k - L_0)] + n_u d_u D(L_k). The choice table (len(units) x (B + 1), int8) gives the
    argmin back; ties go to the lower level of the unit considered later. Objective values are f64. Realized mean:
    mean_bits(result, n) <= target.
    """
    if set(d) != set(n):
        raise ValueError("d and n must have the same units")
    if not d:
        raise ValueError("no units to allocate")
    levels = tuple(sorted(int(b) for b in levels))
    if target_mean_bits < levels[0]:
        raise ValueError(f"target {target_mean_bits} below the lowest level {levels[0]}")
    units = list(d)
    for u in units:
        if d[u] < 0 or n[u] <= 0:
            raise ValueError(f"unit {u!r}: damage {d[u]} must be >= 0 and numel {n[u]} > 0")
        if int(n[u]) != n[u]:
            raise ValueError(f"unit {u!r}: numel {n[u]} must be an integer")
    g = math.gcd(*(int(n[u]) for u in units))
    w = [int(n[u]) // g for u in units]
    total_w = sum(w)
    budget = int(math.floor((target_mean_bits - levels[0]) * total_w + 1e-9))
    extra = [b - levels[0] for b in levels]  # extra bits per unit of weight at each level
    dist = np.array([distortion(b) for b in levels], dtype=np.float64)
    dp = np.zeros(budget + 1, dtype=np.float64)  # no units yet: objective 0 at every budget
    choice = np.zeros((len(units), budget + 1), dtype=np.int8)
    for i, u in enumerate(units):
        cost = float(n[u]) * float(d[u]) * dist  # [levels]
        cand = np.full((len(levels), budget + 1), np.inf, dtype=np.float64)
        for k in range(len(levels)):
            shift = w[i] * extra[k]
            if shift > budget:
                break
            cand[k, shift:] = dp[: budget + 1 - shift] + cost[k]
        k_best = np.argmin(cand, axis=0)  # the first (lowest) level on ties
        dp = cand[k_best, np.arange(budget + 1)]
        choice[i] = k_best
    bits: dict = {}
    j = budget
    for i in range(len(units) - 1, -1, -1):
        k = int(choice[i, j])
        bits[units[i]] = levels[k]
        j -= w[i] * extra[k]
    assert j >= 0
    return {u: bits[u] for u in units}


def _rebalance(new: dict, n: dict, target: float, tol: float, levels: tuple[int, ...], frozen: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """Single-level moves on the units not in `frozen`, each on the unit whose move leaves the mean closest to the
    target, until |mean - target| <= tol; `new` is updated in place. Returns the moves [(unit, from, to)]; raises
    ValueError when no move improves the mean."""
    units = list(new)
    total = float(sum(n[u] for u in units))
    moves: list[tuple[str, int, int]] = []
    while abs(mean_bits(new, n) - target) > tol:
        m = mean_bits(new, n)
        up = m < target
        best: tuple[float, float, str, int] | None = None
        for u in units:
            if u in frozen:
                continue
            i = levels.index(new[u])
            j = i + 1 if up else i - 1
            if not 0 <= j < len(levels):
                continue
            delta = n[u] * (levels[j] - levels[i]) / total
            cand = (abs(m + delta - target), abs(delta), u, levels[j])
            if best is None or cand < best:
                best = cand
        if best is None or best[0] >= abs(m - target):
            raise ValueError(f"cannot rebalance the shuffled allocation to {target:.4f} within {tol} (at {m:.4f})")
        moves.append((best[2], new[best[2]], best[3]))
        new[best[2]] = best[3]
    return moves


def shuffle_allocation_detail(bits: dict, n: dict, seed: int, tol: float = 0.02, levels: tuple[int, ...] = LEVELS,
                              max_draws: int = 200, max_real_share: float = SHUFFLE_MAX_REAL_SHARE) -> tuple[dict, dict]:
    """Shuffle control (DESIGN.md "Shuffle controls"): the widths of `bits` permuted across the units by a seeded
    derangement (no unit keeps its own width slot, so the largest unit always receives another unit's width), then
    rebalanced to the original numel-weighted mean within `tol` by single-level moves on the units other than the
    largest one, whose permuted width is frozen (with the earlier rule, "the move that lands closest to the target"
    over all units, the largest unit was walked back to its real width at every target on the 0.8B, so the control
    shared the real allocation's main decision by construction). Each move goes to the unit whose move leaves the
    mean closest to the target. Draws come from one generator until a derangement rebalances to a valid control: one
    that differs from `bits` and leaves at most `max_real_share` of the parameters at their real width (the rebalancer
    can otherwise walk a permutation back to the real allocation, as it did on the 2B type shape, where several types
    share one width). If none of `max_draws` does, the first derangement is rebalanced with no frozen unit and held to
    the same test; ValueError when that fails too, and when every unit has the same width ("no shuffle control
    possible": no permutation can act on an all-one-width allocation). A single-unit input is returned as is.
    Returns (control, info), info = {"raw_bits" (the permutation before rebalancing), "permutation" (unit -> unit
    whose width it took), "frozen_unit", "moves" [[unit, from, to]], "units_at_real_width",
    "params_share_at_real_width", "max_real_share", "draws", "rejected_draws" (rebalanced draws that were not a valid
    control)}."""
    levels = tuple(sorted(int(b) for b in levels))
    units = list(bits)
    vals = [bits[u] for u in units]
    target = mean_bits(bits, n)
    total = float(sum(n[u] for u in units))
    if len(units) < 2:
        return dict(bits), {"raw_bits": dict(bits), "permutation": {u: u for u in units}, "frozen_unit": None, "moves": [],
                            "units_at_real_width": list(units), "params_share_at_real_width": 1.0, "max_real_share": max_real_share,
                            "draws": 0, "rejected_draws": 0}
    if len(set(vals)) == 1:
        raise ValueError(f"no shuffle control possible: every unit of the real allocation has width {vals[0]} "
                         f"(a permutation of identical widths is the real allocation)")

    def real_share(new: dict) -> float:
        return sum(n[u] for u in units if new[u] == bits[u]) / total

    def valid(new: dict) -> bool:
        return new != bits and real_share(new) <= max_real_share

    largest = max(units, key=lambda u: n[u])
    gen = torch.Generator().manual_seed(int(seed))
    first: tuple[dict, list[int]] | None = None
    result: tuple[dict, dict, list[int], str | None, list] | None = None
    draws = rejected = 0
    while draws < max_draws and result is None:
        perm = torch.randperm(len(units), generator=gen).tolist()
        draws += 1
        if any(p == i for i, p in enumerate(perm)):
            continue  # not a derangement: some unit kept its own width slot
        raw = {u: vals[p] for u, p in zip(units, perm)}
        if first is None:
            first = (raw, perm)
        new = dict(raw)
        try:
            moves = _rebalance(new, n, target, tol, levels, frozen=(largest,))
        except ValueError:
            continue
        if not valid(new):
            rejected += 1  # the rebalancer walked the permutation back to (or next to) the real allocation
            continue
        result = (new, raw, perm, largest, moves)
    if result is None:
        if first is None:
            raise ValueError(f"no derangement of {len(units)} units in {max_draws} draws")
        raw, perm = first
        new = dict(raw)
        moves = _rebalance(new, n, target, tol, levels, frozen=())  # raises when even this cannot rebalance
        if not valid(new):
            raise ValueError(f"no valid shuffle control in {draws} draws (seed {seed}): every rebalanced derangement equals the real "
                             f"allocation or leaves more than {max_real_share:.2f} of the parameters at their real width "
                             f"({rejected} such draws)")
        result = (new, raw, perm, None, moves)
    new, raw, perm, frozen, moves = result
    same = [u for u in units if new[u] == bits[u]]
    info = {"raw_bits": raw, "permutation": {u: units[p] for u, p in zip(units, perm)}, "frozen_unit": frozen,
            "moves": [list(mv) for mv in moves], "units_at_real_width": same,
            "params_share_at_real_width": sum(n[u] for u in same) / total, "max_real_share": max_real_share,
            "draws": draws, "rejected_draws": rejected}
    return new, info


def shuffle_allocation(bits: dict, n: dict, seed: int, tol: float = 0.02, levels: tuple[int, ...] = LEVELS) -> dict:
    """shuffle_allocation_detail()[0]: the shuffled, rebalanced control allocation."""
    return shuffle_allocation_detail(bits, n, seed, tol, levels)[0]


def row_allocate(row_damage: Tensor, bits: int, span: int = ROW_SPAN, lo_min: int = LEVELS[0], hi_max: int = LEVELS[-1]) -> Tensor:
    """Per-row widths in {max(bits - span, lo_min), ..., min(bits + span, hi_max)} at a per-tensor mean of exactly `bits`
    (method G).

    row_damage [N] >= 0 (the Fisher row sum r_i; d_i = r_i / K, the constant K does not change the ranking). Rows
    have equal numel and the widths step by one bit, so the exact optimum (allocate()) is the pooled top-k: each row
    offers hi - lo one-level raises above lo with gains d_i (D(b) - D(b+1)) (decreasing along the levels, so a later
    raise is never taken before an earlier one of the same row), and the N (bits - lo) raises with the largest gains
    are taken. The clamp keeps every row inside [lo_min, hi_max] = LEVELS: a tensor at the lowest level has no room
    below it and a tensor at the highest none above, so both keep all their rows at `bits` (no split; the caller then
    charges no row side information). Returns an int64 CPU tensor [N].
    """
    d = row_damage.detach().to("cpu", torch.float64).reshape(-1)
    N = d.numel()
    if N == 0:
        raise ValueError("empty row damage")
    if bool((d < 0).any()):
        raise ValueError("row damage must be >= 0")
    if span < 0:
        raise ValueError(f"span {span} < 0")
    if not lo_min <= bits <= hi_max:
        raise ValueError(f"bits {bits} outside the row width range {lo_min}..{hi_max}")
    lo, hi = max(bits - span, lo_min), min(bits + span, hi_max)
    steps = hi - lo
    if steps == 0:
        return torch.full((N,), int(bits), dtype=torch.int64)
    gains = torch.stack([d * (distortion(lo + k) - distortion(lo + k + 1)) for k in range(steps)], dim=1)  # [N, steps]
    budget = N * (bits - lo)  # raises that keep the mean at `bits`
    order = torch.argsort(gains.reshape(-1), descending=True, stable=True)
    taken = torch.zeros(N * steps, dtype=torch.bool)
    taken[order[:budget]] = True
    raises = taken.reshape(N, steps).sum(1)
    return (lo + raises).to(torch.int64)


def shuffle_rows(row_bits: Tensor, gen: torch.Generator) -> Tensor:
    """Row-shuffle control: the same multiset of widths assigned to random rows (mean unchanged exactly)."""
    rb = row_bits.reshape(-1)
    return rb[torch.randperm(rb.numel(), generator=gen)]


def row_bits_summary(row_bits: Tensor) -> dict[str, int]:
    """{"3": n_rows_at_3, ...} of a row_bits tensor (for the allocation JSON)."""
    vals, counts = torch.unique(row_bits.to("cpu").long(), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def rel_path(path: Path) -> str:
    """`path` relative to the project root (posix) when it lies under it, else as given (tests use temp dirs)."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


# ---------------------------------------------------------------- allocation inputs


def file_sha(path: Path) -> str:
    """First 10 hex digits of the sha256 of a file's bytes (the provenance of an input file in notes / JSON)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:10]


def is_smoke_path(path: Path) -> bool:
    """True when a "smoke" component is in the path (the rule of scripts/run_baselines.py and damage_c4.py for
    off-protocol output)."""
    return "smoke" in Path(path).resolve().parts


def damage_path(tag: str, out_dir: Path | str | None = None) -> Path:
    """<out_dir>/damage_c4.jsonl when `out_dir` (the runner's output directory) is given and holds one, else the
    protocol file results/phase3/<tag>/damage_c4.jsonl: a smoke run with its own damage file reads that one, a
    protocol run (out_dir == results/phase3/<tag>) the protocol file."""
    if out_dir is not None and (Path(out_dir) / "damage_c4.jsonl").exists():
        return Path(out_dir) / "damage_c4.jsonl"
    return PHASE3_DIR / tag / "damage_c4.jsonl"


def check_damage_rows(rows: list[dict], path: Path) -> None:
    """Refuse damage rows not measured on DAMAGE_C4_SET (a pre-exclusion file has no "c4_set") and, for a file
    outside a smoke tree, not measured on DAMAGE_WINDOWS windows (a smoke damage file in the protocol tree)."""
    for row in rows:
        if row.get("c4_set") != DAMAGE_C4_SET:
            raise ValueError(f"{path}: row {row.get('unit')} {row.get('name')!r} was measured on C4 set {row.get('c4_set')!r}, "
                             f"not {DAMAGE_C4_SET!r}: move the file aside and re-measure it with scripts/damage_c4.py")
        if not is_smoke_path(path) and row.get("windows") != DAMAGE_WINDOWS:
            raise ValueError(f"{path}: row {row.get('unit')} {row.get('name')!r} was measured on {row.get('windows')!r} windows, the "
                             f"protocol needs {DAMAGE_WINDOWS}: a smoke damage file cannot feed a protocol allocation")


def read_damage_file(path: Path) -> list[dict]:
    """All rows of a damage_c4.jsonl, checked by check_damage_rows; raises when the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/damage_c4.py first")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    check_damage_rows(rows, path)
    return rows


def damage_meta(path: Path) -> dict:
    """{"path", "sha256", "c4_set", "windows"} of a (checked) damage file, for the alloc JSON and the row note."""
    rows = read_damage_file(path)
    return {"path": rel_path(path), "sha256": file_sha(path), "c4_set": DAMAGE_C4_SET,
            "windows": sorted({int(r["windows"]) for r in rows if r.get("windows") is not None})}


def load_damage(tag: str, unit: str, out_dir: Path | str | None = None) -> dict[str, dict]:
    """{name: row} of the damage_c4.jsonl rows (damage_path(tag, out_dir)) with the given unit kind ("type" /
    "tensor"); the last row per name wins. Empty when the file has no such rows; raises when the file is missing or
    holds rows of another C4 set / window count (check_damage_rows)."""
    out: dict[str, dict] = {}
    for row in read_damage_file(damage_path(tag, out_dir)):
        if row.get("unit") == unit:
            out[row["name"]] = row
    return out


def damage_per_param(rows: dict[str, dict]) -> dict[str, float]:
    """d_u = KL_u / n_u (nats per parameter) per unit; a negative KL (noise) is floored at 0."""
    return {name: max(float(r["kl"]), 0.0) / float(r["numel"]) for name, r in rows.items()}


def fisher_path(tag: str) -> Path:
    """results/probe/<model>/fisher.json (P6b, the 0.8B) when it exists, else fisher_rows.pt (scripts/fisher_rows.py, whose
    "totals" are the same per-tensor sums; the 2B's are measured with 512-token windows, DESIGN.md A2)."""
    path = PROBE_DIR / tag / "fisher.json"
    return path if path.exists() else fisher_rows_path(tag)


def load_fisher_totals(tag: str) -> dict[str, float]:
    """{tensor: Fisher total} from fisher.json (P6b) or, without one, from fisher_rows.pt (fisher_path)."""
    path = fisher_path(tag)
    if not path.exists():
        raise FileNotFoundError(f"neither {PROBE_DIR / tag / 'fisher.json'} (P6b) nor {path} (scripts/fisher_rows.py) exists")
    if path.suffix == ".pt":
        return {k: float(v) for k, v in torch.load(path)["totals"].items()}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {r["tensor"]: float(r["stat"]["total"]) for r in data["records"]}


def fisher_rows_path(tag: str) -> Path:
    return PROBE_DIR / tag / "fisher_rows.pt"


def load_fisher_rows(tag: str) -> dict[str, Tensor]:
    """{tensor: Fisher row sums [N] f32} written by scripts/fisher_rows.py (results/probe/<model>/fisher_rows.pt)."""
    path = fisher_rows_path(tag)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/fisher_rows.py --model {tag} first")
    data = torch.load(path)
    return {k: v.float() for k, v in data["rows"].items()}


def fisher_rows_meta(tag: str) -> dict:
    """The measurement record of fisher_rows.pt ({"calib_seed", "n_windows", "seqlen", "n_tokens"} when present): one
    file per model, so the row signal is shared by every run seed (the damage set is likewise fixed at seed 2)."""
    data = torch.load(fisher_rows_path(tag))
    return {k: data[k] for k in ("calib_seed", "n_windows", "seqlen", "n_tokens") if k in data}
