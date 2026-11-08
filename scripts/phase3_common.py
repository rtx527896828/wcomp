"""Shared readers and the reference-curve arithmetic of the Phase 3 presentation (scripts/plot_rd3.py,
scripts/phase3_summary.py), after DESIGN.md "Phase 3 contract: methods", rules.

Reference curve: the gptq32-sym-g128-{4,3,2} rows of a model, seed-averaged per bits_blocks (arithmetic mean of KL,
PPL and bits_blocks_huff over the seeds at that width). Between the points log(KL) and log(PPL - PPL_fp16) are linear
in bits_blocks (interp_log); outside the measured range the end segment is extended and the result is flagged
"extrapolated". A method beats the baseline at its bits when its KL is below the curve by more than KL_NOISE (5%)
and its dPPL below by more than PPL_NOISE (4.4%), on every seed of the method. Bits saved = the curve's bits at the
method's KL (invert_log) minus the method's bits: against the fixed-bits curve at bits_blocks, and against the
Huffman curve (the same KLs at the baseline's own bits_blocks_huff) at the method's bits_blocks_huff, so the coding
gain is never counted twice.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_rd  # noqa: E402  (load_runs: latest row per (method, seed, calib_seed, smoke, hessian), smoke rows dropped)

KL_NOISE = 0.05  # measured calibration-seed spread, KL (NOTES.md "GPTQ speed acceptance")
PPL_NOISE = 0.044  # measured calibration-seed spread, PPL; DESIGN.md applies it to dPPL = PPL - PPL_fp16 (see DPPL_RULE_NOTE)
EXTRAP_TOL = 0.01  # bits: a row this close beyond the measured curve is not flagged as extrapolated (G's 2 bits/row land at 4.1266)
# The 4.4% margin was measured as a spread of PPL; applied to dPPL it is inside the reference rows' own dPPL seed spread
# (printed under the reference table). It still binds: on the Phase 3 data several rows clear the KL margin on every seed
# and fail on dPPL alone (the summary marks them), so the verdict is not carried by the KL margin.
DPPL_RULE_NOTE = ("the dPPL margin is DESIGN.md's 4.4% applied to dPPL = PPL - PPL_fp16 as written; 4.4% was measured as a "
                  "spread of PPL and is inside the dPPL seed spread of the reference rows (see the seed spread lines), but it "
                  "binds: a row whose KL ratio clears its margin and whose dPPL ratio does not is not beating the curve "
                  "(\"dPPL decides\" in the rows table)")
REF_RE = re.compile(r"^gptq32-sym-g128-(\d+)$")
FAMILIES = ("reference", "rotation", "codebook", "alloc", "rowalloc", "combo", "other")
ROTATION_KINDS = ("had", "rnd", "hadsign", "eig")
CODEBOOK_KINDS = ("lloyd", "nf", "nfq")


def load_rows(path: Path, include_smoke: bool = False) -> list[dict]:
    """Latest full-protocol row per run key of a runs.jsonl file; [] (reported) when the file is missing."""
    if not path.exists():
        print(f"note: {path.as_posix()} missing", file=sys.stderr)
        return []
    return plot_rd.load_runs(path, include_smoke)


def fp16_ppl(*row_sets: list[dict], include_smoke: bool = False) -> float | None:
    """The full-protocol fp16 PPL from the first row set that has one (Phase 3 file first, then Phase 1); with
    include_smoke a smoke fp16 row counts too (the row sets then hold smoke rows evaluated on the same windows)."""
    for rows in row_sets:
        for r in rows:
            if r["method"] == "fp16" and r.get("ppl") is not None and (include_smoke or plot_rd.eval_tag(r.get("note", "")) == ""):
                return float(r["ppl"])
    return None


def parse_name(method: str) -> dict:
    """Family, kind, width and label of a Phase 3 method name.

    "gptq32-sym-g128-3" -> reference, label "3"; "had-gptq32-sym-g128-3" -> rotation / had, "had 3";
    "lloyd-gptq32-g128-4" -> codebook / lloyd, "lloyd 4" (so "nf" and "nfq"); "alloc-type-gptq32-avg2.5" -> alloc / type, avg 2.5,
    "type 2.5"; "alloc-type-shuf1-gptq32-avg3" -> alloc / shuf1 (shuffle), "shuf1 3"; "alloc-type-rtn-avg3" ->
    alloc / type, quantizer rtn, "type rtn 3"; "rowalloc-fisher-rtn-avg4" -> rowalloc / fisher, "fisher rtn 4";
    "combo-gptq32-avg3" -> combo, "avg 3". Anything else is family "other" with the name as label.
    """
    toks = method.split("-")
    bits = int(toks[-1]) if toks[-1].isdigit() else None
    m = re.search(r"avg([0-9.]+)$", method)
    avg = float(m.group(1)) if m else None
    quant = "rtn" if "rtn" in toks else "gptq32" if "gptq32" in toks else "gptq" if "gptq" in toks else None
    width = f"{bits}" if bits is not None else (f"{avg:g}" if avg is not None else "")
    out = {"method": method, "family": "other", "kind": "", "bits": bits, "avg": avg, "quant": quant,
           "shuffle": "shuf" in method, "label": method}
    if REF_RE.match(method):
        out.update(family="reference", kind="gptq32", label=width)
    elif toks[0] in ROTATION_KINDS and quant == "gptq32":
        out.update(family="rotation", kind=toks[0], label=f"{toks[0]} {width}")
    elif toks[0] in CODEBOOK_KINDS and quant == "gptq32":
        out.update(family="codebook", kind=toks[0], label=f"{toks[0]} {width}")
    elif toks[0] == "alloc" and len(toks) >= 3:
        kind = toks[2] if toks[2].startswith("shuf") else toks[1]
        q = " rtn" if quant == "rtn" else ""
        out.update(family="alloc", kind=kind, label=f"{kind}{q} {width}")
    elif toks[0] == "rowalloc" and len(toks) >= 3:
        q = " rtn" if quant == "rtn" else ""
        out.update(family="rowalloc", kind=toks[1], label=f"{toks[1]}{q} {width}")
    elif toks[0] == "combo":
        out.update(family="combo", kind="combo", label=f"avg {width}")
    return out


def row_label(r: dict) -> str:
    """parse_name label plus " s<seed>" for a non-default seed."""
    lab = parse_name(r["method"])["label"]
    if r.get("seed", 0) != 0:
        lab += f" s{r['seed']}"
    return lab


def dppl(r: dict, fp16: float) -> float:
    return float(r["ppl"]) - fp16


# ---------------------------------------------------------------- interpolation


def _segment(xs: list[float], x: float) -> int:
    """Index i of the segment [xs[i], xs[i+1]] used at x: the enclosing one inside the range, the end segment outside."""
    n = len(xs)
    if n < 2:
        raise ValueError("a curve needs at least two points")
    if x <= xs[0]:
        return 0
    if x >= xs[-1]:
        return n - 2
    for i in range(n - 1):
        if xs[i] <= x <= xs[i + 1]:
            return i
    raise ValueError("xs must be sorted increasing")


def interp_log(xs: list[float], ys: list[float], x: float) -> float:
    """y at x with log(y) linear in x between the points (xs increasing, ys > 0). A point's own x returns its y
    exactly; outside [xs[0], xs[-1]] the end segment is extended (the caller flags that with in_range)."""
    for xi, yi in zip(xs, ys):
        if x == xi:
            return yi
    if len(xs) == 1:
        raise ValueError(f"single-point curve: x {x} is not the point {xs[0]}")
    i = _segment(xs, x)
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return math.exp(math.log(ys[i]) + t * (math.log(ys[i + 1]) - math.log(ys[i])))


def invert_log(xs: list[float], ys: list[float], y: float) -> float:
    """x at which interp_log(xs, ys, x) == y, for ys strictly monotone (decreasing with x on an RD curve). A point's
    own y returns its x exactly; outside [min(ys), max(ys)] the end segment is extended."""
    for xi, yi in zip(xs, ys):
        if y == yi:
            return xi
    if len(xs) < 2:
        raise ValueError("a curve needs at least two points")
    logs = [math.log(v) for v in ys]
    ly = math.log(y)
    decreasing = logs[0] > logs[-1]
    n = len(xs)
    i = None
    for k in range(n - 1):
        lo, hi = sorted((logs[k], logs[k + 1]))
        if lo <= ly <= hi:
            i = k
            break
    if i is None:  # outside the measured KL range: the end segment on that side
        i = (0 if ly > logs[0] else n - 2) if decreasing else (0 if ly < logs[0] else n - 2)
    t = (ly - logs[i]) / (logs[i + 1] - logs[i])
    return xs[i] + t * (xs[i + 1] - xs[i])


def in_range(xs: list[float], x: float, tol: float = 0.0) -> bool:
    """True when x lies within [xs[0] - tol, xs[-1] + tol]."""
    return xs[0] - tol <= x <= xs[-1] + tol


def far_out_of_range(xs: list[float], x: float) -> bool:
    """True when x lies outside the measured range by more than one end segment (the extended end segment is then
    meaningless, e.g. the eig row at ~17.9 bits against a 2.1-4.1 bit curve, where the log-linear KL is ~1e-12)."""
    if len(xs) < 2:
        return not in_range(xs, x)
    return x < xs[0] - (xs[1] - xs[0]) or x > xs[-1] + (xs[-1] - xs[-2])


# ---------------------------------------------------------------- reference curve


class Curve:
    """Seed-mean reference points sorted by bits_blocks, with the individual seed rows."""

    def __init__(self, points: list[dict], seed_rows: list[dict]) -> None:
        self.points = sorted(points, key=lambda p: p["bits"])
        self.seed_rows = seed_rows

    @property
    def bits(self) -> list[float]:
        return [p["bits"] for p in self.points]

    @property
    def bits_huff(self) -> list[float] | None:
        hs = [p["bits_huff"] for p in self.points]
        return None if any(h is None for h in hs) else hs

    @property
    def kls(self) -> list[float]:
        return [p["kl"] for p in self.points]

    @property
    def dppls(self) -> list[float]:
        return [p["dppl"] for p in self.points]

    def usable(self) -> bool:
        return len(self.points) >= 2 and all(p["kl"] > 0 and p["dppl"] > 0 for p in self.points)

    def kl_at(self, bits: float) -> float | None:
        if not self.usable():
            return self.points[0]["kl"] if len(self.points) == 1 and bits == self.points[0]["bits"] else None
        return interp_log(self.bits, self.kls, bits)

    def dppl_at(self, bits: float) -> float | None:
        if not self.usable():
            return self.points[0]["dppl"] if len(self.points) == 1 and bits == self.points[0]["bits"] else None
        return interp_log(self.bits, self.dppls, bits)

    def bits_at_kl(self, kl: float, huff: bool = False) -> float | None:
        """The curve's bits at KL `kl`: bits_blocks, or (huff) the baseline's own bits_blocks_huff."""
        if not self.usable() or kl is None or kl <= 0:
            return None
        xs = self.bits_huff if huff else self.bits
        if xs is None:
            return None
        return invert_log(xs, self.kls, kl)


def reference_curve(rows: list[dict], fp16: float) -> Curve | None:
    """Seed-mean curve of the gptq32-sym-g128-* rows (None when there are none)."""
    ref = [r for r in rows if REF_RE.match(r["method"]) and r.get("kl") is not None]
    if not ref:
        return None
    by_bits: dict[float, list[dict]] = {}
    for r in ref:
        by_bits.setdefault(round(float(r["bits_blocks"]), 6), []).append(r)
    points = []
    for b, rs in by_bits.items():
        huffs = [r.get("bits_blocks_huff") for r in rs]
        points.append({
            "bits": b,
            "bits_huff": sum(huffs) / len(huffs) if all(h is not None for h in huffs) else None,
            "bits_shannon": (sum(r["bits_blocks_shannon"] for r in rs) / len(rs)
                             if all(r.get("bits_blocks_shannon") is not None for r in rs) else None),
            "kl": sum(float(r["kl"]) for r in rs) / len(rs),
            "ppl": sum(float(r["ppl"]) for r in rs) / len(rs),
            "dppl": sum(float(r["ppl"]) for r in rs) / len(rs) - fp16,
            "n_seeds": len(rs),
            "seeds": sorted(r.get("seed", 0) for r in rs),
        })
    return Curve(points, ref)


# ---------------------------------------------------------------- comparison against the curve


def compare(r: dict, curve: Curve | None, fp16: float) -> dict:
    """One row against the reference curve at the row's own bits_blocks: kl_ref, ratio_kl, dppl_ref, ratio_dppl,
    beats_row (None when the curve cannot be read), saved_fixed / saved_huff (bits), extrapolated flags."""
    out = {"kl_ref": None, "ratio_kl": None, "dppl_ref": None, "ratio_dppl": None, "beats_row": None,
           "saved_fixed": None, "saved_huff": None, "extrapolated": False, "saved_extrapolated": False}
    if curve is None or r.get("kl") is None or r.get("bits_blocks") is None:
        return out
    bits = float(r["bits_blocks"])
    kl_ref, dppl_ref = curve.kl_at(bits), curve.dppl_at(bits)
    if kl_ref is None or dppl_ref is None:
        return out
    d = dppl(r, fp16)
    out.update(kl_ref=kl_ref, dppl_ref=dppl_ref, ratio_kl=float(r["kl"]) / kl_ref if kl_ref > 0 else None,
               ratio_dppl=d / dppl_ref if dppl_ref > 0 else None,
               extrapolated=curve.usable() and not in_range(curve.bits, bits, EXTRAP_TOL),
               far_extrapolated=curve.usable() and far_out_of_range(curve.bits, bits))
    if out["ratio_kl"] is not None and out["ratio_dppl"] is not None:
        out["beats_row"] = out["ratio_kl"] < 1 - KL_NOISE and out["ratio_dppl"] < 1 - PPL_NOISE
    kl = float(r["kl"])
    if kl > 0 and curve.usable():
        b_fixed = curve.bits_at_kl(kl)
        out["saved_fixed"] = b_fixed - bits if b_fixed is not None else None
        out["saved_extrapolated"] = not (min(curve.kls) <= kl <= max(curve.kls))
        if r.get("bits_blocks_huff") is not None:
            b_huff = curve.bits_at_kl(kl, huff=True)
            out["saved_huff"] = b_huff - float(r["bits_blocks_huff"]) if b_huff is not None else None
    return out


def kl_gate(cmps: list[dict], margin: float = KL_NOISE) -> str:
    """KL-only gate over the seeds of one method: "passed" when every seed's KL is below the curve by more than `margin`
    (ratio_kl < 1 - margin), "failed" when any seed is not, "incomplete" without a readable ratio. The contract's lloyd
    component rule for the combination ("beats the uniform grid at 3 bits by > 5% KL on both seeds") is this gate, not
    the two-sided beats verdict."""
    ratios = [c["ratio_kl"] for c in cmps if c["ratio_kl"] is not None]
    if not ratios:
        return "incomplete"
    return "passed" if all(r < 1 - margin for r in ratios) else "failed"


def method_verdict(cmps: list[dict], family: str = "") -> str:
    """"beats" per the contract rule over the seeds of one method: "yes" (every seed, two or more), "yes (1 seed)" when
    only one seed row exists, "no", "no (1 of 2 seeds)" when the seeds disagree, "n/a" without a readable curve,
    "n/a (k of n seeds readable)" when some seed row has no readable comparison, "ref" for the reference."""
    if family == "reference":
        return "ref"
    flags = [c["beats_row"] for c in cmps if c["beats_row"] is not None]
    if not flags:
        return "n/a"
    if len(flags) < len(cmps):
        return f"n/a ({len(flags)} of {len(cmps)} seeds readable)"
    if all(flags):
        return "yes" if len(flags) >= 2 else "yes (1 seed)"
    if any(flags):
        return f"no ({sum(flags)} of {len(flags)} seeds)"
    return "no"


def by_method(rows: list[dict]) -> dict[str, list[dict]]:
    """{method: rows sorted by seed}."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["method"], []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: (r.get("seed", 0), r.get("calib_seed") or 0))
    return out


KIND_ORDER = ("gptq32", "had", "rnd", "hadsign", "eig", "lloyd", "nf", "nfq", "type", "tensor", "shuf1", "shuf2", "fisher", "combo")


def sort_key(r: dict) -> tuple:
    """Family order, then kind (KIND_ORDER, unknown kinds last), quantizer, width (bits or avg target, descending), seed."""
    p = parse_name(r["method"])
    width = p["bits"] if p["bits"] is not None else (p["avg"] if p["avg"] is not None else 0)
    kind = KIND_ORDER.index(p["kind"]) if p["kind"] in KIND_ORDER else len(KIND_ORDER)
    return (FAMILIES.index(p["family"]), kind, p["kind"], p["quant"] or "", -width, r.get("seed", 0))


def find(rows: list[dict], method: str, seed: int | None = None) -> list[dict]:
    return [r for r in rows if r["method"] == method and (seed is None or r.get("seed", 0) == seed)]


def spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation (average ranks on ties); None for fewer than 3 pairs."""
    n = len(a)
    if n < 3 or n != len(b):
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return rk

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else None
