"""Phase 3 summary table and hypothesis numbers (DESIGN.md "Phase 3 contract": results/phase3/summary.md).

  uv run python scripts/phase3_summary.py --models qwen3.5-0.8b,qwen3.5-2b,minicpm5-2b

Per model: the fp16 PPL and the reference curve points; a table of every Phase 3 row (method, seed, bits_blocks,
bits_huff, bits_shannon, ppl, dPPL %, kl, top1, the KL of the interpolated reference curve at the row's fixed bits,
the ratio, the "beats" verdict per the contract rule over the seeds of the method, bits saved at equal KL against
the fixed-bits curve and against the Huffman curve, then the dPPL side of the verdict: the curve's dPPL at the row's bits,
the dPPL ratio and "dPPL decides" where the KL margin is cleared and the dPPL margin is not, listed under the table);
a hypotheses block that computes the numbers each H3-* needs
(A: ratio vs curve at avg 3 / 3.5 and vs the shuffles, tensor vs type, Spearman of the C4 damage against the Phase 2
WikiText @only KL; D: had vs curve per bits, rnd / hadsign vs had, eig; E: Huffman savings and the Shannon gap per
bits, lloyd ratio, nf share; G: the RTN gate numbers and the GPTQ rows; combo: bits saved vs gptq32-4 and the dPPL
at avg 2.5); the combination decision (combo.json) when written; and the RD-bound table (E4) from rd_bound.jsonl,
per weight type, median over tensors of "bits above the Gaussian bound" at the three bit figures.

Readers tolerate missing files and rows: every number a hypothesis needs that is not there is printed as "n/a
(missing: <method names>)", never guessed. Smoke rows are ignored (scripts/plot_rd.load_runs rules).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import phase3_common as p3  # noqa: E402
from wcomp import alloc as alloc_mod  # noqa: E402  (read_damage_file: the same checks the allocation methods apply)

TYPE_ORDER = ["q", "k", "v", "o", "gate", "up", "down", "in_proj_qkv", "in_proj_z", "in_proj_ab", "out_proj"]


def fmt(x, nd: int = 4, pct: bool = False, signed: bool = False) -> str:
    """Number formatting; None / NaN -> "—"."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    if pct:
        return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"
    if isinstance(x, int):
        return str(x)
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def pct_below(ratio: float | None) -> str:
    """ratio 0.8 -> "-20.0%" (below the reference)."""
    return "—" if ratio is None else f"{(ratio - 1) * 100:+.1f}%"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [f"| {' | '.join(header)} |", f"|{'|'.join('---' for _ in header)}|"] + [f"| {' | '.join(r)} |" for r in rows]


def seed_str(r: dict) -> str:
    s = str(r.get("seed", 0))
    if r.get("calib_seed") not in (None, r.get("seed", 0)):
        s += f" (calib {r['calib_seed']})"
    return s


# ---------------------------------------------------------------- the rows table


def seed_spread_lines(curve: p3.Curve, fp16: float) -> list[str]:
    """Per reference width with two or more seeds: max / min - 1 of KL, PPL and dPPL over the seed rows, so the two
    margins of the beats rule (KL 5%, dPPL 4.4%) can be read against the measured spread of the quantity they apply to."""
    out = []
    for pnt in curve.points:
        rs = [r for r in curve.seed_rows if round(float(r["bits_blocks"]), 6) == pnt["bits"] and r.get("kl") is not None and r.get("ppl") is not None]
        if len(rs) < 2:
            continue

        def spread(vals: list[float]) -> str:
            lo, hi = min(vals), max(vals)
            return fmt(hi / lo - 1, pct=True) if lo > 0 else "—"

        out.append(f"seed spread at {fmt(pnt['bits'])} bits (seeds {','.join(map(str, pnt['seeds']))}, max / min - 1): "
                   f"KL {spread([float(r['kl']) for r in rs])}, PPL {spread([float(r['ppl']) for r in rs])}, "
                   f"dPPL {spread([p3.dppl(r, fp16) for r in rs])}")
    if out:
        out.append(f"({p3.DPPL_RULE_NOTE})")
    return out


def dppl_decides(c: dict) -> bool:
    """True when a row clears the KL margin but not the dPPL margin, so the dPPL side alone makes it "not beating"."""
    return (c["ratio_kl"] is not None and c["ratio_dppl"] is not None
            and c["ratio_kl"] < 1 - p3.KL_NOISE and not c["ratio_dppl"] < 1 - p3.PPL_NOISE)


def rows_table(rows: list[dict], curve: p3.Curve | None, fp16: float) -> tuple[list[str], dict[str, list[dict]]]:
    """Markdown lines of the per-row table and {method: [compare dicts]} for the hypotheses."""
    # the first 14 columns keep their positions (tests index them); the dPPL side of the verdict follows
    header = ["method", "seed", "bits_blocks", "bits_huff", "bits_shannon", "ppl", "dPPL %", "kl", "top1",
              "KL curve @bits", "KL ratio", "beats", "saved fixed", "saved huff", "dPPL curve @bits", "dPPL ratio", "dPPL decides"]
    groups = p3.by_method([r for r in rows if r["method"] not in ("fp16", "had-fp16")])
    cmps: dict[str, list[dict]] = {m: [p3.compare(r, curve, fp16) for r in rs] for m, rs in groups.items()}
    # RTN-quantized rows are not compared with the gptq32 curve (always "no" there, and not a contract comparison): their
    # gates are the uniform RTN curve (H3-A) and rtn-sym-g128-4 with the shuffled-row controls (H3-G).
    verdicts = {m: ("rtn gate" if p3.parse_name(m)["quant"] == "rtn" else p3.method_verdict(cmps[m], p3.parse_name(m)["family"]))
                for m in groups}
    out = []
    decided: list[str] = []
    for m in sorted(groups, key=lambda m: p3.sort_key(groups[m][0])):
        is_ref = p3.parse_name(m)["family"] == "reference"
        for r, c in zip(groups[m], cmps[m]):
            flag = "*" if c["extrapolated"] else ""
            sflag = "*" if c["saved_extrapolated"] else ""
            far = c.get("far_extrapolated", False)  # more than one segment outside the measured bits: no curve value / ratio
            # a reference seed row against its own seed-mean curve is the seed spread, not a result: no ratio / saved
            cmp_cells = (["—", "—", "—"] if is_ref else
                         ["—" if far else fmt(c["ratio_kl"], 3), fmt(c["saved_fixed"], 3, signed=True) + sflag,
                          fmt(c["saved_huff"], 3, signed=True) + sflag])
            decides = dppl_decides(c) if not is_ref and verdicts[m] != "rtn gate" and not far else None
            if decides:
                decided.append(f"{m} seed {r.get('seed', 0)} (KL ratio {fmt(c['ratio_kl'], 3)}, dPPL ratio {fmt(c['ratio_dppl'], 3)})")
            dppl_cells = (["—", "—", "—"] if is_ref or far else
                          [fmt(c["dppl_ref"], 3) + flag, fmt(c["ratio_dppl"], 3), "yes" if decides else ""])
            out.append([m, seed_str(r), fmt(r.get("bits_blocks")), fmt(r.get("bits_blocks_huff")), fmt(r.get("bits_blocks_shannon")),
                        fmt(r.get("ppl"), 2), fmt(p3.dppl(r, fp16) / fp16, pct=True, signed=True) if r.get("ppl") is not None else "—",
                        fmt(r.get("kl"), 5), fmt(r.get("top1_agree")), "—*" if far else fmt(c["kl_ref"], 5) + flag, cmp_cells[0],
                        verdicts[m], cmp_cells[1], cmp_cells[2]] + dppl_cells)
    lines = table(header, out)
    lines.append("")
    lines.append("Rows whose KL clears its margin but whose dPPL does not (the dPPL margin decides the row): "
                 + ("; ".join(decided) if decided else "none") + ".")
    lines.append("")
    lines.append(f"\"KL curve @bits\" / \"dPPL curve @bits\": the seed-mean gptq32-sym-g128 curve (dPPL = PPL - fp16 PPL, absolute), "
                 f"log-linear in bits_blocks, read at the row's fixed bits; KL ratio = kl / that, dPPL ratio = dPPL / that; "
                 f"\"dPPL decides\" = yes when the KL ratio is below {1 - p3.KL_NOISE:.2f} and the dPPL ratio is not below {1 - p3.PPL_NOISE:.3f}; "
                 f"beats = KL below the curve by > {p3.KL_NOISE:.0%} and dPPL below by > {p3.PPL_NOISE:.1%} on every "
                 f"seed of the method (\"yes (1 seed)\" when only one exists; \"n/a (k of n seeds readable)\" when a seed row has no readable "
                 f"comparison; \"rtn gate\" for RTN-quantized rows, which are judged against the uniform RTN curve in H3-A and against "
                 f"rtn-sym-g128-4 with the shuffled-row controls in H3-G, not against the gptq32 curve); {p3.DPPL_RULE_NOTE}; "
                 f"saved = the curve's bits at the row's KL minus the row's bits, "
                 f"fixed against bits_blocks and huff against the baseline's own bits_blocks_huff; * = more than {p3.EXTRAP_TOL} bits outside the measured curve "
                 f"(end segment extended; —* = more than one segment outside, no curve value or ratio is printed). Reference rows (\"ref\") get no ratio / saved: against their own seed-mean curve those "
                 f"would be the seed spread. bits_shannon = the entropy bound of the coder chosen per tensor (order-0 entropy where "
                 f"the single-symbol or the plain code won, pair-stream entropy per symbol where the pair code won; wcomp.coding), so huff - "
                 f"shannon is the coder's own loss plus its tables, not the gap to the order-0 entropy of P1 (that gap is in H3-E as huff - entropy0).")
    return lines, cmps


# ---------------------------------------------------------------- hypotheses


def _missing(names: list[str]) -> str:
    return f"n/a (missing: {', '.join(names)})"


def _ratio_lines(rows: list[dict], cmps: dict[str, list[dict]], method: str, what: str) -> list[str]:
    """One line per seed of `method`: KL, curve KL, ratio."""
    out = []
    for r, c in zip(rows, cmps.get(method, [])):
        out.append(f"  - {what} seed {r.get('seed', 0)}: bits {fmt(r['bits_blocks'])}, KL {fmt(r.get('kl'), 5)} vs curve "
                   f"{fmt(c['kl_ref'], 5)} -> ratio {fmt(c['ratio_kl'], 3)} ({pct_below(c['ratio_kl'])}), dPPL ratio "
                   f"{fmt(c['ratio_dppl'], 3)}, beats row: {c['beats_row']}")
    return out


def _kl_ratio(a: list[dict], b: list[dict]) -> float | None:
    """Mean KL of a / mean KL of b (None when either is empty)."""
    if not a or not b:
        return None
    ka = [float(r["kl"]) for r in a if r.get("kl") is not None]
    kb = [float(r["kl"]) for r in b if r.get("kl") is not None]
    if not ka or not kb:
        return None
    return (sum(ka) / len(ka)) / (sum(kb) / len(kb))


def rtn_curve(groups: dict[str, list[dict]], p1_rows: list[dict]) -> tuple[list[float], list[float]] | None:
    """(bits, KL) of the uniform rtn-sym-g128-{8,4,3,2} rows, seed mean per width, Phase 3 file first and the Phase 1 file
    for the widths it lacks; None with fewer than two points."""
    pts: dict[float, list[float]] = {}
    for b in (8, 4, 3, 2):
        m = f"rtn-sym-g128-{b}"
        rs = groups.get(m) or [r for r in p1_rows if r["method"] == m and r.get("kl") is not None]
        rs = [r for r in rs if r.get("kl") is not None and float(r["kl"]) > 0]
        if rs:
            pts[round(float(rs[0]["bits_blocks"]), 6)] = [float(r["kl"]) for r in rs]
    if len(pts) < 2:
        return None
    xs = sorted(pts)
    return xs, [sum(pts[x]) / len(pts[x]) for x in xs]


def rtn_gate_lines(groups: dict[str, list[dict]], p1_rows: list[dict]) -> list[str]:
    """H3-A, the RTN gate: alloc-type-rtn-avg{3,3.5} against the uniform RTN curve (rtn-sym-g128-{4,3,2}, log-linear in
    bits_blocks) at the row's own bits; the gptq32 curve says nothing about whether allocation helps RTN."""
    out = []
    curve = rtn_curve(groups, p1_rows)
    for avg in ("3", "3.5"):
        m = f"alloc-type-rtn-avg{avg}"
        if m not in groups:
            continue
        for r in groups[m]:
            if curve is None or r.get("kl") is None:
                out.append(f"  - RTN gate avg {avg} seed {r.get('seed', 0)}: n/a (missing: rtn-sym-g128-{{4,3,2}} rows for the uniform RTN curve)")
                continue
            xs, ys = curve
            bits = float(r["bits_blocks"])
            ref = p3.interp_log(xs, ys, bits)
            ratio = float(r["kl"]) / ref if ref > 0 else None
            flag = "" if p3.in_range(xs, bits) else "*"
            out.append(f"  - RTN gate avg {avg} seed {r.get('seed', 0)}: bits {fmt(bits)}, KL {fmt(r['kl'], 5)} vs uniform RTN curve "
                       f"{fmt(ref, 5)}{flag} (points at {', '.join(fmt(x, 3) for x in xs)} bits) -> ratio {fmt(ratio, 3)} "
                       f"({pct_below(ratio)}; better than uniform RTN by > {p3.KL_NOISE:.0%}: {ratio is not None and ratio < 1 - p3.KL_NOISE})")
    return out


def alloc_record(rdir: Path, model: str, method: str, seed: int) -> dict | None:
    """The alloc_<method>_seed<seed>.json next to a row (written by wcomp.methods), or None when it is missing."""
    path = rdir / model / f"alloc_{method}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def shuffle_control_status(rec: dict | None, method: str, seed: int) -> tuple[bool | None, str]:
    """(valid, text) of a shuffle control from its alloc JSON: invalid when the control equals the real allocation (the
    rebalancer walked the permutation back; share of the parameters at their real width = 1.0), None when the JSON is
    missing. The share is printed beside the real / shuffle ratio either way."""
    if rec is None:
        return None, f"control share at real width n/a (missing: alloc_{method}_seed{seed}.json)"
    info = rec.get("shuffle") or {}
    share = info.get("params_share_at_real_width")
    identical = rec.get("bits") is not None and rec.get("bits") == rec.get("real_bits")
    text = f"control share at real width {fmt(share, 3)}"
    if identical or (share is not None and share >= 1.0):
        return False, text + " CONTROL INVALID (identical to the real allocation: no control was applied)"
    cap = info.get("max_real_share")
    if cap is not None and share is not None and share > cap:
        return False, text + f" CONTROL INVALID (above the {cap:.2f} cap)"
    return True, text


def alloc_signal_text(rec: dict | None, method: str, seed: int) -> str:
    """"signal: damage" / "signal: fisher" of an alloc-tensor row from its alloc JSON ("signal", else the source note)."""
    if rec is None:
        return f"signal: n/a (missing: alloc_{method}_seed{seed}.json)"
    sig = rec.get("signal") or ("fisher" if "Fisher total" in str(rec.get("source", "")) else "damage")
    return f"signal: {sig}"


def hyp_a(rows: list[dict], groups: dict[str, list[dict]], cmps: dict, model: str, rdir: Path, probe_dir: Path, p1_rows: list[dict] | None = None) -> list[str]:
    out = ["- **H3-A** (type allocation beats uniform by 15-30% KL at avg 3.5 / 3, both seeds; shuffles within 5% of uniform; "
           "tensor-level adds < 10%; Spearman(C4 damage, Phase 2 WikiText @only KL) > 0.9):"]
    missing = []
    for avg in ("3.5", "3"):
        m = f"alloc-type-gptq32-avg{avg}"
        if m in groups:
            out += _ratio_lines(groups[m], cmps, m, f"alloc-type avg {avg}")
            out.append(f"    verdict avg {avg}: {p3.method_verdict(cmps[m])}")
        else:
            missing.append(m)
    real = groups.get("alloc-type-gptq32-avg3", [])
    real0 = [r for r in real if r.get("seed", 0) == 0]
    for k in (1, 2):
        m = f"alloc-type-shuf{k}-gptq32-avg3"
        if m in groups:
            c = cmps[m][0]
            rr = _kl_ratio(real0 or real, groups[m])
            seed = groups[m][0].get("seed", 0)
            valid, ctrl = shuffle_control_status(alloc_record(rdir, model, m, seed), m, seed)
            claim = rr is not None and rr < 1 - p3.KL_NOISE and valid is True
            out.append(f"  - shuffle {k}: bits {fmt(groups[m][0]['bits_blocks'])}, KL {fmt(groups[m][0].get('kl'), 5)}, ratio vs curve "
                       f"{fmt(c['ratio_kl'], 3)} ({pct_below(c['ratio_kl'])}; 'no better than uniform' = within 5%: "
                       f"{c['ratio_kl'] is not None and c['ratio_kl'] > 1 - p3.KL_NOISE}); real / shuffle KL = {fmt(rr, 3)} "
                       f"(claim needs < {1 - p3.KL_NOISE:.2f} and a valid control: {claim}); {ctrl}")
        else:
            missing.append(m)
    for avg in ("3", "3.5"):
        mt, my = f"alloc-tensor-gptq32-avg{avg}", f"alloc-type-gptq32-avg{avg}"
        if mt in groups and my in groups:
            for rt in groups[mt]:
                ty = [r for r in groups[my] if r.get("seed", 0) == rt.get("seed", 0)]
                rr = _kl_ratio([rt], ty)
                sig = alloc_signal_text(alloc_record(rdir, model, mt, rt.get("seed", 0)), mt, rt.get("seed", 0))
                out.append(f"  - tensor vs type avg {avg} seed {rt.get('seed', 0)}: KL {fmt(rt.get('kl'), 5)} vs {fmt(ty[0].get('kl'), 5) if ty else '—'} "
                           f"-> tensor / type = {fmt(rr, 3)} (extra gain {pct_below(rr)}; hypothesis: within 10%); {sig}")
        elif mt not in groups:
            missing.append(mt)
    out += rtn_gate_lines(groups, p1_rows or [])
    # Spearman of the C4-measured per-type damage against the Phase 2 WikiText @only KL at 4 bits.
    dmg_path = rdir / model / "damage_c4.jsonl"
    abl_path = probe_dir / model / "ablation.jsonl"
    dmg_rows = None
    rho_line = None
    if dmg_path.exists() and abl_path.exists():
        try:
            dmg_rows = alloc_mod.read_damage_file(dmg_path)  # the allocation methods' own checks (C4 set, protocol windows)
        except ValueError as e:
            out.append(f"  - Spearman: n/a (damage file refused: {e})")
            dmg_rows = None
    if dmg_rows is not None and abl_path.exists():
        dmg = {d["name"]: float(d["kl"]) for d in dmg_rows if d.get("unit") == "type" and d.get("kl") is not None}
        abl = {}
        for r in p3.load_rows(abl_path):
            if r["method"].startswith("rtn-sym-g128-4@only=") and r.get("kl") is not None:
                abl[r["method"].split("=", 1)[1]] = float(r["kl"])
        common = sorted(set(dmg) & set(abl))
        rho = p3.spearman([dmg[t] for t in common], [abl[t] for t in common])
        rho_line = (f"  - Spearman(C4 damage per type, Phase 2 WikiText @only KL at 4 bits) = {fmt(rho, 3)} over {len(common)} types "
                    f"(hypothesis > 0.9: {rho is not None and rho > 0.9}); C4 ranking: "
                    + " > ".join(sorted(common, key=lambda t: -dmg[t])))
    if rho_line is not None:
        out.append(rho_line)
    elif dmg_rows is None and dmg_path.exists() and abl_path.exists():
        pass  # the refusal line was appended
    else:
        out.append(f"  - Spearman: n/a (missing: {dmg_path.as_posix() if not dmg_path.exists() else abl_path.as_posix()})")
    if missing:
        out.append(f"  - {_missing(missing)}")
    return out


def hyp_d(groups: dict[str, list[dict]], cmps: dict) -> list[str]:
    out = ["- **H3-D** (had + gptq32 beats gptq32 by > 20% KL at 3 bits, < 10% at 4, > 40% at 2; rnd within 5% of had; "
           "eig costs > 1 bit/param stored and loses):"]
    missing = []
    for b, target in ((4, "< 10%"), (3, "> 20%"), (2, "> 40%")):
        m = f"had-gptq32-sym-g128-{b}"
        if m in groups:
            out += _ratio_lines(groups[m], cmps, m, f"had {b} (target {target} below)")
            out.append(f"    verdict had {b}: {p3.method_verdict(cmps[m])}")
            if b == 3:
                out.append(f"    had combo gate (= beats at 3 bits, every seed): **{p3.method_verdict(cmps[m])}**")
        else:
            missing.append(m)
    had3 = groups.get("had-gptq32-sym-g128-3", [])
    for kind in ("rnd", "hadsign"):
        m = f"{kind}-gptq32-sym-g128-3"
        if m in groups:
            for r in groups[m]:
                h = [x for x in had3 if x.get("seed", 0) == r.get("seed", 0)]
                rr = _kl_ratio([r], h)
                c = cmps[m][groups[m].index(r)]
                out.append(f"  - {kind} 3 seed {r.get('seed', 0)}: KL {fmt(r.get('kl'), 5)}, ratio vs curve {fmt(c['ratio_kl'], 3)}, "
                           f"{kind} / had (same seed) = {fmt(rr, 3)} ({pct_below(rr)}; within 5%: {rr is not None and abs(rr - 1) <= p3.KL_NOISE})")
        else:
            missing.append(m)
    m = "eig-gptq32-sym-g128-3"
    if m in groups:
        r, c = groups[m][0], cmps[m][0]
        ratio = ("— (its bits lie beyond the measured curve)" if c.get("far_extrapolated") else fmt(c["ratio_kl"], 3))
        out.append(f"  - eig 3 seed {r.get('seed', 0)}: bits_blocks {fmt(r['bits_blocks'])} (stored rotation = {fmt(r['bits_blocks'] - 3.125, 3)} "
                   f"bits/param over 3.125; > 1: {r['bits_blocks'] - 3.125 > 1}), KL {fmt(r.get('kl'), 5)}, ratio vs curve at its bits "
                   f"{ratio}, saved fixed {fmt(c['saved_fixed'], 3, signed=True)} (loses: {c['saved_fixed'] is not None and c['saved_fixed'] < 0}; "
                   f"read from the curve's bits at the row's KL, which lies inside the measured KL range when saved has no *)")
    else:
        missing.append(m)
    if missing:
        out.append(f"  - {_missing(missing)}")
    return out


def hyp_e(groups: dict[str, list[dict]], cmps: dict, curve: p3.Curve | None, rd_summary: dict | None) -> list[str]:
    out = ["- **H3-E** (Huffman saves 0.40-0.50 bits at 4, 0.25-0.40 at 3, < 0.20 at 2; Shannon gap < 0.05 at 4; Lloyd-Max "
           "10-25% lower KL at 3-4 bits for < 0.01 bits of table; NF captures >= 2/3 of it; coders 1.0-1.5 bits above the "
           "Gaussian RD bound at 4 bits, shrinking with fewer bits):",
           "  - codebooks: lloyd = per-tensor Lloyd-Max fit (2^b x 16 table bits); nf = the fixed Lloyd-Max optimum of N(0, 1) (0 bits; "
           "NOTES.md amends DESIGN.md E3, whose quantile codebook clips the tails under RMS normalization); nfq = DESIGN.md E3 as "
           "written, the Gaussian quantiles (0 bits; in the run list at seed 0). The \"NF captures >= 2/3\" clause is read on nf, and nfq "
           "is reported next to it. huff - shannon = the chosen coder's loss plus its tables (see the Rows footnote); huff - entropy0 = "
           "the gap to the order-0 entropy, the contract's \"Shannon figure\" as pre-registered (< 0.05 at 4 bits)."]
    missing = []
    for b in (4, 3, 2):
        m = f"gptq32-sym-g128-{b}"
        rs = groups.get(m, [])
        if not rs:
            missing.append(m)
            continue
        for r in rs:
            h, s, e0 = r.get("bits_blocks_huff"), r.get("bits_blocks_shannon"), r.get("bits_blocks_entropy0")
            sav = r["bits_blocks"] - h if h is not None else None
            gap = h - s if (h is not None and s is not None) else None
            gap0 = h - e0 if (h is not None and e0 is not None) else None
            out.append(f"  - huff {b} seed {r.get('seed', 0)}: fixed {fmt(r['bits_blocks'])}, huff {fmt(h)}, shannon {fmt(s)}, entropy0 {fmt(e0)}: saving "
                       f"{fmt(sav, 3)} bits/param, huff - shannon gap {fmt(gap, 3)}, huff - entropy0 gap {fmt(gap0, 3)}"
                       + (f" (< 0.05: {gap0 < 0.05})" if b == 4 and gap0 is not None else ""))
    for b in (4, 3, 2):
        m = f"lloyd-gptq32-g128-{b}"
        if m in groups:
            for r, c in zip(groups[m], cmps[m]):
                out.append(f"  - lloyd {b} seed {r.get('seed', 0)}: bits {fmt(r['bits_blocks'])} (table {fmt(r['bits_blocks'] - (b + 0.125), 4, signed=True)} "
                           f"vs the uniform grid), KL {fmt(r.get('kl'), 5)}, ratio vs curve {fmt(c['ratio_kl'], 3)} ({pct_below(c['ratio_kl'])}), "
                           f"dPPL ratio {fmt(c['ratio_dppl'], 3)}")
            out.append(f"    verdict lloyd {b} (beats: KL and dPPL): {p3.method_verdict(cmps[m])}")
            if b == 3:
                out.append(f"    lloyd combo gate (KL-only, > {p3.KL_NOISE:.0%} below the curve on every seed): **{p3.kl_gate(cmps[m])}**")
        else:
            missing.append(m)
    for b in (4, 3):
        ml = f"lloyd-gptq32-g128-{b}"
        for kind in ("nf", "nfq"):
            mn = f"{kind}-gptq32-g128-{b}"
            if mn in groups and ml in groups:
                cn, cl = cmps[mn][0], [c for r, c in zip(groups[ml], cmps[ml]) if r.get("seed", 0) == groups[mn][0].get("seed", 0)]
                cl = cl[0] if cl else cmps[ml][0]
                share = None
                if cn["ratio_kl"] is not None and cl["ratio_kl"] is not None and cl["ratio_kl"] < 1:
                    share = (1 - cn["ratio_kl"]) / (1 - cl["ratio_kl"])
                what = "nf (Gaussian Lloyd-Max optimum)" if kind == "nf" else "nfq (DESIGN.md E3 quantiles)"
                out.append(f"  - {what} {b}: ratio vs curve {fmt(cn['ratio_kl'], 3)} ({pct_below(cn['ratio_kl'])}); lloyd gain {pct_below(cl['ratio_kl'])}; "
                           f"{kind} share of the lloyd gain = {fmt(share, 2)} (>= 2/3: {share is not None and share >= 2 / 3})")
            elif mn not in groups:
                missing.append(mn)
    if rd_summary:
        for b in (4, 3, 2):
            m = f"gptq32-sym-g128-{b}"
            if m in rd_summary:
                vals = rd_summary[m]
                out.append(f"  - RD bound gptq32 {b} (numel-weighted over the types): above bound fixed {fmt(vals.get('fixed'), 2)}, "
                           f"huff {fmt(vals.get('huff'), 2)}, shannon {fmt(vals.get('shannon'), 2)} bits (seed 0 row; per seed in the RD bound table)")
    else:
        out.append("  - RD bound: n/a (missing: rd_bound.jsonl)")
    if missing:
        out.append(f"  - {_missing(missing)}")
    return out


def hyp_g(groups: dict[str, list[dict]], cmps: dict, p1_rows: list[dict]) -> list[str]:
    out = ["- **H3-G** (gate on RTN: Fisher row split beats uniform rtn-sym-g128-4 by > 10% KL and both shuffled-row controls by > 5%):"]
    missing = []
    gate = groups.get("rowalloc-fisher-rtn-avg4", [])
    uni = groups.get("rtn-sym-g128-4") or [r for r in p1_rows if r["method"] == "rtn-sym-g128-4"]
    if not gate:
        missing.append("rowalloc-fisher-rtn-avg4")
    if not uni:
        missing.append("rtn-sym-g128-4 (Phase 3 or Phase 1 file)")
    if gate and uni:
        g = gate[0]
        ru = _kl_ratio([g], uni)
        out.append(f"  - rowalloc-fisher-rtn-avg4: bits {fmt(g['bits_blocks'])} vs uniform {fmt(uni[0]['bits_blocks'])}; KL {fmt(g.get('kl'), 5)} vs "
                   f"{fmt(uni[0].get('kl'), 5)} -> ratio {fmt(ru, 3)} ({pct_below(ru)}; > 10% better: {ru is not None and ru < 0.9})")
        checks = [ru is not None and ru < 0.9]
        for k in (1, 2):
            m = f"rowalloc-shuf{k}-rtn-avg4"
            if m in groups:
                rs = _kl_ratio([g], groups[m])
                out.append(f"  - vs shuffled rows {k}: control KL {fmt(groups[m][0].get('kl'), 5)}, real / control = {fmt(rs, 3)} "
                           f"({pct_below(rs)}; > 5% better: {rs is not None and rs < 1 - p3.KL_NOISE})")
                checks.append(rs is not None and rs < 1 - p3.KL_NOISE)
            else:
                missing.append(m)
                checks.append(None)
        gate_verdict = "passed" if all(c is True for c in checks) else ("failed" if any(c is False for c in checks) else "incomplete")
        out.append(f"    gate: **{gate_verdict}**")
    for b in (4, 3):
        m = f"rowalloc-fisher-gptq32-avg{b}"
        if m in groups:
            out += _ratio_lines(groups[m], cmps, m, f"rowalloc-fisher-gptq32 avg {b}")
            out.append(f"    verdict avg {b}: {p3.method_verdict(cmps[m])}")
    if missing:
        out.append(f"  - {_missing(missing)}")
    return out


def hyp_combo(groups: dict[str, list[dict]], cmps: dict, curve: p3.Curve | None, fp16: float, combo_cfg: dict | None) -> list[str]:
    out = ["- **H3-combo** (at avg 3.0 the combination reaches the KL of gptq32-4 (fixed 4.125), i.e. ~1.0-1.3 bits saved at equal "
           "quality; at avg 2.5 it stays below +50% PPL):"]
    if combo_cfg is not None:
        out.append(f"  - components (combo.json): {json.dumps(combo_cfg, sort_keys=True)}")
    else:
        out.append("  - components: n/a (missing: combo.json)")
    ref4 = groups.get("gptq32-sym-g128-4", [])
    kl4 = (sum(float(r["kl"]) for r in ref4) / len(ref4)) if ref4 else None
    huff4 = ([r["bits_blocks_huff"] for r in ref4 if r.get("bits_blocks_huff") is not None]) if ref4 else []
    huff4 = sum(huff4) / len(huff4) if huff4 else None
    missing = []
    for avg in ("2.5", "3", "3.5"):
        m = f"combo-gptq32-avg{avg}"
        if m not in groups:
            missing.append(m)
            continue
        for r, c in zip(groups[m], cmps[m]):
            r4 = float(r["kl"]) / kl4 if (kl4 and r.get("kl") is not None) else None
            d = p3.dppl(r, fp16) / fp16
            saved4_fixed = (4.125 - float(r["bits_blocks"])) if r4 is not None and r4 <= 1 + p3.KL_NOISE else None
            saved4_huff = (huff4 - float(r["bits_blocks_huff"])) if (saved4_fixed is not None and huff4 is not None and r.get("bits_blocks_huff") is not None) else None
            out.append(f"  - combo avg {avg} seed {r.get('seed', 0)}: bits {fmt(r['bits_blocks'])} / huff {fmt(r.get('bits_blocks_huff'))}, KL {fmt(r.get('kl'), 5)}, "
                       f"dPPL {fmt(d, pct=True, signed=True)}, ratio vs curve {fmt(c['ratio_kl'], 3)}, dPPL ratio {fmt(c['ratio_dppl'], 3)}, "
                       f"beats row: {c['beats_row']}; KL / KL(gptq32-4 seed mean {fmt(kl4, 5)}) = {fmt(r4, 3)} "
                       f"(reaches it, within 5%: {r4 is not None and r4 <= 1 + p3.KL_NOISE}); bits saved at equal KL on the curve: fixed "
                       f"{fmt(c['saved_fixed'], 3, signed=True)}, huff {fmt(c['saved_huff'], 3, signed=True)}; vs gptq32-4 directly (only when it reaches "
                       f"its KL): fixed {fmt(saved4_fixed, 3, signed=True)}, huff-vs-huff {fmt(saved4_huff, 3, signed=True)}")
        out.append(f"    verdict avg {avg}: {p3.method_verdict(cmps[m])}" + (f"; dPPL < +50% on every seed: {all(p3.dppl(r, fp16) / fp16 < 0.5 for r in groups[m])}" if avg == "2.5" else ""))
    if not ref4:
        missing.append("gptq32-sym-g128-4")
    if missing:
        out.append(f"  - {_missing(missing)}")
    return out


# ---------------------------------------------------------------- RD bound (E4)


FIGURES = ("fixed", "huff", "shannon")


def load_rd_bound(path: Path) -> list[dict]:
    """rd_bound.jsonl rows (wcomp.rdbound.rd_bound_rows: one per weight type and run, prefixed with the run's model /
    method / seed / calib_seed / bit figures / timestamp), latest per (method, seed, type[, tensor]). Fields used: method,
    seed, type, numel, and gap_fixed / gap_huff / gap_shannon = bits above the Gaussian bound (an older "above_*" spelling
    is accepted); when a gap is absent it is derived from D (or "mse") and bits_fixed / bits_huff / bits_shannon as
    bits - 0.5 log2(1 / D) (R(D) = 0.5 log2(1/D) for a unit-variance source)."""
    if not path.exists():
        return []
    latest: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        D = r.get("D", r.get("mse"))
        for fig in FIGURES:
            if r.get(f"gap_{fig}") is None and r.get(f"above_{fig}") is not None:
                r[f"gap_{fig}"] = r[f"above_{fig}"]
            if r.get(f"gap_{fig}") is None and D is not None and float(D) > 0 and r.get(f"bits_{fig}") is not None:
                r[f"gap_{fig}"] = float(r[f"bits_{fig}"]) - 0.5 * math.log2(1.0 / float(D))
        latest[(r.get("method"), r.get("seed", 0), r.get("type"), r.get("tensor"))] = r
    return list(latest.values())


def _gap(rs: list[dict], fig: str, weighted: bool) -> float | None:
    """Median (per type, over its rows) or numel-weighted mean (over types) of gap_<fig>."""
    v = [(float(r[f"gap_{fig}"]), float(r.get("numel") or 0)) for r in rs if r.get(f"gap_{fig}") is not None]
    if not v:
        return None
    if weighted and all(n > 0 for _, n in v):
        return sum(g * n for g, n in v) / sum(n for _, n in v)
    return statistics.median(g for g, _ in v)


def rd_bound_table(rd_rows: list[dict]) -> tuple[list[str], dict[str, dict]]:
    """Per (method, seed) one table row with per-type cells "fixed / huff / shannon" bits above the bound (median over the
    rows of that type: one per run from wcomp.rdbound, so the value itself) and an "all types" cell (numel-weighted mean
    over the types); also returns {method: {"fixed", "huff", "shannon"}} of the seed-0 "all types" figures."""
    if not rd_rows:
        return ["n/a (missing: rd_bound.jsonl)"], {}
    types = [t for t in TYPE_ORDER if any(r.get("type") == t for r in rd_rows)]
    types += sorted({r.get("type") for r in rd_rows if r.get("type") not in types and r.get("type") is not None})
    groups: dict[tuple, list[dict]] = {}
    for r in rd_rows:
        groups.setdefault((r.get("method"), r.get("seed", 0)), []).append(r)
    lines_rows, overall = [], {}
    for (m, s), rs in sorted(groups.items(), key=lambda kv: (p3.sort_key({"method": kv[0][0] or "", "seed": kv[0][1]}))):
        cells = [m or "—", str(s)]
        for t in types:
            tr = [r for r in rs if r.get("type") == t]
            cells.append(" / ".join(fmt(_gap(tr, f, False), 2) for f in FIGURES) if tr else "—")
        cells.append(" / ".join(fmt(_gap(rs, f, True), 2) for f in FIGURES))
        lines_rows.append(cells)
        if s == 0:
            overall[m] = {f: _gap(rs, f, True) for f in FIGURES}
    lines = table(["method", "seed"] + types + ["all types"], lines_rows)
    lines.append("")
    lines.append("cells: bits above the Gaussian bound, bits - 0.5 log2(1 / D), at the fixed / Huffman / Shannon figure of that type, "
                 "D = MSE of the group-normalized weights of the type (E4, wcomp.rdbound; 6.02 dB per bit); \"all types\" is the "
                 "numel-weighted mean over the types.")
    return lines, overall


# ---------------------------------------------------------------- per model


def model_section(model: str, rdir: Path, p1dir: Path, probe_dir: Path, include_smoke: bool = False) -> list[str]:
    p3_path, p1_path = rdir / model / "runs.jsonl", p1dir / model / "runs.jsonl"
    rows = p3.load_rows(p3_path, include_smoke)
    p1_rows = p3.load_rows(p1_path, include_smoke)
    lines = [f"## {model}", ""]
    if include_smoke:
        lines += ["SMOKE ROWS INCLUDED (--include-smoke): off-protocol numbers, a pipeline check, not a result.", ""]
    if not rows:
        lines += [f"no Phase 3 rows ({p3_path.as_posix()} missing or empty)", ""]
        return lines
    fp16 = p3.fp16_ppl(rows, p1_rows, include_smoke=include_smoke)
    if fp16 is None:
        lines += [f"no full-protocol fp16 row in {p3_path.as_posix()} or {p1_path.as_posix()}; nothing can be differenced", ""]
        return lines
    curve = p3.reference_curve(rows, fp16)
    lines.append(f"fp16 PPL {fp16:.4f} ({'Phase 3 file' if any(r['method'] == 'fp16' for r in rows) else p1_path.as_posix()}); "
                 f"rows from {p3_path.as_posix()}; noise thresholds KL {p3.KL_NOISE:.0%} / PPL {p3.PPL_NOISE:.1%}.")
    lines.append("")
    if curve is None:
        lines += ["reference curve: n/a (missing: gptq32-sym-g128-{4,3,2})", ""]
    else:
        lines.append("Reference curve (gptq32-sym-g128, seed mean):")
        lines.append("")
        lines += table(["bits_blocks", "bits_huff", "bits_shannon", "kl", "ppl", "dPPL %", "seeds"],
                       [[fmt(p["bits"]), fmt(p["bits_huff"]), fmt(p["bits_shannon"]), fmt(p["kl"], 5), fmt(p["ppl"], 2),
                         fmt(p["dppl"] / fp16, pct=True, signed=True), ",".join(map(str, p["seeds"]))] for p in curve.points])
        if not curve.usable():
            lines.append("")
            lines.append("(fewer than two usable points: no interpolation, ratios only at the measured width)")
        spread = seed_spread_lines(curve, fp16)
        if spread:
            lines.append("")
            lines += spread
        lines.append("")
    tbl, cmps = rows_table(rows, curve, fp16)
    lines += ["### Rows", ""] + tbl + [""]
    groups = p3.by_method([r for r in rows if r["method"] not in ("fp16", "had-fp16")])
    rd_rows = load_rd_bound(rdir / model / "rd_bound.jsonl")
    rd_lines, rd_overall = rd_bound_table(rd_rows)
    combo_path = rdir / model / "combo.json"
    combo_cfg = json.loads(combo_path.read_text(encoding="utf-8")) if combo_path.exists() else None
    lines += ["### Hypotheses", ""]
    lines += hyp_a(rows, groups, cmps, model, rdir, probe_dir, p1_rows)
    lines += hyp_d(groups, cmps)
    lines += hyp_e(groups, cmps, curve, rd_overall)
    lines += hyp_g(groups, cmps, p1_rows)
    lines += hyp_combo(groups, cmps, curve, fp16, combo_cfg)
    lines += ["", "### RD bound (E4), bits above the Gaussian bound", ""] + rd_lines + [""]
    return lines


def build_summary(models: list[str], rdir: Path, p1dir: Path, probe_dir: Path, include_smoke: bool = False) -> str:
    lines = ["# Phase 3 summary", "",
             f"Generated by scripts/phase3_summary.py from <results-dir>/<model>/runs.jsonl ({rdir.as_posix()}), rd_bound.jsonl, "
             f"combo.json, damage_c4.jsonl; fp16 and rtn-sym-g128-4 fall back to {p1dir.as_posix()}/<model>/runs.jsonl; Phase 2 @only "
             f"rows from {probe_dir.as_posix()}/<model>/ablation.jsonl. Rules: DESIGN.md \"Phase 3 contract: methods\".", ""]
    for m in models:
        lines += model_section(m, rdir, p1dir, probe_dir, include_smoke)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma-separated model tags")
    ap.add_argument("--results-dir", default="results/phase3")
    ap.add_argument("--phase1-dir", default="results")
    ap.add_argument("--probe-dir", default="results/probe")
    ap.add_argument("--out", default=None, help="default <results-dir>/summary.md")
    ap.add_argument("--include-smoke", action="store_true", help="read off-protocol (smoke) rows too: a pipeline check, marked in the output")
    args = ap.parse_args()

    def absolute(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else ROOT / p

    rdir, p1dir, probe_dir = absolute(args.results_dir), absolute(args.phase1_dir), absolute(args.probe_dir)
    text = build_summary([m.strip() for m in args.models.split(",") if m.strip()], rdir, p1dir, probe_dir, args.include_smoke)
    out = Path(args.out) if args.out else rdir / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
