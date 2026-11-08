"""Structure table for the Phase 2 probes (DESIGN.md "Phase 2 contract").

  uv run python scripts/structure_table.py --models qwen3.5-0.8b,qwen3.5-2b,minicpm5-2b [--results-dir results]
  -> results/structure_table.md

One row per (model, probe, weight type): the key statistic on W and on the control, gain_bits, gain_bits_control,
their difference (= the claimed structure value) and a verdict from the contract's rule:
  shuffle test   passes iff |gain_bits - gain_bits_control| > 0.02 bits AND > the spread between the two control
                 seeds (seed 0 vs seed 1, where the probe recorded a second control; otherwise the 0.02 floor alone,
                 marked "1 seed"). P1 (marginal) uses the same rule against its Gaussian control, which the record
                 states is not a shuffle.
  P5 (acts)      "stable" iff the split-half Spearman of per-channel energy > 0.9 (the contract's control), else "noise".
  P6 (sensitivity) prediction check: Spearman between s_t per type and the measured rtn-sym-g128-4@only KL per type
                 from ablation.jsonl (> 0.7 "predicts"); also per layer against the @layer rows at 3 bits.
  P6b (fisher)   rank-1 fit energy of the position-wise empirical-Fisher approximation (F = mean_t (d_t^2)(x_t^2)^T with
                 d_t = d(sum_s nll_s)/d y_t, the K-FAC independent-positions statistic; it is the per-token Fisher only
                 in a model without token mixing) on W vs the shuffled control (> 0.7 means row + column side info
                 suffices); the per-window "batched" statistic is shown in the note, not judged.
  P2b (lowrank_weighted) gain_bits is the measured RTN-4 output-error ratio of the L-metric residual (not the e_r
                 energy formula, which is kept in the record as formula_gain_r* and quoted in the note).
  P7 (superweight) per candidate: "super weight" iff dPPL > 100 % and the magnitude control < 2 % (H-P7).
Also per model the P0 additivity summary (sum of @only KL over types vs the joint KL at the same bits, 4 and 3).
Aggregates are numel-weighted means over the records of a type (recomputed here, so the layout of the probe's own
"aggregate" block does not matter). Missing files / stats produce "-" cells and a note; nothing is fabricated.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_io as io  # noqa: E402
from probe_io import ALIGNMENTS, R_GRID, aggregate, by_type, get, num, records, types_present  # noqa: E402

GAIN_FLOOR = 0.02  # bits/param, the contract's floor for a shuffle-test pass
# (model, probe) pairs the DESIGN.md run list leaves out: Fisher (P6b) is 0.8B only, P2b needs the full Hessian (Qwen only)
NOT_SCHEDULED = {("qwen3.5-2b", "fisher"), ("minicpm5-2b", "fisher"), ("minicpm5-2b", "lowrank_weighted")}

# probe -> (key statistic label, canonical stat key (probe_io.ALIASES) read under "stat" and "control", number format)
KEY_STATS = {
    "marginal": ("H_4 (index entropy, sym g128 4 bit)", "H_4", "{:.3f}"),
    "lowrank": ("energy captured at r = 0.1 min(N, K)", "energy@0.1", "{:.3f}"),
    "lowrank_weighted": ("energy captured at r = 0.1 (W @ L)", "energy@0.1", "{:.3f}"),
    "outliers": ("column-norm max / median", "col_max_over_median", "{:.2f}"),
    "interlayer": ("min residual energy over alignments, adjacent pairs", "e_best", "{:.3f}"),
    "acts": ("Gini of per-channel input energy", "gini", "{:.3f}"),
    "sensitivity": ("s_t / n_t", "s_per_param", "{:.3g}"),
    "fisher": ("rank-1 fit energy of F", "rank1_energy", "{:.3f}"),
}


def key_stat(recs: list[dict], section: str, key: str) -> float | None:
    """Numel-weighted key statistic; "e_best" is the minimum over the alignments present (the probes store e_raw,
    e_out, e_in, e_perm, not a best)."""
    if key == "e_best":
        vals = [v for v in (aggregate(recs, section, k) for k in ALIGNMENTS) if v is not None]
        return min(vals) if vals else None
    return aggregate(recs, section, key)


def fmt(v, f: str = "{:.3f}") -> str:
    v = num(v)
    return "-" if v is None else f.format(v)


def shuffle_verdict(gain, gain_c, spread) -> str:
    """The contract's rule; the sign of the difference is reported, since a W below its control is not structure."""
    gain, gain_c = num(gain), num(gain_c)
    if gain is None or gain_c is None:
        return "n/a"
    d = gain - gain_c
    thr = GAIN_FLOOR if spread is None else max(GAIN_FLOOR, spread)
    tag = "" if spread is not None else " (1 seed)"
    if abs(d) <= thr:
        return f"fail{tag}"
    return f"pass{tag}" if d > 0 else f"fail (below control){tag}"


def probe_rows(model: str, probe: str, doc: dict | None, extra: dict) -> list[list[str]]:
    """Table rows for one probe of one model (empty when the file is missing)."""
    if doc is None:
        return []
    label, key, f = KEY_STATS[probe]
    if probe == "acts":
        recs = records(doc, types=io.PRIMARY_ACT_TYPES)  # P5 is about the down / o / out_proj inputs
    else:
        recs = records(doc, adjacent_only=(probe == "interlayer"))
    rows = []
    for t in types_present(recs):
        trs = by_type(recs)[t]
        w, c = key_stat(trs, "stat", key), key_stat(trs, "control", key)
        gain, gain_c = aggregate(trs, "gain_bits"), aggregate(trs, "gain_bits_control")
        spread = io.control_spread(trs)
        if probe in ("lowrank", "lowrank_weighted", "outliers", "interlayer", "marginal"):
            note = "Gaussian control, not a shuffle" if probe == "marginal" else ""
            verdict = shuffle_verdict(gain, gain_c, spread)
            if probe == "outliers":
                per_type = io.jaccard_by_type(doc)
                j, jc = per_type.get(t, io.jaccard(doc))
                if j is not None:
                    note = f"top-16 channel Jaccard across layers {j:.3f} vs control {fmt(jc, '{:.3f}')}" + ("" if t in per_type else " (all hidden-input linears)")
            if probe in ("lowrank", "lowrank_weighted"):
                # gain_bits is the numel-weighted mean of the per-record maxima over the r grid; the labelled r is the
                # argmax of the aggregated gain@r, and the aggregated gain at that r is printed so both are visible.
                best = max(((aggregate(trs, "stat", f"gain@{r}") or -math.inf, r) for r in R_GRID), default=(None, None))
                if best[0] is not None and best[0] > -math.inf:
                    best_rs = sorted({f"{num(get(r, 'stat', 'best_r_frac')):.2g}" for r in trs if num(get(r, "stat", "best_r_frac")) is not None})
                    note = (f"best aggregated r = {best[1]} min(N, K) (gain there {best[0]:+.3f}); gain_bits = mean of per-record maxima"
                            + (f", per-record best r in {{{', '.join(best_rs)}}}" if best_rs else ""))
                    if probe == "lowrank_weighted":
                        fg, fgc = aggregate(trs, "stat", "formula_best_gain"), aggregate(trs, "control", "formula_gain_bits_best")
                        note = ("gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V"
                                + (f" (the contract's e_r energy formula would say {fg:+.3f}" + (f" vs control {fgc:+.3f}" if fgc is not None else "")
                                   + ": a statistic, not bits/param)" if fg is not None else "") + "; " + note)
            if probe == "interlayer":
                note = f"control {fmt(c)}; gain at the best alignment"
        elif probe == "acts":
            sp = aggregate(trs, "stat", "spearman_split")
            c = sp
            verdict = "n/a" if sp is None else ("stable" if sp > 0.9 else "noise")
            note = (f"control = split-half Spearman {fmt(sp)}; channels for 50 % / 90 % of the energy: "
                    f"{fmt(aggregate(trs, 'stat', 'share50'))} / {fmt(aggregate(trs, 'stat', 'share90'))}")
            gain = gain_c = None
        elif probe == "sensitivity":
            if w is None:  # fall back to s_t / numel
                s, n = aggregate(trs, "stat", "s_t"), aggregate(trs, "numel")
                w = s / n if s and n else None
            rho = extra.get("spearman_type")
            verdict = "n/a (no ablation rows)" if rho is None else (f"predicts (Spearman {rho:.2f})" if rho > 0.7 else f"no prediction (Spearman {rho:.2f})")
            kl = extra.get("only4", {}).get(t)
            note = f"@only KL 4 bit {fmt(num(kl.get('kl')) if kl else None, '{:.4f}')}"
            c = None
            gain = gain_c = None
        elif probe == "fisher":
            verdict = "n/a" if w is None else ("rank-1 fits (> 0.7)" if w > 0.7 else "rank-1 does not fit")
            if c is not None:
                verdict += f"; control {c:.3f}"
            # the > 0.7 test is on the position-wise empirical-Fisher approximation (stat.rank1_energy: d_t of the summed
            # window loss, not the per-token Fisher of a token-mixing model); the round-1 per-window statistic
            # (stat.batched_*) is printed next to it so a verdict that flips with the definition is visible
            note = f"position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini {fmt(aggregate(trs, 'stat', 'gini'))} vs control {fmt(aggregate(trs, 'control', 'gini'))}"
            wb, gb = aggregate(trs, "stat", "batched_rank1_energy"), aggregate(trs, "stat", "batched_gini")
            if wb is not None:
                note += f"; batched (per-window gradient squared, not used for the verdict): rank-1 {fmt(wb)}, Gini {fmt(gb)}"
            gain = gain_c = None
        else:
            verdict, note = "n/a", ""
        diff = (num(gain) - num(gain_c)) if num(gain) is not None and num(gain_c) is not None else None
        rows.append([model, probe, t, label, fmt(w, f), fmt(c, f), fmt(gain, "{:+.3f}"), fmt(gain_c, "{:+.3f}"), fmt(diff, "{:+.3f}"),
                     fmt(spread, "{:.3f}"), verdict, note])
    return rows


def superweight_rows(model: str, doc: dict | None) -> list[list[str]]:
    if doc is None:
        return []
    rows = []
    for r in records(doc):
        d = num(io.val(r, "stat", "dppl"))
        mag = num(io.val(r, "control", "mag_dppl"))
        rnd = num(io.val(r, "control", "random_dppl"))
        if d is None:
            verdict = "n/a"
        elif d > 100 and (mag is not None and mag < 2):
            verdict = "super weight"
        elif d > 100:
            verdict = "large effect, control not < 2 %" if mag is not None else "large effect (no control)"
        else:
            verdict = "not a super weight"
        rows.append([model, r.get("tensor", "?"), str(r.get("layer", "?")), r.get("type", "?"), f"[{get(r, 'stat', 'row')}, {get(r, 'stat', 'col')}]",
                     fmt(get(r, "stat", "value"), "{:.3g}"), fmt(d, "{:.2f}"), fmt(get(r, "stat", "kl"), "{:.4f}"), fmt(mag, "{:.2f}"), fmt(rnd, "{:.2f}"),
                     str(r.get("note", "")).split(";")[0].strip()[:40], verdict])
    return rows


def ablation_summary(rows: list[dict]) -> tuple[list[list[str]], dict]:
    """P0 additivity per bits: sum of @only KL vs the joint KL; returns (table rows, {"only4": {type: row}})."""
    out = []
    only4 = io.ablation_only(rows, 4)
    for b in (4, 3):
        only = io.ablation_only(rows, b)
        joint = io.find_row(rows, f"rtn-sym-g128-{b}")
        if not only:
            continue
        kls = {t: num(r.get("kl")) for t, r in only.items()}
        kls = {t: v for t, v in kls.items() if v is not None}
        s = sum(kls.values())
        jk = num(joint.get("kl")) if joint else None
        top3 = sorted(kls.items(), key=lambda kv: -kv[1])[:3]
        top3_share = sum(v for _, v in top3) / jk if jk else None
        ratio = s / jk if jk else None
        # the contract's rule is the 50 % band; the direction of a departure is named so 0.51 does not read as 1.0
        direction = "" if ratio is None or 0.9 <= ratio <= 1.1 else (" (sum < joint: super-additive)" if ratio < 1 else " (sum > joint: sub-additive)")
        additive = "-" if ratio is None else ("additive (within 50 %)" if 0.5 <= ratio <= 1.5 else "not additive") + direction
        out.append([str(b), str(len(kls)), f"{s:.4f}", fmt(jk, "{:.4f}"), fmt(ratio, "{:.2f}"),
                    ", ".join(f"{t} {v:.4f}" for t, v in top3), fmt(top3_share, "{:.0%}"), additive])
    return out, {"only4": only4}


def p0_hypothesis_lines(rows: list[dict]) -> list[str]:
    """The pre-registered P0 figures that are not the additivity table: H-P3 (+keepcols=p recovers > 25 % of the
    RTN KL at the same group size, with its bits_blocks cost) and H-P0 group size (KL(g128) / KL(g32) > 2)."""
    lines = []
    for b in (4, 3):
        joint = io.find_row(rows, f"rtn-sym-g128-{b}")
        jk = num(joint.get("kl")) if joint else None
        jb = num(joint.get("bits_blocks")) if joint else None
        keep = []
        for r in rows:
            p = io.parse_method(r["method"])
            if p and p["family"] == "rtn-sym-g128" and p["bits"] == b and p["qual"] is None and p["plus"] == "keepcols" \
                    and r.get("seed", 0) == 0 and r.get("calib_seed") in (None, 0):
                keep.append((float(p["plus_value"]), r))
        for pct, r in sorted(keep):
            k, bb = num(r.get("kl")), num(r.get("bits_blocks"))
            if jk and k is not None:
                rec = (jk - k) / jk
                cost = f"{bb - jb:+.3f} bits/param" if bb is not None and jb is not None else "-"
                verdict = "H-P3 met (> 25 %)" if rec > 0.25 else "H-P3 not met"
                lines.append(f"- +keepcols={pct:g} at {b} bit recovers {rec:.0%} of the RTN KL ({jk:.4f} -> {k:.4f}) at {cost}: {verdict}")
        g32 = io.find_row(rows, f"rtn-sym-g32-{b}")
        k32 = num(g32.get("kl")) if g32 else None
        if jk and k32:
            factor = jk / k32
            verdict = "H-P0 group size met (> 2)" if factor > 2 else "H-P0 group size not met (group size is not the lever)"
            lines.append(f"- KL(g128) / KL(g32) at {b} bit: {factor:.2f} ({jk:.4f} / {k32:.4f}, +{num(g32.get('bits_blocks')) - jb:.3f} bits/param): {verdict}"
                         if jb is not None and num(g32.get("bits_blocks")) is not None else
                         f"- KL(g128) / KL(g32) at {b} bit: {factor:.2f}: {verdict}")
    return lines


def allocation_lines(doc: dict | None) -> list[str]:
    """The P6a model-wide allocation saving, printed only together with its floored fraction and the guarded
    variants (b_u >= 2 bits; unfloored units only), since the 0-bit floor dominates it wherever whole types fall
    below mean * 2^-8; the within-type and relative-output-error tensor numbers follow."""
    alloc = io.allocation(doc, ["tensor", "group", "col"])
    if not alloc:
        return []
    lines = []
    for g, a in alloc.items():
        lines.append(f"- allocation saving, {g} granularity: {fmt(a.get('saving'))} bits/param with {fmt(a.get('frac_floored'), '{:.1%}')} of the weights "
                     f"floored at 0 bits; b_u >= 2 bits: {fmt(a.get('saving_cap2bit'))}; unfloored units alone: {fmt(a.get('saving_unfloored'))}")
    wt = num(io.model_level(doc, "alloc_tensor_within_type_saving"))
    rel = num(io.model_level(doc, "alloc_tensor_rel_saving"))
    if wt is not None or rel is not None:
        lines.append(f"- tensor granularity restricted to within-type units: {fmt(wt)}; on the relative output error s_t / ||W X||^2: {fmt(rel)} "
                     f"(floored {fmt(num(io.model_level(doc, 'alloc_tensor_rel_frac_floored')), '{:.1%}')})")
    lines.append("- s_t is in absolute output units: across types it tracks the input scale of the linear, so the model-wide "
                 "saving is outside the D ~ 2^-2b model for the floored weights; the Spearman above is the test of the proxy")
    return lines


def sensitivity_checks(doc: dict | None, rows: list[dict]) -> dict:
    """Spearman of s_t vs @only KL per type (4 bit) and of sum_t s_t per layer vs @layer KL (3 bit)."""
    out: dict = {}
    if doc is None:
        return out
    recs = records(doc, types=io.TYPES)
    only = io.ablation_only(rows, 4)
    xs, ys = [], []
    for t, trs in by_type(recs).items():
        s = sum(v for v in (num(io.val(r, "stat", "s_t")) for r in trs) if v is not None)
        k = num(only.get(t, {}).get("kl")) if t in only else None
        if s > 0 and k is not None:
            xs.append(s)
            ys.append(k)
    out["spearman_type"] = io.spearman(xs, ys)
    out["n_type"] = len(xs)
    layers = io.ablation_only(rows, 3, "layer")
    per_layer: dict[int, float] = {}
    for r in recs:
        s = num(io.val(r, "stat", "s_t"))
        if s is not None and isinstance(r.get("layer"), int):
            per_layer[r["layer"]] = per_layer.get(r["layer"], 0.0) + s
    xs, ys = [], []
    for l, s in sorted(per_layer.items()):
        k = num(layers.get(str(l), {}).get("kl")) if str(l) in layers else None
        if k is not None:
            xs.append(s)
            ys.append(k)
    out["spearman_layer"] = io.spearman(xs, ys)
    out["n_layer"] = len(xs)
    return out


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def build(results_dir: Path, models: list[str], include_smoke: bool = False) -> str:
    parts = [f"# Phase 2 structure table\n\nGenerated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by scripts/structure_table.py "
             f"from `{results_dir.as_posix()}/probe/<model>/*.json` and `ablation.jsonl`"
             + (" (smoke rows included)" if include_smoke else "") + ".\n",
             "Verdict rule (DESIGN.md Phase 2): a claim passes the shuffle test iff |gain_bits - gain_bits_control| exceeds both "
             f"{GAIN_FLOOR} bits/param and the spread between the two control seeds (where a second seed exists; otherwise marked "
             "\"1 seed\"). P1 is judged against its Gaussian control (the shuffle is vacuous for a marginal). P5: split-half Spearman > 0.9. "
             "P6: Spearman(s_t, measured @only KL) > 0.7. P7: dPPL > 100 % with the magnitude control < 2 %. "
             "`claimed` = gain_bits - gain_bits_control is the structure value in bits/param; `spread` = |control seed 0 - seed 1|. "
             "Values are numel-weighted means over the tensors of a type.\n"]
    header = ["model", "probe", "type", "key statistic", "W", "control", "gain_bits", "gain_bits_control", "claimed", "spread", "verdict", "note"]
    notes = []
    for model in models:
        pdir = io.probe_dir(results_dir, model)
        abl = io.merge_rows(io.load_rows(results_dir / model / "runs.jsonl", include_smoke), io.load_rows(pdir / "ablation.jsonl", include_smoke))
        parts.append(f"\n## {model}\n")
        p0_rows, extra = ablation_summary(abl)
        parts.append("### P0 additivity: sum of @only KL vs joint KL (rtn-sym-g128, same bits)\n")
        if p0_rows:
            parts.append(md_table(["bits", "types", "sum @only KL", "joint KL", "sum / joint", "three largest @only KL", "top-3 share of joint", "H-P0 additivity"], p0_rows))
        else:
            parts.append("_no @only rows in ablation.jsonl_")
            notes.append(f"{model}: no ablation rows")
        h_lines = p0_hypothesis_lines(abl)
        if h_lines:
            parts.append("\n### P0 pre-registered figures: H-P3 (+keepcols recovery) and H-P0 group size\n")
            parts.extend(h_lines)
        parts.append("")
        docs = {p: io.load_probe(pdir / f"{p}.json") for p in io.PROBES}
        checks = sensitivity_checks(docs["sensitivity"], abl)
        extra.update(checks)
        if docs["sensitivity"] is not None:
            parts.append("### P6 prediction check and allocation saving\n")
            parts.append(f"- Spearman(s_t per type, rtn-sym-g128-4@only KL): {fmt(checks.get('spearman_type'), '{:.3f}')} over {checks.get('n_type', 0)} types (contract > 0.7)")
            parts.append(f"- Spearman(sum of s_t per layer, rtn-sym-g128-3@layer KL): {fmt(checks.get('spearman_layer'), '{:.3f}')} over {checks.get('n_layer', 0)} layers")
            parts.extend(allocation_lines(docs["sensitivity"]))
            parts.append("")
        table_rows = []
        for probe in KEY_STATS:
            r = probe_rows(model, probe, docs[probe], extra)
            if not r and docs[probe] is None:
                notes.append(f"{model}: {probe}.json missing" + (" (not in the Phase 2 run list, DESIGN.md)" if (model, probe) in NOT_SCHEDULED else ""))
            table_rows += r
        parts.append("### Probes P1-P6 per weight type\n")
        parts.append(md_table(header, table_rows) if table_rows else "_no probe files_")
        parts.append("")
        sw = superweight_rows(model, docs["superweight"])
        parts.append("### P7 super-weight candidates\n")
        if sw:
            parts.append(md_table(["model", "tensor", "layer", "type", "[row, col]", "value", "dPPL %", "KL", "dPPL % magnitude control", "dPPL % random", "candidate", "verdict"], sw))
        else:
            parts.append("_no candidates_" if docs["superweight"] is not None else "_superweight.json missing_")
        parts.append("")
    if notes:
        parts.append("\n## Skipped\n")
        parts += [f"- {n}" for n in notes]
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma-separated model tags")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="output markdown (default <results-dir>/structure_table.md)")
    ap.add_argument("--include-smoke", action="store_true", help="also read limit-windows / kl-windows / calib-n ablation rows")
    args = ap.parse_args()
    rdir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else ROOT / args.results_dir
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    text = build(rdir, models, args.include_smoke)
    out = Path(args.out) if args.out else rdir / "structure_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
