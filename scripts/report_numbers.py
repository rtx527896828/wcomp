"""Numbers quoted in REPORT.md that are not printed verbatim in NOTES.md / results/phase3/summary.md, recomputed from the
result files (CPU only, no model):

  uv run python scripts/report_numbers.py

Prints: (1) Hadamard vs uniform gptq32 per width and seed (same-seed KL ratio, curve ratio, coding gain fixed - huff);
(2) the equal-KL points of the had curve against the uniform gptq32 points (fixed and Huffman, log-linear chords);
(3) the verdict of every Phase 3 method recomputed with scripts/phase3_common.compare (cross-check of summary.md);
(4) seed counts per method (printed with (3)), so the report never says "both seeds" where only one exists;
(5) model-wide structure values: the per-type gain_bits / gain_bits_control / claimed of results/structure_table.md
    weighted by the type's parameter count (numel from results/phase3/<model>/damage_c4.jsonl, unit "type");
(6) extras: row counts and code versions, reference dPPL per seed, coding-gain range, two-seed spreads, combination
    totals against the Hadamard curve, the had-3 range;
(7) the permutation null of the Phase 2 "damage per parameter tracks 1 / n_t" finding (REPORT.md section 6): per-type
    KL of rtn-sym-g128-4@only=<type> (results/probe/<model>/ablation.jsonl, WikiText-2) and of the Phase 3 C4 damage
    rows (damage_c4.jsonl, unit "type"), numel from damage_c4.jsonl; Spearman(KL / n, 1 / n) against 5000 permutations
    of KL across types (numpy rng seed 1), Spearman(KL, n) and the log-log slope of KL on n.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase3_common as p3  # noqa: E402

MODELS = ("qwen3.5-0.8b", "qwen3.5-2b", "minicpm5-2b")


def mean(v):
    return sum(v) / len(v)


def curve_of(rows, prefix, fp16):
    """Seed-mean points {bits: (kl, huff, dppl, n)} of the rows named prefix + width."""
    pts = {}
    for r in rows:
        m = r["method"]
        if m.startswith(prefix) and m[len(prefix):].isdigit():
            pts.setdefault(float(r["bits_blocks"]), []).append(r)
    out = {}
    for b, rs in sorted(pts.items()):
        out[b] = (mean([r["kl"] for r in rs]), mean([r["bits_blocks_huff"] for r in rs]),
                  mean([r["ppl"] for r in rs]) - fp16, len(rs))
    return out


def main() -> None:
    for model in MODELS:
        rows = p3.load_rows(ROOT / "results" / "phase3" / model / "runs.jsonl")
        p1 = p3.load_rows(ROOT / "results" / model / "runs.jsonl")
        fp16 = p3.fp16_ppl(rows, p1)
        print(f"\n=== {model}  fp16 {fp16:.4f}")
        ref = curve_of(rows, "gptq32-sym-g128-", fp16)
        had = curve_of(rows, "had-gptq32-sym-g128-", fp16)
        print("(1) had vs gptq32, per width and seed")
        for b in sorted(had):
            w = int(round(b - 0.125))
            for s in (0, 1):
                h = p3.find(rows, f"had-gptq32-sym-g128-{w}", s)
                g = p3.find(rows, f"gptq32-sym-g128-{w}", s)
                if not h:
                    continue
                h = h[0]
                g_txt = f"same-seed KL ratio {h['kl'] / g[0]['kl']:.3f}" if g else "no same-seed gptq32 row"
                print(f"  {w}b seed {s}: had KL {h['kl']:.5f} huff {h['bits_blocks_huff']:.4f} (coding gain "
                      f"{h['bits_blocks'] - h['bits_blocks_huff']:.3f}); curve KL {ref[b][0]:.5f} ratio {h['kl'] / ref[b][0]:.3f}; {g_txt}; "
                      f"dPPL {100 * (h['ppl'] / fp16 - 1):+.1f}%")
            print(f"  {w}b gptq32 coding gain (fixed - huff, seed mean) {b - ref[b][1]:.3f}; had {b - had[b][1]:.3f}")
        print("(2) equal-KL points of the had curve (log-linear chord in bits) at the gptq32 seed-mean KLs")
        hb = sorted(had)
        if len(hb) >= 2:
            hk = [had[b][0] for b in hb]
            hh = [had[b][1] for b in hb]
            for b in sorted(ref):
                kl = ref[b][0]
                if not (min(hk) <= kl <= max(hk)):
                    print(f"  gptq32 {b:.3f} (KL {kl:.5f}) outside the had curve's KL range {min(hk):.4f}-{max(hk):.4f}: not read")
                    continue
                xf = p3.invert_log(hb, hk, kl)
                xh = p3.invert_log(hh, hk, kl)
                print(f"  gptq32 {b:.3f} fixed / {ref[b][1]:.3f} huff (KL {kl:.5f}): had reaches it at {xf:.3f} fixed / {xh:.3f} huff "
                      f"-> saves {b - xf:.3f} vs fixed-vs-fixed, {b - xh:.3f} had-huff vs gptq32-fixed, {ref[b][1] - xh:.3f} huff vs huff")
        print("(3) verdicts recomputed with phase3_common.compare")
        curve = p3.reference_curve(rows, fp16)
        groups = p3.by_method([r for r in rows if r["method"] not in ("fp16",)])
        for m in sorted(groups, key=lambda m: p3.sort_key(groups[m][0])):
            fam = p3.parse_name(m)["family"]
            if fam == "reference":
                continue
            cmps = [p3.compare(r, curve, fp16) for r in groups[m]]
            v = p3.method_verdict(cmps, fam)
            ks = " / ".join(f"{c['ratio_kl']:.3f}" if c["ratio_kl"] is not None else "-" for c in cmps)
            ds = " / ".join(f"{c['ratio_dppl']:.3f}" if c["ratio_dppl"] is not None else "-" for c in cmps)
            sv = " / ".join(f"{c['saved_fixed']:+.3f}" if c["saved_fixed"] is not None else "-" for c in cmps)
            sh = " / ".join(f"{c['saved_huff']:+.3f}" if c["saved_huff"] is not None else "-" for c in cmps)
            seeds = ",".join(str(r.get("seed", 0)) for r in groups[m])
            print(f"  {m:32s} seeds {seeds:4s} bits {groups[m][0]['bits_blocks']:.4f} KLratio {ks:13s} dPPLratio {ds:13s} "
                  f"saved fixed {sv:15s} huff {sh:15s} -> {v}")
    structure_model_wide()
    extras()
    ratio_permutation()


def ratio_permutation(n_perm: int = 5000, seed: int = 1) -> None:
    """(7): is Spearman(d = KL / n, 1 / n) above what a KL independent of n gives? KL is permuted across the types."""
    import numpy as np
    from scipy.stats import spearmanr

    print(f"\n(7) permutation null of Spearman(KL/n, 1/n) over weight types ({n_perm} permutations, rng seed {seed})")
    for model in MODELS:
        c4 = [json.loads(x) for x in (ROOT / "results" / "phase3" / model / "damage_c4.jsonl").read_text(encoding="utf-8").splitlines()
              if x.strip()]
        numel = {r["name"]: r["numel"] for r in c4 if r.get("unit") == "type"}
        kl_c4 = {r["name"]: r["kl"] for r in c4 if r.get("unit") == "type"}
        kl_wt = {}
        for x in (ROOT / "results" / "probe" / model / "ablation.jsonl").read_text(encoding="utf-8").splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            if r["method"].startswith("rtn-sym-g128-4@only="):
                kl_wt[r["method"].split("=", 1)[1]] = r["kl"]
        for src, kd in (("WikiText-2 (Phase 2 @only)", kl_wt), ("C4 (Phase 3 damage)", kl_c4)):
            names = [k for k in numel if k in kd]
            if len(names) < 3:
                print(f"  {model:13s} {src}: only {len(names)} types, skipped")
                continue
            n = np.array([numel[k] for k in names], float)
            kl = np.array([kd[k] for k in names], float)
            rho = spearmanr(kl / n, 1 / n)[0]
            rng = np.random.default_rng(seed)
            null = np.array([spearmanr(rng.permutation(kl) / n, 1 / n)[0] for _ in range(n_perm)])
            slope = np.polyfit(np.log(n), np.log(kl), 1)[0]
            print(f"  {model:13s} {src:27s} types {len(names):2d}: rho(KL/n, 1/n) {rho:.3f}, null median {np.median(null):.3f}, "
                  f"p(null >= obs) {(null >= rho).mean():.3f}; rho(KL, n) {spearmanr(kl, n)[0]:+.3f}; log-log slope {slope:.2f}")


def extras() -> None:
    """(6): the remaining REPORT.md figures: row counts and code versions, the reference rows' dPPL per seed at 4 bits,
    the Huffman coding gain range, the two-seed spread over every pair, the combination's total bits saved (coding
    included) and its KL against the Hadamard curve, and the headline had-3 range."""
    print("\n(6) extras")
    for model in MODELS:
        rows = p3.load_rows(ROOT / "results" / "phase3" / model / "runs.jsonl")
        p1 = p3.load_rows(ROOT / "results" / model / "runs.jsonl")
        fp16 = p3.fp16_ppl(rows, p1)
        vers = sorted({r.get("git_or_code_version") for r in rows})
        print(f"  {model}: {len(rows)} rows, code versions {vers}")
        for r in p3.find(rows, "gptq32-sym-g128-4"):
            print(f"    gptq32-4 seed {r['seed']}: ppl {r['ppl']:.4f} dPPL {r['ppl'] - fp16:.4f} ({100 * (r['ppl'] / fp16 - 1):+.2f}%) kl {r['kl']:.5f}")
        q = [r for r in rows if r["method"] != "fp16" and r.get("bits_blocks_huff") is not None]
        ref = [r for r in q if p3.REF_RE.match(r["method"])]
        g = [r["bits_blocks"] - r["bits_blocks_huff"] for r in ref]
        a = [r["bits_blocks"] - r["bits_blocks_huff"] for r in q]
        print(f"    coding gain fixed - huff: gptq32 rows {min(g):.3f}-{max(g):.3f}; all quantized rows {min(a):.3f}-{max(a):.3f}")
        spreads = []
        for m, rs in p3.by_method([r for r in rows if r["method"] != "fp16"]).items():
            if len(rs) == 2:
                k = [float(x["kl"]) for x in rs]
                spreads.append((max(k) / min(k) - 1, m))
        if spreads:
            s = sorted(x for x, _ in spreads)
            med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
            print(f"    two-seed KL spread over {len(s)} pairs: median {100 * med:.1f}% max {100 * s[-1]:.1f}% "
                  f"({max(spreads)[1]}); pairs > 5%: {sum(x > 0.05 for x in s)}")
        curve = p3.reference_curve(rows, fp16)
        had = {}
        for r in rows:
            mm = p3.parse_name(r["method"])
            if mm["family"] == "rotation" and mm["kind"] == "had":
                had.setdefault(float(r["bits_blocks"]), []).append(float(r["kl"]))
        hb = sorted(had)
        hk = [mean(had[b]) for b in hb]
        for r in sorted(p3.find(rows, "combo-gptq32-avg3") + p3.find(rows, "combo-gptq32-avg2.5") + p3.find(rows, "combo-gptq32-avg3.5"),
                        key=lambda r: (r["method"], r["seed"])):
            c = p3.compare(r, curve, fp16)
            total = c["saved_fixed"] + r["bits_blocks"] - r["bits_blocks_huff"]
            vs_had = r["kl"] / p3.interp_log(hb, hk, float(r["bits_blocks"])) if len(hb) >= 2 else (
                r["kl"] / hk[0] if hb else None)
            print(f"    {r['method']} seed {r['seed']}: KL {r['kl']:.5f}, x gptq32-4 seed-mean KL {r['kl'] / curve.kl_at(4.125):.2f}; "
                  f"total saved vs fixed gptq32 (coding incl.) {total:.3f}; KL / had curve {vs_had:.3f}")
        rat = [c for c in (p3.compare(r, curve, fp16) for r in p3.find(rows, "had-gptq32-sym-g128-3"))]
        print("    had-3 KL ratio vs curve: " + ", ".join(f"{c['ratio_kl']:.3f} (-{100 * (1 - c['ratio_kl']):.1f}%)" for c in rat))


def structure_model_wide() -> None:
    """(5): numel-weighted model-wide gain_bits, control and claimed per probe from the per-type rows of structure_table.md."""
    print("\n(5) model-wide (numel-weighted over block-linear types) structure values from results/structure_table.md")
    rows = {}
    for line in (ROOT / "results" / "structure_table.md").read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 12 and c[0] in MODELS and c[1] in ("marginal", "lowrank", "lowrank_weighted", "outliers", "interlayer"):
            try:
                g, gc, cl = float(c[6]), float(c[7]), float(c[8])
            except ValueError:
                continue
            rows.setdefault((c[0], c[1]), {})[c[2]] = (g, gc, cl)
    for model in MODELS:
        numel = {}
        for line in (ROOT / "results" / "phase3" / model / "damage_c4.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("unit") == "type":
                numel[r["name"]] = r["numel"]
        tot = sum(numel.values())
        for probe in ("marginal", "lowrank", "lowrank_weighted", "outliers", "interlayer"):
            t = rows.get((model, probe))
            if not t:
                continue
            types = [k for k in t if k in numel]
            w = sum(numel[k] for k in types)
            vals = [sum(numel[k] * t[k][i] for k in types) / w for i in range(3)]
            print(f"  {model:13s} {probe:17s} types {len(types):2d} ({w / tot:.0%} of block params): gain {vals[0]:+.3f}  "
                  f"control {vals[1]:+.3f}  claimed {vals[2]:+.3f}")


if __name__ == "__main__":
    main()
