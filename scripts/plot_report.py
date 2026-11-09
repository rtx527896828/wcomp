"""REPORT.md rate-distortion figures: every Phase 3 method on one panel per model (CPU only, no model).

  uv run python scripts/plot_report.py
  -> results/report_rd.png      (y = KL(P_ref || P_q) on the first 32 WikiText-2 windows, log)
     results/report_rd_ppl.png  (y = PPL / PPL_fp16 - 1 in %, all WikiText-2 windows, log)

x = bits_blocks (fixed-length indices + all side information, DESIGN.md "Effective bits"). Lines: the gptq32-sym-g128
reference (seed mean, solid) and the same KLs at its own bits_blocks_huff (dashed, hollow); had-gptq32 (Hadamard + GPTQ,
seed mean, solid) and its Huffman figure (dashed, hollow); Phase 1 uniform RTN rtn-sym-g128-{2,3,4} (grey context, from
<phase1-dir>/<model>/runs.jsonl). Every other Phase 3 method is one marker at its fixed bits: the seed mean of the metric
with a vertical bar from the lowest to the highest seed (no bar for a single seed). A point measured on one seed only
(method or reference / Hadamard curve point) gets a thin dark ring, and the footer lists those methods per model, so a
missing bar is never mistaken for a tight two-seed pair. The combination rows and the codebook rows of letter E (lloyd,
nf, nfq) also get a hollow marker at their Huffman bits joined by a thin line (the horizontal distance is their own
coding gain), so E is judged on both curves (fixed against the solid reference, coded against the dashed one). Markers
are not dodged: methods at the same bits share the x position, so the vertical distance to the reference line at that
x is exact. A row beyond the x range (the eig rotation probe, ~17.9 bits of stored bf16 rotations) is drawn with its own
marker at the right edge and its bits in the label. The unquantized model (method name "fp16" in runs.jsonl, weights in
bf16) has KL 0 and dPPL 0: it cannot sit on a log axis and is stated in the footer. The dPPL y limits are set once after
every panel is drawn (sharey), from the lowest plotted value of all panels. Rows are read with
scripts/phase3_common.load_rows (latest full-protocol row per run key; smoke rows dropped).
"""
from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase3_common as p3  # noqa: E402
from plot_rd import GRID, SURFACE, TEXT, TEXT2  # noqa: E402

MODELS = ("qwen3.5-0.8b", "qwen3.5-2b", "minicpm5-2b")
X_LIM = (1.55, 4.40)
# Validated categorical slots (scripts/plot_rd.COLORS = the dataviz reference palette, light mode), one per owner letter;
# identity inside a letter is carried by the marker shape and the legend text, never by colour alone.
INK, RTN_GREY = "#2b2a28", "#9a9993"
C_D, C_E, C_A, C_G, C_COMBO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
# kind key -> (legend text, colour, marker, fill) for the marker-drawn methods; fill: "full", "none" or "bottom" (half).
KIND_STYLE = {
    "rnd": ("D: random orthogonal + GPTQ (0 bit)", C_D, "<", "full"),
    "hadsign": ("D: Hadamard x random signs + GPTQ (0 bit)", C_D, ">", "full"),
    "eig": ("D: Hessian eigenbasis, stored bf16 (probe)", C_D, "8", "full"),
    "lloyd": ("E: Lloyd-Max codebook per tensor + GPTQ", C_E, "D", "full"),
    "nf": ("E: fixed N(0,1) Lloyd codebook (nf) + GPTQ", C_E, "d", "full"),
    "nfq": ("E: Gaussian-quantile codebook (nfq) + GPTQ", C_E, "h", "full"),
    "type": ("A: type-level allocation + GPTQ", C_A, "^", "full"),
    "tensor": ("A: tensor-level allocation + GPTQ", C_A, "p", "full"),
    "type-rtn": ("A: type-level allocation on RTN (gate)", C_A, "*", "full"),
    "shuf": ("A: shuffled allocation (control) + GPTQ", C_A, "X", "full"),
    "fisher": ("G: Fisher row allocation + GPTQ", C_G, "v", "full"),
    "fisher-rtn": ("G: Fisher row allocation on RTN (gate)", C_G, "*", "full"),
    "shuf-rtn": ("G: shuffled rows (control) on RTN", C_G, "X", "full"),
    "combo": ("combination: had + A + G + GPTQ", C_COMBO, "P", "full"),
}
KIND_ORDER = tuple(KIND_STYLE)
HUFF_KINDS = ("lloyd", "nf", "nfq", "combo")  # marker-drawn kinds that also get a hollow twin at bits_blocks_huff
FOOTER_WRAP = 230


def short(method: str) -> str:
    """Compact method name for the footer (drops the quantizer / grid tokens every Phase 3 row shares)."""
    return method.replace("-gptq32", "").replace("-sym-g128", "").replace("-g128", "")


def one_seed_ring(ax, x: float, y: float, marker: str) -> None:
    """Thin dark ring around a point measured on one seed only (no seed bar is possible there)."""
    ax.plot([x], [y], marker="o", markersize=16 if marker == "*" else 14, markerfacecolor="none", markeredgecolor=INK,
            markeredgewidth=0.9, linestyle="none", zorder=7)


def kind_of(method: str) -> str | None:
    """KIND_STYLE key of a Phase 3 method, None for the rows drawn as lines (reference, had) or not drawn."""
    p = p3.parse_name(method)
    fam, kind, rtn = p["family"], p["kind"], p["quant"] == "rtn"
    if fam == "rotation":
        return None if kind == "had" else kind
    if fam == "codebook":
        return kind
    if fam == "alloc":
        if p["shuffle"]:
            return "shuf"
        return "type-rtn" if rtn else kind
    if fam == "rowalloc":
        if p["shuffle"]:
            return "shuf-rtn"
        return "fisher-rtn" if rtn else "fisher"
    if fam == "combo":
        return "combo"
    return None


def metric(r: dict, fp16: float, key: str) -> float | None:
    """KL, or dPPL in percent of the fp16 PPL."""
    if key == "kl":
        return r.get("kl")
    return None if r.get("ppl") is None else 100.0 * (float(r["ppl"]) / fp16 - 1.0)


def positive(y) -> bool:
    return y is not None and math.isfinite(y) and y > 0


def seed_curve(rows: list[dict], prefix: str, fp16: float, key: str) -> list[dict]:
    """Seed-mean points of the uniform-width rows prefix + <bits> ({bits, huff, y, lo, hi, n}), sorted by bits."""
    by_bits: dict[float, list[dict]] = {}
    for r in rows:
        m = r["method"]
        if m.startswith(prefix) and m[len(prefix):].isdigit():
            by_bits.setdefault(round(float(r["bits_blocks"]), 6), []).append(r)
    out = []
    for b, rs in sorted(by_bits.items()):
        ys = [metric(r, fp16, key) for r in rs]
        if not all(positive(y) for y in ys):
            continue
        hs = [r.get("bits_blocks_huff") for r in rs]
        out.append({"bits": b, "huff": sum(hs) / len(hs) if all(h is not None for h in hs) else None,
                    "y": sum(ys) / len(ys), "lo": min(ys), "hi": max(ys), "n": len(rs)})
    return out


def method_points(rows: list[dict], fp16: float, key: str) -> tuple[list[dict], list[str]]:
    """One point per marker-drawn method: {method, kind, label, bits, huff, y, lo, hi, n}; plus stderr notes."""
    notes = []
    pts = []
    groups = p3.by_method([r for r in rows if kind_of(r["method"]) is not None])
    for m in sorted(groups, key=lambda m: p3.sort_key(groups[m][0])):
        rs = groups[m]
        ys = [metric(r, fp16, key) for r in rs]
        if not all(positive(y) for y in ys):
            notes.append(f"{m}: {key} {ys} not all positive, not on the log axis, skipped")
            continue
        hs = [r.get("bits_blocks_huff") for r in rs]
        p = p3.parse_name(m)
        pts.append({"method": m, "kind": kind_of(m), "label": p["label"], "bits": sum(float(r["bits_blocks"]) for r in rs) / len(rs),
                    "huff": sum(hs) / len(hs) if all(h is not None for h in hs) else None,
                    "y": sum(ys) / len(ys), "lo": min(ys), "hi": max(ys), "n": len(rs),
                    "seeds": sorted(r.get("seed", 0) for r in rs)})
    return pts, notes


DPPL_CAP = 2000.0  # percent: an RTN context point above this (2-bit RTN, +1.7e5% / +5.7e6%) would flatten the dPPL panels


def rtn_curve(p1_rows: list[dict], fp16: float, key: str) -> list[dict]:
    """Phase 1 rtn-sym-g128-{2,3,4} (seed 0) as curve points (the 8-bit row is outside the x range). In the dPPL figure a
    point above DPPL_CAP is returned with "off": True (drawn as an up-arrow at the top edge, its value in the footer)."""
    out = []
    for r in p1_rows:
        m = r["method"]
        if m.startswith("rtn-sym-g128-") and m.rsplit("-", 1)[1] in ("2", "3", "4") and r.get("seed", 0) == 0:
            y = metric(r, fp16, key)
            if positive(y):
                out.append({"bits": float(r["bits_blocks"]), "y": y, "off": key == "dppl" and y > DPPL_CAP})
    return sorted(out, key=lambda p: p["bits"])


def draw_panel(ax, model: str, rows: list[dict], p1_rows: list[dict], fp16: float, key: str) -> list[str]:
    notes: list[str] = []
    rtn = rtn_curve(p1_rows, fp16, key)
    on = [p for p in rtn if not p["off"]]
    if on:
        ax.plot([p["bits"] for p in on], [p["y"] for p in on], color=RTN_GREY, linewidth=1.6, marker="o", markersize=5, zorder=2)
    for p in rtn:
        if p["off"]:
            ax.plot([p["bits"]], [DPPL_CAP], marker="^", markersize=8, color=RTN_GREY, linestyle="none", zorder=2)
            ax.annotate(f"RTN {p['bits']:.3f} b: {p['y']:.2g}%", (p["bits"], DPPL_CAP), textcoords="offset points", xytext=(6, -4),
                        ha="left", va="top", fontsize=8.5, color=TEXT2)
            notes.append(f"{model}: RTN at {p['bits']:.3f} bits, dPPL {p['y']:.3g}% above the {DPPL_CAP:g}% cap, drawn at the top edge")
    for prefix, color, marker in (("gptq32-sym-g128-", INK, "o"), ("had-gptq32-sym-g128-", C_D, "s")):
        cur = seed_curve(rows, prefix, fp16, key)
        if not cur:
            continue
        xs, ys = [p["bits"] for p in cur], [p["y"] for p in cur]
        ax.plot(xs, ys, color=color, linewidth=2.2, marker=marker, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1, zorder=5)
        for p in cur:
            if p["n"] > 1 and p["hi"] > p["lo"]:
                ax.plot([p["bits"]] * 2, [p["lo"], p["hi"]], color=color, linewidth=1.2, zorder=4)
            if p["n"] == 1:
                one_seed_ring(ax, p["bits"], p["y"], marker)
        hc = [p for p in cur if p["huff"] is not None]
        if hc:
            ax.plot([p["huff"] for p in hc], [p["y"] for p in hc], color=color, linewidth=1.6, linestyle="--", marker=marker,
                    markersize=7, markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.5, zorder=5)
    pts, n = method_points(rows, fp16, key)
    notes += n
    for p in pts:
        label, color, marker, fill = KIND_STYLE[p["kind"]]
        x = p["bits"]
        if x > X_LIM[1]:
            # Off the x range: drawn with the kind's own marker (so the legend swatch matches) at the right edge.
            xe = X_LIM[1] - 0.04
            ax.plot([xe], [p["y"]], marker=marker, markersize=9, color=color, markeredgecolor=SURFACE, linestyle="none", zorder=6)
            if p["n"] == 1:
                one_seed_ring(ax, xe, p["y"], marker)
            ax.annotate(f"{p['label']}: at {x:.1f} bits, off scale →", (xe, p["y"]), textcoords="offset points",
                        xytext=(-11, 7), ha="right", fontsize=8.5, color=TEXT2)
            notes.append(f"{model}: {p['method']} at {x:.2f} bits drawn at the right edge")
            continue
        if p["n"] > 1 and p["hi"] > p["lo"]:
            ax.plot([x, x], [p["lo"], p["hi"]], color=color, linewidth=1.2, zorder=4)
        kw = dict(marker=marker, markersize=10 if marker == "*" else 8, color=color, linestyle="none", zorder=6)
        if fill == "bottom":
            kw.update(fillstyle="bottom", markerfacecoloralt=SURFACE, markeredgecolor=color, markeredgewidth=1.3)
        else:
            kw.update(markeredgecolor=SURFACE, markeredgewidth=0.9)
        ax.plot([x], [p["y"]], **kw)
        if p["n"] == 1:
            one_seed_ring(ax, x, p["y"], marker)
        if p["kind"] in HUFF_KINDS and p["huff"] is not None:
            ax.plot([p["huff"], x], [p["y"], p["y"]], color=color, linewidth=1.0, alpha=0.8, zorder=3)
            ax.plot([p["huff"]], [p["y"]], marker=marker, markersize=8, markerfacecolor=SURFACE, markeredgecolor=color,
                    markeredgewidth=1.4, linestyle="none", zorder=6)
    return notes


def legend_handles(kinds_present: set[str], have_rtn: bool) -> tuple[list, list[str]]:
    hs, ts = [], []

    def add(h, t):
        hs.append(h)
        ts.append(t)

    add(Line2D([], [], color=INK, linewidth=2.2, marker="o", markersize=7, markeredgecolor=SURFACE),
        "GPTQ g128 uniform (gptq32, reference), fixed bits")
    add(Line2D([], [], color=INK, linewidth=1.6, linestyle="--", marker="o", markersize=7, markerfacecolor=SURFACE),
        "same reference, Huffman-coded bits")
    add(Line2D([], [], color=C_D, linewidth=2.2, marker="s", markersize=7, markeredgecolor=SURFACE),
        "D: Hadamard + GPTQ (had-gptq32), fixed bits")
    add(Line2D([], [], color=C_D, linewidth=1.6, linestyle="--", marker="s", markersize=7, markerfacecolor=SURFACE),
        "D: Hadamard + GPTQ, Huffman-coded bits")
    if have_rtn:
        add(Line2D([], [], color=RTN_GREY, linewidth=1.6, marker="o", markersize=5), "RTN g128 uniform (Phase 1, context)")
    e_kinds = [k for k in KIND_ORDER if k in kinds_present and k in HUFF_KINDS and k != "combo"]
    last_e = e_kinds[-1] if e_kinds else None
    for k in KIND_ORDER:
        if k not in kinds_present:
            continue
        label, color, marker, fill = KIND_STYLE[k]
        kw = dict(color=color, marker=marker, markersize=10 if marker == "*" else 8, linestyle="none")
        if fill == "bottom":
            kw.update(fillstyle="bottom", markerfacecoloralt=SURFACE, markeredgecolor=color)
        else:
            kw.update(markeredgecolor=SURFACE)
        add(Line2D([], [], **kw), label)
        if k == last_e:
            add(Line2D([], [], color=C_E, marker="D", markersize=7, markerfacecolor=SURFACE, linewidth=1.0),
                "E codebook rows at their Huffman-coded bits")
        if k == "combo":
            add(Line2D([], [], color=C_COMBO, marker="P", markersize=8, markerfacecolor=SURFACE, linewidth=1.0),
                "combination at its Huffman-coded bits")
    add(Line2D([], [], color=TEXT2, linewidth=1.2, marker="_", markersize=8), "vertical bar: seed min to max (marker = seed mean)")
    add(Line2D([], [], marker="o", markersize=12, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.9, linestyle="none"),
        "dark ring: one seed only (no bar possible)")
    return hs, ts


def single_seed_methods(rows: list[dict]) -> list[str]:
    """Drawn methods (markers, reference and Hadamard curve points) that have exactly one seed, as compact names."""
    out = []
    groups = p3.by_method(rows)
    for m in sorted(groups, key=lambda m: p3.sort_key(groups[m][0])):
        drawn = kind_of(m) is not None or any(m.startswith(pfx) and m[len(pfx):].isdigit()
                                              for pfx in ("gptq32-sym-g128-", "had-gptq32-sym-g128-"))
        if drawn and len({r.get("seed", 0) for r in groups[m]}) == 1:
            out.append(short(m))
    return out


def rel(p: Path) -> str:
    """p relative to the project root when it is inside it (the footer never prints a machine-specific path)."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def render(models: list[str], rdir: Path, p1dir: Path, key: str, out: Path) -> list[str]:
    notes: list[str] = []
    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 8.6), sharey=True, facecolor=SURFACE, squeeze=False)
    axes = axes[0]
    kinds: set[str] = set()
    have_rtn = False
    fp16s = {}
    seeds_txt = []
    one_seed: list[str] = []
    for ax, model in zip(axes, models):
        rows = p3.load_rows(rdir / model / "runs.jsonl")
        p1 = p3.load_rows(p1dir / model / "runs.jsonl")
        rows = [r for r in rows if r["method"] not in ("fp16", "had-fp16")]
        fp16 = p3.fp16_ppl(p3.load_rows(rdir / model / "runs.jsonl"), p1)
        if fp16 is None:
            raise SystemExit(f"{model}: no full-protocol fp16 row")
        fp16s[model] = fp16
        notes += draw_panel(ax, model, rows, p1, fp16, key)
        kinds |= {kind_of(r["method"]) for r in rows} - {None}
        have_rtn = have_rtn or bool(rtn_curve(p1, fp16, key))
        ref = seed_curve(rows, "gptq32-sym-g128-", fp16, key)
        seeds_txt.append(f"{model} " + ", ".join(f"{p['bits']:.3f}: {p['n']}" for p in ref))
        one_seed.append(f"{model}: " + (", ".join(single_seed_methods(rows)) or "none"))
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.set_xlim(*X_LIM)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.4, alpha=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT2, labelsize=10)
        ax.set_title(f"{model}  (bf16 PPL {fp16:.2f})", color=TEXT, fontsize=12, loc="left")
        ax.set_xlabel("effective bits / parameter of the block linears (bits_blocks)", color=TEXT2, fontsize=10)
    if key == "dppl":
        # sharey: set the limits once, after every panel is drawn, so no panel's data is clipped by the first panel's
        # autoscale (the 2B Hadamard 4.125-bit seed 1 sits at dPPL 0.43%, below the 0.8B panel's minimum).
        lows = [ax.dataLim.y0 for ax in axes if math.isfinite(ax.dataLim.y0) and ax.dataLim.y0 > 0]
        axes[0].set_ylim(0.8 * min(lows) if lows else None, DPPL_CAP * 1.6)
    ylab = ("KL(P_bf16 || P_q), nats / token (65,504 WikiText-2 tokens), log" if key == "kl"
            else "PPL / PPL_bf16 - 1 in %, all WikiText-2 test windows, log")
    axes[0].set_ylabel(ylab, color=TEXT2, fontsize=10)
    title = ("Rate-distortion of every Phase 3 method: KL to the bf16 model" if key == "kl"
             else "Rate-distortion of every Phase 3 method: WikiText-2 perplexity increase")
    fig.suptitle(title, color=TEXT, fontsize=14, x=0.01, ha="left")
    hs, ts = legend_handles(kinds, have_rtn)
    footer = [
        f"Sources: {rel(rdir)}/<model>/runs.jsonl (Phase 3 rows) and {rel(p1dir)}/<model>/runs.jsonl "
        "(fp16 and the Phase 1 RTN context). Reference and Hadamard lines join seed means; reference seeds per width (bits: n): "
        + "; ".join(seeds_txt) + ". Every value, ratio and verdict is tabulated in results/phase3/summary.md.",
        "The unquantized model (method 'fp16' in runs.jsonl; its weights are bf16, the same model that is the KL reference) has KL 0 "
        "and dPPL 0 and cannot be drawn on a log axis. Markers are not dodged: rows at the same bits share x, so their vertical distance "
        "to the reference line is exact. Hollow = the same rows at their Huffman-coded bits (bits_blocks_huff), drawn for the reference, "
        "Hadamard, the E codebook rows (lloyd, nf, nfq) and the combination: codebook rows are compared on both curves (fixed marker vs "
        "solid reference, hollow twin vs dashed reference). The eig probe (stored bf16 rotations, ~17.9 bits) is drawn at the right "
        "edge with its real bits in the label. RTN 8 bit (0.8B) is outside the x range. Phase 1 / 2 rows other than uniform RTN "
        "(had-RTN, RTN asym, f64 GPTQ, +keepcols, g32 / g64) are not drawn here: see results/rd_<model>.png and "
        "results/probe_<model>_ablation.png.",
        "One seed only (dark ring; never read as a two-seed result): " + "; ".join(one_seed) + ".",
    ]
    if key == "dppl":
        footer.append(f"Phase 1 RTN 2-bit points (dPPL > {DPPL_CAP:g}%) are up-arrows at the top edge with their value. At ~4.1 bits dPPL is "
                      "noise-dominated (from runs.jsonl, dPPL = PPL / PPL_bf16 - 1: the 0.8B gptq32-4 reference seed pair is +2.43% vs "
                      "+5.19%, 113% spread; the 2B had-gptq32-4 pair +3.53% vs +0.43%); read KL there.")
    ftext = fig.text(0.01, 0.005, "\n".join(textwrap.fill(t, FOOTER_WRAP) for t in footer), color=TEXT2, fontsize=8.5,
                     ha="left", va="bottom", linespacing=1.35)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    foot_top = inv.transform((0, ftext.get_window_extent().y1))[1]
    leg = fig.legend(hs, ts, loc="lower center", ncol=4, frameon=False, fontsize=9.5, labelcolor=TEXT,
                     bbox_to_anchor=(0.5, foot_top + 0.005), columnspacing=1.6, handletextpad=0.6)
    fig.canvas.draw()
    leg_top = inv.transform((0, leg.get_window_extent().y1))[1]
    fig.tight_layout(rect=(0, leg_top + 0.005, 1, 0.97), w_pad=2.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(out)
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--results-dir", default="results/phase3", help="Phase 3 tree: <results-dir>/<model>/runs.jsonl")
    ap.add_argument("--phase1-dir", default="results", help="Phase 1 tree: fp16 and the RTN context")
    ap.add_argument("--out", default=None, help="KL figure (default <phase1-dir>/report_rd.png)")
    ap.add_argument("--out-ppl", default=None, help="dPPL figure (default <phase1-dir>/report_rd_ppl.png)")
    ap.add_argument("--no-ppl", action="store_true", help="skip the dPPL figure")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else ROOT / p

    rdir, p1dir = resolve(args.results_dir), resolve(args.phase1_dir)
    models = [m for m in args.models.split(",") if m]
    notes = render(models, rdir, p1dir, "kl", resolve(args.out) if args.out else p1dir / "report_rd.png")
    if not args.no_ppl:
        notes += render(models, rdir, p1dir, "dppl", resolve(args.out_ppl) if args.out_ppl else p1dir / "report_rd_ppl.png")
    for n in sorted(set(notes)):
        print(f"note: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
