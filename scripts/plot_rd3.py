"""Phase 3 rate-distortion plot (DESIGN.md "Phase 3 contract", seeds / output: results/rd3_<model>.png).

  uv run python scripts/plot_rd3.py --model qwen3.5-0.8b

x = bits_blocks. Left panel KL(P_ref || P_q) (log), right panel ppl - ppl_fp16 (log). The reference is the
gptq32-sym-g128-{4,3,2} seed-mean curve (solid, dark; the individual seed rows as small points) and the same KLs at
the baseline's own bits_blocks_huff (dashed, hollow). Every other Phase 3 row is one point per family colour: filled
at its fixed bits_blocks, and, when the row has bits_blocks_huff, hollow at the Huffman bits with a thin connector
(the horizontal distance is that row's own coding gain). Shuffle controls are half-filled at their fixed bits (no
connector, label "shuf<k>"; hollow stays reserved for the Huffman figure). Families: rotation (had / rnd / hadsign /
eig), codebook (lloyd / nf / nfq), alloc (type / tensor / shuffles), rowalloc, combo. Seeds: a seed >= 1 row has a
light centre dot, the seeds of one method are joined by a line at their fixed bits (the seed spread), and the method
carries one label: bit width or avg target with the kind ("had 3", "type 2.5", "avg 3"). fp16 (y = 0, off the log
axes: stated in the legend) and the Phase 1 gptq-sym-g128 rows (f64 Hessian) from results/<model>/runs.jsonl are grey
context. A value that cannot be placed on a log axis (dPPL <= 0) is reported on stderr and skipped. Smoke rows are
ignored (scripts/plot_rd.load_runs rules). Labels are placed after the figure layout is final (place_labels works in
display space); the legend and the wrapped footer are stacked under the panels.
The x range is set from the reference curve and the rows at <= MAX_PLOT_BITS bits (fixed and Huffman figures, x_range);
a row beyond it (the eig probe at ~17.9 bits: its stored bf16 rotation is the point of that row) is drawn as a right-
pointing marker at the right edge, labelled with its bits ("eig 3: 17.9 b"), so the 2-4 bit region where every decision
is read keeps the panel.
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

# Validated categorical palette (scripts/plot_rd.py COLORS), one fixed slot per family; never cycled.
FAMILY_STYLE = {
    "rotation": ("#2a78d6", "s"),
    "codebook": ("#eb6834", "D"),
    "alloc": ("#1baf7a", "^"),
    "rowalloc": ("#eda100", "v"),
    "combo": ("#e87ba4", "P"),
    "other": ("#6250d6", "X"),
}
REF_COLOR, P1_COLOR = "#2b2a28", "#a9a8a2"
MAX_PLOT_BITS = 8.0  # rows above this many bits/param are probes (stored matrices), drawn at the right edge
X_PAD = 0.35  # bits of padding on both sides of the in-range rows
LABEL_GAP, LABEL_STEP = 9, 9  # points: minimum vertical distance between labels, and the push step
LABEL_MAX_PUSH = 2  # steps a label may move from its slot without a leader line to its point
LABEL_MAX_STEPS = 6  # steps tried away from the point (3 towards it)
MARKER_HALF = 5  # points: half size of a marker box a label may not cover
FOOTER_WRAP = 190  # characters per footer line (7.5 pt on the 13 in figure: ~225 fit)
SEED_DOT = 2.6  # points: the light centre dot that marks a seed >= 1 row
FIG_HEIGHT = 8.2  # inches: the 0.8B panels hold ~35 labels in the 3.1-4.1 bit columns
LABEL_SLOTS = {"reference": ((7, -11), "left"), "rotation": ((7, 5), "left"), "codebook": ((-7, 5), "right"),
               "alloc": ((7, -11), "left"), "rowalloc": ((-7, -11), "right"), "combo": ((8, 6), "left"),
               "other": ((-7, 5), "right")}  # "other" is rtn-sym-g128-4 at the right end: its long name goes left


def positive(y: float | None) -> bool:
    return y is not None and math.isfinite(y) and y > 0


def panel_values(r: dict, fp16: float) -> tuple[float | None, float | None]:
    """(kl, dppl) of a row for the two panels."""
    return r.get("kl"), p3.dppl(r, fp16) if r.get("ppl") is not None else None


def place_labels(ax, labels: list[tuple], markers: list[tuple[float, float]] = ()) -> list[tuple[str, float, bool]]:
    """Annotate (x, y, text, color, (dx, dy), ha, bold) with collision avoidance: a label whose box would land within
    LABEL_GAP of an already placed label at an overlapping x, or on one of the drawn `markers` (data (x, y) of every
    point other than its own), is moved in steps of LABEL_STEP points, alternating away from and towards the point
    (LABEL_MAX_STEPS away, 3 towards), until it is clear or the candidates run out. A label moved by more than
    LABEL_MAX_PUSH steps gets a leader line to its point, so it is never read as the caption of a neighbouring marker
    (in the crowded 3.125-bit column a bare label pushed 3-5 steps landed beside another family's marker). Returns
    [(text, chosen dy, leader)] in input order."""
    fig = ax.figure
    fig.canvas.draw()
    placed: list[tuple[float, float, float]] = []  # (x_disp, y_disp, half width in px)
    px = fig.dpi / 72.0
    marks = [tuple(ax.transData.transform((mx, my))) for mx, my in markers]
    out: list[tuple[str, float, bool]] = []
    for x, y, text, color, (dx, dy), ha, bold in labels:
        xd, yd = ax.transData.transform((x, y))
        half_w = 0.5 * len(text) * 4.3 * px  # ~4.3 pt per character at 7.5 pt
        sign = 1 if dy >= 0 else -1
        candidates = [dy]
        for k in range(1, LABEL_MAX_STEPS + 1):
            candidates.append(dy + sign * LABEL_STEP * k)
            if k <= 3:
                candidates.append(dy - sign * LABEL_STEP * k)
        others = [(mx, my) for mx, my in marks if abs(mx - xd) > 0.5 or abs(my - yd) > 0.5]  # not the label's own point
        chosen = candidates[-1]
        for cand in candidates:
            lx = xd + dx * px + (half_w if ha == "left" else -half_w)
            ly = yd + cand * px
            clear_labels = all(abs(ly - py) >= LABEL_GAP * px or abs(lx - pxx) >= hw + half_w for pxx, py, hw in placed)
            clear_marks = all(abs(ly - my) >= (0.5 * LABEL_GAP + MARKER_HALF) * px or abs(lx - mx) >= half_w + MARKER_HALF * px
                              for mx, my in others)
            if clear_labels and clear_marks:
                chosen = cand
                break
        placed.append((xd + dx * px + (half_w if ha == "left" else -half_w), yd + chosen * px, half_w))
        leader = abs(chosen - dy) > LABEL_MAX_PUSH * LABEL_STEP
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, chosen), ha=ha, fontsize=7.5, color=color,
                    fontweight="bold" if bold else "normal",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none", alpha=0.85), zorder=7,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6, alpha=0.8, shrinkA=0, shrinkB=4) if leader else None)
        out.append((text, chosen, leader))
    return out


def x_range(rows: list[dict], curve: p3.Curve | None, p1_rows: list[dict], max_bits: float = MAX_PLOT_BITS, pad: float = X_PAD) -> tuple[float, float]:
    """(lo, hi) of the x axis: the reference curve (fixed and Huffman bits), the Phase 1 context and every method row
    with bits_blocks <= max_bits (its Huffman figure too), padded by `pad` bits. Rows above max_bits do not widen it."""
    xs: list[float] = []
    if curve is not None:
        xs += curve.bits
        xs += curve.bits_huff or []
    xs += [float(r["bits_blocks"]) for r in p1_rows if r.get("bits_blocks") is not None and float(r["bits_blocks"]) <= max_bits]
    for r in rows:
        b = r.get("bits_blocks")
        if b is None or float(b) > max_bits:
            continue
        xs.append(float(b))
        if r.get("bits_blocks_huff") is not None:
            xs.append(float(r["bits_blocks_huff"]))
    if not xs:
        return 1.5, max_bits + pad
    return min(xs) - pad, max(xs) + pad


def draw(ax, key: str, rows: list[dict], curve: p3.Curve | None, p1_rows: list[dict], fp16: float,
         xlim: tuple[float, float] | None = None) -> tuple[list[str], list[tuple], list[tuple[float, float]]]:
    """The points of one panel; returns (stderr notes for skipped values, labels, markers) for place_labels, which the
    caller runs after the figure layout is final. Rows beyond xlim[1] are drawn at the right edge."""
    notes = []
    labels: list[tuple] = []
    markers: list[tuple[float, float]] = []  # every drawn point, so a label never sits on another row's marker

    def val(r: dict) -> float | None:
        kl, d = panel_values(r, fp16)
        return kl if key == "kl" else d

    # Reference: seed-mean curve (solid), seed points, and the Huffman-bits curve (dashed, hollow).
    if curve is not None:
        ys = [p["kl"] if key == "kl" else p["dppl"] for p in curve.points]
        ok = [positive(y) for y in ys]
        xs = curve.bits
        ax.plot([x for x, o in zip(xs, ok) if o], [y for y, o in zip(ys, ok) if o], color=REF_COLOR, linewidth=2,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1, zorder=4)
        for p, y, o in zip(curve.points, ys, ok):
            if o:
                labels.append((p["bits"], y, p3.parse_name(f"gptq32-sym-g128-{int(round(p['bits']))}")["label"], REF_COLOR,
                               LABEL_SLOTS["reference"][0], LABEL_SLOTS["reference"][1], True))
                markers.append((p["bits"], y))
        for r in curve.seed_rows:
            y = val(r)
            if positive(y):
                ax.plot([r["bits_blocks"]], [y], marker="o", markersize=3.5, color=REF_COLOR, linestyle="none", zorder=5)
                markers.append((float(r["bits_blocks"]), y))
        if curve.bits_huff is not None:
            hx = curve.bits_huff
            ax.plot([x for x, o in zip(hx, ok) if o], [y for y, o in zip(ys, ok) if o], color=REF_COLOR, linewidth=1.5,
                    linestyle="--", marker="o", markersize=7, markerfacecolor=SURFACE, markeredgewidth=1.5, zorder=4)
            markers += [(x, y) for x, y, o in zip(hx, ys, ok) if o]
    # Phase 1 gptq-sym-g128 context (grey): one label per width, the seeds as points.
    p1_labelled: set[str] = set()
    for r in sorted(p1_rows, key=lambda r: (r["method"], r.get("seed", 0))):
        y = val(r)
        if not positive(y):
            continue
        ax.plot([r["bits_blocks"]], [y], marker="o", markersize=5, color=P1_COLOR, linestyle="none", zorder=2)
        markers.append((float(r["bits_blocks"]), y))
        if r["method"] not in p1_labelled:
            p1_labelled.add(r["method"])
            labels.append((float(r["bits_blocks"]), y, f"P1 {p3.parse_name(r['method'])['bits'] or ''}".strip(), P1_COLOR,
                           (-6, -10), "right", False))
    # Phase 3 method rows, one method at a time: its seeds share a marker (seed >= 1 with a light centre dot), are joined by
    # a thin line at their fixed bits, and carry one label.
    groups = p3.by_method([r for r in rows if p3.parse_name(r["method"])["family"] != "reference"])
    for m in sorted(groups, key=lambda m: p3.sort_key(groups[m][0])):
        p = p3.parse_name(m)
        color, marker = FAMILY_STYLE.get(p["family"], FAMILY_STYLE["other"])
        drawn: list[tuple[float, float]] = []
        for r in groups[m]:
            y = val(r)
            if not positive(y):
                notes.append(f"{r['method']} (seed {r.get('seed', 0)}): {key} = {y} not on the log axis, skipped")
                continue
            x = float(r["bits_blocks"])
            seed_dot = r.get("seed", 0) != 0  # seed >= 1: the same filled marker with a light centre dot
            if xlim is not None and x > xlim[1]:
                # off the right edge: a filled arrow marker at the edge, the bits in the label, no Huffman connector
                xe = xlim[1] - 0.03 * (xlim[1] - xlim[0])
                ax.plot([xe], [y], marker=">", markersize=9, color=color, markeredgecolor=SURFACE, markeredgewidth=1,
                        linestyle="none", zorder=6)
                if seed_dot:
                    ax.plot([xe], [y], marker="o", markersize=SEED_DOT, color=SURFACE, linestyle="none", zorder=7)
                labels.append((xe, y, f"{p3.row_label(r)}: {x:.1f} b", color, (-8, 6), "right", False))
                markers.append((xe, y))
                notes.append(f"{r['method']} (seed {r.get('seed', 0)}): bits_blocks {x:.2f} beyond the x range, drawn at the right edge")
                continue
            if p["shuffle"]:
                # half-filled: a control at its fixed bits (hollow is reserved for the Huffman figure)
                ax.plot([x], [y], marker=marker, markersize=8, color=color, fillstyle="bottom", markerfacecoloralt=SURFACE,
                        markeredgecolor=color, markeredgewidth=1.4, linestyle="none", zorder=5)
            else:
                ax.plot([x], [y], marker=marker, markersize=7, color=color, markeredgecolor=SURFACE, markeredgewidth=1,
                        linestyle="none", zorder=6)
                if seed_dot:
                    ax.plot([x], [y], marker="o", markersize=SEED_DOT, color=SURFACE, linestyle="none", zorder=7)
                hx = r.get("bits_blocks_huff")
                if hx is not None:
                    ax.plot([x, hx], [y, y], color=color, linewidth=0.9, alpha=0.7, zorder=3)
                    ax.plot([hx], [y], marker=marker, markersize=6, markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.2,
                            linestyle="none", zorder=5)
                    markers.append((float(hx), y))
            markers.append((x, y))
            drawn.append((x, y))
        if not drawn:
            continue
        if len(drawn) >= 2:
            xs_, ys_ = zip(*sorted(drawn, key=lambda t: t[1]))
            ax.plot(xs_, ys_, color=color, linewidth=1.4, alpha=0.6, zorder=4)
        (dx, dy), ha = LABEL_SLOTS.get(p["family"], LABEL_SLOTS["other"])
        # the label sits beside the outer point of the seed pair, on the side of its slot, so it never covers the other seed
        lx, ly = (min if dy < 0 else max)(drawn, key=lambda t: t[1])
        labels.append((lx, ly, p["label"], color, (dx, dy), ha, False))
    if xlim is not None:
        ax.set_xlim(*xlim)
    return notes, labels, markers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-dir", default="results/phase3", help="Phase 3 tree: <results-dir>/<model>/runs.jsonl")
    ap.add_argument("--phase1-dir", default="results", help="Phase 1 tree for fp16 and the gptq-sym-g128 context")
    ap.add_argument("--out", default=None, help="output png (default <phase1-dir>/rd3_<model>.png)")
    ap.add_argument("--include-smoke", action="store_true")
    args = ap.parse_args()

    rdir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else ROOT / args.results_dir
    p1dir = Path(args.phase1_dir) if Path(args.phase1_dir).is_absolute() else ROOT / args.phase1_dir
    p3_path, p1_path = rdir / args.model / "runs.jsonl", p1dir / args.model / "runs.jsonl"
    rows = p3.load_rows(p3_path, args.include_smoke)
    p1_all = p3.load_rows(p1_path, args.include_smoke)
    if not rows:
        raise SystemExit(f"no Phase 3 rows in {p3_path}")
    fp16 = p3.fp16_ppl(rows, p1_all, include_smoke=args.include_smoke)
    if fp16 is None:
        raise SystemExit(f"no full-protocol fp16 row in {p3_path} or {p1_path}")
    curve = p3.reference_curve(rows, fp16)
    if curve is None:
        print("warning: no gptq32-sym-g128-* reference rows; methods are drawn without a curve", file=sys.stderr)
    p1_rows = [r for r in p1_all if r["method"].startswith("gptq-sym-g128-") and r.get("kl") is not None]
    methods = [r for r in rows if r["method"] not in ("fp16", "had-fp16")]

    fig, axes = plt.subplots(1, 2, figsize=(13, FIG_HEIGHT), facecolor=SURFACE)
    kl_tokens = sorted({r.get("kl_tokens") for r in methods if r.get("kl_tokens")})
    n_tokens = sorted({r.get("n_tokens") for r in methods if r.get("n_tokens")})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT2, labelsize=9)
        ax.margins(x=0.1, y=0.15)
        ax.set_xlabel("bits per parameter of the block linears (filled: bits_blocks; hollow: bits_blocks_huff)", color=TEXT2, fontsize=9)
    axes[0].set_ylabel("kl: KL(P_ref || P_q), nats per token, log scale", color=TEXT2)
    axes[1].set_ylabel(f"ppl - ppl(fp16 = {fp16:.4g}), log scale", color=TEXT2)
    axes[0].set_title(f"KL to the bf16 model ({', '.join(f'{k:,}' for k in kl_tokens) or '?'} tokens)", color=TEXT, fontsize=10, loc="left")
    axes[1].set_title(f"WikiText-2 perplexity increase over fp16 ({', '.join(f'{k:,}' for k in n_tokens) or '?'} tokens)",
                      color=TEXT, fontsize=10, loc="left")
    notes = []
    xlim = x_range(methods, curve, p1_rows)
    panel_labels = []
    for ax, key in ((axes[0], "kl"), (axes[1], "dppl")):
        n, labels, markers = draw(ax, key, methods, curve, p1_rows, fp16, xlim)
        notes += n
        panel_labels.append((labels, markers))
        ax.set_xlim(*xlim)
    for n in sorted(set(notes)):
        print(f"warning: {n}", file=sys.stderr)

    families = sorted({p3.parse_name(r["method"])["family"] for r in methods} - {"reference"}, key=p3.FAMILIES.index)
    handles: list = []
    texts: list[str] = []

    def add(handle, text: str) -> None:
        handles.append(handle)
        texts.append(text)

    if curve is not None:
        if len({tuple(p["seeds"]) for p in curve.points}) == 1:
            seed_txt = f"seeds {', '.join(map(str, curve.points[0]['seeds']))}"
        else:  # e.g. MiniCPM: seeds 0, 1 at 3 bits, seed 0 alone at 4 bits
            seed_txt = "; ".join(f"{p['bits']:.3f}: {','.join(map(str, p['seeds']))}" for p in curve.points)
        add(Line2D([], [], color=REF_COLOR, linewidth=2, marker="o", markersize=6),
            f"gptq32-sym-g128 reference, seed mean ({seed_txt}); dots: seed rows")
        if curve.bits_huff is not None:
            add(Line2D([], [], color=REF_COLOR, linewidth=1.5, linestyle="--", marker="o", markersize=6, markerfacecolor=SURFACE),
                "the same reference at its own Huffman bits")
    names = {"rotation": "rotation (had / rnd / hadsign / eig) + gptq32", "codebook": "codebook (lloyd / nf / nfq) + gptq32",
             "alloc": "bit allocation (type / tensor / type rtn / shuf<k>)", "rowalloc": "row allocation (fisher / fisher rtn / shuf<k> rtn)",
             "combo": "combination (label: avg target)", "other": "other (label: method name)"}
    for fam in families:
        color, marker = FAMILY_STYLE[fam]
        add(Line2D([], [], color=color, marker=marker, markersize=7, markeredgecolor=SURFACE, linestyle="none"), names[fam])
    if p1_rows:
        add(Line2D([], [], color=P1_COLOR, marker="o", markersize=5, linestyle="none"), "Phase 1 gptq-sym-g128 (f64 Hessian), context")
    add((Line2D([], [], color=TEXT2, linewidth=1.4, alpha=0.6, marker="o", markersize=7, markeredgecolor=SURFACE),
         Line2D([], [], color=SURFACE, marker="o", markersize=SEED_DOT, linestyle="none")),
        "seed 1 has a light centre dot; a method's seeds are joined")
    add(Line2D([], [], color=TEXT2, marker="o", markersize=6, markerfacecolor=SURFACE, linewidth=0.9),
        "hollow: the same row at its Huffman bits (to the left)")
    if any(p3.parse_name(r["method"])["shuffle"] for r in methods):
        add(Line2D([], [], color=TEXT2, marker="o", markersize=7, fillstyle="bottom", markerfacecoloralt=SURFACE, linestyle="none"),
            "half-filled: shuffled-allocation control (fixed bits)")
    if any(float(r["bits_blocks"]) > xlim[1] for r in methods if r.get("bits_blocks") is not None):
        add(Line2D([], [], marker=">", markersize=8, color=TEXT2, markeredgecolor=SURFACE, linestyle="none"),
            f"> {MAX_PLOT_BITS:g} bits/param (stored rotation): at the right edge")
    add(Line2D([], [], linestyle="none"), f"fp16: ppl {fp16:.4g}, kl 0 (y = 0, off the log axes)")
    fig.suptitle(f"{args.model}: Phase 3 methods against the gptq32 reference curve", color=TEXT, fontsize=12, x=0.02, ha="left")
    footer_parts = [
        f"source: {p3_path.relative_to(ROOT).as_posix() if p3_path.is_relative_to(ROOT) else p3_path.as_posix()} "
        f"(bits_blocks, bits_blocks_huff, ppl, kl); fp16 and Phase 1 context: "
        f"{p1_path.relative_to(ROOT).as_posix() if p1_path.is_relative_to(ROOT) else p1_path.as_posix()}; "
        f"every plotted value, ratio and verdict is tabulated in results/phase3/summary.md.",
        f"labels: bit width or avg target with the method kind (a thin leader line joins a displaced label to its point). Beating the "
        f"curve needs KL below it by > {p3.KL_NOISE:.0%} and dPPL below it by > {p3.PPL_NOISE:.1%} on every seed (DESIGN.md Phase 3 "
        f"rules); both margins bind: rows whose KL clears its margin but whose dPPL does not are 'no' (marked in summary.md).",
    ]
    footer = "\n".join(textwrap.fill(t, FOOTER_WRAP) for t in footer_parts)
    ftext = fig.text(0.02, 0.008, footer, color=TEXT2, fontsize=7.5, ha="left", va="bottom", linespacing=1.4)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    foot_top = inv.transform((0, ftext.get_window_extent().y1))[1]
    leg = fig.legend(handles, texts, loc="lower center", ncol=3, frameon=False, fontsize=7.5, labelcolor=TEXT,
                     bbox_to_anchor=(0.5, foot_top + 0.005), columnspacing=1.5, handletextpad=0.6)
    fig.canvas.draw()
    leg_top = inv.transform((0, leg.get_window_extent().y1))[1]
    fig.tight_layout(rect=(0, leg_top + 0.005, 1, 1), w_pad=3)
    # labels last: collision avoidance works in display space, which tight_layout has just fixed
    for ax, (labels, markers) in zip(axes, panel_labels):
        place_labels(ax, labels, markers)
    out = Path(args.out) if args.out else p1dir / f"rd3_{args.model}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
