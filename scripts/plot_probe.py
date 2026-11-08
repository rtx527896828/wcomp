"""Figures for the Phase 2 structure probes (DESIGN.md "Phase 2 contract").

  uv run python scripts/plot_probe.py --model qwen3.5-0.8b [--probes marginal,lowrank,...] [--results-dir results]

Reads results/probe/<model>/<probe>.json (+ ablation.jsonl and results/<model>/runs.jsonl for the ablation figure)
and writes results/probe_<model>_<probe>.png, one figure per probe:
  marginal     excess kurtosis per tensor and type; 4-bit index-entropy gain and the bf16 lossless bound per type,
               W vs the iid-Gaussian control (P1).
  lowrank      energy captured at rank r per type (every tensor thin, numel-weighted mean bold) over the shuffled
               control band (seeds 0 / 1); gain vs r for W and control; P2b (W @ L) dotted when present (P2).
  outliers     column / row norm max-over-median per layer per type vs the shuffled control; keepcols gain at
               p = 0.5 / 1 / 2 % per type, W filled vs control dashed outline; top-16 channel Jaccard (P3).
  interlayer   residual energy of adjacent layer pairs after each alignment per type vs the control; differential
               coding gain per type (P4).
  acts         Lorenz curves of per-channel input energy for down / o / out_proj by layer; split-half Spearman and
               Gini per layer (P5).
  sensitivity  s_t / n_t per layer per type (log); allocation saving at tensor / group / column granularity; the
               prediction check s_t vs measured @only KL when ablation rows exist (P6a).
  fisher       rank-1 fit energy of the Fisher diagonal per layer per type vs control; allocation saving (P6b).
  superweight  dPPL of zeroing each candidate vs the magnitude and random controls; down_proj output spikes (P7).
  ablation     @only KL and dPPL per type at 4 / 3 bits against the joint run, @layer KL per layer, and bits vs KL for
               the group-size sweep, +keepcols and GPTQ (P0).
Missing files or stats are skipped with a line on stderr; the figure is still written with what exists.
Conventions: one fixed colour per weight type (the type is always also named in text), the control always grey and
dashed (band = spread of the two control seeds), log axes where values span decades, footer citing the source file.
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
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_io as io  # noqa: E402
from probe_io import P_GRID, R_GRID, aggregate, aggregate_by_type, by_type, get, num, per_layer, records, skip, types_present  # noqa: E402
from probe_io import numel as numel_of  # noqa: E402

# Validated categorical palette (light mode) for the first eight types; the three further DeltaNet types take
# supplementary hues. A type is never identified by colour alone: panels are titled, bars are ticked, lines are
# marked and legended.
TYPE_COLORS = {"q": "#2a78d6", "k": "#eb6834", "v": "#1baf7a", "o": "#eda100", "gate": "#e87ba4", "up": "#008300",
               "down": "#6250d6", "in_proj_qkv": "#e34948", "in_proj_z": "#104281", "in_proj_ab": "#8c6d1f", "out_proj": "#7a3b8f"}
TYPE_MARKERS = {"q": "o", "k": "s", "v": "D", "o": "^", "gate": "v", "up": "P", "down": "X", "in_proj_qkv": "*",
                "in_proj_z": "<", "in_proj_ab": ">", "out_proj": "h"}
OTHER = "#52514e"  # types outside the contract list (embedding)
CONTROL, CONTROL_BAND = "#898781", "#c3c2b7"
SURFACE, TEXT, TEXT2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE_RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal: steps 250 / 450 / 650
ORANGE_RAMP = ["#f3b08f", "#eb6834", "#9a3a12"]
GPTQ = "#eda100"  # the gptq family colour of plot_rd.py
ALIGN_MARKERS = {"e_raw": ("o", "raw"), "e_out": ("s", "Procrustes (out)"), "e_in": ("D", "Procrustes (in)"), "e_perm": ("^", "neuron permutation")}
ALLOC_LABELS = {"tensor": "tensor", "group": "128-col group", "col": "column", "row": "row", "elem": "element"}


def tcolor(t: str) -> str:
    return TYPE_COLORS.get(t, OTHER)


def tmarker(t: str) -> str:
    return TYPE_MARKERS.get(t, "o")


# ---------------------------------------------------------------- figure chrome


def new_fig(figsize: tuple, nrows: int | None = None, ncols: int | None = None, **kw):
    """A constrained-layout figure (axes, outside legends, subfigure titles are all laid out) with the bottom
    4% reserved for the footer. Returns fig, or (fig, axes) when nrows / ncols are given."""
    if nrows is None:
        fig = plt.figure(figsize=figsize, facecolor=SURFACE, layout="constrained")
        fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
        return fig
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, layout="constrained", **kw)
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    return fig, axes


def below(ncol: int = 2, drop: float = 0.3) -> dict:
    """Legend keyword arguments placing the legend under the axes (below rotated tick labels)."""
    return {"loc": "upper center", "bbox_to_anchor": (0.5, -drop), "ncol": ncol}


def style(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, logx=False, logy=False) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9.5, loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT2, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT2, fontsize=8.5)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")


def legend(ax, handles=None, **kw) -> None:
    kw.setdefault("frameon", False)
    kw.setdefault("fontsize", 7.5)
    kw.setdefault("labelcolor", TEXT)
    if handles is not None:
        ax.legend(handles=handles, **kw)
    else:
        ax.legend(**kw)


def fig_legend(fig, handles: list, ncol: int = 2) -> None:
    """Legend for a grid of small panels, placed outside the axes at the top right of the figure (an axes legend
    inside the first panel hides that panel's data)."""
    fig.legend(handles=handles, loc="outside upper right", ncol=ncol, frameon=False, fontsize=7.5, labelcolor=TEXT)


def plain_ticks(ax, axis: str = "y") -> None:
    """Plain tick labels (1, 2, 5) on a log axis instead of 4 x 10^0, for values within one or two decades."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(fmt)
    a.set_minor_formatter(fmt)


def symlog_ticks(ax, linthresh: float, axis: str = "y") -> None:
    """Explicit 1-2-5 ticks for a symlog axis within its current limits; the default locator labels only the decades
    and 0, which leaves values between 0 and the first decade without a readable tick."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    ticks = {0.0}
    for sign in (1, -1):
        for t in (sign * linthresh / 2,):
            if lo <= t <= hi:
                ticks.add(t)
        v = linthresh
        while v <= max(abs(lo), abs(hi)):
            for m in (1, 2, 5):
                t = sign * v * m
                if lo <= t <= hi:
                    ticks.add(t)
            v *= 10
    a.set_major_locator(FixedLocator(sorted(ticks)))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def wrap(title: str, width: int) -> str:
    return textwrap.fill(title, width)


def no_data(ax, what: str) -> None:
    ax.text(0.5, 0.5, f"no data: {what}", transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)
    skip(what)


def grid(parent, n: int, ncols: int = 4, sharey: bool = False) -> list:
    """n axes in a grid on a Figure or SubFigure; surplus axes are hidden."""
    ncols = max(1, min(ncols, n))
    nrows = max(1, math.ceil(n / ncols))
    axes = parent.subplots(nrows, ncols, squeeze=False, sharey=sharey)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return flat[:n]


def type_ticks(ax, types: list[str]) -> None:
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=30, ha="right", fontsize=8)


def finish(fig, title: str, source: str, out: Path, note: str = "") -> Path:
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.02, ha="left")
    footer = "source: " + source.replace("; ", "\nsource: ")
    if note:
        footer += "\n" + note
    # The footer is outside the layout engine's knowledge: reserve its height (0.16 in per line) explicitly.
    height_in = fig.get_size_inches()[1]
    bottom = min(0.2, (0.1 + 0.16 * footer.count("\n") + 0.16) / height_in)
    fig.get_layout_engine().set(rect=(0, bottom, 1, 1 - bottom))
    fig.text(0.02, 0.008, footer, color=TEXT2, fontsize=7.5, ha="left", va="bottom", linespacing=1.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(out)
    return out


def source_of(doc: dict | None, path: str) -> str:
    if doc is None:
        return path
    return f"{path} (code {doc.get('code_version', '?')}, {doc.get('timestamp', '?')})"


def paired_bars(ax, types: list[str], w: dict, c: dict, spread: dict | None = None, fmt: str = "{:+.3f}") -> int:
    """Per type: W bar (type colour) next to the control bar (grey dashed outline, hatched), the difference W - control
    written above the pair, an error bar for the control-seed spread when given. Returns the number of pairs drawn."""
    drawn = 0
    for i, t in enumerate(types):
        wv, cv = num(w.get(t)), num(c.get(t))
        if wv is None and cv is None:
            continue
        drawn += 1
        if wv is not None:
            ax.bar(i - 0.2, wv, 0.38, color=tcolor(t), edgecolor=SURFACE, linewidth=1)
        if cv is not None:
            ax.bar(i + 0.2, cv, 0.38, facecolor="none", edgecolor=CONTROL, hatch="///", linewidth=1, linestyle="--")
            if spread and num(spread.get(t)):
                ax.errorbar(i + 0.2, cv, yerr=spread[t] / 2, color=CONTROL, capsize=2, linewidth=1)
        if wv is not None and cv is not None:
            top = max(wv, cv, 0)
            ax.annotate(fmt.format(wv - cv), (i, top), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7, color=TEXT2)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    type_ticks(ax, types)
    headroom(ax)
    return drawn


def headroom(ax, frac: float = 0.08) -> None:
    """Extra space above the highest mark so annotations at the top (e.g. at y = 0 over negative bars) clear the title."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def control_handles(w_label: str = "W", with_band: bool = True) -> list:
    hs = [Patch(facecolor=TEXT2, label=w_label),
          Patch(facecolor="none", edgecolor=CONTROL, hatch="///", linestyle="--", label="control (shuffled within tensor)")]
    if with_band:
        hs.append(Line2D([], [], color=CONTROL, linewidth=1, label="error bar: spread of control seeds 0 / 1"))
    return hs


def seed_spread(recs: list[dict], *path) -> dict[str, float | None]:
    """Per type: |aggregate(control seed 0) - aggregate(control seed 1)| of the gain or stat at `path`; None without a
    second seed. path names the field for seed 0 ("gain_bits_control" or "control", key); seed 1 is read from
    "gain_bits_control_seed1" / "control_seed1"."""
    out = {}
    for t, rs in by_type(recs).items():
        if path == ("gain_bits_control",):
            a, b = aggregate(rs, "gain_bits_control"), aggregate(rs, "gain_bits_control_seed1")
        else:
            a, b = aggregate(rs, *path), aggregate(rs, "control_seed1", *path[1:])
        out[t] = abs(a - b) if a is not None and b is not None else None
    return out


# ---------------------------------------------------------------- the figures


class Ctx:
    def __init__(self, results_dir: Path, model: str, out_dir: Path, include_smoke: bool = False):
        self.rdir, self.model, self.out_dir = results_dir, model, out_dir
        self.include_smoke = include_smoke  # read limit-windows / kl-windows / calib-n ablation rows too
        self.pdir = io.probe_dir(results_dir, model)
        self._docs: dict[str, dict | None] = {}

    def path(self, probe: str) -> Path:
        return self.pdir / f"{probe}.json"

    def probe(self, probe: str) -> dict | None:
        if probe not in self._docs:
            self._docs[probe] = io.load_probe(self.path(probe))
        return self._docs[probe]

    def out(self, probe: str) -> Path:
        return self.out_dir / f"probe_{self.model}_{probe}.png"

    def rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def fig_marginal(ctx: Ctx) -> Path | None:
    doc = ctx.probe("marginal")
    if doc is None:
        return None
    recs = records(doc)
    types = types_present(recs)
    fig, axes = new_fig((15, 5.6), 1, 3)
    rng = np.random.RandomState(0)

    ax = axes[0]
    style(ax, "excess kurtosis per tensor (marker) and type (bar = numel-weighted mean)", None, "excess kurtosis (symlog, linear below 1)")
    drawn = 0
    all_ys: list[float] = []
    for i, t in enumerate(types):
        ys = [y for y in (num(io.val(r, "stat", "kurtosis")) for r in by_type(recs)[t]) if y is not None]
        if not ys:
            continue
        drawn += 1
        all_ys += ys
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(ys)), ys, s=14, color=tcolor(t), alpha=0.55, edgecolor="none")
        m = aggregate(by_type(recs)[t], "stat", "kurtosis")
        ax.plot([i - 0.3, i + 0.3], [m, m], color=tcolor(t), linewidth=2.2)
    ck = aggregate(recs, "control", "kurtosis")
    if drawn:
        ax.axhline(ck if ck is not None else 0.0, color=CONTROL, linestyle="--", linewidth=1.2,
                   label=f"Gaussian control (excess kurtosis {ck if ck is not None else 0:.2g})")
        ax.axhline(1.0, color=TEXT2, linestyle=":", linewidth=0.9, label="1: H-P1 heavy-tail threshold")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(min(-0.2, min(all_ys) * 1.2), max(all_ys) * 1.5)
        symlog_ticks(ax, 1.0)
        legend(ax, **below(2))
        type_ticks(ax, types)
    else:
        no_data(ax, "marginal: stat.kurtosis")

    ax = axes[1]
    style(ax, "4-bit index-entropy gain: 4 - H_4 - table cost, W vs Gaussian control", None, "gain_bits (bits/param)")
    if paired_bars(ax, types, aggregate_by_type(recs, "gain_bits"), aggregate_by_type(recs, "gain_bits_control")):
        hs = [Patch(facecolor=TEXT2, label="W (sym g128, b = 4)"),
              Patch(facecolor="none", edgecolor=CONTROL, hatch="///", linestyle="--", label="iid Gaussian, matched per-group std, same grid"),
              Line2D([], [], linestyle="none", label="number: W - control = shape value beyond Gaussian")]
        legend(ax, hs, **below(1))
    else:
        no_data(ax, "marginal: gain_bits / gain_bits_control")

    ax = axes[2]
    style(ax, "bf16 lossless bound: 16 - H(16-bit word) per type", None, "bits saved per param")
    w = aggregate_by_type(recs, "stat", "lossless_bits_saved")
    c = aggregate_by_type(recs, "control", "lossless_bits_saved")
    if all(v is None for v in w.values()):  # fall back to the word entropy
        hw, hc = aggregate_by_type(recs, "stat", "H_word"), aggregate_by_type(recs, "control", "H_word")
        w = {t: 16 - v if v is not None else None for t, v in hw.items()}
        c = {t: 16 - v if v is not None else None for t, v in hc.items()}
    if paired_bars(ax, types, w, c, fmt="{:+.2f}"):
        legend(ax, control_handles("W", with_band=False)[:1] + [Patch(facecolor="none", edgecolor=CONTROL, hatch="///", linestyle="--", label="Gaussian control")],
               **below(2))
    else:
        no_data(ax, "marginal: stat.lossless_bits_saved / H_word")
    return finish(fig, f"{ctx.model}: P1 marginals and entropy (W vs iid Gaussian; the shuffle control is vacuous here)",
                  source_of(doc, ctx.rel(ctx.path("marginal"))), ctx.out("marginal"),
                  "excess kurtosis > 1 and a gain above the Gaussian bar mean the shape beyond Gaussian carries value; table cost = 2^b x 16 bits per tensor")


def fig_lowrank(ctx: Ctx) -> Path | None:
    doc = ctx.probe("lowrank")
    if doc is None:
        return None
    wdoc = ctx.probe("lowrank_weighted")
    recs = records(doc)
    wrecs = records(wdoc) if wdoc else []
    types = types_present(recs)
    rs = [float(r) for r in R_GRID]
    fig = new_fig((15, 11.5))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1], hspace=0.06)
    top.suptitle("energy fraction captured at rank r (x = r / min(N, K)); thin: each tensor, bold: numel-weighted mean", color=TEXT, fontsize=9.5, x=0.01, ha="left")
    bottom.suptitle("gain of low-rank + quantized residual at rank r: P2 (solid) 0.5 log2(1 / e_r) - 16 r (N + K) / (N K); "
                    "P2b (dotted) measured 0.5 log2(||dW L||^2 / ||dE_r L||^2) - overhead with dX = X - RTN-4(X); max over r is the estimate",
                    color=TEXT, fontsize=9.5, x=0.01, ha="left")
    axes_e, axes_g = grid(top, len(types), 6), grid(bottom, len(types), 6)
    wb = by_type(wrecs)
    any_e = any_g = False
    for ax_e, ax_g, t in zip(axes_e, axes_g, types):
        trs = by_type(recs)[t]
        style(ax_e, t, "r / min(N, K)", "energy captured", logx=True)
        style(ax_g, t, "r / min(N, K)", "gain_bits (symlog)", logx=True)
        for r in trs:
            ys = [num(io.val(r, "stat", f"energy@{k}")) for k in R_GRID]
            if all(y is not None for y in ys):
                ax_e.plot(rs, ys, color=tcolor(t), linewidth=0.7, alpha=0.3)
        for ax, key, label in ((ax_e, "energy@", "energy"), (ax_g, "gain@", "gain")):
            mean = [aggregate(trs, "stat", f"{key}{k}") for k in R_GRID]
            c0 = [aggregate(trs, "control", f"{key}{k}") for k in R_GRID]
            c1 = [aggregate(trs, "control_seed1", f"{key}{k}") for k in R_GRID]
            if all(v is None for v in mean):
                no_data(ax, f"lowrank {t}: stat.{key}*")
                continue
            if ax is ax_e:
                any_e = True
            else:
                any_g = True
            if all(v is not None for v in c0):
                if all(v is not None for v in c1):
                    ax.fill_between(rs, np.minimum(c0, c1), np.maximum(c0, c1), color=CONTROL_BAND, alpha=0.5, linewidth=0)
                    cm = [(a + b) / 2 for a, b in zip(c0, c1)]
                else:
                    cm = c0
                ax.plot(rs, cm, color=CONTROL, linestyle="--", linewidth=1.3, marker="o", markersize=3)
            ax.plot(rs, mean, color=tcolor(t), linewidth=2.2, marker="o", markersize=4)
            if t in wb:
                wm = [aggregate(wb[t], "stat", f"{key}{k}") for k in R_GRID]
                wc = [aggregate(wb[t], "control", f"{key}{k}") for k in R_GRID]
                if all(v is not None for v in wm):
                    ax.plot(rs, wm, color=tcolor(t), linewidth=1.6, linestyle=":", marker="x", markersize=4)
                if all(v is not None for v in wc):
                    ax.plot(rs, wc, color=CONTROL, linewidth=1.1, linestyle=":", marker="x", markersize=4)
        ax_e.set_ylim(0, 1.02)
        ax_g.axhline(0, color=MUTED, linewidth=0.8)
        ax_e.set_xticks(rs)
        ax_g.set_xticks(rs)
        ax_e.set_xticklabels([f"{r:g}" for r in rs])
        ax_g.set_xticklabels([f"{r:g}" for r in rs])
        # The gains span -0.03 (r = 1 %) to -7 (r = 25 %): a linear axis hides the W - control distance at small r,
        # which is where the estimate (the maximum over r) lives, so the axis is symlog and the maxima are written out.
        gw, gc = aggregate(trs, "gain_bits"), aggregate(trs, "gain_bits_control")
        if gw is not None or gc is not None:
            ax_g.set_yscale("symlog", linthresh=0.1)
            ax_g.set_ylim(top=0.15 if max(gw or 0, gc or 0) <= 0 else max(gw or 0, gc or 0) * 2)
            symlog_ticks(ax_g, 0.1)
            ax_g.text(0.97, 0.95, f"max over r: W {gw:+.3f}\ncontrol {gc:+.3f}" if gw is not None and gc is not None
                      else f"max over r: W {gw:+.3f}" if gw is not None else f"max over r: control {gc:+.3f}",
                      transform=ax_g.transAxes, ha="right", va="top", fontsize=7, color=TEXT2)
    hs = [Line2D([], [], color=TEXT2, linewidth=2.2, marker="o", label="W (mean over tensors of the type)"),
          Line2D([], [], color=CONTROL, linestyle="--", marker="o", label="shuffled control seeds 0 / 1 (band = spread)")]
    if wrecs:
        hs += [Line2D([], [], color=TEXT2, linestyle=":", marker="x", label="P2b: W @ L (activation-weighted)"),
               Line2D([], [], color=CONTROL, linestyle=":", marker="x", label="P2b control: shuffled W @ L")]
    if axes_e:
        fig_legend(fig, hs, ncol=2)
    if not (any_e or any_g):
        skip("lowrank: no energy / gain stats at the r grid")
    src = source_of(doc, ctx.rel(ctx.path("lowrank")))
    if wdoc:
        src += f"; {ctx.rel(ctx.path('lowrank_weighted'))}"
    return finish(fig, f"{ctx.model}: P2 low rank (plain and activation-weighted) vs shuffled control", src, ctx.out("lowrank"),
                  "a W curve inside the grey control band has no rank structure beyond what any matrix of that marginal shows")


def fig_outliers(ctx: Ctx) -> Path | None:
    doc = ctx.probe("outliers")
    if doc is None:
        return None
    recs = records(doc, types=io.TYPES)
    types = types_present(recs)
    fig = new_fig((15, 11))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.3, 1], hspace=0.06)
    top.suptitle("column (input) and row (output) L2-norm max / median per layer; grey dashed: shuffled control (band = seeds 0 / 1)",
                 color=TEXT, fontsize=9.5, x=0.01, ha="left")
    axes = grid(top, len(types), 6)
    any_top = False
    for ax, t in zip(axes, types):
        trs = by_type(recs)[t]
        style(ax, t, "layer", "max / median (log)", logy=True)
        col = per_layer(trs, "stat", "col_max_over_median")
        row = per_layer(trs, "stat", "row_max_over_median")
        c0 = per_layer(trs, "control", "col_max_over_median")
        c1 = per_layer(trs, "control_seed1", "col_max_over_median")
        if not col and not row:
            no_data(ax, f"outliers {t}: stat.col_max_over_median / row_max_over_median")
            continue
        any_top = True
        if c0:
            xs = [l for l, _ in c0]
            if c1 and len(c1) == len(c0):
                a, b = np.array([v for _, v in c0]), np.array([v for _, v in c1])
                ax.fill_between(xs, np.minimum(a, b), np.maximum(a, b), color=CONTROL_BAND, alpha=0.5, linewidth=0)
                ax.plot(xs, (a + b) / 2, color=CONTROL, linestyle="--", linewidth=1.2)
            else:
                ax.plot(xs, [v for _, v in c0], color=CONTROL, linestyle="--", linewidth=1.2)
        if col:
            ax.plot([l for l, _ in col], [v for _, v in col], color=tcolor(t), linewidth=1.8, marker="o", markersize=3.5)
        if row:
            ax.plot([l for l, _ in row], [v for _, v in row], color=tcolor(t), linewidth=1.2, linestyle=":", marker="s", markersize=3, alpha=0.8)
        ax.axhline(5, color=TEXT2, linestyle=":", linewidth=0.9)
        plain_ticks(ax)
    if axes:
        fig_legend(fig, [Line2D([], [], color=TEXT2, marker="o", linewidth=1.8, label="columns (input channels)"),
                         Line2D([], [], color=TEXT2, marker="s", linestyle=":", linewidth=1.2, label="rows (output channels)"),
                         Line2D([], [], color=CONTROL, linestyle="--", label="control, columns (band = seeds 0 / 1)"),
                         Line2D([], [], color=TEXT2, linestyle=":", linewidth=0.9, label="5 x median: the contract's outlier-column threshold")], ncol=2)
    if not any_top:
        skip("outliers: no norm ratio stats")

    axb = bottom.subplots(1, 1)
    style(axb, "keepcols gain per type: 0.5 log2(rho) - ((16-b) n_keep/K + 1/N) (= p/100 x 12 + 1/N at 4 bit, K p/100 integer), n_keep = round(K p/100), rho = MSE(RTN-4) / MSE(RTN-4, top-p% columns kept in bf16)",
          None, "gain_bits (bits/param)")
    width = 0.26
    drawn = 0
    for i, t in enumerate(types):
        trs = by_type(recs)[t]
        spread = io.control_spread(trs)  # |best gain of control seed 0 - seed 1|
        for j, p in enumerate(P_GRID):
            x = i + (j - 1) * width
            w = aggregate(trs, "stat", f"gain@p{p}")
            c0 = aggregate(trs, "control", f"gain@p{p}")
            if w is None and c0 is None:
                continue
            drawn += 1
            if w is not None:
                axb.bar(x, w, width * 0.92, color=BLUE_RAMP[j], edgecolor=SURFACE, linewidth=0.8)
            if c0 is not None:
                axb.bar(x, c0, width * 0.92, facecolor="none", edgecolor=CONTROL, linestyle="--", linewidth=1, hatch="///")
                if spread is not None:
                    axb.errorbar(x, c0, yerr=spread / 2, color=CONTROL, capsize=2, linewidth=1)
            if w is not None and c0 is not None and j == 1:
                axb.annotate(f"{w - c0:+.3f}", (x, max(w, c0, 0)), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7, color=TEXT2)
    if drawn:
        axb.axhline(0, color=MUTED, linewidth=0.8)
        type_ticks(axb, types)
        headroom(axb)
        hs = [Patch(facecolor=BLUE_RAMP[j], label=f"W, p = {p}%") for j, p in enumerate(P_GRID)]
        hs += control_handles()[1:]
        hs.append(Line2D([], [], linestyle="none", label="number: W - control at p = 1%"))
        jw, jc = io.jaccard(doc)
        per_type = io.jaccard_by_type(doc)
        if jw is not None or per_type:
            txt = "cross-layer consistency of the top-16 outlier input channels, mean pairwise Jaccard W / control (contract > 0.3 vs < 0.05): "
            if jw is not None:
                txt += f"all hidden-input linears {jw:.2f} / {'-' if jc is None else f'{jc:.3f}'}"
            if per_type:
                txt += ("; " if jw is not None else "") + ", ".join(f"{t} {w:.2f} / {'-' if c is None else f'{c:.3f}'}" for t, (w, c) in per_type.items())
            axb.set_title(axb.get_title(loc="left") + "\n" + "\n".join(textwrap.wrap(txt, 150)), color=TEXT, fontsize=9.5, loc="left")
        else:
            skip("outliers: cross-layer Jaccard summary records")
        legend(axb, hs, **below(3))
    else:
        no_data(axb, "outliers: stat.gain_at_p*")
    return finish(fig, f"{ctx.model}: P3 outlier rows / columns vs shuffled control", source_of(doc, ctx.rel(ctx.path("outliers"))),
                  ctx.out("outliers"), "direct check of the keepcols estimate: the +keepcols rows of ablation.jsonl (see the ablation figure)")


def fig_interlayer(ctx: Ctx) -> Path | None:
    doc = ctx.probe("interlayer")
    if doc is None:
        return None
    recs = records(doc, types=io.TYPES, adjacent_only=True)  # the best-pair records are not adjacent pairs
    types = types_present(recs)
    fig = new_fig((15, 11))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.3, 1], hspace=0.06)
    top.suptitle("residual energy e = ||W_j - align(W_i)||^2 / ||W_j||^2 for adjacent pairs (i = pair_with, j = layer), per alignment; grey dashed: both tensors shuffled",
                 color=TEXT, fontsize=9.5, x=0.01, ha="left")
    axes = grid(top, len(types), 6)
    any_top = False
    used_aligns: set[str] = set()
    for ax, t in zip(axes, types):
        trs = by_type(recs)[t]
        style(ax, t, "layer j", "residual energy e")
        drawn = False
        lo, hi = 1.0, 1.0
        for key, (marker, _) in ALIGN_MARKERS.items():
            w = per_layer(trs, "stat", key)
            c = per_layer(trs, "control", key)
            if not w:
                continue
            drawn = True
            used_aligns.add(key)
            if c:
                ax.plot([l for l, _ in c], [v for _, v in c], color=CONTROL, linestyle="--", linewidth=1, marker=marker, markersize=3, alpha=0.9)
            ax.plot([l for l, _ in w], [v for _, v in w], color=tcolor(t), linewidth=1.3, marker=marker, markersize=3.5)
            lo = min(lo, min(v for _, v in w), min((v for _, v in c), default=1.0))
            hi = max(hi, max(v for _, v in w), max((v for _, v in c), default=1.0))
        if not drawn:
            no_data(ax, f"interlayer {t}: stat.e_*")
            continue
        any_top = True
        # e_raw of two uncorrelated tensors of equal norm is 2 (not 1): the axis must show it, not clip at 1
        ax.set_ylim(max(0, lo - 0.05), hi + 0.05)
        ax.axhline(0.9, color=TEXT2, linestyle=":", linewidth=0.9)
        ax.axhline(1.0, color=GRID, linewidth=0.8)
    if axes and used_aligns:
        hs = [Line2D([], [], color=TEXT2, marker=ALIGN_MARKERS[k][0], linewidth=1.3, label=ALIGN_MARKERS[k][1]) for k in ALIGN_MARKERS if k in used_aligns]
        hs.append(Line2D([], [], color=CONTROL, linestyle="--", label="control (same alignment)"))
        hs.append(Line2D([], [], color=TEXT2, linestyle=":", linewidth=0.9, label="0.9: H-P4 'no redundancy' line (e > 0.9); e = 1 is W_i unused, e = 2 is uncorrelated equal-norm tensors"))
        fig_legend(fig, hs, ncol=3)
    if not any_top:
        skip("interlayer: no residual energies")
    axb = bottom.subplots(1, 1)
    gains, gains_c = aggregate_by_type(recs, "gain_bits"), aggregate_by_type(recs, "gain_bits_control")
    span = max((abs(v) for v in list(gains.values()) + list(gains_c.values()) if v is not None), default=0.0)
    style(axb, "differential-coding gain at the best alignment: 0.5 log2(1 / e_best) - overhead(R at bf16 or permutation), W vs control"
          + (" (symlog)" if span > 2 else ""), None, "gain_bits (bits/param)")
    if paired_bars(axb, types, gains, gains_c):
        if span > 2:  # a stored N x N rotation costs tens of bits/param on wide tensors
            axb.set_yscale("symlog", linthresh=0.1)
            symlog_ticks(axb, 0.1)
        legend(axb, control_handles(with_band=False), **below(2))
    else:
        no_data(axb, "interlayer: gain_bits")
    return finish(fig, f"{ctx.model}: P4 inter-layer correlation vs shuffled control", source_of(doc, ctx.rel(ctx.path("interlayer"))),
                  ctx.out("interlayer"), "alignment always fits something: only the distance between the type-coloured line and the grey control is structure")


def _ramp(n: int) -> list[str]:
    """n ordinal colours from the blue ramp (steps 250..700) for layer order."""
    steps = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
    if n <= 1:
        return [steps[4]]
    return [steps[round(i * (len(steps) - 1) / (n - 1))] for i in range(n)]


def fig_acts(ctx: Ctx) -> Path | None:
    doc = ctx.probe("acts")
    if doc is None:
        return None
    recs = records(doc)
    types = [t for t in io.PRIMARY_ACT_TYPES if t in types_present(recs)] or types_present(recs)
    fig = new_fig((15, 10))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.25, 1], hspace=0.06)
    top.suptitle("Lorenz curves of per-channel input energy (channels sorted by energy; colour = layer, light to dark); grey dashed: uniform usage",
                 color=TEXT, fontsize=9.5, x=0.01, ha="left")
    axes = grid(top, len(types), 4)
    for ax, t in zip(axes, types):
        trs = by_type(recs)[t]
        style(ax, f"{t} input", "fraction of channels", "cumulative energy share")
        colors = _ramp(len(trs))
        drawn = 0
        for r, col in zip(trs, colors):
            lx, ly = get(r, "stat", "lorenz_x"), get(r, "stat", "lorenz_y")
            if isinstance(lx, list) and isinstance(ly, list) and len(lx) == len(ly) and len(lx) >= 2:
                ax.plot(lx, ly, color=col, linewidth=1)
            else:
                # The probe stores three points of the Lorenz curve: the channel fractions carrying 50 % and 90 % of
                # the energy, and the energy share of the top 20 % of channels.
                pts = [(0.0, 0.0), (1.0, 1.0)]
                s50, s90, top20 = num(io.val(r, "stat", "share50")), num(io.val(r, "stat", "share90")), num(get(r, "stat", "energy_share_top20pct"))
                if s50 is not None:
                    pts.append((s50, 0.5))
                if s90 is not None:
                    pts.append((s90, 0.9))
                if top20 is not None:
                    pts.append((0.2, top20))
                if len(pts) < 3:
                    continue
                pts.sort()
                ax.plot([x for x, _ in pts], [y for _, y in pts], color=col, linewidth=1, marker="o", markersize=2.5)
            drawn += 1
        if not drawn:
            no_data(ax, f"acts {t}: stat.frac_channels_50 / frac_channels_90 / energy_share_top20pct")
            continue
        ax.plot([0, 1], [0, 1], color=CONTROL, linestyle="--", linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        layers = [r.get("layer") for r in trs if isinstance(r.get("layer"), int)]
        if layers:
            legend(ax, [Line2D([], [], color=colors[0], label=f"layer {min(layers)}"), Line2D([], [], color=colors[-1], label=f"layer {max(layers)}"),
                        Line2D([], [], color=CONTROL, linestyle="--", label="uniform usage")], loc="lower right")
    axl, axr = bottom.subplots(1, 2)
    style(axl, "split-half Spearman of per-channel energy (C4 seeds 0 vs 2); the control for Gini", "layer", "Spearman")
    style(axr, "Gini of per-channel energy per layer", "layer", "Gini")
    any_s = any_g = False
    for t in types:
        trs = by_type(recs)[t]
        sp = per_layer(trs, "stat", "spearman_split")
        gi = per_layer(trs, "stat", "gini")
        if sp:
            any_s = True
            axl.plot([l for l, _ in sp], [v for _, v in sp], color=tcolor(t), marker=tmarker(t), markersize=4, linewidth=1.3, label=t)
        if gi:
            any_g = True
            axr.plot([l for l, _ in gi], [v for _, v in gi], color=tcolor(t), marker=tmarker(t), markersize=4, linewidth=1.3, label=t)
    if any_s:
        axl.axhline(0.9, color=CONTROL, linestyle="--", linewidth=1.2, label="0.9: stable usage pattern (contract)")
        axl.set_ylim(min(0.5, axl.get_ylim()[0]), 1.02)
        legend(axl, loc="lower left")
    else:
        no_data(axl, "acts: stat.spearman_split")
    if any_g:
        axr.axhline(0.5, color=CONTROL, linestyle="--", linewidth=1.2, label="0.5: H-P5 threshold")
        axr.set_ylim(0, 1)
        legend(axr, loc="lower left")
    else:
        no_data(axr, "acts: stat.gini")
    return finish(fig, f"{ctx.model}: P5 activation sparsity and neuron usage (C4 inputs of down / o / out_proj)",
                  source_of(doc, ctx.rel(ctx.path("acts"))), ctx.out("acts"),
                  "a permutation of channel energies has the same Gini, so the control is the split-half agreement, not a shuffle\n"
                  "each Lorenz curve is drawn through the three recorded points (channel fraction carrying 50 % / 90 % of the energy, energy share of the top 20 % channels)")


def _allocation_bars(ax, alloc, title: str) -> bool:
    style(ax, wrap(title, 52), "allocation granularity", "bits/param")
    if not isinstance(alloc, dict) or not alloc:
        no_data(ax, f"{title}: global.alloc_<granularity>_{{saving,overhead,gain}}")
        return False
    alloc = {ALLOC_LABELS.get(k, k): v for k, v in alloc.items()}
    names = list(alloc)
    sav = [num(get(alloc, n, "saving")) for n in names]
    ovh = [num(get(alloc, n, "overhead")) for n in names]
    gain = [num(get(alloc, n, "gain")) for n in names]
    cap = [num(get(alloc, n, "saving_cap2bit")) for n in names]
    floored = [num(get(alloc, n, "frac_floored")) for n in names]
    xs = np.arange(len(names))
    ax.bar(xs - 0.27, [s or 0 for s in sav], 0.26, color=BLUE_RAMP[1], edgecolor=SURFACE, label="saving 0.5 log2(mean / geomean of s_u / n_u), units floored at 0 bits")
    if any(c is not None for c in cap):
        ax.bar(xs, [c or 0 for c in cap], 0.26, color=BLUE_RAMP[0], edgecolor=SURFACE, label="same with b_u >= 2 bits (floor mean 2^-4)")
    ax.bar(xs + 0.27, [o or 0 for o in ovh], 0.26, facecolor="none", edgecolor=CONTROL, hatch="///", label="side information (allocation table)")
    for x, g in zip(xs, gain):
        if g is not None:
            ax.annotate(f"net {g:+.3f}", (x - 0.27, sav[x] or 0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5, color=TEXT2)
    ax.set_xticks(xs)
    # the floored fraction travels with the headline: where it is large the saving is the floor, not measured spread
    ax.set_xticklabels([n + (f"\n{f:.0%} floored" if f is not None else "") for n, f in zip(names, floored)])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    headroom(ax, 0.12)
    legend(ax, **below(1, 0.22))  # inside the axes it covered the bars and their "net" labels
    return True


def _ablation_rows(ctx: Ctx) -> list[dict]:
    """ablation.jsonl rows merged with the Phase 1 runs.jsonl (probe_io.merge_rows: Phase 1 names from runs.jsonl,
    else latest per key; the same merge structure_table.py uses)."""
    return io.merge_rows(io.load_rows(ctx.rdir / ctx.model / "runs.jsonl", ctx.include_smoke), io.load_rows(ctx.pdir / "ablation.jsonl", ctx.include_smoke))


def fig_sensitivity(ctx: Ctx) -> Path | None:
    doc = ctx.probe("sensitivity")
    if doc is None:
        return None
    recs = records(doc, types=io.TYPES)
    types = types_present(recs)
    fig, axes = new_fig((17, 5.5), 1, 3, gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axes[0]
    style(ax, "output-error weight per parameter s_t / n_t, RTN-4 g128 (s_t = sum_j H_jj sum_i dW_ij^2)", "layer", "s_t / n_t (log)", logy=True)
    drawn = 0
    for t in types:
        trs = by_type(recs)[t]
        pts = per_layer(trs, "stat", "s_per_param")
        if not pts:  # s_t per layer divided by the layer's parameter count of this type
            pts = []
            for layer, s_mean in per_layer(trs, "stat", "s_t"):
                n = sum(numel_of(r) for r in trs if r.get("layer") == layer)
                if n:
                    pts.append((layer, s_mean * len([r for r in trs if r.get("layer") == layer]) / n))
        if not pts:
            continue
        drawn += 1
        ax.plot([l for l, _ in pts], [v for _, v in pts], color=tcolor(t), marker=tmarker(t), markersize=4, linewidth=1.2, label=t)
    if drawn:
        legend(ax, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    else:
        no_data(ax, "sensitivity: stat.s_per_param or stat.s")
    _allocation_bars(axes[1], io.allocation(doc, ["tensor", "group", "col"]), "allocation saving by granularity (model-wide)")

    ax = axes[2]
    style(ax, "prediction check: s_t per type vs @only KL (4 bit)", "sum of s_t over the type (log)", "KL of @only=<type>, 4 bit (log)", logx=True, logy=True)
    rows = _ablation_rows(ctx)
    only = io.ablation_only(rows, 4)
    xs, ys, labels = [], [], []
    for t in types:
        s = sum(v for v in (num(io.val(r, "stat", "s_t")) for r in by_type(recs)[t]) if v is not None)
        k = num(only.get(t, {}).get("kl"))
        if s > 0 and k is not None and k > 0:
            xs.append(s)
            ys.append(k)
            labels.append(t)
    if len(xs) >= 2:
        for x, y, t in zip(xs, ys, labels):
            ax.scatter([x], [y], color=tcolor(t), marker=tmarker(t), s=45, edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7.5, color=TEXT2)
        rho = io.spearman(xs, ys)
        ax.set_title(f"prediction check: Spearman {rho:.2f}, n = {len(xs)} (> 0.7 predicts)" if rho is not None
                     else "prediction check: too few types", color=TEXT, fontsize=9.5, loc="left", pad=12)
        ax.margins(x=0.35, y=0.15)  # room for the type labels right of the markers
    else:
        no_data(ax, "sensitivity prediction check: ablation.jsonl @only rows at 4 bit")
    src = source_of(doc, ctx.rel(ctx.path("sensitivity")))
    if only:
        src += f"; {ctx.rel(ctx.pdir / 'ablation.jsonl')}"
    return finish(fig, f"{ctx.model}: P6a Hessian-diagonal sensitivity", src, ctx.out("sensitivity"),
                  "the allocation saving is permutation invariant (no shuffle control): the prediction check against measured KL replaces it")


def fig_fisher(ctx: Ctx) -> Path | None:
    doc = ctx.probe("fisher")
    if doc is None:
        return None
    recs = records(doc, types=io.TYPES)
    types = types_present(recs)
    fig, axes = new_fig((15, 5.5), 1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    style(ax, wrap("energy of the best rank-1 fit F_ij ~ r_i c_j of the position-wise empirical-Fisher approximation (d_t of the summed "
                   "window NLL; the per-token Fisher only without token mixing), per layer and type; grey dashed: shuffled F", 110),
          "layer", "rank-1 energy fraction")
    drawn = 0
    for t in types:
        trs = by_type(recs)[t]
        w = per_layer(trs, "stat", "rank1_energy")
        c = per_layer(trs, "control", "rank1_energy")
        if not w:
            continue
        drawn += 1
        ax.plot([l for l, _ in w], [v for _, v in w], color=tcolor(t), marker=tmarker(t), markersize=4, linewidth=1.2, label=t)
        if c:
            ax.plot([l for l, _ in c], [v for _, v in c], color=CONTROL, linestyle="--", linewidth=0.9)
    if drawn:
        ax.axhline(0.7, color=TEXT2, linestyle=":", linewidth=1.1, label="0.7: per-row + per-column side info suffices")
        ax.set_ylim(0, 1.02)
        hs, _ = ax.get_legend_handles_labels()
        hs.insert(-1, Line2D([], [], color=CONTROL, linestyle="--", linewidth=0.9, label="control: F shuffled within tensor (one line per type)"))
        legend(ax, hs, loc="center left", bbox_to_anchor=(1.0, 0.5))
    else:
        no_data(ax, "fisher: stat.rank1_energy")
    # element / row / column allocation is within-tensor (per record, averaged); the tensor granularity is model-wide.
    alloc = io.record_allocation(recs, ["elem", "row", "col"])
    alloc.update(io.allocation(doc, ["tensor"]))
    _allocation_bars(axes[1], alloc, "Fisher allocation saving by granularity (element / row / column within tensor; tensor model-wide)")
    return finish(fig, f"{ctx.model}: P6b Fisher diagonal (measured gradients, nothing applied)", source_of(doc, ctx.rel(ctx.path("fisher"))),
                  ctx.out("fisher"), "control: Fisher values shuffled within the tensor (the rank-1 fit of the control is the floor)")


def fig_superweight(ctx: Ctx) -> Path | None:
    doc = ctx.probe("superweight")
    if doc is None:
        return None
    recs = records(doc)
    per_layer_stats = io.layer_stats(doc)
    fig, axes = new_fig((14, 5.5 if len(recs) <= 4 else 7), 1, 2)
    ax = axes[0]
    style(ax, "dPPL % of zeroing one weight: candidate vs magnitude control\n(largest |w| non-candidate of the tensor) vs one random weight", None, "dPPL % (symlog)")
    labels, cand, mag, rnd = [], [], [], []
    for r in recs:
        d = num(io.val(r, "stat", "dppl"))
        if d is None:
            continue
        kind = str(r.get("note", "")).split(";")[0].strip()[:24]
        labels.append(f"L{r.get('layer')} {r.get('type')} [{get(r, 'stat', 'row')}, {get(r, 'stat', 'col')}] w={num(get(r, 'stat', 'value')) or 0:.2g} {kind}")
        cand.append(d)
        mag.append(num(io.val(r, "control", "mag_dppl")))
        rnd.append(num(io.val(r, "control", "random_dppl")))
    if cand:
        xs = np.arange(len(cand))
        ax.bar(xs - 0.27, cand, 0.26, color=BLUE_RAMP[1], edgecolor=SURFACE, label="candidate zeroed")
        ax.bar(xs, [m if m is not None else 0 for m in mag], 0.26, facecolor="none", edgecolor=CONTROL, hatch="///", label="magnitude control")
        ax.bar(xs + 0.27, [v if v is not None else 0 for v in rnd], 0.26, facecolor=CONTROL_BAND, edgecolor=SURFACE, label="random weight (seed 0)")
        ax.axhline(100, color=TEXT2, linestyle=":", linewidth=1.1, label="100 %: H-P7 threshold")
        ax.set_yscale("symlog", linthresh=1.0)
        vals = [v for v in cand + mag + rnd if v is not None]
        # symlog autoscaling pads the negative side out to a full decade below tiny negative controls
        ax.set_ylim(min(-0.5, min(vals) * 1.5), max(300.0, max(vals) * 3))
        symlog_ticks(ax, 1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7 if len(labels) <= 6 else 6, rotation=0 if len(labels) <= 4 else 40, ha="center" if len(labels) <= 4 else "right")
        if len(labels) <= 4:
            legend(ax, **below(2, 0.18))
        else:  # long rotated labels below: the legend goes inside, where a symlog axis has room above the bars
            legend(ax, loc="upper right")
    else:
        no_data(ax, "superweight: stat.dppl")
    ax = axes[1]
    style(ax, "down_proj max |output| and max |input| per layer (one C4 window)\ndashed: 20 x median layer max (spike threshold)", "layer", "max |activation| (log)", logy=True)
    if per_layer_stats:
        items = sorted((s["layer"], num(s.get("out_max")), num(s.get("in_max"))) for s in per_layer_stats)
        outs = [(l, o) for l, o, _ in items if o is not None]
        ins = [(l, i) for l, _, i in items if i is not None]
        if outs:
            ax.plot([l for l, _ in outs], [v for _, v in outs], color=BLUE_RAMP[1], marker="o", markersize=4, linewidth=1.3, label="max |output|")
            med = float(np.median([v for _, v in outs]))
            ax.axhline(20 * med, color=CONTROL, linestyle="--", linewidth=1.2, label=f"20 x median ({20 * med:.3g})")
        if ins:
            ax.plot([l for l, _ in ins], [v for _, v in ins], color=ORANGE_RAMP[1], marker="s", markersize=3.5, linewidth=1.1, label="max |input|")
        legend(ax, loc="upper right")
    else:
        no_data(ax, "superweight: global.layer_stats")
    return finish(fig, f"{ctx.model}: P7 super weights (Yu et al. 2024 recipe; 16 WikiText-2 windows)", source_of(doc, ctx.rel(ctx.path("superweight"))),
                  ctx.out("superweight"), "a super weight: dPPL > 100 % while the magnitude control stays < 2 % (H-P7)")


def fig_ablation(ctx: Ctx) -> Path | None:
    abl_path = ctx.pdir / "ablation.jsonl"
    rows = _ablation_rows(ctx)
    if not any("@" in r["method"] or "+" in r["method"] or io.parse_method(r["method"]) and io.parse_method(r["method"])["group"] != 128 for r in rows):
        skip(f"{ctx.rel(abl_path)}: no ablation rows")
        return None
    fp16 = io.fp16_ppl(rows)
    joint = {b: io.find_row(rows, f"rtn-sym-g128-{b}") for b in (4, 3)}
    only = {b: io.ablation_only(rows, b) for b in (4, 3)}
    types = [t for t in io.TYPES if t in only[4] or t in only[3]]
    fig, axes = new_fig((15, 11), 2, 2)
    axes = axes.ravel()

    def dppl(r) -> float | None:
        p = num(r.get("ppl")) if r else None
        return 100 * (p / fp16 - 1) if p is not None and fp16 else None

    for ax, key, ylabel, title in ((axes[0], "kl", "KL(P_ref || P_q), nats (log)", "@only=<type>: KL of quantizing one weight type (rest bf16)"),
                                   (axes[1], "dppl", "dPPL % (log)", "@only=<type>: PPL increase over fp16")):
        style(ax, title, None, ylabel, logy=True)
        drawn = 0
        sums = {}
        omitted = []  # values at or below 0 have no place on the log axis; they are listed in the panel, not dropped silently
        for j, b in enumerate((4, 3)):
            vals = []
            for i, t in enumerate(types):
                r = only[b].get(t)
                v = (num(r.get("kl")) if key == "kl" else dppl(r)) if r else None
                vals.append(v)
                if v is not None and v > 0:
                    drawn += 1
                    ax.bar(i + (j - 0.5) * 0.4, v, 0.38, color=BLUE_RAMP[1 + j], edgecolor=SURFACE, linewidth=0.8)
                elif v is not None:
                    omitted.append(f"{t} {b} bit {v:+.2f}" + (" %" if key == "dppl" else ""))
            good = [v for v in vals if v is not None]
            if good:
                sums[b] = sum(good)
            jv = (num(joint[b].get("kl")) if key == "kl" else dppl(joint[b])) if joint[b] else None
            if jv is not None and jv > 0:
                ax.axhline(jv, color=CONTROL, linestyle="--" if b == 4 else ":", linewidth=1.3,
                           label=f"all types, {b} bit: {jv:.3g}" + (f" (sum of @only = {sums[b]:.3g})" if b in sums and key == "kl" else ""))
        if drawn:
            type_ticks(ax, types)
            hs = [Patch(facecolor=BLUE_RAMP[1 + j], label=f"rtn-sym-g128-{b}@only") for j, b in enumerate((4, 3)) if only[b]]
            h2, _ = ax.get_legend_handles_labels()
            if omitted:
                hs.append(Line2D([], [], linestyle="none", label="no bar (value <= 0, off the log axis): " + ", ".join(omitted)))
            legend(ax, hs + h2, **below(2))
        else:
            no_data(ax, f"ablation: @only rows ({key})" + ("" if key == "kl" or fp16 else "; no fp16 row for dPPL"))

    ax = axes[2]
    style(ax, "@layer=<i>: KL of quantizing one decoder layer at 3 bits (rest bf16)", "layer", "KL, nats (log)", logy=True)
    layers = io.ablation_only(rows, 3, "layer")
    pts = sorted((int(k), num(r.get("kl"))) for k, r in layers.items() if k.lstrip("-").isdigit() and num(r.get("kl")))
    if pts:
        ax.bar([l for l, _ in pts], [v for _, v in pts], 0.8, color=BLUE_RAMP[2], edgecolor=SURFACE, linewidth=0.8, label="rtn-sym-g128-3@layer")
        if joint[3] and num(joint[3].get("kl")):
            ax.axhline(joint[3]["kl"], color=CONTROL, linestyle=":", linewidth=1.3,
                       label=f"all layers, 3 bit: {joint[3]['kl']:.3g} (sum of @layer = {sum(v for _, v in pts):.3g})")
        ax.set_xticks([l for l, _ in pts])
        ax.tick_params(axis="x", labelsize=7)
        legend(ax, **below(2, 0.15))
    else:
        no_data(ax, "ablation: @layer rows at 3 bits")

    ax = axes[3]
    style(ax, "bits vs KL: group size (g32 / g64 / g128), +keepcols p, +keepsuper and GPTQ", "bits_blocks", "KL, nats (log)", logy=True)
    series: dict[str, list] = {}
    for r in rows:
        p = io.parse_method(r["method"])
        if p is None or p["qual"] is not None or num(r.get("kl")) is None or r["kl"] <= 0:
            continue
        if p["plus"] == "keepcols":
            name = f"rtn-sym-g128+keepcols={p['plus_value']}"
        elif p["plus"] == "keepsuper":
            name = "rtn-sym-g128+keepsuper"
        elif p["plus"] is None and p["family"] in ("rtn-sym-g32", "rtn-sym-g64", "rtn-sym-g128", "gptq-sym-g128"):
            name = p["family"]
        else:
            continue
        series.setdefault(name, []).append((r["bits_blocks"], r["kl"], p["bits"], r.get("seed", 0), r.get("calib_seed"),
                                            bool(io.hessian_tag(r.get("note", "")))))
    palette = {"rtn-sym-g32": (BLUE_RAMP[0], "o"), "rtn-sym-g64": (BLUE_RAMP[1], "o"), "rtn-sym-g128": (BLUE_RAMP[2], "o"),
               "rtn-sym-g128+keepcols=0.5": (ORANGE_RAMP[0], "P"), "rtn-sym-g128+keepcols=1": (ORANGE_RAMP[1], "P"),
               "rtn-sym-g128+keepcols=2": (ORANGE_RAMP[2], "P"), "rtn-sym-g128+keepsuper": ("#7a3b8f", "X"), "gptq-sym-g128": (GPTQ, "^")}
    if series:
        for name in palette:
            pts = sorted(series.get(name, []))
            if not pts:
                continue
            color, marker = palette[name]
            main = [p for p in pts if p[3] == 0 and p[4] in (None, 0) and not p[5]]
            ax.plot([x for x, *_ in main], [y for _, y, *_ in main], color=color, marker=marker, markersize=6, linewidth=1.6, label=name,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
            for x, y, b, seed, cs, h32 in pts:
                # the f32-Hessian acceptance row is a detached point like a seed repeat (plot_rd uses the same label)
                lab = f"{b}" + (f" s{seed}" if seed else "") + (f" c{cs}" if cs not in (None, 0) else "") + (" H32" if h32 else "")
                detached = (seed, cs) not in ((0, None), (0, 0)) or h32
                if detached:
                    ax.plot([x], [y], color=color, marker=marker, markersize=6, linestyle="none", markeredgecolor=TEXT2, markeredgewidth=0.8)
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -18 if h32 else -9 if detached else 3), fontsize=7, color=TEXT2)
        ax.margins(x=0.1)
        legend(ax, **below(3, 0.15))
    else:
        no_data(ax, "ablation: g32/g64/keepcols/gptq rows")
    src = f"{ctx.rel(abl_path)}; fp16 and joint RTN rows from {ctx.rel(ctx.rdir / ctx.model / 'runs.jsonl')}"
    return finish(fig, f"{ctx.model}: P0 direct RTN ablation (measured PPL / KL, WikiText-2 protocol)", src, ctx.out("ablation"),
                  (f"fp16 ppl {fp16:.4g}; " if fp16 else "") + "@only bits_blocks mix 16-bit and b-bit tensors; a point with s<seed> / c<calib_seed> is a seed repeat, "
                  "H32 the f32-Hessian GPTQ acceptance row (detached marker)")


FIGURES = {"marginal": fig_marginal, "lowrank": fig_lowrank, "outliers": fig_outliers, "interlayer": fig_interlayer, "acts": fig_acts,
           "sensitivity": fig_sensitivity, "fisher": fig_fisher, "superweight": fig_superweight, "ablation": fig_ablation}


def render(results_dir: Path, model: str, probes: list[str] | None = None, out_dir: Path | None = None,
           include_smoke: bool = False) -> dict[str, Path | None]:
    ctx = Ctx(results_dir, model, out_dir or results_dir, include_smoke)
    out = {}
    for name in probes or FIGURES:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name!r}; known: {sorted(FIGURES)}")
        out[name] = FIGURES[name](ctx)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--probes", default=None, help=f"comma-separated subset of {','.join(FIGURES)} (default all)")
    ap.add_argument("--out-dir", default=None, help="directory for the png files (default <results-dir>)")
    ap.add_argument("--include-smoke", action="store_true", help="also read limit-windows / kl-windows / calib-n ablation rows")
    args = ap.parse_args()
    rdir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else ROOT / args.results_dir
    probes = [p.strip() for p in args.probes.split(",") if p.strip()] if args.probes else None
    out_dir = Path(args.out_dir) if args.out_dir else rdir
    done = render(rdir, args.model, probes, out_dir, args.include_smoke)
    missing = [k for k, v in done.items() if v is None]
    if missing:
        print(f"not rendered (no data): {', '.join(missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
