"""Rate-distortion plot from results/<model>/runs.jsonl (DESIGN.md, scripts/plot_rd.py).

Left panel: bits_blocks vs ppl - ppl_fp16 (log y). Right panel: bits_blocks vs KL(P_ref || P_q) (log y).
One line per method family (name without the trailing bit width) through its default-seed runs, points labelled
with the bit width in the family color; a run with a non-default seed (" s<seed>" / " c<calib_seed>") is a detached
marker next to its family. Labels of different families sit in different quadrants of their point, so families
that share an x stay readable. A run whose ppl is not above fp16 cannot be placed on the log axis: it is drawn
as a hollow marker at a floor (half the smallest positive delta), its label carries the measured value, and it
is reported on stderr, never silently dropped. The fp16 reference (y = 0, off the log axis) is stated in each
legend; the PPL panel also carries a right-hand axis in absolute WikiText-2 PPL. Rotation-only rows (had-fp16:
no quantization, bits 16) are drawn as a dashed reference line, so the had-* deltas can be read against the bf16
rotation cost.

Smoke rows are off the DESIGN.md protocol: their note carries "limit-windows N", "kl-windows N" or "calib-n N"
(older rows: a "calib Nx2048" token with N != 128). They are ignored unless --include-smoke is given, and each
smoke row is differenced against the fp16 row of the *same* eval windows (same "limit-windows" marker); a smoke
row without such an fp16 row is reported and skipped. Rows of one (method, seed, calib_seed, smoke setting) are
deduplicated to the latest timestamp; when they disagree in note, code version, ppl or kl a warning lists them.
Output: <results-dir>/rd_<model>.png

--extra <jsonl> (repeatable, Phase 2): further files in the same row schema, e.g. results/probe/<model>/ablation.jsonl.
Only full-model quantizer rows are drawn from them (rtn-sym-g32/g64, +keepcols, +keepsuper, had-*, GPTQ seed
repeats); @only / @except / @layer rows quantize part of the model and are never RD points, so they are skipped
(counted on stderr). A variant of a Phase 1 family (another group size, or a +keepcols / +keepsuper suffix) keeps
the parent family's colour and takes its own line style and marker; +keepcols rows form one family with the
percentage in the point label. Without --extra the output is unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wcomp.methods import METHODS, family  # noqa: E402

# Fixed family order and colors (validated categorical palette, light mode); never cycled.
FAMILIES = ["rtn-sym-g128", "rtn-asym-g128", "had-rtn-sym-g128", "gptq-sym-g128", "had-gptq-sym-g128"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Label placement per family slot (offset in points, horizontal alignment): right-up, right-down, left-up,
# left-down, then further right. Families that share an x (all of Phase 1) get non-overlapping labels.
LABEL_SLOTS = [((7, 5), "left"), ((7, -11), "left"), ((-7, 5), "right"), ((-7, -11), "right"), ((18, 5), "left")]
# Candidate ticks of the absolute-PPL axis on the right of the PPL panel.
PPL_TICKS = [12, 15, 20, 25, 30, 40, 50, 70, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 1e5, 1e6, 1e7]

PROTOCOL_CALIB_N = 128  # DESIGN.md 4
SMOKE_PREFIXES = ("limit-windows", "kl-windows", "calib-n")
_CALIB_RE = re.compile(r"^calib (\d+)x\d+ seed (\d+)$")
# Line style and marker of a variant family (--extra rows), drawn in its parent Phase 1 family's colour.
VARIANT_STYLES = {"g32": ("--", "s"), "g64": (":", "D"), "+keepcols": ("-.", "P"), "+keepsuper": ((0, (3, 1, 1, 1, 1, 1)), "X")}


def run_family(name: str) -> tuple[str, int | None]:
    """family() extended to the Phase 2 parametric names: "rtn-sym-g128-4+keepcols=1" -> ("rtn-sym-g128+keepcols", 4),
    "rtn-sym-g128-4+keepsuper=<path>" -> ("rtn-sym-g128+keepsuper", 4). Names without "+" behave exactly as family()."""
    base, plus, suffix = name.partition("+")
    fam, bits = family(base)
    if plus:
        fam = f"{fam}+{suffix.split('=')[0]}"
    return fam, bits


def plus_value(name: str) -> str | None:
    """The value after "+<key>=" of a parametric name (the keepcols percentage), else None."""
    _, plus, suffix = name.partition("+")
    if not plus or "=" not in suffix:
        return None
    return suffix.split("=", 1)[1]


def variant_of(fam: str) -> tuple[str, str] | None:
    """(parent family, variant key) when fam is a group-size or +suffix variant of a FAMILIES member, else None."""
    base, plus, suffix = fam.partition("+")
    parent = re.sub(r"-g\d+$", "-g128", base)
    if parent not in FAMILIES:
        return None
    keys = []
    m = re.search(r"-g(\d+)$", base)
    if m and m.group(1) != "128":
        keys.append(f"g{m.group(1)}")
    if plus:
        keys.append(f"+{suffix}")
    if len(keys) != 1 or keys[0] not in VARIANT_STYLES:
        return None
    return parent, keys[0]


def is_full_model(method: str) -> bool:
    """False for @only / @except / @layer rows (part of the model quantized): never a rate-distortion point."""
    return "@" not in method


def merge_runs(primary: list[dict], extra: list[dict]) -> list[dict]:
    """Union of rows by run_key. A Phase 1 method name (wcomp.methods.METHODS) present in the primary file is never
    replaced by an --extra row; any other key keeps the latest timestamp (the primary file wins ties). Rows of one
    key that disagree (rows_disagree) are reported on stderr like load_runs does, never resolved silently. The
    Hessian path is part of run_key, so the gptq-sym-g128-3 acceptance row (hessian-f32) is a separate point."""
    latest: dict[tuple, dict] = {run_key(r): r for r in primary}
    for r in extra:
        key = run_key(r)
        prev = latest.get(key)
        if prev is None:
            latest[key] = r
            continue
        phase1 = r["method"] in METHODS
        newer = r.get("timestamp", "") > prev.get("timestamp", "")
        if rows_disagree(prev, r):
            print(f"warning: {r['method']} (seed {key[1]}, calib_seed {key[2]}): primary row (ppl {prev.get('ppl')}, {prev.get('timestamp')}) "
                  f"and --extra row (ppl {r.get('ppl')}, {r.get('timestamp')}) disagree; keeping "
                  f"{'the primary (Phase 1 name)' if phase1 else ('the later --extra row' if newer else 'the primary')}", file=sys.stderr)
        if not phase1 and newer:
            latest[key] = r
    return list(latest.values())


def smoke_tag(note: str) -> str:
    """The off-protocol markers of a note, sorted and joined ("calib-n 16; kl-windows 4; limit-windows 8").

    "" means a full protocol run. Rows written before the "calib-n" marker existed are recognized by their
    "calib Nx2048 seed s" token with N != 128.
    """
    tags = set()
    for tok in note.split(";"):
        tok = tok.strip()
        if tok.startswith(SMOKE_PREFIXES):
            tags.add(tok)
            continue
        m = _CALIB_RE.match(tok)
        if m and int(m.group(1)) != PROTOCOL_CALIB_N:
            tags.add(f"calib-n {m.group(1)}")
    return "; ".join(sorted(tags))


def eval_tag(note: str) -> str:
    """The markers that change the *evaluated windows* ("limit-windows N", or ""), which is what an fp16 baseline
    must match; kl-windows and calib-n change the KL token count / calibration, not the fp16 PPL."""
    return "; ".join(sorted(t for t in smoke_tag(note).split("; ") if t.startswith("limit-windows")))


def is_smoke(r: dict) -> bool:
    return smoke_tag(r.get("note", "")) != ""


HESSIAN_F32_TOKEN = "hessian-f32"  # wcomp.quant.gptq note token of the f32-product Hessian path


def hessian_tag(note: str) -> str:
    """"hessian-f32" for a GPTQ row that ran the f32-product Hessian path, else "" (the Phase 1 f64 path, whether the
    note says "hessian-f64" or predates the token)."""
    return HESSIAN_F32_TOKEN if HESSIAN_F32_TOKEN in re.split(r"[;\s]+", note) else ""


def run_key(r: dict) -> tuple:
    """(method, seed, calib_seed, smoke tag, hessian tag). A calibrated row without a calib_seed field (older code)
    gets the seed recorded in its "calib Nx2048 seed s" note token. The Hessian path is part of the identity so a
    differently configured run of a Phase 1 method name is a separate (detached) point, never a replacement."""
    note = r.get("note", "")
    calib_seed = r.get("calib_seed")
    if calib_seed is None:
        for tok in note.split(";"):
            m = _CALIB_RE.match(tok.strip())
            if m:
                calib_seed = int(m.group(2))
    return (r["method"], r.get("seed", 0), calib_seed, smoke_tag(note), hessian_tag(note))


CONFLICT_FIELDS = ("note", "git_or_code_version", "ppl", "kl")


def rows_disagree(a: dict, b: dict) -> bool:
    """True when two rows of the same run_key differ in settings, code version or result (ppl / kl).

    A rerun under identical settings that lands on another ppl is a non-reproducible run (or a GPU fault) and
    must be reported, not silently replaced; byte-identical duplicates are not a conflict.
    """
    return any(a.get(k) != b.get(k) for k in CONFLICT_FIELDS)


def load_runs(path: Path, include_smoke: bool = False) -> list[dict]:
    """Latest run (by timestamp, then file order) per run_key; smoke rows only on request.

    Rows that share a key but differ in note, git_or_code_version, ppl or kl (rows_disagree) come from different
    code versions / settings or from a non-reproducible rerun: the latest wins and the conflict is reported on
    stderr. Identical duplicates collapse silently.
    """
    latest: dict[tuple, dict] = {}
    conflicts: dict[tuple, list[dict]] = {}
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if is_smoke(r) and not include_smoke:
                skipped += 1
                continue
            key = run_key(r)
            prev = latest.get(key)
            if prev is not None and rows_disagree(prev, r):
                conflicts.setdefault(key, [prev]).append(r)
            if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
                latest[key] = r
    if skipped:
        print(f"note: ignored {skipped} smoke rows (--include-smoke to plot them)", file=sys.stderr)
    for key, rows in conflicts.items():
        kept = latest[key]
        print(f"warning: {key[0]} (seed {key[1]}, calib_seed {key[2]}): {len(rows)} rows that disagree in note, code "
              f"version, ppl or kl, keeping the latest ({kept.get('timestamp')}, ppl {kept['ppl']:.6g}):", file=sys.stderr)
        for r in rows:
            print(f"    {r.get('timestamp')} version {r.get('git_or_code_version')} ppl {r['ppl']:.6g} "
                  f"note '{r.get('note', '')}'", file=sys.stderr)
    return list(latest.values())


def baselines(runs: list[dict]) -> dict[str, float]:
    """fp16 ppl per eval_tag ("" = all windows). fp16 rows of one tag that agree in ppl (e.g. seeds 0 and 1: the
    seed does not affect an unquantized model, nor does kl-windows) collapse; disagreeing ones are an error."""
    out: dict[str, float] = {}
    for r in runs:
        if r["method"] != "fp16":
            continue
        tag = eval_tag(r.get("note", ""))
        if tag in out and (out[tag] != r["ppl"]):
            raise SystemExit(f"fp16 rows with different ppl for the same windows ({tag or 'full run'}): "
                             f"{out[tag]:.6g} vs {r['ppl']:.6g}; keep one")
        out[tag] = r["ppl"]
    if not out:
        raise SystemExit("no fp16 run to subtract from")
    return out


def baseline_ppl(runs: list[dict]) -> float:
    """The full-protocol fp16 ppl (error if there is none or if several disagree)."""
    b = baselines(runs)
    if "" not in b:
        raise SystemExit("no full-protocol fp16 run to subtract from")
    return b[""]


def family_style(fam: str, order: list[str]) -> tuple[str, str]:
    """(color, marker) of a family: fixed by its position in FAMILIES, so gptq is the same color on every model
    whatever other families are present; families outside FAMILIES take the slots after it, in `order`."""
    if fam in FAMILIES:
        i = FAMILIES.index(fam)
    else:
        extra = [f for f in order if f not in FAMILIES]
        i = len(FAMILIES) + extra.index(fam)
    return COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]


def line_ys(ys: list[float], hollow: list[bool]) -> list[float]:
    """y values for the family's connecting line: NaN at hollow (floored) points, so the line is drawn through
    measured values only and never descends to the artificial floor."""
    return [math.nan if h else y for y, h in zip(ys, hollow)]


def default_seed(r: dict) -> bool:
    """True for the protocol's default configuration (seed 0, calib_seed 0 or none, Phase 1 f64 Hessian path): the
    rows the family line goes through; an f32-Hessian row is a detached marker like a seed repeat."""
    return r.get("seed", 0) == 0 and r.get("calib_seed") in (None, 0) and hessian_tag(r.get("note", "")) == ""


def point_label(r: dict) -> str:
    bits = run_family(r["method"])[1]
    lab = f"{bits}"
    p = plus_value(r["method"])
    if p is not None and "+keepcols" in r["method"]:
        lab += f" p{p}"
    if r.get("seed", 0) != 0:
        lab += f" s{r['seed']}"
    if r.get("calib_seed") not in (None, 0):
        lab += f" c{r['calib_seed']}"
    if hessian_tag(r.get("note", "")):
        lab += " H32"
    return lab


def label_offset(slot: int, flip: bool = False) -> tuple[tuple[int, int], str]:
    """((dx, dy) in points, ha) of a family's labels; flip mirrors dy for a detached (non-default-seed) marker so
    its label does not land on the default-seed label at the same x."""
    (dx, dy), ha = LABEL_SLOTS[slot % len(LABEL_SLOTS)]
    if flip:
        dy = 5 if dy < 0 else -11
    return (dx, dy), ha


LEGEND_LOCS = ("upper right", "lower left", "center right", "upper left", "lower right")


def place_legend(ax, handles: list, points: list[tuple[float, float]], **kw) -> None:
    """Legend at the candidate location whose box covers the fewest plotted points (ties: first in LEGEND_LOCS).
    matplotlib's "best" counts the full-width had-fp16 reference line as an obstacle everywhere, so it ends up
    crossing that line or sitting on the point labels; a reference line under the legend is harmless, a hidden
    point is not. Labels sit right of their points, so each point is widened by 60 px on that side."""
    fig = ax.figure
    best = None
    for loc in LEGEND_LOCS:
        leg = ax.legend(handles=handles, loc=loc, **kw)
        fig.canvas.draw()
        box = leg.get_window_extent()
        disp = ax.transData.transform(points) if points else []
        n = sum(1 for x, y in disp if box.x0 - 60 <= x <= box.x1 and box.y0 <= y <= box.y1)
        if best is None or n < best[0]:
            best = (n, loc)
        if n == 0:
            break
    ax.legend(handles=handles, loc=best[1], **kw)


def absolute_ppl_axis(ax, fp16: float) -> None:
    """Right-hand axis of the PPL panel in absolute WikiText-2 PPL: ticks at round PPL values placed at
    PPL - fp16 on the shared log scale, thinned so neighbouring labels are at least a factor 1.6 apart."""
    lo, hi = ax.get_ylim()
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(lo, hi)
    pos, labels = [], []
    for t in PPL_TICKS:
        d = t - fp16
        if not (lo <= d <= hi) or (pos and d / pos[-1] < 1.6):
            continue
        pos.append(d)
        labels.append(f"{t:,.0f}")
    ax2.yaxis.set_major_locator(FixedLocator(pos))
    ax2.yaxis.set_major_formatter(FixedFormatter(labels))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.set_ylabel(f"absolute PPL (fp16 = {fp16:.4g})", color=TEXT2)
    ax2.tick_params(colors=TEXT2, labelsize=8)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.spines["right"].set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="output png (default <results-dir>/rd_<model>.png)")
    ap.add_argument("--include-smoke", action="store_true", help="also plot limit-windows / kl-windows / calib-n rows")
    ap.add_argument("--extra", action="append", default=[], metavar="JSONL",
                    help="additional runs.jsonl-schema file (repeatable); only full-model quantizer rows are drawn")
    args = ap.parse_args()

    rdir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else ROOT / args.results_dir
    runs = load_runs(rdir / args.model / "runs.jsonl", args.include_smoke)
    sources = [(Path(args.results_dir) / args.model / "runs.jsonl").as_posix()]
    for extra in args.extra:
        epath = Path(extra) if Path(extra).is_absolute() else ROOT / extra
        rows = load_runs(epath, args.include_smoke)
        full = [r for r in rows if is_full_model(r["method"])]
        if len(full) < len(rows):
            print(f"note: {extra}: {len(rows) - len(full)} @only / @except / @layer rows are not full-model points, skipped",
                  file=sys.stderr)
        runs = merge_runs(runs, full)
        sources.append(Path(extra).as_posix())
    base = baselines(runs)
    smoke = {smoke_tag(r.get("note", "")) for r in runs if is_smoke(r)}
    if smoke:
        print(f"warning: plotting smoke runs ({' | '.join(sorted(smoke))})", file=sys.stderr)
    if len(base) > 1:
        print("note: fp16 baselines per window subset: " + ", ".join(f"[{t or 'all'}] {p:.6g}" for t, p in sorted(base.items())),
              file=sys.stderr)

    # Every plotted row is differenced against the fp16 row of its own windows; rows without one are skipped.
    plotted, dppl = [], {}
    for r in runs:
        if r["method"] == "fp16":
            continue
        tag = eval_tag(r.get("note", ""))
        if tag not in base:
            print(f"warning: {r['method']} ({r.get('note', '')}): no fp16 row for these windows, skipped", file=sys.stderr)
            continue
        plotted.append(r)
        dppl[id(r)] = r["ppl"] - base[tag]

    by_family: dict[str, list[dict]] = {}
    refs: list[dict] = []  # unquantized reference rows other than fp16, e.g. had-fp16 (rotation only)
    for r in plotted:
        fam, bits = run_family(r["method"])
        if bits is None:
            refs.append(r)
            continue
        by_family.setdefault(fam, []).append(r)
    order = [f for f in FAMILIES if f in by_family] + sorted(set(by_family) - set(FAMILIES))

    # Values that cannot be placed on a log axis (dppl <= 0, KL <= 0, NaN) are drawn at a floor, hollow.
    pos_dppl = [dppl[id(r)] for r in plotted if dppl[id(r)] > 0]
    pos_kl = [r["kl"] for r in plotted if r.get("kl") is not None and r["kl"] > 0]
    floor = {"dppl": min(pos_dppl) / 2 if pos_dppl else 1e-3, "kl": min(pos_kl) / 2 if pos_kl else 1e-5}
    kl_tokens = sorted({r.get("kl_tokens") for r in plotted if r.get("kl_tokens")})
    n_tokens = sorted({r.get("n_tokens") for r in plotted if r.get("n_tokens")})
    fp16_rows = [r for r in runs if r["method"] == "fp16" and eval_tag(r.get("note", "")) == ""]
    fp16_bits = fp16_rows[0]["bits_blocks"] if fp16_rows else 16.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_yscale("log")
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT2, labelsize=9)
        ax.margins(x=0.08)
        ax.set_xlabel("bits_blocks: bits per parameter of the quantized block linears", color=TEXT2)
    if len(base) == 1:
        axes[0].set_ylabel(f"ppl - ppl(fp16 = {next(iter(base.values())):.4g}), log scale", color=TEXT2)
    else:
        axes[0].set_ylabel("ppl - ppl(fp16, same windows), log scale", color=TEXT2)
    axes[1].set_ylabel("kl: KL(P_ref || P_q), nats per token, log scale", color=TEXT2)
    axes[0].set_title(f"WikiText-2 perplexity increase over fp16 ({', '.join(f'{k:,}' for k in n_tokens) or '?'} tokens)",
                      color=TEXT, fontsize=10, loc="left")
    axes[1].set_title(f"KL to the bf16 model ({', '.join(f'{k:,}' for k in kl_tokens) or '?'} tokens)", color=TEXT, fontsize=10, loc="left")

    any_hollow = False
    drawn_points: dict[int, list[tuple[float, float]]] = {id(ax): [] for ax in axes}  # for the legend placement
    for slot, fam in enumerate(order):
        rows = sorted(by_family[fam], key=lambda r: r["bits_blocks"])
        variant = variant_of(fam)
        if variant is not None:  # parent colour, variant line style / marker
            color = family_style(variant[0], order)[0]
            linestyle, marker = VARIANT_STYLES[variant[1]]
        else:
            color, marker = family_style(fam, order)
            linestyle = "-"
        for ax, key in ((axes[0], "dppl"), (axes[1], "kl")):
            xs, ys, labels, hollow, extra = [], [], [], [], []
            for r in rows:
                y = dppl[id(r)] if key == "dppl" else r.get("kl")
                if y is None:
                    continue
                ok = math.isfinite(y) and y > 0
                if not ok:
                    what = f"ppl {r['ppl']:.6g} vs fp16 {r['ppl'] - dppl[id(r)]:.6g}" if key == "dppl" else f"kl {y}"
                    print(f"warning: {r['method']} ({r.get('note', '')}): {what} not on the log axis, drawn hollow at "
                          f"{floor[key]:.3g}", file=sys.stderr)
                    any_hollow = True
                lab = point_label(r)
                if not ok:  # the floor is not a measurement: the label states the measured value
                    lab += f" ({'ppl-fp16' if key == 'dppl' else 'kl'} = {y:+.3g})"
                if not default_seed(r):
                    extra.append((r["bits_blocks"], y if ok else floor[key], lab, not ok))
                    continue
                xs.append(r["bits_blocks"])
                ys.append(y if ok else floor[key])
                labels.append(lab)
                hollow.append(not ok)
            if not xs and not extra:
                continue
            drawn_points[id(ax)] += list(zip(xs, ys)) + [(x, y) for x, y, _, _ in extra]
            # The line and the solid markers go through the default-seed measured points only; floored points are
            # isolated hollow markers (a line into the floor would assert a positive delta that was not measured)
            # and other seeds are detached markers (a seed is a repeat of the same point, not a new rate).
            if xs:
                ax.plot(xs, line_ys(ys, hollow), color=color, marker=marker, markersize=6, linewidth=2, label=fam,
                        markeredgecolor=SURFACE, markeredgewidth=1, linestyle=linestyle)
            for x, y, lab, h in extra:
                if not h:
                    ax.plot([x], [y], marker=marker, markersize=6, color=color, markeredgecolor=TEXT2,
                            markeredgewidth=0.8, linestyle="none")
            for x, y, lab, h in [(x, y, lab, h) for x, y, lab, h in zip(xs, ys, labels, hollow)] + extra:
                if h:
                    ax.plot([x], [y], marker=marker, markersize=8, markerfacecolor=SURFACE, markeredgecolor=color,
                            markeredgewidth=1.5, linestyle="none")
            for flip, pts in ((False, list(zip(xs, ys, labels))), (True, [(x, y, lab) for x, y, lab, _ in extra])):
                (dx, dy), ha = label_offset(slot, flip)
                for x, y, lab in pts:
                    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8, color=color,
                                bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none", alpha=0.85))
    for r in refs:
        for ax, y in ((axes[0], dppl[id(r)]), (axes[1], r.get("kl"))):
            if y is None or not (math.isfinite(y) and y > 0):
                print(f"note: {r['method']} ({r.get('note', '')}): value {y} not on the log axis, no reference line", file=sys.stderr)
                continue
            ax.axhline(y, color=TEXT2, linestyle="--", linewidth=1, label=f"{r['method']} (no quantization): {y:.3g}")
    # The fp16 reference sits at y = 0 on both panels, below any log axis: it is stated in the legend instead.
    fp16_text = [f"fp16 reference: ppl {next(iter(base.values())):.4g} at {fp16_bits:g} bits (y = 0, off this axis)"
                 if len(base) == 1 else "fp16 reference: y = 0, off this axis",
                 f"fp16 reference: kl 0 at {fp16_bits:g} bits (y = 0, off this axis)"]
    for ax, text in zip(axes, fp16_text):
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle="none", label=text))
        place_legend(ax, handles, drawn_points[id(ax)], frameon=False, fontsize=8, labelcolor=TEXT)
    if len(base) == 1:
        absolute_ppl_axis(axes[0], next(iter(base.values())))
    what = "Phase 1 baselines" if not args.extra else "Phase 1 baselines + Phase 2 full-model ablation rows"
    fig.suptitle(f"{args.model}: rate-distortion of {what}", color=TEXT, fontsize=12, x=0.02, ha="left")
    footer = (f"source: {', '.join(sources)} (fields bits_blocks, ppl, kl); "
              "point labels: bit width, s<seed> / c<calib_seed> when not 0, H32 for the f32-Hessian GPTQ path (detached marker)")
    if args.extra:
        footer += ", p<percent> for +keepcols"
    if any_hollow:
        footer += "\nhollow marker: value <= 0 cannot be placed on the log axis; drawn at the floor, measured value in its label"
    fig.text(0.02, 0.012, footer, color=TEXT2, fontsize=7.5, ha="left", va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.035 + 0.025 * any_hollow, 1, 1), w_pad=3)
    out = Path(args.out) if args.out else rdir / f"rd_{args.model}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
