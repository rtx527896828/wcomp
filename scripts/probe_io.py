"""Readers for the Phase 2 probe output (DESIGN.md "Phase 2 contract", Output), shared by plot_probe.py and
structure_table.py.

Files: results/probe/<model>/<probe>.json with {"model", "probe", "code_version", "timestamp", "params", "records",
"aggregate"[, "global"]}; each record {"tensor", "layer", "type", "stat", "control", "gain_bits", "gain_bits_control",
"note", "numel"}, and results/probe/<model>/ablation.jsonl in the run_baselines.py row schema with parametric method
names. Stat names follow wcomp/probe/{weights,acts}.py; where the two modules name the same quantity differently
(lowrank "energy_f0.01" vs lowrank_weighted "energy_r1pct" / control "seed0_energy_r1pct") the canonical key below
resolves through ALIASES, so callers write val(rec, "control", "energy@0.01").

Everything here is tolerant: a missing file, record field or stat yields None / an empty list and a line on stderr
saying what was skipped; nothing raises on absent data. Aggregates are recomputed from the records (numel-weighted
means), so the readers do not depend on the layout of the probe's own "aggregate" block; that block and "global" are
consulted only for model-level entries (allocation, Jaccard, down_proj spikes, prediction check).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

TYPES = ["q", "k", "v", "o", "gate", "up", "down", "in_proj_qkv", "in_proj_z", "in_proj_ab", "out_proj"]
PRIMARY_ACT_TYPES = ["down", "o", "out_proj"]  # P5 inputs the contract names
PROBES = ["marginal", "lowrank", "lowrank_weighted", "outliers", "interlayer", "acts", "sensitivity", "fisher", "superweight"]
R_GRID = ["0.01", "0.05", "0.1", "0.25"]  # rank fractions of the contract; "0.1" is spelled as the probes spell it
R_PCT = {"0.01": "1", "0.05": "5", "0.1": "10", "0.25": "25"}
P_GRID = ["0.5", "1", "2"]
ALIGNMENTS = ["e_raw", "e_out", "e_in", "e_perm"]

# canonical key -> candidate keys, first hit wins. "@r" / "@p" keys are expanded per grid value.
ALIASES: dict[str, list[str]] = {
    "kurtosis": ["kurtosis_excess", "kurtosis"],
    "frac_4sigma": ["frac_gt4sigma", "frac_4sigma"],
    "frac_6sigma": ["frac_gt6sigma", "frac_6sigma"],
    "best_fit": ["best_family", "best_fit"],
    "H_4": ["H_idx_b4", "H_4"], "H_3": ["H_idx_b3", "H_3"], "H_2": ["H_idx_b2", "H_2"],
    "gain_bits_4": ["gain_bits_b4", "gain_bits_4"], "gain_bits_3": ["gain_bits_b3", "gain_bits_3"], "gain_bits_2": ["gain_bits_b2", "gain_bits_2"],
    "rank90": ["rank90", "rank_90"], "rank99": ["rank99", "rank_99"],
    "gain_best": ["gain_bits_best", "best_gain"],
    "col_n_5x": ["col_n_gt5x_median", "col_n_5x"], "col_n_10x": ["col_n_gt10x_median", "col_n_10x"],
    "row_n_5x": ["row_n_gt5x_median", "row_n_5x"], "row_n_10x": ["row_n_gt10x_median", "row_n_10x"],
    "col_norm_kurtosis": ["col_kurtosis_excess", "col_norm_kurtosis"], "row_norm_kurtosis": ["row_kurtosis_excess", "row_norm_kurtosis"],
    "e_out": ["e_out", "e_procrustes_out"], "e_in": ["e_in", "e_procrustes_in"],
    "share50": ["frac_channels_50", "share50"], "share90": ["frac_channels_90", "share90"],
    "spearman_split": ["spearman_split_half", "spearman_split"],
    "frac_inactive": ["frac_active_lt_1pct", "frac_inactive"],
    "s_t": ["s", "s_t"], "fisher_total": ["total", "fisher_total"],
    "dppl": ["dppl_pct", "dppl"],
    "mag_dppl": ["magnitude_dppl_pct", "dppl_mag_control"], "mag_kl": ["magnitude_kl", "kl_mag_control"],
    "random_dppl": ["random_dppl_pct", "dppl_random"], "random_kl": ["random_kl", "kl_random"],
}
for _r in R_GRID:
    ALIASES[f"energy@{_r}"] = [f"energy_f{_r}", f"energy_r{R_PCT[_r]}pct", f"energy_f{float(_r):.2f}"]
    ALIASES[f"gain@{_r}"] = [f"gain_f{_r}", f"gain_r{R_PCT[_r]}pct", f"gain_f{float(_r):.2f}"]
for _p in P_GRID:
    ALIASES[f"rho@{_p}"] = [f"rho_p{_p}", f"rho_p{float(_p):g}"]
    ALIASES[f"gain@p{_p}"] = [f"gain_p{_p}", f"gain_p{float(_p):g}"]

_skipped: set[str] = set()


def skip(msg: str) -> None:
    """Report a skipped file / stat once on stderr."""
    if msg not in _skipped:
        _skipped.add(msg)
        print(f"skipped: {msg}", file=sys.stderr)


def probe_dir(results_dir: Path, model: str) -> Path:
    return results_dir / "probe" / model


def load_probe(path: Path) -> dict | None:
    """The probe JSON, or None (reported) when the file is missing or unreadable."""
    if not path.exists():
        skip(f"{path.as_posix()} (missing)")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        skip(f"{path.as_posix()} ({e})")
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("records"), list):
        skip(f"{path.as_posix()} (no records list)")
        return None
    return doc


def get(d, *keys, default=None):
    """Nested lookup tolerant of None / missing keys / non-dicts; non-finite numbers come back as default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if isinstance(cur, float) and not math.isfinite(cur):
        return default
    return cur


def num(x) -> float | None:
    """x as a finite float, else None (strings, bools, None, NaN)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def val(rec: dict, section: str, key: str):
    """A stat of a record by canonical key. section is "stat", "control" or "control_seed1" (the second shuffle
    seed: read from rec["control_seed1"][key], else rec["control"]["seed1_<key>"] / "<key>_seed1"); the "control"
    section also accepts "seed0_<key>". Returns None when absent."""
    cands = ALIASES.get(key, [key])
    if section == "control_seed1":
        sub = rec.get("control_seed1")
        for k in cands:
            if isinstance(sub, dict) and k in sub:
                return get(sub, k)
        ctrl = rec.get("control")
        for k in cands:
            for kk in (f"seed1_{k}", f"{k}_seed1"):
                if isinstance(ctrl, dict) and kk in ctrl:
                    return get(ctrl, kk)
        return None
    sub = rec.get(section)
    if not isinstance(sub, dict):
        return None
    for k in cands:
        if k in sub:
            return get(sub, k)
    if section == "control":
        for k in cands:
            if f"seed0_{k}" in sub:
                return get(sub, f"seed0_{k}")
    return None


def numel(rec: dict) -> float:
    """Weight of a record in aggregates: its numel if recorded (record or stat level), else 1."""
    for v in (rec.get("numel"), get(rec, "stat", "numel"), get(rec, "stat", "n_t")):
        n = num(v)
        if n is not None:
            return n
    skip(f"record {rec.get('tensor', '?')} without numel: counted with weight 1")
    return 1.0


def is_summary(rec: dict) -> bool:
    """True for a cross-layer / model-level summary record (P3 Jaccard: tensor "<type>:...:cross-layer" or
    "all-hidden-input:cross-layer", numel 0), which is not a tensor and must not enter per-tensor statistics."""
    return "cross-layer" in str(rec.get("tensor", "")) or rec.get("type") in ("", "all_hidden_input") or num(rec.get("numel")) == 0


def records(doc: dict | None, types=None, adjacent_only: bool = False, summaries: bool = False) -> list[dict]:
    """Tensor records of a probe (optionally restricted to `types`; `adjacent_only` keeps interlayer records whose
    stat.is_adjacent is 1 or absent; `summaries=True` returns the summary records instead); malformed entries are
    dropped."""
    if doc is None:
        return []
    out = []
    for r in doc["records"]:
        if not isinstance(r, dict) or "type" not in r:
            continue
        if is_summary(r) != summaries:
            continue
        if types is not None and r["type"] not in types:
            continue
        if adjacent_only and get(r, "stat", "is_adjacent") not in (None, 1, 1.0, True):
            continue
        out.append(r)
    return out


def types_present(recs: list[dict], include_other: bool = True) -> list[str]:
    """Weight types in TYPES order, then any other type (e.g. "embed") in first-seen order."""
    seen = []
    for r in recs:
        if r["type"] not in seen:
            seen.append(r["type"])
    ordered = [t for t in TYPES if t in seen]
    if include_other:
        ordered += [t for t in seen if t not in TYPES]
    return ordered


def by_type(recs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in recs:
        out.setdefault(r["type"], []).append(r)
    return {t: sorted(v, key=lambda r: r.get("layer", -1)) for t, v in out.items()}


def wmean(pairs) -> float | None:
    """Weighted mean of (value, weight) pairs; None values are skipped; None when nothing is left."""
    s = w = 0.0
    for v, wt in pairs:
        v = num(v)
        if v is None:
            continue
        s += v * wt
        w += wt
    return s / w if w > 0 else None


def field(rec: dict, *path):
    """val() for a (section, canonical key) path, plain nested get() otherwise (e.g. ("gain_bits",))."""
    if len(path) == 2 and path[0] in ("stat", "control", "control_seed1"):
        return val(rec, path[0], path[1])
    return get(rec, *path)


def aggregate(recs: list[dict], *path) -> float | None:
    """Numel-weighted mean over records of the value at `path` (("stat", key) with aliases, or ("gain_bits",))."""
    return wmean((field(r, *path), numel(r)) for r in recs)


def aggregate_by_type(recs: list[dict], *path) -> dict[str, float | None]:
    return {t: aggregate(rs, *path) for t, rs in by_type(recs).items()}


def per_layer(recs: list[dict], *path) -> list[tuple[int, float]]:
    """(layer, value) pairs for records with that stat, sorted by layer; several tensors in one layer (in_proj_a
    and in_proj_b) are numel-averaged."""
    acc: dict[int, list] = {}
    for r in recs:
        v = num(field(r, *path))
        if v is None or not isinstance(r.get("layer"), int):
            continue
        acc.setdefault(r["layer"], []).append((v, numel(r)))
    return [(layer, wmean(vals)) for layer, vals in sorted(acc.items())]


def control_seed_gains(recs: list[dict]) -> tuple[float | None, float | None]:
    """Numel-weighted (gain of control seed 0, gain of control seed 1) for a type's records, read from
    control.gain_bits_best_seed{0,1} (lowrank, outliers), control.seed{0,1}_best_gain (lowrank_weighted) or
    gain_bits_control / gain_bits_control_seed1 at record level; None where a seed is absent."""
    g0 = wmean(((get(r, "control", "gain_bits_best_seed0") if get(r, "control", "gain_bits_best_seed0") is not None
                 else get(r, "control", "seed0_best_gain") if get(r, "control", "seed0_best_gain") is not None
                 else r.get("gain_bits_control")), numel(r)) for r in recs)
    g1 = wmean(((get(r, "control", "gain_bits_best_seed1") if get(r, "control", "gain_bits_best_seed1") is not None
                 else get(r, "control", "seed1_best_gain") if get(r, "control", "seed1_best_gain") is not None
                 else r.get("gain_bits_control_seed1")), numel(r)) for r in recs)
    return g0, g1


def control_spread(recs: list[dict]) -> float | None:
    """|gain(control seed 0) - gain(control seed 1)| per type, None without a second seed."""
    g0, g1 = control_seed_gains(recs)
    return abs(g0 - g1) if g0 is not None and g1 is not None else None


def model_level(doc: dict | None, key: str):
    """A model-level entry, looked up under "global", "aggregate" and the top level."""
    if doc is None:
        return None
    for root in (doc.get("global"), doc.get("aggregate"), doc):
        if isinstance(root, dict) and key in root:
            return root[key]
    return None


ALLOC_FIELDS = ("saving", "overhead", "gain", "frac_floored", "saving_cap2bit", "saving_unfloored")


def allocation(doc: dict | None, granularities: list[str]) -> dict[str, dict]:
    """{granularity: {"saving", "overhead", "gain"[, "frac_floored", "saving_cap2bit", "saving_unfloored"]}} from
    global.alloc_<g>_<field> (the probes' layout) or from an "allocation" mapping; granularities missing from the
    file are left out. The floored fraction and the guarded savings travel with the headline saving, which is
    dominated by the 0-bit floor wherever frac_floored is large."""
    out: dict[str, dict] = {}
    if doc is None:
        return out
    mapping = model_level(doc, "allocation")
    for g in granularities:
        entry = {}
        for f in ALLOC_FIELDS:
            v = num(model_level(doc, f"alloc_{g}_{f}"))
            if v is None and isinstance(mapping, dict):
                v = num(get(mapping, g, f))
            if v is not None:
                entry[f] = v
        if entry:
            out[g] = entry
    return out


def record_allocation(recs: list[dict], granularities: list[str]) -> dict[str, dict]:
    """Per-record (within-tensor) allocation fields alloc_<g>_<field>, numel-averaged over the records."""
    out: dict[str, dict] = {}
    for g in granularities:
        entry = {f: aggregate(recs, "stat", f"alloc_{g}_{f}") for f in ALLOC_FIELDS}
        entry = {f: v for f, v in entry.items() if v is not None}
        if entry:
            out[g] = entry
    return out


def jaccard(doc: dict | None, typ: str | None = None) -> tuple[float | None, float | None]:
    """(W, control) mean pairwise top-16 Jaccard of outlier input channels across layers: the mean over the
    cross-layer summary records of that type (tensor "<type>:<leaf>:cross-layer"; in_proj_ab has two, in_proj_a and
    in_proj_b), else the pooled "all-hidden-input:cross-layer" record, else an aggregate["all_hidden_input"] entry.
    The probe's aggregate cannot carry it per type (summary records have numel 0, so their numel-weighted mean is
    empty)."""
    if doc is None:
        return None, None
    if typ is not None:
        per_type = jaccard_by_type(doc)
        if typ in per_type:
            return per_type[typ]
    for r in records(doc, summaries=True):
        if r.get("type") != "all_hidden_input":
            continue
        w = num(get(r, "stat", "jaccard_mean"))
        if w is not None:
            return w, num(get(r, "control", "jaccard_mean"))
    agg = get(doc, "aggregate", "all_hidden_input")
    w = num(get(agg, "stat", "jaccard_mean"))
    return (w, num(get(agg, "control", "jaccard_mean"))) if w is not None else (None, None)


def jaccard_by_type(doc: dict | None) -> dict[str, tuple[float, float | None]]:
    """{type: (W, control)} for every type with cross-layer summary records: the plain mean over that type's records
    (one per leaf series, so in_proj_ab averages in_proj_a and in_proj_b instead of keeping the first); the control
    is None when no record of the type carries one."""
    acc: dict[str, tuple[list[float], list[float]]] = {}
    for r in records(doc, summaries=True):
        w = num(get(r, "stat", "jaccard_mean"))
        if w is None or r.get("type") in ("", "all_hidden_input", None):
            continue
        ws, cs = acc.setdefault(r["type"], ([], []))
        ws.append(w)
        c = num(get(r, "control", "jaccard_mean"))
        if c is not None:
            cs.append(c)
    return {t: (sum(ws) / len(ws), sum(cs) / len(cs) if cs else None) for t, (ws, cs) in acc.items()}


def layer_stats(doc: dict | None) -> list[dict]:
    """P7 per-layer down_proj spikes: global.layer_stats (list of {layer, in_max, out_max, ...})."""
    ls = model_level(doc, "layer_stats")
    if isinstance(ls, list):
        return [s for s in ls if isinstance(s, dict) and isinstance(s.get("layer"), int)]
    if isinstance(ls, dict):  # {layer: {...}} layout
        return [dict(v, layer=int(k)) for k, v in ls.items() if str(k).lstrip("-").isdigit() and isinstance(v, dict)]
    return []


# ---------------------------------------------------------------- parametric method names (P0)

_METHOD_RE = re.compile(r"^(?P<fam>(?:had-)?(?:rtn|gptq)-(?:sym|asym)-g(?P<group>\d+))-(?P<bits>\d+)(?P<rest>(?:[@+][a-z]+=[^@+]*)*)$")


def parse_method(name: str) -> dict | None:
    """Split a (parametric) method name. "rtn-sym-g128-4@only=down" -> {"family": "rtn-sym-g128", "group": 128,
    "bits": 4, "qual": "only", "qual_value": "down", "plus": None, "plus_value": None, "base": "rtn-sym-g128-4"}.
    "+keepcols=1" fills plus / plus_value. None for fp16, had-fp16 and anything else that does not match."""
    m = _METHOD_RE.match(name)
    if m is None:
        return None
    out = {"family": m.group("fam"), "group": int(m.group("group")), "bits": int(m.group("bits")), "qual": None,
           "qual_value": None, "plus": None, "plus_value": None, "base": f"{m.group('fam')}-{m.group('bits')}"}
    for tok in re.findall(r"[@+][a-z]+=[^@+]*", m.group("rest")):
        key, _, value = tok[1:].partition("=")
        if tok[0] == "@":
            out["qual"], out["qual_value"] = key, value
        else:
            out["plus"], out["plus_value"] = key, value
    return out


def is_full_model(name: str) -> bool:
    """True when the method quantizes every block linear (no @only / @except / @layer qualifier)."""
    return "@" not in name


SMOKE_PREFIXES = ("limit-windows", "kl-windows", "calib-n")  # the run_baselines.py off-protocol note tokens
_CALIB_RE = re.compile(r"^calib (\d+)x\d+ seed (\d+)$")
PROTOCOL_CALIB_N = 128


def smoke_tag(note: str) -> str:
    """The off-protocol markers of a row note, sorted and joined; "" for a full DESIGN.md protocol row (the same
    rule as scripts/plot_rd.py: marker tokens, or an old "calib Nx2048 seed s" token with N != 128)."""
    tags = set()
    for tok in str(note or "").split(";"):
        tok = tok.strip()
        if tok.startswith(SMOKE_PREFIXES):
            tags.add(tok)
            continue
        m = _CALIB_RE.match(tok)
        if m and int(m.group(1)) != PROTOCOL_CALIB_N:
            tags.add(f"calib-n {m.group(1)}")
    return "; ".join(sorted(tags))


def is_smoke(row: dict) -> bool:
    return smoke_tag(row.get("note", "")) != ""


HESSIAN_F32_TOKEN = "hessian-f32"  # the wcomp.quant.gptq note token of the f32-product Hessian path (DESIGN.md GPTQ speed)


def hessian_tag(note: str) -> str:
    """"hessian-f32" when the row ran GPTQ's f32-product Hessian path, else "" (the Phase 1 f64 path, whether its note
    says "hessian-f64" or, for rows written before the token existed, nothing). Part of a row's identity: the
    acceptance row must never replace the Phase 1 point of the same method name."""
    return HESSIAN_F32_TOKEN if HESSIAN_F32_TOKEN in re.split(r"[;\s]+", str(note or "")) else ""


def row_key(r: dict) -> tuple:
    """(method, seed, calib_seed, smoke tag, hessian tag): the identity under which rows deduplicate (same rule as
    scripts/plot_rd.run_key), so a smoke row never shadows a full-protocol row and an f32-Hessian GPTQ row never
    shadows the f64 one."""
    note = r.get("note", "")
    return (r["method"], r.get("seed", 0), r.get("calib_seed"), smoke_tag(note), hessian_tag(note))


_PHASE1_RE = re.compile(r"^(had-)?(rtn-(sym|asym)|gptq-sym)-g128-\d+$")


def is_phase1_name(method: str) -> bool:
    """True for the Phase 1 method names (wcomp.methods.METHODS: fp16, had-fp16, {had-}rtn-{sym,asym}-g128-b,
    {had-}gptq-sym-g128-b), whose numbers live in results/<model>/runs.jsonl."""
    return method in ("fp16", "had-fp16") or _PHASE1_RE.match(method) is not None


ROW_CONFLICT_FIELDS = ("note", "git_or_code_version", "ppl", "kl")


def merge_rows(runs_rows: list[dict], ablation_rows: list[dict]) -> list[dict]:
    """The one merge of results/<model>/runs.jsonl with results/probe/<model>/ablation.jsonl, per row_key: a Phase 1
    method name is taken from runs.jsonl whenever it has the row (ablation.jsonl copies of fp16 / joint RTN never
    replace the Phase 1 point); any other key keeps the latest timestamp. Two rows of one key that differ in note,
    code version, ppl or kl are reported on stderr, never silently resolved."""
    out: dict[tuple, dict] = {row_key(r): r for r in runs_rows}
    for r in ablation_rows:
        key = row_key(r)
        prev = out.get(key)
        if prev is None:
            out[key] = r
            continue
        if any(prev.get(k) != r.get(k) for k in ROW_CONFLICT_FIELDS):
            keep_runs = is_phase1_name(r["method"]) or r.get("timestamp", "") <= prev.get("timestamp", "")
            skip(f"{r['method']} (seed {key[1]}, calib_seed {key[2]}): runs.jsonl (ppl {prev.get('ppl')}) and ablation.jsonl "
                 f"(ppl {r.get('ppl')}) disagree; keeping {'runs.jsonl' if keep_runs else 'the later ablation.jsonl row'}")
        if not is_phase1_name(r["method"]) and r.get("timestamp", "") > prev.get("timestamp", ""):
            out[key] = r
    return list(out.values())


def load_rows(path: Path, include_smoke: bool = False) -> list[dict]:
    """Rows of a runs.jsonl-schema file, deduplicated to the latest timestamp per row_key (method, seed, calib_seed,
    smoke tag, hessian tag); missing file -> [] (reported). Smoke rows (limit-windows / kl-windows / calib-n markers
    in the note) are dropped (reported once) unless include_smoke; with or without the flag a later smoke row never
    replaces a full-protocol row, since the smoke tag is part of the key."""
    if not path.exists():
        skip(f"{path.as_posix()} (missing)")
        return []
    latest: dict[tuple, dict] = {}
    n_smoke = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skip(f"{path.as_posix()}: unparsable line")
                continue
            if not isinstance(r, dict) or "method" not in r:
                continue
            if is_smoke(r) and not include_smoke:
                n_smoke += 1
                continue
            key = row_key(r)
            prev = latest.get(key)
            if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
                latest[key] = r
    if n_smoke:
        skip(f"{path.as_posix()}: {n_smoke} smoke rows ignored (include_smoke / --include-smoke to read them)")
    return list(latest.values())


def find_row(rows: list[dict], method: str, seed: int = 0) -> dict | None:
    for r in rows:
        if r["method"] == method and r.get("seed", 0) == seed and r.get("calib_seed") in (None, 0):
            return r
    return None


def fp16_ppl(rows: list[dict]) -> float | None:
    r = find_row(rows, "fp16")
    return num(r.get("ppl")) if r else None


def ablation_only(rows: list[dict], bits: int, qual: str = "only") -> dict[str, dict]:
    """{type (or layer string): row} of the rtn-sym-g128-<bits>@<qual>=<x> rows (without a +keepcols/+keepsuper)."""
    out = {}
    for r in rows:
        p = parse_method(r["method"])
        if p and p["family"] == "rtn-sym-g128" and p["bits"] == bits and p["qual"] == qual and p["plus"] is None:
            out[p["qual_value"]] = r
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation (scipy), None with fewer than 3 pairs or a constant input."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    from scipy.stats import spearmanr

    rho = float(spearmanr(xs, ys).statistic)
    return rho if math.isfinite(rho) else None
