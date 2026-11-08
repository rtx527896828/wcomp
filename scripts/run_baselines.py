"""Run Phase 1 methods on one model and append one JSON line per run (DESIGN.md, scripts/run_baselines.py).

Example:
  uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods fp16,rtn-sym-g128-4
  uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods gptq-sym-g128-3 --seed 1
Smoke runs: --limit-windows 8 --kl-windows 4 --calib-n 16. Any of --limit-windows, --kl-windows != 32 or
--calib-n != 128 is off the DESIGN.md protocol: the run is marked in the note ("limit-windows N", "kl-windows N",
"calib-n N"; the calib-n marker only on rows of methods that calibrate) and the default --results-dir becomes
results/smoke, so smoke rows are never mixed with full runs (plot_rd.py and the Phase 2 readers ignore them unless
asked). An explicit --results-dir without a "smoke" path component is redirected to <results-dir>/smoke for such a
run, so results/probe/<model>/ablation.jsonl only ever holds full-protocol rows.

--hessian-f32 selects the f32-product Hessian path of wcomp.quant.gptq (DESIGN.md "GPTQ speed") for this run; the
default stays the Phase 1 f64 path until the acceptance row is recorded in NOTES.md. A GPTQ row of the f32 path
carries "hessian-f32" in its note; the default path leaves the Phase 1 note byte-identical. A full-protocol GPTQ row
into results/probe (the Phase 2 ablation tree) is refused without an explicit --hessian-f32 / --hessian-f64, so the
acceptance row cannot be recorded by accident on the path it does not test; a smoke run aimed there is redirected to
results/probe/smoke and needs no flag (planned_results_dir decides from the CLI before any model loads).
A +keepsuper row whose P7 JSON has no co-occurring spike candidate is skipped (the run list's "if P7 finds any")
unless --keepsuper-any-kind is given. An acceptance miss inside the 4.4% seed spread is GPTQ rounding chaos in H, not f32 precision
(wcomp.quant.gptq.hessian_path_check, scripts/hessian_paths.py measure that directly).

Memory: the reference's final hidden states on the KL windows are computed once (wcomp.eval.reference_hidden),
then the reference's decoder layers are moved to the CPU and only its lm_head stays resident, so the quantized
copy plus the f32 log-softmax chunks fit in the ~10 GB available even for the 2B models.

Phase 3 (DESIGN.md "Phase 3 contract"; additive, every Phase 1 / 2 name and path produces the same row values):
  uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods gptq32-sym-g128-3 --results-dir results/phase3
The "gptq32-*" names and every other Phase 3 name select the f32 per-window Hessian product by name, so they need
no --hessian-f32 / --hessian-f64 (the flags keep their Phase 1 / 2 meaning for the "gptq-sym-g128-*" names and
are not consulted by the new names). Ctx.tag is set to the model tag so allocation methods can read
results/phase3/<model>/damage_c4.jsonl, fisher.json and combo.json at call time. After a method runs, if any
TensorBits of its account carries `indices`, wcomp.coding.coded_bits(account) gives bits_blocks_huff /
bits_blocks_shannon (the fixed figure bits_blocks is untouched); "alloc" and "rotation" are read from the account
notes (alloc_of / rotation_of). wcomp.rdbound.rd_bound_rows(account=, model=, ref=, row=) rows (E4, per tensor)
are appended to <results-dir>/<model>/rd_bound.jsonl. Both modules are imported lazily: when one is not there the
row carries None in its fields (bits_blocks_huff / bits_blocks_shannon) or nothing is appended (rd_bound) and the
log says so, so the runner never blocks on a module that another builder has not landed yet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import logging
import math
import re
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wcomp.bits import BitAccount  # noqa: E402
from wcomp.data import c4_calibration, wikitext2_test  # noqa: E402
from wcomp.eval import evaluate, reference_hidden  # noqa: E402
from wcomp.methods import METHODS, Ctx, needs_calib, resolve, split_name  # noqa: E402
from wcomp.models import MODELS, load_model  # noqa: E402
from wcomp.quant.rtn import super_candidate_kinds  # noqa: E402
from wcomp.version import code_hash, code_version  # noqa: E402,F401  (shared with scripts/probe_weights.py)

log = logging.getLogger("run_baselines")

# DESIGN.md protocol defaults; anything else is a smoke run.
PROTOCOL_CALIB_N = 128
PROTOCOL_KL_WINDOWS = 32

# Fields of one runs.jsonl row (DESIGN.md scripts/run_baselines.py) plus the extensions calib_seed,
# time_method_s and env (library versions / kernel availability, which git_or_code_version does not cover), and
# the Phase 3 fields (DESIGN.md "Phase 3 contract", seeds / output): bits_blocks_huff and bits_blocks_shannon
# (wcomp.coding on the quantizers' index streams; None when no tensor carries indices or the module is absent),
# alloc (the per-unit bit allocation as a short string, or the path of the alloc JSON; None for uniform methods)
# and rotation ("hadamard" / "hadsign" / "rnd" / "eig" / "none", read from the account notes).
ROW_FIELDS = (
    "model", "method", "seed", "calib_seed", "bits_blocks", "bits_all", "bits_blocks_huff", "bits_blocks_shannon",
    "bits_blocks_entropy0", "ppl", "kl", "top1_agree", "n_tokens", "kl_tokens", "time_s", "time_method_s", "note", "alloc", "rotation",
    "git_or_code_version", "env", "timestamp",
)
# bits_blocks_entropy0 (additive): the plain order-0 entropy figure of wcomp.coding (aux + n * H1 per tensor), the
# contract's "Shannon figure" definition, next to bits_blocks_shannon (the chosen coder's bound; NOTES.md "Decisions").

ROTATIONS = ("hadsign", "eig", "hadamard", "rnd")  # note keywords in precedence order; "none" when no note matches
_ROTATION_WORDS = {"hadsign": ("hadsign",), "eig": ("eig", "eigen", "eigenbasis", "eigenvector", "eigenvectors"),
                   "hadamard": ("hadamard",), "rnd": ("rnd", "random_orthogonal")}
_ROTATION_TOKEN = re.compile(r"(?:^|[\s;])rotation=(hadsign|eig|hadamard|rnd)\b")  # the fixed token of wcomp.methods.rotated
_ALLOC_RE = re.compile(r"(?:^|\s)(?:(?:row)?alloc\b|combo-)")  # alloc-*, rowalloc-* and combo-* note clauses carry the widths


def env_version() -> str:
    """torch / transformers versions and whether the fused Qwen3.5 kernels (fla, causal_conv1d) are importable."""
    import transformers

    kernels = ",".join(k for k in ("fla", "causal_conv1d") if importlib.util.find_spec(k) is not None) or "none"
    return f"torch {torch.__version__}; transformers {transformers.__version__}; kernels {kernels}"


def smoke_markers(limit_windows: int | None, kl_windows: int, calib_n: int, n_windows: int | None = None,
                  calib_used: bool = True) -> list[str]:
    """Note tokens for every off-protocol setting; an empty list means a full DESIGN.md protocol run.

    With `n_windows` (the evaluated window count after --limit-windows) the kl-windows marker names the KL
    window count actually used, min(kl_windows, n_windows), which is what wcomp.eval.evaluate clamps to.
    `calib_used=False` (no method of the run calibrates) leaves the calib-n marker out: a reduced calibration set
    is not a property of an RTN row.
    """
    markers = []
    if limit_windows is not None:
        markers.append(f"limit-windows {limit_windows}")
    if n_windows is not None:
        kl_windows = min(kl_windows, n_windows)
    if kl_windows != PROTOCOL_KL_WINDOWS:
        markers.append(f"kl-windows {kl_windows}")
    if calib_n != PROTOCOL_CALIB_N and calib_used:
        markers.append(f"calib-n {calib_n}")
    return markers


def default_results_dir(markers: list[str]) -> str:
    return "results/smoke" if markers else "results"


def smoke_results_dir(results_dir: str, markers: list[str]) -> str:
    """An explicit --results-dir for a smoke run (non-empty markers) is redirected to <results-dir>/smoke unless a
    path component already says "smoke"; a full-protocol run keeps it."""
    if not markers or "smoke" in Path(results_dir).parts:
        return results_dir
    return str(Path(results_dir) / "smoke")


def hessian_choice_required(methods: list[str], results_dir: str | None) -> bool:
    """True when a calibrated (GPTQ) method would write a full-protocol row into the Phase 2 tree (a --results-dir with
    a "probe" and no "smoke" path component): the Hessian path must then be chosen explicitly, since the default f64
    path reproduces the Phase 1 number and the pre-registered acceptance row only tests the f32 path."""
    if results_dir is None or not any(needs_calib(m) for m in methods):
        return False
    parts = Path(results_dir).parts
    return "probe" in parts and "smoke" not in parts


def planned_results_dir(results_dir: str | None, methods: list[str], limit_windows: int | None, kl_windows: int,
                        calib_n: int) -> str | None:
    """The results dir a run will write to, decided from the CLI alone (before any model loads): None keeps the default
    rule (default_results_dir), otherwise smoke_results_dir with the markers the CLI implies. The later kl-windows clamp
    to the evaluated window count can only add a marker (and only under --limit-windows, itself a marker), so a smoke run
    is never taken for a full-protocol one here."""
    if results_dir is None:
        return None
    markers = smoke_markers(limit_windows, kl_windows, calib_n, calib_used=any(needs_calib(m) for m in methods))
    return smoke_results_dir(results_dir, markers)


def skip_keepsuper(name: str, any_kind: bool = False) -> bool:
    """The run list's "+keepsuper (if P7 finds any)": a +keepsuper row whose P7 JSON lists no co-occurring spike
    candidate is skipped (logged, no row written) unless any_kind. Such a row would restore only global top-|w| weights,
    each by construction its 128-group amax, which the unsigned sym grid clips to 7/8 (4 bit) when positive and maps
    exactly when negative: an un-clipping of a handful of the ~50 % positive-amax groups, not a super-weight test."""
    _, suffix = split_name(name)
    if "+keepsuper=" not in suffix:
        return False
    path = suffix.partition("+keepsuper=")[2]
    try:
        kinds = super_candidate_kinds(path)
    except (OSError, ValueError, KeyError, TypeError) as e:  # missing, unreadable or malformed P7 JSON: refuse before any model loads
        raise SystemExit(f"{name}: cannot read the P7 JSON {path!r} ({type(e).__name__}: {e}); run probe_acts first or fix the path") from e
    if any("spike" in k.split("+") for k in kinds):
        return False
    if any_kind:
        log.warning("%s: the P7 JSON has no co-occurring spike candidate; running anyway (--keepsuper-any-kind): the row restores %s "
                    "only, i.e. it undoes the sym grid's positive-amax clip on those groups, and is not a super-weight test; "
                    "the row note records the kinds", name, kinds or "nothing")
        return False
    log.warning("%s: skipped, the P7 JSON has no co-occurring spike candidate (run list: '+keepsuper if P7 finds any'); it lists %s, "
                "whose restore would only undo the sym grid's positive-amax clip (--keepsuper-any-kind runs it regardless)",
                name, kinds or "nothing")
    return True


def timed(fn, *args):
    """Call fn(*args) and return (result, wall seconds of that call alone), synchronizing CUDA first and after."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.time() - t0


def assert_disjoint(model: torch.nn.Module, ref: torch.nn.Module) -> None:
    """The quantized model must not share any parameter storage with the reference copy.

    Storage is identified by (device, data_ptr): raw pointers of different devices live in different address
    spaces (the reference's layers are on the CPU while the quantized copy is on the GPU).
    """
    if model is ref:
        raise AssertionError("quantized model is the reference object")
    ref_ptrs = {(str(p.device), p.data_ptr()) for p in ref.parameters()}
    shared = [n for n, p in model.named_parameters() if (str(p.device), p.data_ptr()) in ref_ptrs]
    if shared:
        raise AssertionError(f"{len(shared)} parameters share storage with the reference, e.g. {shared[0]}")


def check_metrics(name: str, metrics: dict) -> None:
    """Refuse to record a run whose ppl or kl is not finite (a NaN would otherwise become a 'result')."""
    bad = [k for k in ("ppl", "kl", "top1_agree") if metrics[k] is not None and not math.isfinite(metrics[k])]
    if bad:
        raise RuntimeError(f"{name}: non-finite metrics {bad}: {metrics}")


def check_free_memory(need_bytes: int) -> None:
    """Raise before loading a model copy that cannot fit in the free GPU memory."""
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    log.info("gpu memory: %.2f GB free of %.2f GB, loading %.2f GB", free / 1e9, total / 1e9, need_bytes / 1e9)
    if need_bytes > free:
        raise RuntimeError(f"need {need_bytes / 1e9:.2f} GB for the model copy but only {free / 1e9:.2f} GB are free")


def rotation_of(notes: list[str]) -> str:
    """"hadsign" / "eig" / "hadamard" / "rnd" / "none" from the account notes (the rotation methods write
    "input rotation hadamard(1024,2048,3584)", "... hadsign(...)", "... random_orthogonal(...)" / "rnd", "eig ...").
    hadsign and eig take precedence over the plain words they contain or accompany; a Phase 1 had-* row whose widths
    without a Hadamard fell back to random_orthogonal is "hadamard" (the primary rotation). A fixed "rotation=<kind>"
    token (wcomp.methods.rotated writes one) wins over the word matching, so a note rewording cannot change the field;
    the words are matched exactly ("eight" is not an eig rotation)."""
    text = " ".join(notes).lower()
    m = _ROTATION_TOKEN.search(text)
    if m:
        return m.group(1)
    words = set(re.findall(r"[a-z_]+", text))
    for kind in ROTATIONS:
        if any(w in words for w in _ROTATION_WORDS[kind]):
            return kind
    return "none"


def alloc_of(notes: list[str], out_dir: Path | None = None, method: str = "", seed: int = 0) -> str | None:
    """The allocation summary of a row: the note tokens that start with "alloc" / "rowalloc" / "combo-" (the allocation
    methods write e.g. "alloc-type-gptq32-avg3: q=5,k=6,..." ) joined with " | ", else the path of the alloc JSON
    (<out_dir>/alloc_<method>_seed<seed>.json) when that file exists, else None (uniform methods)."""
    tokens = []
    for note in notes:
        for tok in note.split(";"):
            tok = tok.strip()
            if _ALLOC_RE.match(tok) and tok not in tokens:
                tokens.append(tok)
    if tokens:
        return " | ".join(tokens)
    if out_dir is not None and method:
        path = out_dir / f"alloc_{method}_seed{seed}.json"
        if path.exists():
            return path.as_posix()
    return None


def has_indices(account: BitAccount) -> bool:
    return any(getattr(t, "indices", None) is not None for t in account.tensors)


CODING_FAILED = "coding-failed"  # note token of a row whose coded figures could not be computed (the row is still written)


def coded_detail(account: BitAccount) -> list[dict] | None:
    """wcomp.coding.coded_bits_detail(account) computed once per row and shared by coded_figures and the rd_bound rows
    (one Huffman pass over the index streams, not two); None (logged) when the module or the call fails."""
    if not has_indices(account):
        return None
    try:
        from wcomp.coding import coded_bits_detail

        return coded_bits_detail(account)
    except Exception:  # noqa: BLE001  (a bug here must not cost the quantized, evaluated row)
        log.warning("coded_bits_detail failed; the coded figures are computed from scratch or left None\n%s", traceback.format_exc())
        return None


def coded_figures(account: BitAccount, detail: list[dict] | None = None) -> tuple[float | None, float | None, float | None]:
    """(bits_blocks_huff, bits_blocks_shannon, bits_blocks_entropy0) from wcomp.coding.coded_bits(account) when any
    tensor carries an index stream, else (None, None, None). coded_bits may return a dict with those keys (or "huff" /
    "shannon") or a (huff, shannon) pair. A missing wcomp.coding, or any exception inside it, is logged with its
    traceback and gives (None, None, None), never an error: the quantized, evaluated row is written with None figures
    and the note token CODING_FAILED (main) instead of being lost. `detail` = coded_detail(account) when available."""
    if not has_indices(account):
        return None, None, None
    try:
        from wcomp.coding import coded_bits

        out = coded_bits(account, detail) if detail is not None else coded_bits(account)
        if isinstance(out, dict):
            huff = out.get("bits_blocks_huff", out.get("huff"))
            shannon = out.get("bits_blocks_shannon", out.get("shannon"))
            entropy0 = out.get("bits_blocks_entropy0")
        else:
            huff, shannon = out[0], out[1]
            entropy0 = out[2] if len(out) > 2 else None
    except Exception:  # noqa: BLE001
        log.warning("wcomp.coding failed: bits_blocks_huff / bits_blocks_shannon left None, the row is kept\n%s", traceback.format_exc())
        return None, None, None
    return tuple(float(x) if x is not None else None for x in (huff, shannon, entropy0))


def rd_bound_rows_for(account: BitAccount, model: torch.nn.Module, ref: torch.nn.Module, row: dict,
                      detail: list[dict] | None = None) -> list[dict] | None:
    """wcomp.rdbound.rd_bound_rows(account=, model=, ref=, row=) (E4: per weight type, the group-normalized MSE D against
    the Gaussian bound R(D) at the row's three bit figures, rows prefixed with the run's identifying fields); None
    (logged with the traceback) when the module is not there or fails, so a bug in it never aborts the run list.
    `detail` (coded_detail) is passed on when given, so the index streams are coded once per row."""
    try:
        from wcomp.rdbound import rd_bound_rows

        kwargs = {"detail": detail} if detail is not None else {}
        rows = rd_bound_rows(account=account, model=model, ref=ref, row=row, **kwargs)
    except Exception:  # noqa: BLE001
        log.warning("wcomp.rdbound failed: no rd_bound.jsonl rows for %s\n%s", row.get("method"), traceback.format_exc())
        return None
    return list(rows) if rows is not None else None


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False) + "\n")


def make_row(
    model_tag: str,
    method: str,
    seed: int,
    calib_seed: int | None,
    account: BitAccount,
    model: torch.nn.Module,
    metrics: dict,
    time_s: float,
    time_method_s: float,
    notes: list[str],
    version: str,
    env: str,
    coded: tuple = (None, None, None),
    alloc: str | None = None,
    rotation: str | None = None,
) -> dict:
    """One runs.jsonl row (keys ROW_FIELDS); calib_seed is None for methods without calibration. `coded` is
    (bits_blocks_huff, bits_blocks_shannon[, bits_blocks_entropy0]); rotation defaults to rotation_of(notes)."""
    row = {
        "model": model_tag,
        "method": method,
        "seed": seed,
        "calib_seed": calib_seed,
        "bits_blocks": account.bits_blocks(),
        "bits_all": account.bits_all(model),
        "bits_blocks_huff": coded[0],
        "bits_blocks_shannon": coded[1],
        "bits_blocks_entropy0": coded[2] if len(coded) > 2 else None,
        "ppl": metrics["ppl"],
        "kl": metrics["kl"],
        "top1_agree": metrics["top1_agree"],
        "n_tokens": metrics["n_tokens"],
        "kl_tokens": metrics["kl_tokens"],
        "time_s": round(time_s, 2),
        "time_method_s": round(time_method_s, 2),
        "note": "; ".join(notes),
        "alloc": alloc,
        "rotation": rotation_of(notes) if rotation is None else rotation,
        "git_or_code_version": version,
        "env": env,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    assert tuple(row) == ROW_FIELDS
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--methods", required=True,
                    help="comma-separated names: wcomp.methods.METHODS or parametric P0 names (wcomp.methods.resolve)")
    ap.add_argument("--seed", type=int, default=0, help="method seed (rotations); also the default GPTQ calibration seed")
    ap.add_argument("--calib-n", type=int, default=PROTOCOL_CALIB_N, help="GPTQ calibration windows (!= 128 is a smoke run)")
    ap.add_argument("--calib-seed", type=int, default=None, help="GPTQ calibration sampling seed (default: --seed)")
    ap.add_argument("--kl-windows", type=int, default=PROTOCOL_KL_WINDOWS, help="(!= 32 is a smoke run)")
    ap.add_argument("--limit-windows", type=int, default=None, help="evaluate only the first N windows (smoke runs)")
    ap.add_argument("--results-dir", default=None,
                    help="runs go to <results-dir>/<model>/<out-name> (default results, or results/smoke for a smoke run)")
    ap.add_argument("--out-name", default="runs.jsonl",
                    help="output file name under <results-dir>/<model>/ (Phase 2 ablation rows: ablation.jsonl)")
    ap.add_argument("--hessian-f32", action="store_true",
                    help="GPTQ: f32 per-window x^T x into the f64 Hessian (DESIGN.md GPTQ speed); default is the Phase 1 f64 path")
    ap.add_argument("--hessian-f64", action="store_true",
                    help="GPTQ: the Phase 1 f64 Hessian path, stated explicitly (required, as is --hessian-f32, for GPTQ rows into results/probe)")
    ap.add_argument("--keepsuper-any-kind", action="store_true",
                    help="run a +keepsuper row even when its P7 JSON has no co-occurring spike candidate (default: the row is skipped)")
    args = ap.parse_args()
    if args.hessian_f32 and args.hessian_f64:
        raise SystemExit("--hessian-f32 and --hessian-f64 exclude each other")
    if args.hessian_f32 or args.hessian_f64:
        import wcomp.quant.gptq as gptq_mod

        gptq_mod.HESSIAN_F32 = bool(args.hessian_f32)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = []
    for m in methods:
        try:
            resolve(m)
        except ValueError:
            unknown.append(m)
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; known: {sorted(METHODS)} or a parametric P0 name (wcomp.methods.resolve)")
    planned_dir = planned_results_dir(args.results_dir, methods, args.limit_windows, args.kl_windows, args.calib_n)
    if hessian_choice_required(methods, planned_dir) and not (args.hessian_f32 or args.hessian_f64):
        raise SystemExit("a GPTQ row into the Phase 2 tree (results/probe) needs an explicit Hessian path: --hessian-f32 (the "
                         "DESIGN.md 'GPTQ speed' acceptance run, gptq-sym-g128-3 seed 0) or --hessian-f64 (the Phase 1 path, which "
                         "reproduces the Phase 1 number by construction and tests nothing); an f32 row's note carries 'hessian-f32', "
                         "the f64 path leaves the Phase 1 note unchanged")
    methods = [m for m in methods if not skip_keepsuper(m, args.keepsuper_any_kind)]
    if not methods:
        log.warning("nothing left to run")
        return
    calib_seed = args.seed if args.calib_seed is None else args.calib_seed
    torch.manual_seed(args.seed)
    version = code_version()
    env = env_version()

    log.info("loading reference %s", args.model)
    ref, tokenizer = load_model(args.model)
    model_bytes = sum(p.numel() * p.element_size() for p in ref.parameters())
    eval_ids = wikitext2_test(tokenizer, args.model)
    if args.limit_windows is not None:
        eval_ids = eval_ids[: args.limit_windows]
    log.info("eval windows %s", tuple(eval_ids.shape))

    # The markers name the settings actually used: the KL window count is clamped to the evaluated windows; the
    # calib-n marker goes only on the rows of methods that calibrate (added per method below).
    any_calib = any(needs_calib(m) for m in methods)
    run_markers = smoke_markers(args.limit_windows, args.kl_windows, args.calib_n, n_windows=eval_ids.shape[0], calib_used=any_calib)
    notes_common = smoke_markers(args.limit_windows, args.kl_windows, args.calib_n, n_windows=eval_ids.shape[0], calib_used=False)
    if run_markers:
        log.info("smoke run (%s)", "; ".join(run_markers))
    if args.results_dir is None:
        results_dir = default_results_dir(run_markers)
    else:
        results_dir = smoke_results_dir(args.results_dir, run_markers)
        if results_dir != args.results_dir:
            log.warning("off-protocol run with an explicit --results-dir %s: rows go to %s instead", args.results_dir, results_dir)
    out_dir = (ROOT / results_dir / args.model) if not Path(results_dir).is_absolute() else Path(results_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    t0 = time.time()
    ref_hidden = reference_hidden(ref, eval_ids, args.kl_windows)
    ref.model.layers.to("cpu")  # embed_tokens / norm / lm_head stay on the GPU for the KL logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("reference hidden states %s in %.1fs; reference layers moved to the CPU", tuple(ref_hidden.shape), time.time() - t0)
    calib_ids = None
    if any(needs_calib(m) for m in methods):
        calib_ids = c4_calibration(tokenizer, args.model, n=args.calib_n, seed=calib_seed)
        log.info("calibration windows %s (seed %d)", tuple(calib_ids.shape), calib_seed)

    for name in methods:
        log.info("=== %s ===", name)
        # Fresh global RNG state per method (DESIGN.md 7): a method's draws must not depend on what ran before it.
        torch.manual_seed(args.seed)
        t0 = time.time()
        check_free_memory(model_bytes)
        model, _ = load_model(args.model)
        assert_disjoint(model, ref)
        # tag: the Phase 3 allocation methods read their inputs by it; out_dir: they write alloc_<method>_seed<s>.json there
        ctx = Ctx(calib_ids if needs_calib(name) else None, args.seed, tokenizer, tag=args.model, out_dir=out_dir)
        account, t_method = timed(resolve(name), model, ctx)  # the method alone, not the load
        metrics = evaluate(model, ref, eval_ids, kl_windows=args.kl_windows, ref_hidden=ref_hidden)
        check_metrics(name, metrics)
        detail = coded_detail(account)  # one Huffman pass per row, shared with the rd_bound rows
        coded = coded_figures(account, detail)  # (None, None, None) unless a quantizer attached index streams
        time_s = time.time() - t0
        notes = list(notes_common)
        if needs_calib(name):
            if args.calib_n != PROTOCOL_CALIB_N:
                notes.append(f"calib-n {args.calib_n}")
            notes.append(f"calib {args.calib_n}x{calib_ids.shape[1]} seed {calib_seed}")
        if has_indices(account) and coded[0] is None:
            notes.append(CODING_FAILED)  # the figures are missing because wcomp.coding failed, not because the row has no indices
        notes.extend(account.summary()["notes"])
        row = make_row(args.model, name, args.seed, calib_seed if needs_calib(name) else None, account, model,
                       metrics, time_s, t_method, notes, version, env, coded=coded,
                       alloc=alloc_of(notes, out_dir, name, args.seed))
        line = json.dumps(row, allow_nan=False)  # a NaN can never reach the file
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        log.info(
            "%s: ppl %.4g kl %s top1 %s bits_blocks %.4f bits_all %.4f huff %s shannon %s rotation %s time %.1fs (method %.1fs)",
            name, row["ppl"], f"{row['kl']:.5f}" if row["kl"] is not None else None,
            f"{row['top1_agree']:.4f}" if row["top1_agree"] is not None else None,
            row["bits_blocks"], row["bits_all"],
            f"{row['bits_blocks_huff']:.4f}" if row["bits_blocks_huff"] is not None else None,
            f"{row['bits_blocks_shannon']:.4f}" if row["bits_blocks_shannon"] is not None else None,
            row["rotation"], time_s, t_method,
        )
        if has_indices(account):
            rd_rows = rd_bound_rows_for(account, model, ref, row, detail)
            if rd_rows:
                append_jsonl(out_dir / "rd_bound.jsonl", rd_rows)
                log.info("%s: %d rd_bound rows appended to %s", name, len(rd_rows), out_dir / "rd_bound.jsonl")
        del model, account
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
