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
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import logging
import math
import sys
import time
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
# time_method_s and env (library versions / kernel availability, which git_or_code_version does not cover).
ROW_FIELDS = (
    "model", "method", "seed", "calib_seed", "bits_blocks", "bits_all", "ppl", "kl", "top1_agree", "n_tokens",
    "kl_tokens", "time_s", "time_method_s", "note", "git_or_code_version", "env", "timestamp",
)


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
) -> dict:
    """One runs.jsonl row (keys ROW_FIELDS); calib_seed is None for methods without calibration."""
    row = {
        "model": model_tag,
        "method": method,
        "seed": seed,
        "calib_seed": calib_seed,
        "bits_blocks": account.bits_blocks(),
        "bits_all": account.bits_all(model),
        "ppl": metrics["ppl"],
        "kl": metrics["kl"],
        "top1_agree": metrics["top1_agree"],
        "n_tokens": metrics["n_tokens"],
        "kl_tokens": metrics["kl_tokens"],
        "time_s": round(time_s, 2),
        "time_method_s": round(time_method_s, 2),
        "note": "; ".join(notes),
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
        ctx = Ctx(calib_ids if needs_calib(name) else None, args.seed, tokenizer)
        account, t_method = timed(resolve(name), model, ctx)  # the method alone, not the load
        metrics = evaluate(model, ref, eval_ids, kl_windows=args.kl_windows, ref_hidden=ref_hidden)
        check_metrics(name, metrics)
        time_s = time.time() - t0
        notes = list(notes_common)
        if needs_calib(name):
            if args.calib_n != PROTOCOL_CALIB_N:
                notes.append(f"calib-n {args.calib_n}")
            notes.append(f"calib {args.calib_n}x{calib_ids.shape[1]} seed {calib_seed}")
        notes.extend(account.summary()["notes"])
        row = make_row(args.model, name, args.seed, calib_seed if needs_calib(name) else None, account, model,
                       metrics, time_s, t_method, notes, version, env)
        line = json.dumps(row, allow_nan=False)  # a NaN can never reach the file
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        log.info(
            "%s: ppl %.4g kl %s top1 %s bits_blocks %.4f bits_all %.4f time %.1fs (method %.1fs)",
            name, row["ppl"], f"{row['kl']:.5f}" if row["kl"] is not None else None,
            f"{row['top1_agree']:.4f}" if row["top1_agree"] is not None else None,
            row["bits_blocks"], row["bits_all"], time_s, t_method,
        )
        del model, account
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
