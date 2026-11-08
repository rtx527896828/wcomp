"""Run Phase 1 methods on one model and append one JSON line per run (DESIGN.md, scripts/run_baselines.py).

Example:
  uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods fp16,rtn-sym-g128-4
  uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods gptq-sym-g128-3 --seed 1
Smoke runs: --limit-windows 8 --kl-windows 4 --calib-n 16. Any of --limit-windows, --kl-windows != 32 or
--calib-n != 128 is off the DESIGN.md protocol: the run is marked in the note ("limit-windows N", "kl-windows N",
"calib-n N") and the default --results-dir becomes results/smoke, so smoke rows are never mixed with full runs
(plot_rd.py ignores them unless asked).

Memory: the reference's final hidden states on the KL windows are computed once (wcomp.eval.reference_hidden),
then the reference's decoder layers are moved to the CPU and only its lm_head stays resident, so the quantized
copy plus the f32 log-softmax chunks fit in the ~10 GB available even for the 2B models.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wcomp.bits import BitAccount  # noqa: E402
from wcomp.data import c4_calibration, wikitext2_test  # noqa: E402
from wcomp.eval import evaluate, reference_hidden  # noqa: E402
from wcomp.methods import METHODS, Ctx, needs_calib  # noqa: E402
from wcomp.models import MODELS, load_model  # noqa: E402

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


def code_hash() -> str:
    """Short sha256 over the framework sources (wcomp/**/*.py, scripts/*.py) and uv.lock: changes whenever they do."""
    h = hashlib.sha256()
    files = list(ROOT.glob("wcomp/**/*.py")) + list(ROOT.glob("scripts/*.py")) + [ROOT / "uv.lock"]
    for p in sorted(p for p in files if p.exists()):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:10]


def code_version() -> str:
    """'<git short hash>[+dirty]/<source hash>'; the source hash identifies uncommitted code too."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
        git = rev.stdout.strip() + ("+dirty" if status.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        git = "unknown"
    return f"{git}/{code_hash()}"


def env_version() -> str:
    """torch / transformers versions and whether the fused Qwen3.5 kernels (fla, causal_conv1d) are importable."""
    import transformers

    kernels = ",".join(k for k in ("fla", "causal_conv1d") if importlib.util.find_spec(k) is not None) or "none"
    return f"torch {torch.__version__}; transformers {transformers.__version__}; kernels {kernels}"


def smoke_markers(limit_windows: int | None, kl_windows: int, calib_n: int, n_windows: int | None = None) -> list[str]:
    """Note tokens for every off-protocol setting; an empty list means a full DESIGN.md protocol run.

    With `n_windows` (the evaluated window count after --limit-windows) the kl-windows marker names the KL
    window count actually used, min(kl_windows, n_windows), which is what wcomp.eval.evaluate clamps to.
    """
    markers = []
    if limit_windows is not None:
        markers.append(f"limit-windows {limit_windows}")
    if n_windows is not None:
        kl_windows = min(kl_windows, n_windows)
    if kl_windows != PROTOCOL_KL_WINDOWS:
        markers.append(f"kl-windows {kl_windows}")
    if calib_n != PROTOCOL_CALIB_N:
        markers.append(f"calib-n {calib_n}")
    return markers


def default_results_dir(markers: list[str]) -> str:
    return "results/smoke" if markers else "results"


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
    ap.add_argument("--methods", required=True, help="comma-separated names from wcomp.methods.METHODS")
    ap.add_argument("--seed", type=int, default=0, help="method seed (rotations); also the default GPTQ calibration seed")
    ap.add_argument("--calib-n", type=int, default=PROTOCOL_CALIB_N, help="GPTQ calibration windows (!= 128 is a smoke run)")
    ap.add_argument("--calib-seed", type=int, default=None, help="GPTQ calibration sampling seed (default: --seed)")
    ap.add_argument("--kl-windows", type=int, default=PROTOCOL_KL_WINDOWS, help="(!= 32 is a smoke run)")
    ap.add_argument("--limit-windows", type=int, default=None, help="evaluate only the first N windows (smoke runs)")
    ap.add_argument("--results-dir", default=None,
                    help="runs go to <results-dir>/<model>/runs.jsonl (default results, or results/smoke for a smoke run)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; known: {sorted(METHODS)}")
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

    # The markers name the settings actually used: the KL window count is clamped to the evaluated windows.
    notes_common = smoke_markers(args.limit_windows, args.kl_windows, args.calib_n, n_windows=eval_ids.shape[0])
    if notes_common:
        log.info("smoke run (%s)", "; ".join(notes_common))
    results_dir = args.results_dir or default_results_dir(notes_common)
    out_dir = (ROOT / results_dir / args.model) if not Path(results_dir).is_absolute() else Path(results_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runs.jsonl"
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
        account, t_method = timed(METHODS[name], model, ctx)  # the method alone, not the load
        metrics = evaluate(model, ref, eval_ids, kl_windows=args.kl_windows, ref_hidden=ref_hidden)
        check_metrics(name, metrics)
        time_s = time.time() - t0
        notes = list(notes_common)
        if needs_calib(name):
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
