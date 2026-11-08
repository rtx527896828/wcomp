"""Activation-based Phase 2 probes P5, P6a, P6b, P7, P2b on one model (DESIGN.md, "Phase 2 contract").

  uv run python scripts/probe_acts.py --model qwen3.5-0.8b [--probes p5,p6a,p6b,p7,p2b] [--windows 32] [--layer-chunk 6]

Writes results/probe/<model>/{acts,sensitivity,fisher,superweight,lowrank_weighted}.json in the contract schema
(wcomp.probe.common.write_probe_json) plus a "global" key with the model-level numbers of that probe (allocation
across tensors, P7 layer spikes and reference, timing and peak GPU memory). Calibration: C4 seed 0 (`--windows`
windows), seed 2 for the P5 split half. P6b (Fisher) defaults to on for qwen3.5-0.8b only (run list). Smoke runs
(--windows != 32, --fisher-windows != 32, --fisher-tokens != 1024, --eval-windows != 16) are marked in params.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_io  # noqa: E402  (the one ablation.jsonl reader: dedup, smoke rows dropped unless asked)
from wcomp.data import c4_calibration, wikitext2_test  # noqa: E402
from wcomp.models import MODELS, load_model  # noqa: E402
from wcomp.probe import acts  # noqa: E402
from wcomp.probe.common import write_probe_json  # noqa: E402

log = logging.getLogger("probe_acts")

ALL_PROBES = ("p5", "p6a", "p6b", "p7", "p2b")
FILES = {"p5": "acts", "p6a": "sensitivity", "p6b": "fisher", "p7": "superweight", "p2b": "lowrank_weighted"}
PROTOCOL = {"windows": 32, "fisher_windows": 32, "fisher_tokens": 1024, "eval_windows": 16}


def _mem_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _mem_peak_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def _write(path: Path, model_tag: str, probe: str, params: dict, records, glob: dict | None) -> None:
    doc = write_probe_json(path, model_tag, probe, params, records)
    if glob is not None:
        doc["global"] = json.loads(json.dumps(glob, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, allow_nan=False)
    log.info("wrote %s (%d records)", path, len(records))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--probes", default=None, help="comma list of p5,p6a,p6b,p7,p2b (default: all; p6b only on qwen3.5-0.8b)")
    ap.add_argument("--windows", type=int, default=PROTOCOL["windows"], help="C4 windows x 2048 tokens per half")
    ap.add_argument("--layer-chunk", type=int, default=6)
    ap.add_argument("--fisher-windows", type=int, default=PROTOCOL["fisher_windows"])
    ap.add_argument("--fisher-tokens", type=int, default=PROTOCOL["fisher_tokens"])
    ap.add_argument("--eval-windows", type=int, default=PROTOCOL["eval_windows"], help="P7 WikiText-2 windows (= kl_windows)")
    ap.add_argument("--damp", type=float, default=0.01, help="P2b Cholesky damping (x mean diag H)")
    ap.add_argument("--out-dir", default=None, help="default results/probe/<model>")
    ap.add_argument("--include-smoke", action="store_true",
                    help="P6a prediction check: also read limit-windows / kl-windows / calib-n rows of <out-dir>/ablation.jsonl")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    if args.probes is None:
        probes = [p for p in ALL_PROBES if p != "p6b" or args.model == "qwen3.5-0.8b"]
    else:
        probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    unknown = [p for p in probes if p not in ALL_PROBES]
    if unknown:
        raise SystemExit(f"unknown probes {unknown}; known {ALL_PROBES}")
    smoke = [f"{k} {getattr(args, k)}" for k in PROTOCOL if getattr(args, k) != PROTOCOL[k]]
    # off-protocol output goes under results/smoke/probe/<model> (the smoke convention), never next to the real probe files
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / ("smoke/probe" if smoke else "probe") / args.model
    if smoke and "smoke" not in out_dir.parts:
        out_dir = out_dir.parent / "smoke" / out_dir.name
        log.warning("off-protocol run with an explicit --out-dir: writing to %s instead", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        log.info("off-protocol settings: %s", "; ".join(smoke))
    common = {"windows": args.windows, "seqlen": 2048, "calib_seed": 0, "layer_chunk": args.layer_chunk, "smoke": smoke}

    torch.manual_seed(0)
    t0 = time.time()
    model, tokenizer = load_model(args.model)
    log.info("loaded %s in %.1fs", args.model, time.time() - t0)
    need_stats = any(p in probes for p in ("p5", "p6a", "p2b"))
    calib = c4_calibration(tokenizer, args.model, n=args.windows, seed=0) if (need_stats or "p6b" in probes or "p7" in probes) else None

    stats = stats_half = None
    t_stats = 0.0
    if need_stats:
        _mem_reset()
        t1 = time.time()
        stats = acts.collect_input_stats(model, calib, full_hessian="p2b" in probes, layer_chunk=args.layer_chunk)
        if "p5" in probes:
            calib2 = c4_calibration(tokenizer, args.model, n=args.windows, seed=2)
            stats_half = acts.collect_input_stats(model, calib2, full_hessian=False, layer_chunk=args.layer_chunk)
        t_stats = time.time() - t1
        log.info("input stats: %.1fs, peak %.2f GB, full_hessian=%s", t_stats, _mem_peak_gb() or 0, "p2b" in probes)

    if "p5" in probes:
        t1 = time.time()
        recs = acts.usage_stats(stats, stats_half)
        params = dict(common, split_half_seed=2, tau_fraction=acts.TAU_FRACTION, primary_types=list(acts.USAGE_TYPES),
                      time_s=round(time.time() - t1 + t_stats, 1), peak_gb=_mem_peak_gb())
        prim = [r for r in recs if r.type in acts.USAGE_TYPES]
        glob = {"n_primary": len(prim),
                "min_split_half_spearman_primary": min((r.stat.get("spearman_split_half", 1.0) for r in prim), default=None)}
        _write(out_dir / "acts.json", args.model, "P5", params, recs, glob)

    if "p6a" in probes:
        _mem_reset()
        t1 = time.time()
        recs, glob = acts.hessian_sensitivity(model, stats)
        abl = out_dir / "ablation.jsonl"
        rows = probe_io.load_rows(abl, include_smoke=args.include_smoke) if abl.exists() else []
        glob["prediction_check"] = acts.prediction_check(recs, rows)
        glob["prediction_check"]["ablation_rows_seen"] = len(rows)
        glob["prediction_check"]["ablation_smoke_included"] = bool(args.include_smoke)
        params = dict(common, bits=acts.BITS, group=acts.GROUP, time_s=round(time.time() - t1, 1), peak_gb=_mem_peak_gb())
        _write(out_dir / "sensitivity.json", args.model, "P6a", params, recs, glob)

    if "p2b" in probes:
        _mem_reset()
        t1 = time.time()
        recs = acts.weighted_low_rank(model, stats, damp=args.damp)
        params = dict(common, damp=args.damp, control_seeds=[0, 1], rank_fractions=list(acts.RANK_FRACTIONS),
                      time_s=round(time.time() - t1, 1), peak_gb=_mem_peak_gb())
        _write(out_dir / "lowrank_weighted.json", args.model, "P2b", params, recs, None)

    if stats is not None:
        del stats, stats_half

    if "p6b" in probes:
        _mem_reset()
        t1 = time.time()
        recs, glob = acts.fisher_diag(model, calib, n_windows=args.fisher_windows, seqlen=args.fisher_tokens)
        params = dict(common, fisher_windows=args.fisher_windows, fisher_tokens=args.fisher_tokens, bits=acts.BITS,
                      time_s=round(time.time() - t1, 1), peak_gb=_mem_peak_gb())
        _write(out_dir / "fisher.json", args.model, "P6b", params, recs, glob)

    if "p7" in probes:
        _mem_reset()
        t1 = time.time()
        eval_ids = wikitext2_test(tokenizer, args.model)[: args.eval_windows]
        recs, glob = acts.super_weights(model, calib[0], eval_ids, kl_windows=args.eval_windows)
        params = dict(common, calib_window=0, eval_windows=args.eval_windows, spike_factor=glob["spike_factor"], top_k=glob["top_k"],
                      time_s=round(time.time() - t1, 1), peak_gb=_mem_peak_gb())
        _write(out_dir / "superweight.json", args.model, "P7", params, recs, glob)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
