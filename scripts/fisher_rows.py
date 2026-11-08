"""Fisher row (and column) sums per block linear (DESIGN.md Phase 3, method G input): the P6b position-wise empirical
Fisher F = mean_t (d_t^2)(x_t^2)^T summed over each row, r_i = sum_j F_ij = mean_t d_ti^2 ||x_t||^2, accumulated directly
from the hooks (an [N] accumulator per linear instead of the [N, K] matrix of wcomp.probe.acts.fisher_accumulate, so the
2B fits too). Same measurement protocol as P6b: C4 calibration windows (seed 0), 32 windows x 1024 tokens, bf16 forward,
f32 accumulators, summed-NLL backward through a leaf on the embedding output, no weight gradient formed, nothing written
back (parameter checksum asserted). Measurement, not training.

  uv run python scripts/fisher_rows.py --model qwen3.5-0.8b        -> results/probe/<model>/fisher_rows.pt

The file holds {"rows": {tensor: f32 [N]}, "cols": {tensor: f32 [K]}, "totals": {tensor: float}, "n_windows", "seqlen",
"n_tokens", "calib_seed", "model", "git_or_code_version", "timestamp"}; when results/probe/<model>/fisher.json exists the
per-tensor totals are compared with its "total" and the largest relative difference is logged (the two paths differ only by
summation order). --windows / --tokens != 32 / 1024 write to results/probe/smoke/<model>/ instead, except --tokens 512, the
contract's setting for the 2B's per-tensor Fisher totals (DESIGN.md A2; the file records "seqlen"). A model without a
fisher.json (the 2B, MiniCPM) gets its alloc-tensor-* totals from this file (wcomp.alloc.load_fisher_totals).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wcomp.data import c4_calibration  # noqa: E402
from wcomp.models import MODELS, block_linears, load_model  # noqa: E402
from wcomp.probe.acts import param_checksum  # noqa: E402
from wcomp.version import code_version  # noqa: E402

log = logging.getLogger("fisher_rows")

PROTOCOL = {"windows": 32, "tokens": 1024}
TOKENS_2B = 512  # DESIGN.md A2: the 2B's per-tensor Fisher totals use 512-token windows (memory); not a smoke run


def fisher_row_col_sums(model: torch.nn.Module, calib_ids: torch.Tensor, n_windows: int, seqlen: int) -> tuple[dict, dict, int]:
    """({tensor: row sums [N]}, {tensor: column sums [K]}, n_tokens) of the position-wise empirical Fisher, f32 on the CPU.

    For y = W x with input x_t [K] and output gradient d_t [N] of the window's summed NLL: row sum_i = mean_t d_ti^2 ||x_t||^2,
    column sum_j = mean_t ||d_t||^2 x_tj^2 (both exact marginals of F = mean_t (d_t^2)(x_t^2)^T)."""
    dev = next(model.parameters()).device
    linears = block_linears(model)
    before = param_checksum(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rows = {n: torch.zeros(lin.weight.shape[0], dtype=torch.float32, device=dev) for n, lin in linears}
    cols = {n: torch.zeros(lin.weight.shape[1], dtype=torch.float32, device=dev) for n, lin in linears}

    def make_fwd(name: str):
        def fwd(mod, args, out):
            x = args[0]

            def on_grad(d: torch.Tensor) -> None:
                xx = x.reshape(-1, x.shape[-1]).float().square()  # [T, K]
                dd = d.reshape(-1, d.shape[-1]).float().square()  # [T, N]
                rows[name].add_(dd.T @ xx.sum(1))
                cols[name].add_(dd.sum(1) @ xx)

            if out.requires_grad:
                out.register_hook(on_grad)

        return fwd

    def leaf(mod, args, out):
        return out.detach().requires_grad_(True)

    handles = [lin.register_forward_hook(make_fwd(n)) for n, lin in linears]
    embed = model.get_input_embeddings()
    handles.append(embed.register_forward_hook(leaf))
    n_tokens = 0
    n = min(n_windows, calib_ids.shape[0])
    try:
        for w in range(n):
            ids = calib_ids[w : w + 1, :seqlen].to(dev)
            with torch.enable_grad():
                h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
                logits = model.lm_head(h[:-1]).float()
                loss = F.cross_entropy(logits, ids[0, 1:], reduction="sum")
                loss.backward()
            n_tokens += int(ids.shape[1]) - 1
            del h, logits, loss
    finally:
        for hd in handles:
            hd.remove()
        for p in model.parameters():
            p.requires_grad_(False)
            p.grad = None
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if param_checksum(model) != before:
        raise AssertionError("fisher_row_col_sums changed a parameter")
    rows = {k: (v / max(n_tokens, 1)).cpu() for k, v in rows.items()}
    cols = {k: (v / max(n_tokens, 1)).cpu() for k, v in cols.items()}
    return rows, cols, n_tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--windows", type=int, default=PROTOCOL["windows"])
    ap.add_argument("--tokens", type=int, default=PROTOCOL["tokens"])
    ap.add_argument("--calib-seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    smoke = args.windows != PROTOCOL["windows"] or args.tokens not in (PROTOCOL["tokens"], TOKENS_2B)
    out_dir = ROOT / "results" / "probe" / ("smoke/" if smoke else "") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args.model)
    calib = c4_calibration(tok, args.model, n=max(32, args.windows), seed=args.calib_seed)
    t0 = time.time()
    rows, cols, n_tokens = fisher_row_col_sums(model, calib, args.windows, args.tokens)
    log.info("%d linears, %d tokens, %.1fs, peak %.2f GB", len(rows), n_tokens, time.time() - t0,
             torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
    totals = {k: float(v.sum()) for k, v in rows.items()}
    for k, v in cols.items():
        if abs(float(v.sum()) - totals[k]) > 1e-3 * max(totals[k], 1e-30):
            raise AssertionError(f"{k}: row total {totals[k]} != column total {float(v.sum())}")

    fisher_json = ROOT / "results" / "probe" / args.model / "fisher.json"
    if fisher_json.exists() and not smoke:
        ref = {r["tensor"]: float(r["stat"]["total"]) for r in json.load(open(fisher_json, encoding="utf-8"))["records"]}
        rel = [abs(totals[k] - ref[k]) / max(ref[k], 1e-30) for k in totals if k in ref]
        log.info("vs fisher.json totals: %d tensors, max relative difference %.2e, median %.2e", len(rel), max(rel), sorted(rel)[len(rel) // 2])

    out = {"rows": rows, "cols": cols, "totals": totals, "n_windows": min(args.windows, calib.shape[0]), "seqlen": args.tokens,
           "n_tokens": n_tokens, "calib_seed": args.calib_seed, "model": args.model, "git_or_code_version": code_version(),
           "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "definition": "position-wise empirical Fisher F = mean_t (d_t^2)(x_t^2)^T (P6b), rows = F.sum(1), cols = F.sum(0)"}
    path = out_dir / "fisher_rows.pt"
    torch.save(out, path)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
