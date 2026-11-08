"""Objective-level comparison of the two GPTQ Hessian paths (DESIGN.md "GPTQ speed"), read next to the acceptance row.

  uv run python scripts/hessian_paths.py --model qwen3.5-0.8b [--layer 0] [--calib-n 128] [--bits 3,4]
  -> results/<model>/hessian_paths.json

For every nn.Linear of one decoder layer: the f32-product and the f64 Hessian of the same C4 windows, their relative
Frobenius / diagonal difference, the fraction of weights GPTQ quantizes differently from the two, and the ratio of the
GPTQ objectives tr(dW H dW^T) (wcomp.quant.gptq.hessian_path_check). This is what the PPL / KL acceptance rule
cannot separate: frac_changed > 0 with objective_ratio ~ 1 is rounding chaos in H, not lost precision. Reads C4 only
(calibration), never WikiText-2; the model is left untouched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wcomp.data import c4_calibration  # noqa: E402
from wcomp.models import MODELS, load_model  # noqa: E402
from wcomp.quant.gptq import _capture_layer_inputs, _run_layer, accumulate_hessian_paths, hessian_path_check  # noqa: E402
from wcomp.version import code_version  # noqa: E402

log = logging.getLogger("hessian_paths")


@torch.no_grad()
def layer_hessians(model: nn.Module, calib_ids: torch.Tensor, layer: int) -> dict[str, tuple[torch.Tensor, torch.Tensor, list[int]]]:
    """(H32 sum, H64 sum, [rows]) of every nn.Linear of decoder layer `layer` over the calibration windows, both
    accumulation paths, formed inside the pre-hooks (two K x K f64 sums per linear on the GPU; the inputs themselves
    are never stored, which for the 0.8B's layer 0 and 128 windows would have been ~7 GB of host RAM)."""
    layers = model.model.layers
    inps, kwargs = _capture_layer_inputs(model, layers, calib_ids, torch.device("cpu"))
    device = next(model.parameters()).device
    for i in range(layer):
        for j in range(len(inps)):
            inps[j] = _run_layer(layers[i], inps[j].to(device), kwargs[i]).cpu()
    linears = [(n, m) for n, m in layers[layer].named_modules() if isinstance(m, nn.Linear)]
    acc: dict[str, tuple[torch.Tensor, torch.Tensor, list[int]] | None] = {n: None for n, _ in linears}

    def make_hook(name: str):
        def hook(mod, args):
            acc[name] = accumulate_hessian_paths(acc[name], args[0])

        return hook

    handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in linears]
    try:
        for x in inps:
            _run_layer(layers[layer], x.to(device), kwargs[layer])
    finally:
        for h in handles:
            h.remove()
    return {n: a for n, a in acc.items() if a is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--bits", default="3,4")
    ap.add_argument("--out", default=None, help="default results/<model>/hessian_paths.json (results/smoke/... when --calib-n != 128)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    bits_list = [int(b) for b in args.bits.split(",") if b.strip()]

    model, tokenizer = load_model(args.model)
    calib = c4_calibration(tokenizer, args.model, n=args.calib_n, seed=args.calib_seed)
    hs = layer_hessians(model, calib, args.layer)
    linears = dict((n, m) for n, m in model.model.layers[args.layer].named_modules() if isinstance(m, nn.Linear))
    results = []
    for name, acc in hs.items():
        W = linears[name].weight.detach()
        for b in bits_list:
            r = hessian_path_check(W, None, b, 128, True, hessians=acc)
            r.update({"linear": name, "bits": b, "shape": list(W.shape)})
            results.append(r)
            log.info("%s %db: relF %.2e, max rel diag %.2e, changed %.3f%%, objective ratio %.6f",
                     name, b, r["rel_frob"], r["max_rel_diag"], 100 * r["frac_changed"], r["objective_ratio"])
    del model, hs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    doc = {"model": args.model, "layer": args.layer, "calib_n": args.calib_n, "calib_seed": args.calib_seed,
           "code_version": code_version(), "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "reading": "frac_changed > 0 with objective_ratio ~ 1 and rel_frob ~ 1e-7..1e-6: the paths differ by GPTQ rounding chaos, not precision",
           "results": results}
    out = Path(args.out) if args.out else ROOT / ("results" if args.calib_n == 128 else "results/smoke") / args.model / "hessian_paths.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
