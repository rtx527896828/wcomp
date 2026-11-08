"""Time GPTQ (f32 Hessian path) on layer 0 of a model: the Phase 3 quantizer-extension acceptance check
("within 10% before / after"). Captures the layer-0 inputs of the 128 C4 calibration windows, accumulates the
f32-product Hessians once, then times gptq_quantize over the layer's linears `--repeat` times (best and mean).

  uv run python scripts/time_gptq_layer0.py --model qwen3.5-0.8b [--bits 3] [--repeat 3] [--codebook lloyd]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wcomp.data import c4_calibration  # noqa: E402
from wcomp.models import MODELS, load_model  # noqa: E402
from wcomp.quant import gptq as gptq_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=sorted(MODELS))
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--calib-n", type=int, default=128)
    ap.add_argument("--codebook", default=None, choices=[None, "lloyd", "nf"], help="Phase 3 codebook path (needs the extended gptq.py)")
    ap.add_argument("--row-bits", action="store_true", help="Phase 3 row_bits path: random b-1/b/b+1 per row (needs the extended gptq.py)")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    calib = c4_calibration(tok, args.model, n=args.calib_n, seed=0)
    layers = model.model.layers
    device = next(model.parameters()).device
    with torch.no_grad():
        inps, kwargs = gptq_mod._capture_layer_inputs(model, layers, calib, device)
        layer = layers[0]
        linears = [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]
        hess = {n: [torch.zeros(m.in_features, m.in_features, dtype=torch.float64, device=device), 0] for n, m in linears}

        def make_hook(name: str):
            def hook(mod, a):
                x = a[0].reshape(-1, a[0].shape[-1])
                gptq_mod.add_xtx(hess[name][0], x, f32=True)
                hess[name][1] += x.shape[0]

            return hook

        handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in linears]
        torch.cuda.synchronize()
        t0 = time.time()
        for x in inps:
            gptq_mod._run_layer(layer, x, kwargs[0])
        torch.cuda.synchronize()
        t_hess = time.time() - t0
        for h in handles:
            h.remove()
        for n in hess:
            hess[n][0] /= hess[n][1]
        del inps

        extra = {}
        if args.codebook:
            extra["codebook"] = {"kind": args.codebook, "bits": args.bits}
        if args.row_bits:
            g = torch.Generator(device="cpu").manual_seed(0)
            extra["row_bits"] = {n: (args.bits - 1 + torch.randint(0, 3, (m.out_features,), generator=g)) for n, m in linears}

        times = []
        for r in range(args.repeat):
            torch.cuda.synchronize()
            t0 = time.time()
            for n, m in linears:
                kw = dict(extra)
                if "row_bits" in kw:
                    kw["row_bits"] = kw["row_bits"][n].to(device)
                gptq_mod.gptq_quantize(m.weight.float(), hess[n][0], args.bits, 128, True, 128, 0.01, False, **kw)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    print(f"{args.model} layer 0: {len(linears)} linears, hessian (f32 path) {t_hess:.2f}s")
    print(f"quantize {args.bits}b" + (f" codebook={args.codebook}" if args.codebook else "") + (" row_bits" if args.row_bits else "")
          + f": runs {['%.3f' % t for t in times]} best {min(times):.3f}s mean {sum(times) / len(times):.3f}s")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
