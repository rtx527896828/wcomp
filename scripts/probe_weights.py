"""Weight-only probes P1-P4 of the Phase 2 contract (DESIGN.md) on one model.

  uv run python scripts/probe_weights.py --model qwen3.5-0.8b [--probes p1,p2,p3,p4] [--limit-tensors N] [--include-embedding]
  -> results/probe/<model>/{marginal,lowrank,outliers,interlayer}.json

The model is loaded to the GPU in bf16; each tensor is converted to f32 on its own, with its controls, and freed
before the next one (never more than one f32 copy plus controls). Embedding / lm_head (tied on Qwen3.5: one
record) go to P1 and P2 only, with --include-embedding. --limit-tensors N keeps the first N block linears in
forward order (a smoke setting, recorded in params).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wcomp.models import MODELS, block_linears, layer_index, load_model, weight_type  # noqa: E402
from wcomp.probe import weights as pw  # noqa: E402
from wcomp.probe.common import ProbeRecord, shuffle_within_tensor, write_probe_json  # noqa: E402

log = logging.getLogger("probe_weights")

PROBE_FILES = {"p1": "marginal", "p2": "lowrank", "p3": "outliers", "p4": "interlayer"}
CONTROL_SEEDS = (0, 1)


def embedding_entries(model: torch.nn.Module) -> list[tuple[str, int, str, torch.Tensor]]:
    """(name, layer=-1, type, weight) for the input embedding and, if not tied to it, the lm_head."""
    out = []
    names = {id(m): n for n, m in model.named_modules()}
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    out.append((names.get(id(emb), "embed_tokens"), -1, "embed", emb.weight))
    if head is not None and head.weight.data_ptr() != emb.weight.data_ptr():
        out.append((names.get(id(head), "lm_head"), -1, "lm_head", head.weight))
    return out


def fill(rec: ProbeRecord, name: str, layer: int, wtype: str) -> ProbeRecord:
    rec.tensor, rec.layer, rec.type = name, layer, wtype
    return rec


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--probes", default="p1,p2,p3,p4")
    ap.add_argument("--limit-tensors", type=int, default=None, help="first N block linears only (smoke)")
    ap.add_argument("--include-embedding", action="store_true", help="add embedding / lm_head to P1 and P2")
    ap.add_argument("--out-dir", default=None, help="default results/probe/<model>")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    probes = [p.strip().lower() for p in args.probes.split(",") if p.strip()]
    unknown = [p for p in probes if p not in PROBE_FILES]
    if unknown:
        raise SystemExit(f"unknown probes {unknown}; known: {sorted(PROBE_FILES)}")
    # --limit-tensors is off protocol: such output goes under results/smoke/probe/<model> (the smoke convention), never
    # next to the real results/probe/<model>/*.json.
    smoke = args.limit_tensors is not None
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / ("smoke/probe" if smoke else "probe") / args.model
    if smoke and "smoke" not in out_dir.parts:
        out_dir = out_dir.parent / "smoke" / out_dir.name
        log.warning("off-protocol run (--limit-tensors) with an explicit --out-dir: writing to %s instead", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, _ = load_model(args.model)
    linears = block_linears(model)
    if args.limit_tensors is not None:
        linears = linears[: args.limit_tensors]
    blocks = [(name, layer_index(name), weight_type(name), lin.weight) for name, lin in linears]
    embeds = embedding_entries(model) if args.include_embedding else []
    log.info("loaded %s in %.1fs: %d block linears%s", args.model, time.time() - t0, len(blocks),
             f" + {len(embeds)} embedding tensors" if embeds else "")
    common_params = {"limit_tensors": args.limit_tensors, "include_embedding": args.include_embedding,
                     "n_block_linears": len(blocks), "embedding_tensors": [e[0] for e in embeds]}

    with torch.no_grad():
        if "p1" in probes:
            sync()
            t = time.time()
            records = []
            for i, (name, layer, wtype, w) in enumerate(blocks + embeds):
                records.append(fill(pw.marginal_entropy(w.detach(), pw.GROUP, pw.BITS, seed=0), name, layer, wtype))
                if (i + 1) % 20 == 0 or i + 1 == len(blocks) + len(embeds):
                    log.info("P1 %d/%d (%s) %.1fs", i + 1, len(blocks) + len(embeds), name, time.time() - t)
            sync()
            params = {**common_params, "group": pw.GROUP, "bits": list(pw.BITS), "control": "gaussian_control seed 0",
                      "student_nu_grid": list(pw.STUDENT_NU_GRID), "time_s": round(time.time() - t, 2)}
            write_probe_json(out_dir / "marginal.json", args.model, "P1 marginal", params, records)
            log.info("P1 marginal: %d records in %.1fs -> %s", len(records), time.time() - t, out_dir / "marginal.json")
            torch.cuda.empty_cache()

        if "p2" in probes:
            sync()
            t = time.time()
            records = []
            for i, (name, layer, wtype, w) in enumerate(blocks + embeds):
                W = w.detach().float()
                controls = [shuffle_within_tensor(W, s) for s in CONTROL_SEEDS]
                records.append(fill(pw.low_rank(W, controls, pw.R_FRACS), name, layer, wtype))
                del W, controls
                if (i + 1) % 20 == 0 or i + 1 == len(blocks) + len(embeds):
                    log.info("P2 %d/%d (%s) %.1fs", i + 1, len(blocks) + len(embeds), name, time.time() - t)
            sync()
            params = {**common_params, "r_fracs": list(pw.R_FRACS), "control_seeds": list(CONTROL_SEEDS),
                      "time_s": round(time.time() - t, 2)}
            write_probe_json(out_dir / "lowrank.json", args.model, "P2 low rank", params, records)
            log.info("P2 lowrank: %d records in %.1fs -> %s", len(records), time.time() - t, out_dir / "lowrank.json")
            torch.cuda.empty_cache()

        if "p3" in probes:
            sync()
            t = time.time()
            records = []
            col_norms: dict[str, list[tuple[int, torch.Tensor]]] = {}  # per (type, leaf) series of hidden-input linears
            for i, (name, layer, wtype, w) in enumerate(blocks):
                W = w.detach().float()
                controls = [shuffle_within_tensor(W, s) for s in CONTROL_SEEDS]
                records.append(fill(pw.outliers(W, controls, pw.PCTS, 4, pw.GROUP), name, layer, wtype))
                if wtype in pw.HIDDEN_INPUT_TYPES:
                    col_norms.setdefault(f"{wtype}:{name.split('.')[-1]}", []).append((layer, W.norm(dim=0).cpu()))
                del W, controls
                if (i + 1) % 20 == 0 or i + 1 == len(blocks):
                    log.info("P3 %d/%d (%s) %.1fs", i + 1, len(blocks), name, time.time() - t)
            pooled: list[tuple[int, torch.Tensor]] = []
            for key, series in sorted(col_norms.items()):
                pooled.extend(series)
                if len(series) >= 2:
                    wtype = key.split(":")[0]
                    records.append(fill(pw.outlier_consistency(series, top=16, seed=0), f"{key}:cross-layer", -1, wtype))
            if len(pooled) >= 2:
                records.append(fill(pw.outlier_consistency(pooled, top=16, seed=0), "all-hidden-input:cross-layer", -1, "all_hidden_input"))
            sync()
            params = {**common_params, "group": pw.GROUP, "bits": 4, "pcts": list(pw.PCTS), "control_seeds": list(CONTROL_SEEDS),
                      "consistency_top": 16, "hidden_input_types": sorted(pw.HIDDEN_INPUT_TYPES), "time_s": round(time.time() - t, 2)}
            write_probe_json(out_dir / "outliers.json", args.model, "P3 outliers", params, records)
            log.info("P3 outliers: %d records in %.1fs -> %s", len(records), time.time() - t, out_dir / "outliers.json")
            torch.cuda.empty_cache()

        if "p4" in probes:
            sync()
            t = time.time()
            records = []
            series: dict[str, list[tuple[str, int, torch.Tensor]]] = {}
            for name, layer, wtype, w in blocks:
                series.setdefault(f"{wtype}:{name.split('.')[-1]}", []).append((name, layer, w.detach()))
            for key, entries in sorted(series.items()):
                if len(entries) < 2:
                    continue
                wtype = key.split(":")[0]
                ts = time.time()
                recs = pw.interlayer(entries, perm_axis=pw.INTERMEDIATE_AXIS.get(wtype), seed=0)
                for r in recs:
                    r.type = wtype
                records.extend(recs)
                sync()
                log.info("P4 %s: %d layers, %d records (perm_axis %s) %.1fs", key, len(entries), len(recs),
                         pw.INTERMEDIATE_AXIS.get(wtype), time.time() - ts)
            params = {**common_params, "control": "one shuffled pair per record (W_i seed 0, W_j seed 1); no second control replicate, no control spread",
                      "perm_axis": pw.INTERMEDIATE_AXIS,
                      "pairs_recorded": "adjacent + best (largest overhead-corrected raw/out/in gain) pair; all pairs summarized on the best",
                      "time_s": round(time.time() - t, 2)}
            write_probe_json(out_dir / "interlayer.json", args.model, "P4 interlayer", params, records)
            log.info("P4 interlayer: %d records in %.1fs -> %s", len(records), time.time() - t, out_dir / "interlayer.json")
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    log.info("done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
