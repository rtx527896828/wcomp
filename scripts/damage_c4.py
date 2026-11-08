"""A1 damage measurement on C4 (DESIGN.md Phase 3, method A): per unit, quantize only that unit with rtn-sym-g128-4 and
evaluate against the bf16 reference on the seed-2 C4 windows (never WikiText). The seed-2 set is disjoint from the GPTQ
calibration sets by document, not only by seed: the documents of the seed-0 and seed-1 128-window sets
(wcomp.data.c4_calibration_documents; the smaller --calib-n sets are prefixes of them) are excluded before the seed-2
draw (c4_calibration(..., exclude_docs=...)). Without the exclusion 6 of the 32 seed-2 documents were calibration
documents of seeds 0 / 1 and 8.3% of the damage-set tokens were token-identical to seed-1 calibration tokens. Every row
records the set ("c4_set"); a file measured on another set is refused (move it aside, the damage must be re-measured).

  uv run python scripts/damage_c4.py --model qwen3.5-0.8b                      # types and tensors, 32 windows
  uv run python scripts/damage_c4.py --model qwen3.5-2b --units type
  uv run python scripts/damage_c4.py --model qwen3.5-0.8b --units type --windows 2   # smoke (results/phase3/smoke/)

Rows go to results/phase3/<model>/damage_c4.jsonl: {"unit": "type"|"tensor", "name", "numel", "kl", "ppl", "n_tokens",
"kl_tokens", "top1_agree", "method", "model", "c4_seed", "c4_set", "windows", "time_s", "git_or_code_version", "timestamp"}. Units already
in the file are skipped (the script appends and resumes). The fp16 self-check (KL 0, the reference PPL) and the
all-block-linears rtn-sym-g128-4 row (for the additivity check) go to damage_c4_ref.json next to it. A run with
--windows != 32 or --bits != 4 is a smoke run and writes under results/phase3/smoke/<model>/ instead.

Memory / time: the reference's final hidden states on the C4 windows are computed once (wcomp.eval.reference_hidden), the
reference's decoder layers then move to the CPU and only its lm_head stays on the GPU; the working copy is loaded once and
every quantized unit is restored from an exact bf16 snapshot of its weights (identical to a fresh load, without the 5 s
reload). On the 0.8B one unit is one evaluate() over 32 windows, ~5-10 s.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wcomp import alloc as alloc_mod  # noqa: E402
from wcomp.data import c4_calibration, c4_calibration_documents  # noqa: E402
from wcomp.eval import evaluate, reference_hidden  # noqa: E402
from wcomp.models import MODELS, block_linears, load_model, weight_type  # noqa: E402
from wcomp.quant.rtn import apply_rtn  # noqa: E402
from wcomp.version import code_version  # noqa: E402

log = logging.getLogger("damage_c4")

# The set definition lives in wcomp.alloc so the allocation methods can refuse a file measured on any other set.
PROTOCOL_WINDOWS = alloc_mod.DAMAGE_WINDOWS
PROTOCOL_BITS = 4
C4_SEED = alloc_mod.DAMAGE_C4_SEED  # the disjoint seed-2 C4 set (DESIGN.md Phase 3 rules); calibration runs use the run seed
CALIB_SEEDS = alloc_mod.DAMAGE_CALIB_SEEDS  # the run seeds whose calibration documents are excluded from the seed-2 set
CALIB_N, CALIB_SEQLEN = alloc_mod.DAMAGE_CALIB_N, alloc_mod.DAMAGE_CALIB_SEQLEN  # the protocol calibration set (smaller --calib-n sets are prefixes)
C4_SET = alloc_mod.DAMAGE_C4_SET


def damage_windows(tokenizer, model_tag: str, n: int) -> tuple[torch.Tensor, list[int]]:
    """(the first n seed-2 damage windows, the excluded document ids): c4_calibration at seed C4_SEED with the documents
    of the seed-0 / seed-1 protocol calibration sets excluded (max(n, PROTOCOL_WINDOWS) windows are drawn and cached)."""
    exclude = sorted(set().union(*(c4_calibration_documents(tokenizer, model_tag, n=CALIB_N, seqlen=CALIB_SEQLEN, seed=s) for s in CALIB_SEEDS)))
    c4 = c4_calibration(tokenizer, model_tag, n=max(n, PROTOCOL_WINDOWS), seed=C4_SEED, exclude_docs=exclude)[:n]
    return c4, exclude


def check_c4_set(path: Path, expected: str) -> None:
    """Refuse to resume a damage_c4.jsonl whose rows were measured on another C4 set (or on the pre-exclusion set)."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("c4_set") != expected:
                    raise SystemExit(f"{path} has rows measured on C4 set {row.get('c4_set')!r}, this run uses {expected!r}: move the file "
                                     f"aside and re-measure (the damage set definition changed)")


def out_dir_for(model_tag: str, windows: int, bits: int, results_dir: str | None = None) -> Path:
    """results/phase3/<model> for a protocol run, results/phase3/smoke/<model> for anything else (a --results-dir that
    already has a "smoke" path component, e.g. results/smoke/phase3, is kept as it is, the rule of run_baselines.py)."""
    base = ROOT / "results" / "phase3" if results_dir is None else Path(results_dir)
    if (windows != PROTOCOL_WINDOWS or bits != PROTOCOL_BITS) and "smoke" not in base.parts:
        base = base / "smoke"
    return base / model_tag


def done_units(path: Path) -> set[tuple[str, str]]:
    """(unit, name) pairs already recorded in a damage_c4.jsonl."""
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out.add((row["unit"], row["name"]))
    return out


def plan_units(linears: list[tuple[str, torch.nn.Module]], kinds: list[str]) -> list[tuple[str, str, list[str]]]:
    """[(unit kind, name, member tensor names)] for the requested kinds ("type", "tensor"), types in first-appearance order."""
    plan: list[tuple[str, str, list[str]]] = []
    if "type" in kinds:
        types: dict[str, list[str]] = {}
        for n, _ in linears:
            types.setdefault(weight_type(n), []).append(n)
        plan.extend(("type", t, names) for t, names in types.items())
    if "tensor" in kinds:
        plan.extend(("tensor", n, [n]) for n, _ in linears)
    return plan


@torch.no_grad()
def restore(modules: dict[str, torch.nn.Module], snapshot: dict[str, torch.Tensor], names: list[str]) -> None:
    for n in names:
        modules[n].weight.data.copy_(snapshot[n])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--units", default="type,tensor", help="comma-separated unit kinds: type, tensor")
    ap.add_argument("--windows", type=int, default=PROTOCOL_WINDOWS, help="C4 seed-2 windows evaluated (PPL and KL); != 32 is a smoke run")
    ap.add_argument("--bits", type=int, default=PROTOCOL_BITS, help="RTN width of the quantized unit (!= 4 is a smoke run)")
    ap.add_argument("--results-dir", default=None, help="base directory (default results/phase3)")
    args = ap.parse_args()
    kinds = [k.strip() for k in args.units.split(",") if k.strip()]
    if not kinds or any(k not in ("type", "tensor") for k in kinds):
        raise SystemExit(f"--units must name type and/or tensor, got {args.units!r}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    out_dir = out_dir_for(args.model, args.windows, args.bits, args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "damage_c4.jsonl"
    ref_path = out_dir / "damage_c4_ref.json"
    version = code_version()
    method_base = f"rtn-sym-g128-{args.bits}"

    log.info("loading reference %s", args.model)
    ref, tok = load_model(args.model)
    check_c4_set(out_path, C4_SET)
    c4, excluded = damage_windows(tok, args.model, args.windows)
    log.info("C4 seed-%d windows %s, %d calibration documents of seeds %s excluded", C4_SEED, tuple(c4.shape), len(excluded), CALIB_SEEDS)
    t0 = time.time()
    ref_hidden = reference_hidden(ref, c4, args.windows)
    ref.model.layers.to("cpu")
    torch.cuda.empty_cache()
    log.info("reference hidden states %s in %.1fs; reference layers on the CPU", tuple(ref_hidden.shape), time.time() - t0)

    model, _ = load_model(args.model)
    linears = block_linears(model)
    modules = dict(linears)
    snapshot = {n: lin.weight.data.detach().clone() for n, lin in linears}

    def run_eval() -> dict:
        m = evaluate(model, ref, c4, kl_windows=args.windows, ref_hidden=ref_hidden)
        return {k: m[k] for k in ("ppl", "kl", "top1_agree", "n_tokens", "kl_tokens")}

    ref_info = json.load(open(ref_path, encoding="utf-8")) if ref_path.exists() else {}
    if ref_info and ref_info.get("c4_set") != C4_SET:
        raise SystemExit(f"{ref_path} was measured on C4 set {ref_info.get('c4_set')!r}, this run uses {C4_SET!r}: move it aside")
    if "fp16" not in ref_info:
        t1 = time.time()
        m = run_eval()
        if m["kl"] is None or abs(m["kl"]) > 1e-6:
            raise RuntimeError(f"fp16 self-check KL {m['kl']} != 0")
        ref_info["fp16"] = dict(m, time_s=round(time.time() - t1, 1))
        log.info("fp16: ppl %.4f kl %.2e (%.1fs)", m["ppl"], m["kl"], time.time() - t1)
    if f"{method_base}@all" not in ref_info:
        t1 = time.time()
        apply_rtn(model, args.bits, 128, True, linears=linears)
        m = run_eval()
        restore(modules, snapshot, [n for n, _ in linears])
        ref_info[f"{method_base}@all"] = dict(m, numel=sum(lin.weight.numel() for _, lin in linears), time_s=round(time.time() - t1, 1))
        log.info("%s@all: ppl %.4f kl %.5f (%.1fs)", method_base, m["ppl"], m["kl"], time.time() - t1)
    ref_info.update({"model": args.model, "c4_seed": C4_SEED, "c4_set": C4_SET, "excluded_documents": len(excluded), "windows": args.windows, "bits": args.bits,
                     "git_or_code_version": version, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(ref_info, f, indent=1)

    done = done_units(out_path)
    plan = [u for u in plan_units(linears, kinds) if (u[0], u[1]) not in done]
    log.info("%d units to measure (%d already in %s)", len(plan), len(done), out_path)
    for i, (kind, name, members) in enumerate(plan):
        t1 = time.time()
        selected = [(n, modules[n]) for n in members]
        apply_rtn(model, args.bits, 128, True, linears=selected)
        m = run_eval()
        restore(modules, snapshot, members)
        if not all(torch.isfinite(torch.tensor(float(m[k]))) for k in ("ppl", "kl")):
            raise RuntimeError(f"{kind} {name}: non-finite metrics {m}")
        row = {
            "unit": kind, "name": name, "numel": sum(modules[n].weight.numel() for n in members),
            "kl": m["kl"], "ppl": m["ppl"], "n_tokens": m["n_tokens"], "kl_tokens": m["kl_tokens"], "top1_agree": m["top1_agree"],
            "method": f"{method_base}@only={name}", "model": args.model, "c4_seed": C4_SEED, "c4_set": C4_SET, "windows": args.windows,
            "time_s": round(time.time() - t1, 2), "git_or_code_version": version,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        log.info("[%d/%d] %s %s: numel %d kl %.5f ppl %.4f (%.1fs)", i + 1, len(plan), kind, name, row["numel"], m["kl"], m["ppl"], row["time_s"])
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
