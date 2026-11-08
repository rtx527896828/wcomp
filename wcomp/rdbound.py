"""Gaussian rate-distortion bound table (DESIGN.md Phase 3, method E4).

Per weight type: D = MSE of the group-normalized weights, (W - W_hat) / rms_g with rms_g the RMS of the *original*
weights over each 128-column group (so D is relative to a unit-variance source and the Gaussian bound is
R(D) = 0.5 log2(1 / D), 6.02 dB per bit), the row's three bit figures (fixed, Huffman, Shannon) restricted to the
type, and "bits above the bound" = bits - 0.5 log2(1 / D) (D clipped at 1e-12) for each figure.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn

from wcomp.bits import BitAccount
from wcomp.coding import coded_bits_detail
from wcomp.models import block_linears, weight_type

GROUP = 128
D_FLOOR = 1e-12

RD_FIELDS = ("type", "D", "bits_fixed", "bits_huff", "bits_shannon", "gap_fixed", "gap_huff", "gap_shannon", "numel", "n_tensors")


def rd_bound_bits(D: float) -> float:
    """Gaussian rate-distortion function R(D) = 0.5 log2(1 / D) for a unit-variance source, D clipped at D_FLOOR."""
    return 0.5 * math.log2(1.0 / max(D, D_FLOOR))


def normalized_sq_error(W: Tensor, W_hat: Tensor, group: int = GROUP) -> tuple[float, int]:
    """(sum of ((W - W_hat) / rms_g)^2, count) over W [N, K] with rms_g the RMS of W over each `group` consecutive
    columns; groups with rms 0 (all-zero original weights) contribute 0 error. K must divide by `group`."""
    if W.shape != W_hat.shape or W.dim() != 2:
        raise ValueError(f"shapes {tuple(W.shape)} vs {tuple(W_hat.shape)}")
    N, K = W.shape
    g = K if group is None or group <= 0 else group
    if K % g != 0:
        raise ValueError(f"K={K} not divisible by group={g}")
    W32 = W.to(torch.float32).reshape(N, K // g, g)
    E = (W32 - W_hat.to(torch.float32).reshape(N, K // g, g)).to(torch.float64)
    rms = W32.to(torch.float64).square().mean(dim=-1, keepdim=True).sqrt()
    E = torch.where(rms > 0, E / rms.clamp_min(1e-300), torch.zeros_like(E))
    return float(E.square().sum()), N * K


def original_weights_from_disk(model: nn.Module, names: list[str]) -> dict[str, Tensor]:
    """{name: original weight (CPU, stored dtype)} of the named modules, read from the safetensors files of the
    directory the model was loaded from (config._name_or_path); the quantized model's weights are already overwritten."""
    from safetensors import safe_open

    path = Path(getattr(model.config, "_name_or_path", "") or getattr(model.config, "name_or_path", ""))
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else []
    if not files:
        raise FileNotFoundError(f"no safetensors files under {path!r} (model.config._name_or_path) to read the original weights from")
    # Checkpoint keys may carry an extra prefix that from_pretrained strips (Qwen3.5: "model.language_model.layers..."
    # for the module "model.layers..."), so a key matches on the suffix after the leading "model." of the module name
    # and must start with that same head component ("model."), which excludes the unloaded MTP head ("mtp.layers...").
    suffixes = {n: ("." + n.split(".", 1)[1] + ".weight" if n.startswith("model.") else "." + n + ".weight") for n in names}
    out: dict[str, Tensor] = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
            for n, suf in suffixes.items():
                head = n.split(".", 1)[0] + "."
                hits = [k for k in keys if k == suf[1:] or (k.endswith(suf) and k.startswith(head))]
                if len(hits) > 1:
                    raise KeyError(f"{n}: ambiguous checkpoint keys {hits}")
                if hits:
                    if n in out:
                        raise KeyError(f"{n}: found in more than one safetensors file")
                    out[n] = sf.get_tensor(hits[0])
    missing = [n for n in names if n not in out]
    if missing:
        raise KeyError(f"{len(missing)} original weights not found in {path}, e.g. {missing[0]}")
    return out


ROW_ID_FIELDS = ("model", "method", "seed", "calib_seed", "bits_blocks", "bits_blocks_huff", "bits_blocks_shannon", "timestamp")


def rd_bound_rows(model: nn.Module, account: BitAccount, ref: nn.Module | None = None, group: int = GROUP,
                  row: dict | None = None, detail: list[dict] | None = None) -> list[dict]:
    """One row per weight type (RD_FIELDS) for the quantized `model` and its BitAccount. Original weights come from
    `ref` (the runner's resident reference copy, its block linears looked up by name) or, without one, from the
    safetensors files the model was loaded from. Tensors in the account that are not block linears of the model are
    skipped; block linears missing from the account are not quantized and not counted. With `row` (the runs.jsonl
    row of this run) every output row is prefixed with the identifying fields ROW_ID_FIELDS that `row` has.
    A linear whose input basis was rotated (wcomp.quant.rotate leaves the fold matrix on the module as
    `wcomp_input_fold`) is compared in that basis: the original is folded the same way, W (R^T)^-1 in bf16, which is
    exactly the tensor the quantizer was given; without this the distortion of every rotated row would be O(1).
    `detail` = wcomp.coding.coded_bits_detail(account) when the caller already computed it (one pass per row)."""
    linears = dict(block_linears(model))
    names = [t.name for t in account.tensors if t.name in linears]
    if ref is not None:
        ref_linears = dict(block_linears(ref))
        originals = {n: ref_linears[n].weight.data for n in names}
    else:
        originals = original_weights_from_disk(model, names)
    detail = {r["name"]: r for r in (coded_bits_detail(account) if detail is None else detail)}

    per_type: dict[str, dict] = {}
    with torch.no_grad():
        for name in names:
            W_hat = linears[name].weight.data
            W = originals[name].to(W_hat.device)
            fold = getattr(linears[name], "wcomp_input_fold", None)
            if fold is not None:
                W = (W.float() @ fold.to(W.device)).to(W_hat.dtype)
            err, n = normalized_sq_error(W, W_hat, group)
            d = detail[name]
            agg = per_type.setdefault(weight_type(name), {"err": 0.0, "numel": 0, "fixed": 0, "huff": 0, "shannon": 0.0, "n_tensors": 0})
            agg["err"] += err
            agg["numel"] += n
            agg["fixed"] += d["fixed"]
            agg["huff"] += d["huff"]
            agg["shannon"] += d["shannon"]
            agg["n_tensors"] += 1

    prefix = {k: row[k] for k in ROW_ID_FIELDS if k in row} if row is not None else {}
    rows = []
    for typ, a in per_type.items():
        D = a["err"] / a["numel"]
        bound = rd_bound_bits(D)
        bf, bh, bs = a["fixed"] / a["numel"], a["huff"] / a["numel"], a["shannon"] / a["numel"]
        core = {
            "type": typ, "D": D, "bits_fixed": bf, "bits_huff": bh, "bits_shannon": bs,
            "gap_fixed": bf - bound, "gap_huff": bh - bound, "gap_shannon": bs - bound,
            "numel": a["numel"], "n_tensors": a["n_tensors"],
        }
        assert tuple(core) == RD_FIELDS
        rows.append({**prefix, **core})
    return rows
