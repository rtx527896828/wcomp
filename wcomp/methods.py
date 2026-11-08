"""Method registry for Phase 1 (DESIGN.md, wcomp/methods.py contract) and the Phase 2 parametric names (P0).

Every method is callable(model, ctx) -> BitAccount and modifies the model's block linears in place.
Phase 1 names live in METHODS; resolve(name) returns those objects unchanged and builds the parametric
P0 methods ("rtn-sym-g32-4", "rtn-sym-g128-4@only=down", "rtn-sym-g128-3@layer=5",
"rtn-sym-g128-4+keepcols=1", "rtn-sym-g128-4+keepsuper=<path>") on demand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from wcomp.bits import BitAccount
from wcomp.models import WEIGHT_TYPES, block_linears, layer_index, weight_type
from wcomp.quant.gptq import gptq_model
from wcomp.quant.rotate import hadamard, rotate_inputs
from wcomp.quant.rtn import apply_rtn, apply_rtn_keepcols, load_super_candidates, super_candidate_records, super_weight_bits  # noqa: F401

GROUP = 128


@dataclass
class Ctx:
    """What a method may need besides the model. calib_ids is None unless the method needs calibration."""

    calib_ids: torch.Tensor | None
    seed: int
    tokenizer: PreTrainedTokenizerBase | None = None


Method = Callable[[nn.Module, Ctx], BitAccount]


def fp16(model: nn.Module, ctx: Ctx) -> BitAccount:
    """No quantization: every block linear counted at 16 bits, so bits_blocks = bits_all = 16."""
    account = BitAccount()
    for name, lin in block_linears(model):
        n = lin.weight.numel()
        account.add(name, n, 16 * n, 0, "bf16, not quantized")
    return account


def rtn(bits: int, sym: bool) -> Method:
    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        return apply_rtn(model, bits, GROUP, sym)

    return method


def gptq(bits: int, sym: bool = True) -> Method:
    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        if ctx.calib_ids is None:
            raise ValueError("GPTQ needs ctx.calib_ids (C4 calibration windows)")
        return gptq_model(model, ctx.calib_ids, bits, GROUP, sym)

    return method


def had(inner: Method) -> Method:
    """Rotate every block linear's input space (Hadamard, or seeded random orthogonal where no Hadamard
    exists), then apply `inner`. Rotations cost 0 bits (deterministic / seeded); the note records which.

    Every tensor gets the same rotation summary ("input rotation hadamard(1024,2048,3584)"), so the de-duplicated
    notes of BitAccount.summary() carry the inner quantizer's note once, not once per input width."""

    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        rotations = rotate_inputs(model, seed=ctx.seed)
        account = inner(model, ctx)
        account_widths = sorted({rotations[t.name].shape[0] for t in account.tensors})
        had_widths, rnd_widths = [], []
        for K in account_widths:
            try:
                hadamard(K)
                had_widths.append(K)
            except ValueError:
                rnd_widths.append(K)
        kinds = []
        if had_widths:
            kinds.append(f"hadamard({','.join(map(str, had_widths))})")
        if rnd_widths:
            kinds.append(f"random_orthogonal({','.join(map(str, rnd_widths))}, seed={ctx.seed})")
        summary = f"input rotation {' + '.join(kinds)}"
        for t in account.tensors:
            t.note = f"{t.note}; {summary}"
        return account

    return method


def split_name(name: str) -> tuple[str, str]:
    """("rtn-sym-g128-4", "@only=down+keepcols=1"): the base method name and its P0 suffix (empty for Phase 1)."""
    m = re.match(r"^([^@+]*)(.*)$", name)
    return m.group(1), m.group(2)


def needs_calib(name: str) -> bool:
    """True for data-dependent methods (the runner then loads C4 calibration windows)."""
    return "gptq" in split_name(name)[0]


# "had-fp16" is the rotation with no quantization: it measures the bf16 rounding cost of the rotation itself
# (about +0.07% PPL on the 0.8B), the reference the had-* rows are read against.
METHODS: dict[str, Method] = {"fp16": fp16, "had-fp16": had(fp16)}
for _b in (8, 4, 3, 2):
    METHODS[f"rtn-sym-g{GROUP}-{_b}"] = rtn(_b, True)
for _b in (4, 3, 2):
    METHODS[f"rtn-asym-g{GROUP}-{_b}"] = rtn(_b, False)
    METHODS[f"had-rtn-sym-g{GROUP}-{_b}"] = had(rtn(_b, True))
    METHODS[f"gptq-sym-g{GROUP}-{_b}"] = gptq(_b, True)
METHODS[f"had-gptq-sym-g{GROUP}-3"] = had(gptq(3, True))


def family(name: str) -> tuple[str, int | None]:
    """Split "rtn-sym-g128-4" into ("rtn-sym-g128", 4); "fp16" -> ("fp16", None).

    P0 names keep their suffix in the family: "rtn-sym-g128-4@only=down" -> ("rtn-sym-g128@only=down", 4),
    "rtn-sym-g128-3+keepcols=1" -> ("rtn-sym-g128+keepcols=1", 3); a +keepsuper path is reduced to its file stem.
    """
    base, suffix = split_name(name)
    if "+keepsuper=" in suffix:
        sel, _, path = suffix.partition("+keepsuper=")
        suffix = f"{sel}+keepsuper={Path(path).stem}"
    head, _, tail = base.rpartition("-")
    if tail.isdigit() and head:
        return head + suffix, int(tail)
    return name, None


# ---------------------------------------------------------------- Phase 2 P0: parametric RTN ablations

GROUPS = (32, 64, 128)  # the P0 group sizes (DESIGN.md: rtn-sym-g{32,64,128}-{4,3}); every block linear's K is a multiple

_PARAM_RE = re.compile(
    r"^rtn-(?P<sym>sym|asym)-g(?P<group>\d+)-(?P<bits>\d+)"
    r"(?:@(?P<sel>only|except|layer)=(?P<selval>[^@+]+))?"
    r"(?:\+(?P<mod>keepcols|keepsuper)=(?P<modval>.+))?$"
)


@dataclass(frozen=True)
class ParamSpec:
    """A parsed P0 name. select = ("only" | "except", weight type) or ("layer", "<i>") or None (all block linears)."""

    bits: int
    group: int
    sym: bool
    select: tuple[str, str] | None = None
    keepcols: float | None = None
    keepsuper: str | None = None


def parse(name: str) -> ParamSpec:
    """Parse a parametric P0 name; raises ValueError for anything else."""
    m = _PARAM_RE.match(name)
    if m is None:
        raise ValueError(f"not a parametric method name: {name!r}")
    bits, group = int(m["bits"]), int(m["group"])
    if not 1 <= bits <= 16 or group not in GROUPS:
        raise ValueError(f"{name}: bits must be in 1..16 and the group one of {list(GROUPS)} (DESIGN.md P0)")
    select = None
    if m["sel"] is not None:
        val = m["selval"]
        if m["sel"] == "layer":
            if not val.isdigit():
                raise ValueError(f"{name}: @layer= needs a non-negative integer, got {val!r}")
        elif val not in WEIGHT_TYPES:
            raise ValueError(f"{name}: unknown weight type {val!r}; known: {sorted(WEIGHT_TYPES)}")
        select = (m["sel"], val)
    keepcols = keepsuper = None
    if m["mod"] == "keepcols":
        keepcols = float(m["modval"])
        if not 0 < keepcols <= 100:
            raise ValueError(f"{name}: keepcols percentage must be in (0, 100]")
    elif m["mod"] == "keepsuper":
        keepsuper = m["modval"]
    return ParamSpec(bits, group, m["sym"] == "sym", select, keepcols, keepsuper)


def select_linears(linears: list[tuple[str, nn.Linear]], spec: ParamSpec) -> list[tuple[str, nn.Linear]]:
    """The block linears a spec quantizes: all, one weight type, all but one type, or one decoder layer."""
    if spec.select is None:
        return list(linears)
    kind, val = spec.select
    if kind == "only":
        return [(n, m) for n, m in linears if weight_type(n) == val]
    if kind == "except":
        return [(n, m) for n, m in linears if weight_type(n) != val]
    return [(n, m) for n, m in linears if layer_index(n) == int(val)]


def parametric(name: str, spec: ParamSpec) -> Method:
    """RTN on the selected block linears; unselected block linears are counted at 16 bits like fp16, so
    bits_blocks is the honest mix. +keepsuper restores the P7 weights after quantization and charges
    16 + ceil(log2(N*K)) bits each on top of the tensor's index bits (the b-bit slot is still stored); the tensor
    note says how many weights of which P7 kind ("spike" = co-occurring down_proj activation spike, "top16" =
    global top-16 |w|) were restored model-wide, so a row that restored only top-16 weights is not read as a
    super-weight test: a global top-|w| weight is by construction its 128-group's amax, which the unsigned sym grid
    (wcomp.quant.rtn.group_grid) maps exactly when negative and clips to (2^(b-1)-1)/2^(b-1) of itself when positive,
    so a top-|w| restore only undoes that clip on a handful of the ~50 % of groups whose amax is positive (null effect:
    smoke KL 0.19948 -> 0.19957). scripts/run_baselines.py skips such rows unless told otherwise."""

    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        linears = block_linears(model)
        selected = select_linears(linears, spec)
        if not selected:
            raise ValueError(f"{name}: selects no block linear of this model")
        selected_names = {n for n, _ in selected}
        modules = dict(linears)

        saved: dict[str, list[tuple[int, int, torch.Tensor]]] = {}
        kinds: dict[str, int] = {}
        if spec.keepsuper is not None:
            with torch.no_grad():
                for c in super_candidate_records(spec.keepsuper):
                    tensor, row, col = c["tensor"], c["row"], c["col"]
                    if tensor not in modules:
                        raise ValueError(f"{name}: {tensor!r} is not a block linear of this model")
                    if tensor not in selected_names:
                        continue  # not quantized by this spec: nothing to restore, nothing to charge
                    saved.setdefault(tensor, []).append((row, col, modules[tensor].weight[row, col].clone()))
                    kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1

        if spec.keepcols is not None:
            quantized = apply_rtn_keepcols(model, spec.bits, spec.group, spec.keepcols, spec.sym, linears=selected)
        else:
            quantized = apply_rtn(model, spec.bits, spec.group, spec.sym, linears=selected)

        by_name = {t.name: t for t in quantized.tensors}
        if spec.keepsuper is not None:
            if not saved:
                raise ValueError(f"{name}: no listed super weight lies in a quantized tensor")
            n_saved = sum(len(e) for e in saved.values())
            kind_summary = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
            # One note text for every touched tensor (size-independent), so BitAccount.summary() carries it once; the
            # count is therefore the model-wide one, not that tensor's (its own charge is in aux_bits).
            clause = f"; keepsuper: {n_saved} bf16 weights restored ({kind_summary}) model-wide, 16+ceil(log2(N*K)) bits each"
            with torch.no_grad():
                for tensor, entries in saved.items():
                    W = modules[tensor].weight
                    for row, col, value in entries:
                        W[row, col] = value
                    t = by_name[tensor]
                    t.aux_bits += len(entries) * super_weight_bits(*W.shape)
                    t.note += clause

        account = BitAccount()
        for n, lin in linears:  # block_linears order, quantized and 16-bit tensors mixed as in the model
            if n in by_name:
                t = by_name[n]
                account.add(t.name, t.numel, t.index_bits, t.aux_bits, t.note)
            else:
                k = lin.weight.numel()
                account.add(n, k, 16 * k, 0, "bf16, not quantized")
        return account

    return method


def resolve(name: str) -> Method:
    """METHODS[name] for every Phase 1 name (the identical object), else the parametric P0 method; ValueError
    (with the known names) for anything else."""
    if name in METHODS:
        return METHODS[name]
    try:
        spec = parse(name)
    except ValueError as e:
        raise ValueError(f"unknown method {name!r} ({e}); known: {sorted(METHODS)} or a parametric rtn name") from e
    return parametric(name, spec)
