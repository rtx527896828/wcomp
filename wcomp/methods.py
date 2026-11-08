"""Method registry for Phase 1 (DESIGN.md, wcomp/methods.py contract) and the Phase 2 parametric names (P0).

Every method is callable(model, ctx) -> BitAccount and modifies the model's block linears in place.
Phase 1 names live in METHODS; resolve(name) returns those objects unchanged and builds the parametric
P0 methods ("rtn-sym-g32-4", "rtn-sym-g128-4@only=down", "rtn-sym-g128-3@layer=5",
"rtn-sym-g128-4+keepcols=1", "rtn-sym-g128-4+keepsuper=<path>") on demand, and the Phase 3 allocation names
("alloc-type-gptq32-avg3", "alloc-tensor-gptq32-avg3.5", "alloc-type-rtn-avg3", "alloc-type-shuf1-gptq32-avg3",
"rowalloc-fisher-rtn-avg4", "rowalloc-shuf2-rtn-avg4", "rowalloc-fisher-gptq32-avg3", "combo-gptq32-avg3"); those
read results/phase3/<ctx.tag>/damage_c4.jsonl, results/probe/<ctx.tag>/fisher_rows.pt and combo.json at call time.
The other Phase 3 names are in METHODS: "gptq32-sym-g128-{4,3,2}" (f32 Hessian baseline), "had-gptq32-sym-g128-{4,3,2}",
"rnd-/hadsign-/eig-gptq32-sym-g128-3" (method D), "lloyd-gptq32-g128-{4,3,2}", "nf-gptq32-g128-{4,3}" (the fixed
Gaussian Lloyd-Max codebook) and "nfq-gptq32-g128-{4,3}" (the literal DESIGN.md E3 codebook, the Gaussian quantiles) (method E).
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from wcomp import alloc as alloc_mod
from wcomp.bits import BitAccount
from wcomp.models import WEIGHT_TYPES, block_linears, layer_index, weight_type
from wcomp.quant.gptq import gptq_model
from wcomp.quant.rotate import ROTATION_KINDS, hadamard, rotate_inputs, rotation_bits
from wcomp.quant.rtn import (CODEBOOK_KINDS, apply_rtn, apply_rtn_keepcols, load_super_candidates,  # noqa: F401
                             super_candidate_records, super_weight_bits)

GROUP = 128


@dataclass
class Ctx:
    """What a method may need besides the model. calib_ids is None unless the method needs calibration."""

    calib_ids: torch.Tensor | None
    seed: int
    tokenizer: PreTrainedTokenizerBase | None = None
    tag: str | None = None  # model tag (wcomp.models.MODELS key); Phase 3 allocation methods read their inputs by it
    # Where the runner writes its rows; the allocation methods put alloc_<method>_seed<s>.json next to them (default,
    # when None: results/phase3/<tag>/), so a smoke run never overwrites a protocol run's allocation record.
    out_dir: Path | None = None


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


def rotation_summary(rotations: dict[str, torch.Tensor], account: BitAccount, seed: int) -> str:
    """The note clause of had(): "input rotation hadamard(1024,2048,3584)[ + random_orthogonal(..., seed=s)]"."""
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
        kinds.append(f"random_orthogonal({','.join(map(str, rnd_widths))}, seed={seed})")
    return f"input rotation {' + '.join(kinds)}"


def had(inner: Method) -> Method:
    """Rotate every block linear's input space (Hadamard, or seeded random orthogonal where no Hadamard
    exists), then apply `inner`. Rotations cost 0 bits (deterministic / seeded); the note records which.

    Every tensor gets the same rotation summary ("input rotation hadamard(1024,2048,3584)"), so the de-duplicated
    notes of BitAccount.summary() carry the inner quantizer's note once, not once per input width."""

    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        rotations = rotate_inputs(model, seed=ctx.seed)
        account = inner(model, ctx)
        summary = rotation_summary(rotations, account, ctx.seed)
        for t in account.tensors:
            t.note = f"{t.note}; {summary}"
        return account

    return method


def split_name(name: str) -> tuple[str, str]:
    """("rtn-sym-g128-4", "@only=down+keepcols=1"): the base method name and its P0 suffix (empty for Phase 1)."""
    m = re.match(r"^([^@+]*)(.*)$", name)
    return m.group(1), m.group(2)


def needs_calib(name: str) -> bool:
    """True for data-dependent methods (the runner then loads C4 calibration windows): GPTQ (incl. gptq32) and the
    eig rotation (calibration Hessians)."""
    base = split_name(name)[0]
    return "gptq" in base or base.startswith("eig-")


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
# The Phase 1 registry as it stood before the Phase 3 additions below (the names whose numbers live in
# results/<model>/runs.jsonl; scripts/probe_io.is_phase1_name agrees with this set).
PHASE1_METHODS: frozenset[str] = frozenset(METHODS)


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


# ---------------------------------------------------------------- Phase 3 (DESIGN.md "Phase 3 contract"): gptq32 and rotations.
# Additive; nothing above this line changed. "gptq32" = GPTQ with the f32 per-window Hessian product, selected by the
# name alone (the Phase 1 "gptq-sym-g128-*" names keep the f64 path and their Phase 1 numbers).


def gptq32_model(model: nn.Module, calib_ids: torch.Tensor, bits: int, group: int = GROUP, sym: bool = True,
                 codebook: dict | None = None) -> BitAccount:
    """gptq_model on the f32 Hessian path (selected by argument; the module flag of the Phase 1 names is not touched)."""
    return gptq_model(model, calib_ids, bits, group, sym, hessian_f32=True, codebook=codebook)


def gptq32(bits: int, sym: bool = True) -> Method:
    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        if ctx.calib_ids is None:
            raise ValueError("gptq32 needs ctx.calib_ids (C4 calibration windows)")
        return gptq32_model(model, ctx.calib_ids, bits, GROUP, sym)

    return method


def codebook_gptq32(kind: str, bits: int) -> Method:
    """E2 / E3: GPTQ (f32 Hessian) rounding onto a per-tensor codebook (wcomp.quant.rtn.fit_codebook): "lloyd" = Lloyd-Max
    fit on the RMS-normalized weights (2^b * 16 table bits per tensor), "nf" = the fixed Gaussian Lloyd-Max codebook
    (0 bits; the literal Gaussian-quantile codebook is "nfq", see wcomp.quant.codebook). `sym` is irrelevant."""
    if kind not in CODEBOOK_KINDS:
        raise ValueError(f"unknown codebook kind {kind!r}; known: {CODEBOOK_KINDS}")

    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        if ctx.calib_ids is None:
            raise ValueError(f"{kind}-gptq32 needs ctx.calib_ids (C4 calibration windows)")
        return gptq32_model(model, ctx.calib_ids, bits, GROUP, True, codebook={"kind": kind})

    return method


def rotated(kind: str, inner: Method) -> Method:
    """Rotate every block linear's input space with `kind` (wcomp.quant.rotate.ROTATION_KINDS), then apply `inner`.
    "hadamard" is had(inner) (same notes as Phase 1). "hadsign" / "rnd" cost 0 bits (seeded by ctx.seed) and are
    recorded in the note once per row like had(). "eig" collects the calibration Hessians (ctx.calib_ids), rotates
    each shared-input group with its eigenvectors and charges the stored bf16 matrices (16 K K bits per distinct R,
    split over the group by numel; wcomp.quant.rotate.rotation_bits) as aux bits of the rotated tensors."""
    if kind == "hadamard":
        return had(inner)
    if kind not in ROTATION_KINDS:
        raise ValueError(f"unknown rotation kind {kind!r}; known: {ROTATION_KINDS}")

    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        linears = block_linears(model)
        if kind == "eig":
            if ctx.calib_ids is None:
                raise ValueError("eig rotation needs ctx.calib_ids (C4 calibration windows)")
            rotations = rotate_inputs(model, linears, seed=ctx.seed, kind="eig", calib_ids=ctx.calib_ids)
        else:
            rotations = rotate_inputs(model, linears, seed=ctx.seed, kind=kind)
        account = inner(model, ctx)
        widths = sorted({rotations[t.name].shape[0] for t in account.tensors})
        # "rotation=<kind>" is the fixed token scripts/run_baselines.py reads the row's rotation field from (the free
        # text after it is for people); the Phase 1 had() note is unchanged and is still read by its words.
        if kind == "eig":
            charge = rotation_bits(rotations, linears, "eig")
            groups = len({id(R) for R in rotations.values()})
            summary = (f"rotation=eig; input rotation eig({','.join(map(str, widths))}): calibration-Hessian eigenvectors, "
                       f"{groups} bf16 matrices stored, 16 K K bits each shared by linears with one input")
            for t in account.tensors:
                t.aux_bits += charge[t.name]
                t.note = f"{t.note}; {summary}"
            return account
        if kind == "rnd":
            summary = f"rotation=rnd; input rotation random_orthogonal({','.join(map(str, widths))}, seed={ctx.seed})"
        else:
            had_widths, rnd_widths = [], []
            for K in widths:
                try:
                    hadamard(K)
                    had_widths.append(K)
                except ValueError:
                    rnd_widths.append(K)
            kinds = []
            if had_widths:
                kinds.append(f"hadsign({','.join(map(str, had_widths))}, seed={ctx.seed})")
            if rnd_widths:
                kinds.append(f"random_orthogonal({','.join(map(str, rnd_widths))}, seed={ctx.seed})")
            summary = f"rotation={kind}; input rotation {' + '.join(kinds)}"
        for t in account.tensors:
            t.note = f"{t.note}; {summary}"
        return account

    return method


for _b in (4, 3, 2):
    METHODS[f"gptq32-sym-g{GROUP}-{_b}"] = gptq32(_b, True)
    METHODS[f"had-gptq32-sym-g{GROUP}-{_b}"] = rotated("hadamard", gptq32(_b, True))
METHODS[f"rnd-gptq32-sym-g{GROUP}-3"] = rotated("rnd", gptq32(3, True))
METHODS[f"hadsign-gptq32-sym-g{GROUP}-3"] = rotated("hadsign", gptq32(3, True))
METHODS[f"eig-gptq32-sym-g{GROUP}-3"] = rotated("eig", gptq32(3, True))
for _b in (4, 3, 2):
    METHODS[f"lloyd-gptq32-g{GROUP}-{_b}"] = codebook_gptq32("lloyd", _b)
for _b in (4, 3):
    METHODS[f"nf-gptq32-g{GROUP}-{_b}"] = codebook_gptq32("nf", _b)
    METHODS[f"nfq-gptq32-g{GROUP}-{_b}"] = codebook_gptq32("nfq", _b)  # DESIGN.md E3 as written: the Gaussian quantiles


# ---------------------------------------------------------------- Phase 3: allocation (A), row allocation (G), combination.
# Additive. The allocation inputs are read at call time by ctx.tag (set by scripts/run_baselines.py).

_ALLOC_RE = re.compile(r"^alloc-(?P<unit>type|tensor)(?:-shuf(?P<shuf>[12]))?-(?P<q>gptq32|rtn)-avg(?P<avg>\d+(?:\.\d+)?)$")
# Models whose run list measures per-tensor damage (DESIGN.md A1: "per tensor where cheap (0.8B)"). For them the tensor
# allocation signal is the damage rows and a file without them is an error; the A2 Fisher fallback is for the others
# (the 2B), so one method name never runs on two signals depending on the state of damage_c4.jsonl.
TENSOR_DAMAGE_MODELS: tuple[str, ...] = ("qwen3.5-0.8b",)
_ROWALLOC_RE = re.compile(r"^rowalloc-(?P<src>fisher|shuf[12])-(?P<q>gptq32|rtn)-avg(?P<avg>\d+)$")
_COMBO_RE = re.compile(r"^combo-gptq32-avg(?P<avg>\d+(?:\.\d+)?)$")


def is_phase3_name(name: str) -> bool:
    """True for the alloc-* / rowalloc-* / combo-* names."""
    return any(r.match(name) for r in (_ALLOC_RE, _ROWALLOC_RE, _COMBO_RE))


def _require_tag(name: str, ctx: Ctx) -> str:
    if ctx.tag is None:
        raise ValueError(f"{name}: ctx.tag (the model tag) is required to read results/phase3/<model>/ and results/probe/<model>/")
    return ctx.tag


def _require_calib(name: str, ctx: Ctx) -> torch.Tensor:
    if ctx.calib_ids is None:
        raise ValueError(f"{name}: GPTQ needs ctx.calib_ids (C4 calibration windows)")
    return ctx.calib_ids


def type_units(linears: list[tuple[str, nn.Linear]]) -> dict[str, list[str]]:
    """{weight type: [tensor names in block_linears order]}."""
    out: dict[str, list[str]] = {}
    for n, _ in linears:
        out.setdefault(weight_type(n), []).append(n)
    return out


def damage_clause(meta: dict) -> str:
    """"damage_c4.jsonl sha 0123456789, c4 seed2 excl. calib docs of seeds 0,1, 32 windows" for a note / source string."""
    windows = ",".join(map(str, meta["windows"])) or "?"
    return (f"{Path(meta['path']).name} sha {meta['sha256']}, c4 seed{alloc_mod.DAMAGE_C4_SEED} excl. calib docs of seeds "
            f"{','.join(map(str, alloc_mod.DAMAGE_CALIB_SEEDS))}, {windows} windows")


def unit_allocation(model: nn.Module, tag: str, unit: str, target: float, out_dir: Path | None = None
                    ) -> tuple[dict[str, int], dict[str, float], dict[str, int], dict[str, list[str]], str, dict]:
    """(bits per unit, damage per param per unit, numel per unit, tensors per unit, source note, damage file meta) for
    "type" or "tensor" units of `model` at the target mean. The damage file is wcomp.alloc.damage_path(tag, out_dir)
    (the runner's output directory when it holds one, else the protocol file) and must pass wcomp.alloc.check_damage_rows
    (the C4 set and, in the protocol tree, 32 windows); its path, sha256 and set go into the source note and `meta`.
    Type units need damage_c4.jsonl type rows; tensor units take its tensor rows (all of them: a partial tensor section,
    e.g. an interrupted scripts/damage_c4.py run, raises instead of silently changing the signal) and, for a model not
    in TENSOR_DAMAGE_MODELS, fall back to the P6b Fisher per-tensor totals (fisher.json) when the file has no tensor row
    at all (DESIGN.md A2); for a model in TENSOR_DAMAGE_MODELS a file without tensor rows raises. The source note and
    the alloc JSON ("signal": alloc_signal) say which signal was used."""
    linears = block_linears(model)
    numel = {n: lin.weight.numel() for n, lin in linears}
    path = alloc_mod.damage_path(tag, out_dir)
    meta = alloc_mod.damage_meta(path)  # raises on a missing / pre-exclusion / smoke-in-protocol-tree file
    where = f"{alloc_mod.rel_path(path)} ({damage_clause(meta)})"
    if unit == "type":
        members = type_units(linears)
        rows = alloc_mod.load_damage(tag, "type", out_dir)
        missing = [t for t in members if t not in rows]
        if missing:
            raise ValueError(f"{alloc_mod.rel_path(path)} has no type rows for {missing}")
        d = alloc_mod.damage_per_param({t: rows[t] for t in members})
        source = f"d = KL/numel per type from {where}"
    else:
        members = {n: [n] for n, _ in linears}
        rows = alloc_mod.load_damage(tag, "tensor", out_dir)
        if not rows and tag in TENSOR_DAMAGE_MODELS:
            raise ValueError(f"{alloc_mod.rel_path(path)} has no tensor rows and {tag} measures per-tensor damage (TENSOR_DAMAGE_MODELS): "
                             f"run scripts/damage_c4.py --model {tag} --units tensor first; the Fisher fallback is for models without "
                             f"a tensor damage measurement")
        if rows:
            missing = [n for n in members if n not in rows]
            if missing:
                raise ValueError(f"{alloc_mod.rel_path(path)} has {len(rows)} tensor rows but lacks "
                                 f"{len(missing)} of the {len(members)} block linears (e.g. {missing[0]}): finish "
                                 f"scripts/damage_c4.py --model {tag} --units tensor (it resumes a file of the same C4 set) or remove "
                                 f"the partial tensor rows")
            d = alloc_mod.damage_per_param({n: rows[n] for n in members})
            source = f"d = KL/numel per tensor from {where}"
        else:
            totals = alloc_mod.load_fisher_totals(tag)
            missing = [n for n in members if n not in totals]
            if missing:
                raise ValueError(f"{alloc_mod.rel_path(alloc_mod.fisher_path(tag))} lacks {len(missing)} tensors, e.g. {missing[0]}")
            d = {n: totals[n] / numel[n] for n in members}
            source = (f"no per-tensor damage rows in {alloc_mod.rel_path(path)}: d = Fisher total/numel "
                      f"per tensor from {alloc_mod.rel_path(alloc_mod.fisher_path(tag))}")
    n = {u: sum(numel[t] for t in ts) for u, ts in members.items()}
    for u in members:
        if u in rows and int(rows[u]["numel"]) != n[u]:
            raise ValueError(f"{u}: damage row numel {rows[u]['numel']} != model numel {n[u]}")
    bits = alloc_mod.allocate(d, n, target)
    return bits, d, n, members, source, meta


def alloc_signal(source: str) -> str:
    """"fisher" when a unit_allocation source note says the Fisher totals were used, else "damage"."""
    return "fisher" if "Fisher total" in source else "damage"


def _write_alloc_json(ctx: Ctx, name: str, payload: dict) -> Path:
    """results/phase3/<tag>/alloc_<name>_seed<s>.json, or the same file name in ctx.out_dir when the runner set one."""
    tag, seed = ctx.tag, ctx.seed
    out_dir = Path(ctx.out_dir) if ctx.out_dir is not None else alloc_mod.PHASE3_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"alloc_{name}_seed{seed}.json"
    payload = dict(payload, method=name, model=tag, seed=seed, timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return path


def _charge_unit_side_info(account: BitAccount, members: dict[str, list[str]]) -> None:
    """UNIT_SIDE_BITS on the first tensor of every unit (block_linears order): the unit's stored level index."""
    by_name = {t.name: t for t in account.tensors}
    for ts in members.values():
        by_name[ts[0]].aux_bits += alloc_mod.UNIT_SIDE_BITS


def _bits_string(bits: dict[str, int], unit: str) -> str:
    """"down=5,gate=3,..." for types; a width histogram "tensors at 2b:10,3b:50,..." for tensors."""
    if unit == "type":
        return ",".join(f"{u}={b}" for u, b in bits.items())
    hist: dict[int, int] = {}
    for b in bits.values():
        hist[b] = hist.get(b, 0) + 1
    return "tensors at " + ",".join(f"{b}b:{c}" for b, c in sorted(hist.items()))


def alloc_method(name: str, unit: str, quantizer: str, target: float, shuf: int | None) -> Method:
    """A3: widths over types or tensors from the C4-measured damage (wcomp.alloc.allocate), then GPTQ (f32 Hessian)
    or RTN with per-tensor widths. shuf = 1 / 2 replaces the allocation by its shuffled, rebalanced control. Writes
    results/phase3/<model>/alloc_<name>_seed<s>.json and charges UNIT_SIDE_BITS per unit."""

    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        tag = _require_tag(name, ctx)
        real, d, n, members, source, meta = unit_allocation(model, tag, unit, target, ctx.out_dir)
        shuffle_info = None
        if shuf is not None:
            bits, shuffle_info = alloc_mod.shuffle_allocation_detail(real, n, seed=shuf)
        else:
            bits = real
        per_tensor = {t: bits[u] for u, ts in members.items() for t in ts}
        realized = alloc_mod.mean_bits(bits, n)
        payload = {"unit": unit, "target_mean_bits": target, "realized_mean_bits": realized, "levels": list(alloc_mod.LEVELS),
                   "bits": bits, "damage_per_param": d, "numel": n, "source": source, "signal": alloc_signal(source), "damage_file": meta,
                   "side_info_bits_per_unit": alloc_mod.UNIT_SIDE_BITS, "n_units": len(bits), "shuffle_seed": shuf}
        if shuffle_info is not None:
            payload["real_bits"] = real
            payload["real_mean_bits"] = alloc_mod.mean_bits(real, n)
            payload["shuffle"] = shuffle_info
        path = _write_alloc_json(ctx, name, payload)
        if quantizer == "rtn":
            account = apply_rtn(model, per_tensor, GROUP, True)
        else:
            account = gptq_model(model, _require_calib(name, ctx), per_tensor, GROUP, True, hessian_f32=True)
        _charge_unit_side_info(account, members)
        clause = (f"{name}: {_bits_string(bits, unit)}; realized mean {realized:.4f}; {alloc_mod.UNIT_SIDE_BITS} bits/unit "
                  f"side info on the unit's first tensor; {alloc_mod.rel_path(path)}; {source}")
        if shuffle_info is not None:
            same = shuffle_info["units_at_real_width"]
            clause += (f"; shuffled control (seed {shuf}, derangement, {shuffle_info['frozen_unit']} frozen at its permuted width, "
                       f"{len(shuffle_info['moves'])} rebalancing moves) of {_bits_string(real, unit)}; {len(same)} of {len(bits)} units at "
                       f"their real width ({shuffle_info['params_share_at_real_width']:.3f} of the parameters)")
        for t in account.tensors:
            t.note = f"{t.note}; {clause}"
        return account

    return method


def row_allocation(model: nn.Module, tag: str, bits_per_tensor: dict[str, int], shuf: int | None) -> dict[str, torch.Tensor]:
    """{tensor: row widths} by Fisher row energy at each tensor's own mean width (wcomp.alloc.row_allocate, widths
    clamped to LEVELS: a tensor at LEVELS[0] or LEVELS[-1] (2 or 8 bits) is not split and is left out of the dict, so it
    is quantized at its width with no row side information, which the decoder infers from the width). A tensor at an
    interior width keeps its row table even when water-filling leaves every row at the tensor's width (Fisher row
    energies within a factor of 4), so its ROW_SIDE_BITS per row are charged: the decoder expects a row table for every
    interior-width tensor and cannot tell otherwise. shuf = 1 / 2 permutes every split tensor's widths over its rows
    (one generator seeded with shuf, drawn in block_linears order)."""
    rows = alloc_mod.load_fisher_rows(tag)
    gen = torch.Generator().manual_seed(int(shuf)) if shuf is not None else None
    out: dict[str, torch.Tensor] = {}
    for n, lin in block_linears(model):
        if n not in rows:
            raise ValueError(f"fisher_rows.pt of {tag} has no row sums for {n}")
        r = rows[n]
        if r.numel() != lin.weight.shape[0]:
            raise ValueError(f"{n}: {r.numel()} Fisher row sums for {lin.weight.shape[0]} rows")
        rb = alloc_mod.row_allocate(r, bits_per_tensor[n])
        if bits_per_tensor[n] in (alloc_mod.LEVELS[0], alloc_mod.LEVELS[-1]):
            assert bool((rb == bits_per_tensor[n]).all())  # no room below / above: not split, no row side information
            continue
        if gen is not None:
            rb = alloc_mod.shuffle_rows(rb, gen)
        out[n] = rb
    return out


def row_widths_clause(bits_per_tensor: dict[str, int], row_bits: dict[str, torch.Tensor]) -> str:
    """"rows at b-1/b/b+1 bits within 2..8 (k tensors at 2 / 8 bits not split)" for a note."""
    unsplit = [n for n in bits_per_tensor if n not in row_bits]
    lo, hi = alloc_mod.LEVELS[0], alloc_mod.LEVELS[-1]
    text = f"rows at b-{alloc_mod.ROW_SPAN}/b/b+{alloc_mod.ROW_SPAN} bits within {lo}..{hi}"
    if unsplit:
        text += f" ({len(unsplit)} tensors at {lo} or {hi} bits not split, no row side info)"
    return text


def rowalloc_method(name: str, src: str, quantizer: str, avg: int) -> Method:
    """G: rows of every tensor at avg-1 / avg / avg+1 bits by Fisher row energy (src "fisher") or its row-shuffled
    control (src "shuf1" / "shuf2"), RTN or GPTQ (f32 Hessian); ROW_SIDE_BITS per row are charged by the quantizer."""
    shuf = int(src[4:]) if src.startswith("shuf") else None

    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        tag = _require_tag(name, ctx)
        linears = block_linears(model)
        per_tensor = {n: avg for n, _ in linears}
        row_bits = row_allocation(model, tag, per_tensor, shuf)
        summary = {n: alloc_mod.row_bits_summary(rb) for n, rb in row_bits.items()}
        meta = alloc_mod.fisher_rows_meta(tag)
        path = _write_alloc_json(ctx, name, {
            "unit": "row", "mean_bits_per_tensor": avg, "levels": sorted({int(b) for rb in row_bits.values() for b in rb.unique().tolist()} or {avg}),
            "rows_per_width": summary, "unsplit_tensors": [n for n in per_tensor if n not in row_bits], "shuffle_seed": shuf,
            "source": f"Fisher row sums from {alloc_mod.rel_path(alloc_mod.fisher_rows_path(tag))} {meta} (one measurement per model, "
                      f"shared by every run seed)"})
        if quantizer == "rtn":
            account = apply_rtn(model, avg, GROUP, True, row_bits=row_bits)
        else:
            account = gptq_model(model, _require_calib(name, ctx), avg, GROUP, True, row_bits=row_bits, hessian_f32=True)
        clause = (f"{name}: {row_widths_clause(per_tensor, row_bits)} by Fisher row energy"
                  + (f", row-shuffled control seed {shuf}" if shuf else "") + f"; {alloc_mod.rel_path(path)}")
        for t in account.tensors:
            t.note = f"{t.note}; {clause}"
        return account

    return method


def combo_path(tag: str, out_dir: Path | str | None = None) -> Path:
    """<out_dir>/combo.json when the runner's output directory holds one (a smoke run's own decision file), else the
    protocol file results/phase3/<tag>/combo.json; a smoke run never has to write into the protocol tree."""
    if out_dir is not None and (Path(out_dir) / "combo.json").exists():
        return Path(out_dir) / "combo.json"
    return alloc_mod.PHASE3_DIR / tag / "combo.json"


def combo_method(name: str, target: float) -> Method:
    """Combination: reads combo.json (combo_path: the runner's output directory or results/phase3/<model>/) {"had": bool,
    "lloyd": bool, "rowalloc": bool} (written after the decision rules) and composes Hadamard rotation (had) + type-level
    allocation (always) + Fisher row allocation around each type's width (rowalloc; rows stay within LEVELS, so a type
    at 2 or 8 bits is not split) + Lloyd-Max codebook (lloyd) + GPTQ with the f32 Hessian. The decision file's path,
    sha256 and content go into the alloc JSON and the note, so the row proves which decision it used."""

    def method(model: nn.Module, ctx: Ctx) -> BitAccount:
        tag = _require_tag(name, ctx)
        path = combo_path(tag, ctx.out_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: the combination's components are decided after the single-method rows")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ("had", "lloyd", "rowalloc"):
            if not isinstance(cfg.get(key), bool):
                raise ValueError(f"{path}: needs a boolean {key!r}")
        combo_file = {"path": alloc_mod.rel_path(path), "sha256": alloc_mod.file_sha(path)}
        calib = _require_calib(name, ctx)
        rotations = rotate_inputs(model, seed=ctx.seed) if cfg["had"] else None
        bits, d, n, members, source, meta = unit_allocation(model, tag, "type", target, ctx.out_dir)
        per_tensor = {t: bits[u] for u, ts in members.items() for t in ts}
        row_bits = row_allocation(model, tag, per_tensor, None) if cfg["rowalloc"] else None
        kwargs: dict = {"hessian_f32": True, "row_bits": row_bits}
        if cfg["lloyd"]:
            kwargs["codebook"] = {"kind": "lloyd"}
        realized = alloc_mod.mean_bits(bits, n)
        payload = {"unit": "type", "target_mean_bits": target, "realized_mean_bits": realized, "levels": list(alloc_mod.LEVELS), "bits": bits,
                   "damage_per_param": d, "numel": n, "source": source, "damage_file": meta,
                   "side_info_bits_per_unit": alloc_mod.UNIT_SIDE_BITS, "combo": cfg, "combo_file": combo_file}
        if row_bits is not None:
            payload["rows_per_width"] = {t: alloc_mod.row_bits_summary(rb) for t, rb in row_bits.items()}
            payload["unsplit_tensors"] = [t for t in per_tensor if t not in row_bits]
            payload["row_levels"] = f"max(b-{alloc_mod.ROW_SPAN}, {alloc_mod.LEVELS[0]}) .. min(b+{alloc_mod.ROW_SPAN}, {alloc_mod.LEVELS[-1]})"
        jpath = _write_alloc_json(ctx, name, payload)
        account = gptq_model(model, calib, per_tensor, GROUP, True, **kwargs)
        _charge_unit_side_info(account, members)
        parts = [k for k in ("had", "lloyd", "rowalloc") if cfg[k]] + ["alloc-type", "gptq32"]
        clause = (f"{name} = {'+'.join(parts)} ({combo_file['path']} sha {combo_file['sha256']}): {_bits_string(bits, 'type')}; "
                  f"realized mean {realized:.4f}; {alloc_mod.UNIT_SIDE_BITS} bits/unit side info on the unit's first tensor; "
                  f"{alloc_mod.rel_path(jpath)}; {source}")
        if row_bits is not None:
            clause += f"; {row_widths_clause(per_tensor, row_bits)} by Fisher row energy"
        if rotations is not None:
            clause += "; " + rotation_summary(rotations, account, ctx.seed)
        for t in account.tensors:
            t.note = f"{t.note}; {clause}"
        return account

    return method


def resolve_phase3(name: str) -> Method | None:
    """The Phase 3 allocation method for `name`, or None when the name is not one."""
    m = _ALLOC_RE.match(name)
    if m:
        return alloc_method(name, m["unit"], m["q"], float(m["avg"]), int(m["shuf"]) if m["shuf"] else None)
    m = _ROWALLOC_RE.match(name)
    if m:
        avg = int(m["avg"])
        if not alloc_mod.LEVELS[0] < avg < alloc_mod.LEVELS[-1]:
            raise ValueError(f"{name}: rowalloc avg must be in {alloc_mod.LEVELS[0] + 1}..{alloc_mod.LEVELS[-1] - 1} (rows get avg-1 / avg+1 "
                             f"bits and stay within {alloc_mod.LEVELS[0]}..{alloc_mod.LEVELS[-1]})")
        return rowalloc_method(name, m["src"], m["q"], avg)
    m = _COMBO_RE.match(name)
    if m:
        return combo_method(name, float(m["avg"]))
    return None


def resolve(name: str) -> Method:
    """METHODS[name] for every Phase 1 / Phase 3 registry name (the identical object), else the parametric P0 method or
    a Phase 3 allocation method; ValueError (with the known names) for anything else."""
    if name in METHODS:
        return METHODS[name]
    phase3 = resolve_phase3(name)
    if phase3 is not None:
        return phase3
    try:
        spec = parse(name)
    except ValueError as e:
        raise ValueError(f"unknown method {name!r} ({e}); known: {sorted(METHODS)}, a parametric rtn name or a Phase 3 "
                         "alloc-* / rowalloc-* / combo-* name") from e
    return parametric(name, spec)
