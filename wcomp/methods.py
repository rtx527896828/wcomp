"""Method registry for Phase 1 (DESIGN.md, wcomp/methods.py contract).

Every method is callable(model, ctx) -> BitAccount and modifies the model's block linears in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from wcomp.bits import BitAccount
from wcomp.models import block_linears
from wcomp.quant.gptq import gptq_model
from wcomp.quant.rotate import hadamard, rotate_inputs
from wcomp.quant.rtn import apply_rtn

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


def needs_calib(name: str) -> bool:
    """True for data-dependent methods (the runner then loads C4 calibration windows)."""
    return "gptq" in name


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
    """Split "rtn-sym-g128-4" into ("rtn-sym-g128", 4); "fp16" -> ("fp16", None)."""
    head, _, tail = name.rpartition("-")
    if tail.isdigit() and head:
        return head, int(tail)
    return name, None
