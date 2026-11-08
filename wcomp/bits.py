"""Effective-bits accounting for quantized tensors (DESIGN.md, protocol decision 6)."""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TensorBits:
    """Bit cost of one quantized tensor: index bits plus all side information at its stored width.

    Phase 3 (optional, default None, so every Phase 1 / 2 call is unchanged): `indices` is the flattened int16 CPU
    tensor of quantization indices 0..levels-1 in row-major order, `levels` the alphabet size (2^b; for mixed
    widths within a tensor 2^max_bits with the row's own width in `row_bits`). wcomp.coding reads them for the
    Huffman / Shannon bit figures; a tensor without indices is coded at its fixed width.
    """

    name: str
    numel: int
    index_bits: int
    aux_bits: int
    note: str = ""
    indices: torch.Tensor | None = None
    levels: int | None = None
    row_bits: list[int] | None = None

    @property
    def total_bits(self) -> int:
        return self.index_bits + self.aux_bits

    def as_dict(self) -> dict:
        """JSON-able fields (the index tensor itself stays out; `has_indices` says whether one is attached)."""
        return {
            "name": self.name,
            "numel": self.numel,
            "index_bits": self.index_bits,
            "aux_bits": self.aux_bits,
            "note": self.note,
            "levels": self.levels,
            "row_bits": self.row_bits,
            "has_indices": self.indices is not None,
        }


def count_params(model: nn.Module) -> dict[tuple, int]:
    """numel per unique parameter storage, keyed by (device, data_ptr); a meta parameter (data_ptr 0) is keyed
    by its identity instead, since every meta tensor reports pointer 0."""
    out: dict[tuple, int] = {}
    for p in model.parameters():
        key = (str(p.device), p.data_ptr()) if p.data_ptr() != 0 else ("id", id(p))
        out[key] = max(out.get(key, 0), p.numel())
    return out


class BitAccount:
    """Running total of the two-part code over all quantized tensors."""

    def __init__(self) -> None:
        self.tensors: list[TensorBits] = []

    def add(self, name: str, numel: int, index_bits: int, aux_bits: int, note: str = "",
            indices: torch.Tensor | None = None, levels: int | None = None, row_bits: list[int] | None = None) -> None:
        """Record one quantized tensor. `name` must be unique within the account.

        `indices` (int16, flattened, CPU; any integer tensor is converted) and `levels` are the Phase 3 index
        stream for wcomp.coding; both or neither must be given, and the stream must have `numel` entries."""
        if any(t.name == name for t in self.tensors):
            raise ValueError(f"tensor {name!r} already in BitAccount")
        if (indices is None) != (levels is None):
            raise ValueError(f"tensor {name!r}: indices and levels must be given together")
        if indices is not None:
            indices = indices.reshape(-1).to("cpu", torch.int16)
            if indices.numel() != int(numel):
                raise ValueError(f"tensor {name!r}: {indices.numel()} indices for {numel} weights")
            if int(levels) < 2 or int(levels) > 2**15:
                raise ValueError(f"tensor {name!r}: levels must be in 2..32768 for int16 indices, got {levels}")
            levels = int(levels)
        self.tensors.append(TensorBits(name, int(numel), int(index_bits), int(aux_bits), note, indices, levels, row_bits))

    def numel(self) -> int:
        return sum(t.numel for t in self.tensors)

    def total_bits(self) -> int:
        return sum(t.total_bits for t in self.tensors)

    def names(self) -> set[str]:
        return {t.name for t in self.tensors}

    def bits_blocks(self) -> float:
        """Total bits of the quantized tensors divided by their parameter count."""
        n = self.numel()
        return self.total_bits() / n if n else 0.0

    def bits_all(self, model: nn.Module) -> float:
        """(quantized bits + 16 bits for every other parameter) / total parameters of `model`.

        Parameters of `model` are counted once each: tied weights are deduplicated by storage, (device, data_ptr),
        not only by Parameter identity, so a tie made of two Parameter objects over one storage (not what
        transformers' tie_weights does today, but a legal tie) is not double counted either.
        """
        total = sum(count_params(model).values())
        other = total - self.numel()
        if other < 0:
            raise ValueError(f"account covers {self.numel()} params but model has only {total}")
        return (self.total_bits() + 16 * other) / total

    def summary(self) -> dict:
        return {
            "n_tensors": len(self.tensors),
            "numel": self.numel(),
            "index_bits": sum(t.index_bits for t in self.tensors),
            "aux_bits": sum(t.aux_bits for t in self.tensors),
            "bits_blocks": self.bits_blocks(),
            "notes": sorted({t.note for t in self.tensors if t.note}),
            "tensors": [t.as_dict() for t in self.tensors],
        }
