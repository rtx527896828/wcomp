"""Effective-bits accounting for quantized tensors (DESIGN.md, protocol decision 6)."""
from dataclasses import dataclass, asdict

import torch.nn as nn


@dataclass
class TensorBits:
    """Bit cost of one quantized tensor: index bits plus all side information at its stored width."""

    name: str
    numel: int
    index_bits: int
    aux_bits: int
    note: str = ""

    @property
    def total_bits(self) -> int:
        return self.index_bits + self.aux_bits


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

    def add(self, name: str, numel: int, index_bits: int, aux_bits: int, note: str = "") -> None:
        """Record one quantized tensor. `name` must be unique within the account."""
        if any(t.name == name for t in self.tensors):
            raise ValueError(f"tensor {name!r} already in BitAccount")
        self.tensors.append(TensorBits(name, int(numel), int(index_bits), int(aux_bits), note))

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
            "tensors": [asdict(t) for t in self.tensors],
        }
