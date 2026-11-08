"""Entropy coding of quantization indices (DESIGN.md Phase 3, method E1: the Huffman / Shannon bit figures).

Per tensor the index stream (TensorBits.indices, 0..levels-1) is coded with a canonical Huffman code built from the
tensor's exact symbol counts; the code-length table is charged at b bits per symbol (b * 2^b), the pair-symbol code
over consecutive index pairs (alphabet 2^(2b), table 2b * 2^(2b)) is tried as well, and so is the plain b-bit code
(no table: on a small 8-bit or a near-uniform 3-bit tensor the Huffman tables cost more than the code saves); the
best of the three is taken and the choice costs CHOICE_BITS = 2 bits per tensor, so huff <= fixed + 2 bits per
tensor (DESIGN.md's two-way choice with 1 bit lacked the fallback and charged such tensors above their fixed width;
NOTES.md records the amendment). The Shannon figure is the entropy bound of the code chosen for that tensor (no
table): n * H1 (order-0 entropy of the indices) where the single-symbol or the plain code won, n * H2 / 2 (pair-stream
entropy per symbol) where the pair code won, so huff - shannon is that coder's own loss plus its tables; it is not the
order-0 entropy when the pair code wins (NOTES.md "Decisions fixed at integration"; the order-0 figure is kept per
tensor in coded_bits_detail()["coded"]["entropy"] and summed in coded_bits()["bits_blocks_entropy0"]). Tensors
without indices are counted at their fixed width. All side information (aux_bits) is in every figure.
"""
from __future__ import annotations

import heapq
import math
from typing import Sequence

import torch
from torch import Tensor

from wcomp.bits import BitAccount

CHOICE_BITS = 2  # per tensor: which of the single-symbol Huffman, pair-symbol Huffman and plain b-bit codes is used
CHOICES = ("huff1", "huff2", "plain")  # tie order


def huffman_code_lengths(counts: Sequence[int] | Tensor) -> list[int]:
    """Huffman code lengths (bits) for an alphabet with the given counts; symbols with count 0 get length 0 (not
    coded). A lone symbol gets length 1 (a one-symbol stream still needs a bit per symbol in a prefix code).
    The lengths satisfy Kraft, sum 2^-l <= 1 (asserted), and define the canonical code of canonical_codes()."""
    if isinstance(counts, Tensor):
        counts = counts.reshape(-1).tolist()
    counts = [int(c) for c in counts]
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    lengths = [0] * len(counts)
    alive = [(c, i) for i, c in enumerate(counts) if c > 0]
    if not alive:
        return lengths
    if len(alive) == 1:
        lengths[alive[0][1]] = 1
        return lengths
    # heap entries: (weight, tie-break id, list of symbols under this node); merging two nodes adds a bit to every
    # symbol beneath them.
    heap = [(c, i, [i]) for c, i in alive]
    heapq.heapify(heap)
    tie = len(counts)
    while len(heap) > 1:
        w1, _, s1 = heapq.heappop(heap)
        w2, _, s2 = heapq.heappop(heap)
        for s in s1:
            lengths[s] += 1
        for s in s2:
            lengths[s] += 1
        heapq.heappush(heap, (w1 + w2, tie, s1 + s2))
        tie += 1
    kraft = sum(2.0 ** -l for l in lengths if l > 0)
    assert kraft <= 1.0 + 1e-12, f"Kraft sum {kraft} > 1"
    return lengths


def canonical_codes(lengths: Sequence[int]) -> dict[int, tuple[int, int]]:
    """{symbol: (code, length)} of the canonical Huffman code for these lengths: symbols sorted by (length, symbol)
    get consecutive codewords, each shorter length's last code shifted left when the length grows."""
    order = sorted((l, s) for s, l in enumerate(lengths) if l > 0)
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    prev_len = order[0][0] if order else 0
    for l, s in order:
        code <<= l - prev_len
        codes[s] = (code, l)
        code += 1
        prev_len = l
    return codes


def huffman_encode(symbols: Sequence[int] | Tensor, lengths: Sequence[int]) -> tuple[int, int]:
    """Bit stream of `symbols` under the canonical code of `lengths`: (bits as a Python int, msb first, n_bits)."""
    codes = canonical_codes(lengths)
    if isinstance(symbols, Tensor):
        symbols = symbols.reshape(-1).tolist()
    stream = 0
    n_bits = 0
    for s in symbols:
        code, l = codes[int(s)]
        stream = (stream << l) | code
        n_bits += l
    return stream, n_bits


def huffman_decode(stream: int, n_bits: int, lengths: Sequence[int], n_symbols: int) -> list[int]:
    """Inverse of huffman_encode: reads n_symbols symbols from the msb-first bit stream."""
    codes = canonical_codes(lengths)
    lookup = {(code, l): s for s, (code, l) in codes.items()}
    max_len = max((l for _, l in codes.values()), default=0)
    out: list[int] = []
    pos = n_bits
    code = 0
    l = 0
    while len(out) < n_symbols:
        if pos == 0:
            raise ValueError("bit stream exhausted before n_symbols symbols were decoded")
        pos -= 1
        code = (code << 1) | ((stream >> pos) & 1)
        l += 1
        s = lookup.get((code, l))
        if s is not None:
            out.append(s)
            code = 0
            l = 0
        elif l > max_len:
            raise ValueError("invalid code word in the bit stream")
    return out


def _entropy_bits(counts: Tensor) -> float:
    """Entropy in bits of the empirical distribution `counts` (any shape), 0 for an empty stream."""
    c = counts.reshape(-1).to(torch.float64)
    n = float(c.sum())
    if n == 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * torch.log2(p)).sum())


def _coded_length(symbols: Tensor, alphabet: int) -> tuple[int, Tensor]:
    """(sum len * count of the Huffman code over `symbols`, the counts) for an alphabet 0..alphabet-1."""
    counts = torch.bincount(symbols, minlength=alphabet)
    lengths = huffman_code_lengths(counts)
    coded = int(sum(l * c for l, c in zip(lengths, counts.tolist())))
    return coded, counts


def tensor_coded_bits(indices: Tensor, levels: int) -> dict:
    """Coded size in bits of one tensor's index stream (indices 0..levels-1, b = ceil(log2 levels)):
    "huff1" = single-symbol Huffman + b * 2^b table; "huff2" = Huffman over consecutive index pairs (alphabet 2^(2b),
    table 2b * 2^(2b)) + the odd last element at b bits; "plain" = n * b (the fixed-width code, no table); "best" =
    min(huff1, huff2, plain) + CHOICE_BITS (ties in CHOICES order); "shannon" = the entropy bound of the chosen code,
    n * H1 (single-symbol entropy) when huff1 or plain is taken and n * H2 / 2 (entropy of the pair stream per symbol,
    the dependence between adjacent indices included) when the pair code is, so the Shannon figure never exceeds the
    table-free part of the code that produced the Huffman figure and huff - shannon is that coder's loss plus its
    tables. (The empirical pair entropy underestimates on a stream short relative to its 2^(2b) pair alphabet; it is
    used only where the pair code also paid its 2b * 2^(2b) table and won.) Also "n", "levels", "bits", "entropy" (H1,
    bits per symbol), "pair_entropy" (H2 / 2) and "choice" ("huff1" | "huff2" | "plain")."""
    idx = indices.reshape(-1).to(torch.int64)
    n = idx.numel()
    if n == 0:
        return {"shannon": 0.0, "huff1": 0, "huff2": 0, "plain": 0, "best": 0, "n": 0, "levels": levels, "bits": 0, "entropy": 0.0,
                "pair_entropy": 0.0, "choice": "huff1"}
    if levels < 2:
        raise ValueError(f"levels must be >= 2, got {levels}")
    if int(idx.min()) < 0 or int(idx.max()) >= levels:
        raise ValueError(f"indices outside 0..{levels - 1}: min {int(idx.min())}, max {int(idx.max())}")
    b = math.ceil(math.log2(levels))
    coded1, counts1 = _coded_length(idx, levels)
    entropy = _entropy_bits(counts1)
    huff1 = coded1 + b * 2**b
    n_pairs = n // 2
    pairs = idx[: 2 * n_pairs].reshape(n_pairs, 2)
    pair_sym = pairs[:, 0] * levels + pairs[:, 1]
    coded2, counts2 = _coded_length(pair_sym, levels * levels)
    huff2 = coded2 + 2 * b * 2 ** (2 * b) + (n - 2 * n_pairs) * b
    # the odd last symbol of the pair stream is counted at the single-symbol entropy
    pair_entropy = (_entropy_bits(counts2) * n_pairs + entropy * (n - 2 * n_pairs)) / n if n_pairs else entropy
    plain = n * b
    sizes = {"huff1": huff1, "huff2": huff2, "plain": plain}
    choice = min(CHOICES, key=lambda k: (sizes[k], CHOICES.index(k)))
    shannon = (pair_entropy if choice == "huff2" else entropy) * n
    best = sizes[choice] + CHOICE_BITS
    return {
        "shannon": shannon,
        "huff1": int(huff1),
        "huff2": int(huff2),
        "plain": int(plain),
        "best": int(best),
        "n": n,
        "levels": levels,
        "bits": b,
        "entropy": entropy,
        "pair_entropy": pair_entropy,
        "choice": choice,
    }


def tensor_coded_bits_rows(indices: Tensor, row_bits: Sequence[int]) -> dict:
    """tensor_coded_bits for a row-allocated tensor (TensorBits.row_bits, method G): the decoder knows every row's
    width from the side information, so the rows of each distinct width b form their own stream on the alphabet 2^b
    (own Huffman table(s), own entropy); the figures are summed over the widths. Coding the pooled stream on the
    2^max alphabet would charge the width mixture itself (its Huffman length and entropy can exceed the fixed
    sum of row widths), which the side information already paid for. `indices` is the row-major [N*K] stream."""
    rb = torch.as_tensor(list(row_bits), dtype=torch.int64)
    N = rb.numel()
    idx = indices.reshape(-1)
    if N == 0 or idx.numel() % N != 0:
        raise ValueError(f"{idx.numel()} indices do not split into {N} rows")
    idx = idx.reshape(N, -1)
    total = {"shannon": 0.0, "huff1": 0, "huff2": 0, "plain": 0, "best": 0, "n": 0, "levels": 2 ** int(rb.max()), "bits": int(rb.max()),
             "entropy": 0.0, "pair_entropy": 0.0, "choice": [], "widths": {}}
    for b in torch.unique(rb).tolist():
        c = tensor_coded_bits(idx[rb == b], 2**b)
        for k in ("shannon", "huff1", "huff2", "plain", "best", "n"):
            total[k] += c[k]
        total["entropy"] += c["entropy"] * c["n"]
        total["pair_entropy"] += c["pair_entropy"] * c["n"]
        total["choice"].append(c["choice"])
        total["widths"][b] = c["n"]
    total["entropy"] /= total["n"]
    total["pair_entropy"] /= total["n"]
    total["choice"] = ",".join(total["choice"])
    return total


def coded_bits_detail(account: BitAccount) -> list[dict]:
    """Per tensor: {"name", "numel", "aux_bits", "fixed", "huff", "shannon"} total bits (aux included); a tensor
    without indices has huff = shannon = fixed = its total_bits; a tensor with row_bits is coded per width class
    (tensor_coded_bits_rows)."""
    rows = []
    for t in account.tensors:
        if t.indices is None:
            rows.append({"name": t.name, "numel": t.numel, "aux_bits": t.aux_bits, "fixed": t.total_bits,
                         "huff": t.total_bits, "shannon": float(t.total_bits), "coded": None})
            continue
        c = tensor_coded_bits_rows(t.indices, t.row_bits) if t.row_bits is not None else tensor_coded_bits(t.indices, t.levels)
        rows.append({"name": t.name, "numel": t.numel, "aux_bits": t.aux_bits, "fixed": t.total_bits,
                     "huff": t.aux_bits + c["best"], "shannon": t.aux_bits + c["shannon"], "coded": c})
    return rows


def coded_bits(account: BitAccount, detail: list[dict] | None = None) -> dict[str, float]:
    """{"bits_blocks_huff", "bits_blocks_shannon", "bits_blocks_entropy0"} of the account (DESIGN.md Phase 3 bit figures):
    index streams coded per tensor (tensor_coded_bits "best" / the chosen coder's Shannon bound), side information and
    index-less tensors at their fixed widths, divided by the quantized parameter count. "bits_blocks_entropy0" is the
    plain order-0 entropy figure (aux + n * H1 per tensor, per width class for row-allocated tensors), >= shannon.
    `detail` = coded_bits_detail(account) computed by the caller (the runner shares one pass with wcomp.rdbound)."""
    n = account.numel()
    if n == 0:
        return {"bits_blocks_huff": 0.0, "bits_blocks_shannon": 0.0, "bits_blocks_entropy0": 0.0}
    rows = coded_bits_detail(account) if detail is None else detail
    entropy0 = sum(r["aux_bits"] + r["coded"]["entropy"] * r["coded"]["n"] if r["coded"] is not None else r["fixed"] for r in rows)
    return {
        "bits_blocks_huff": sum(r["huff"] for r in rows) / n,
        "bits_blocks_shannon": sum(r["shannon"] for r in rows) / n,
        "bits_blocks_entropy0": entropy0 / n,
    }
