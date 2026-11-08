"""Evaluation and calibration data: WikiText-2 test windows and C4 validation windows (DESIGN.md 1, 4)."""
from __future__ import annotations

import gzip
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

WIKITEXT2_TEST_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)
C4_VALIDATION_URL = (
    "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz"
)


def download(url: str, dest: Path) -> Path:
    """Download url to dest (atomically, via a .part file) unless it already exists."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "wcomp/0.1"})
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(part, dest)
    return dest


def load_tokenizer(model_dir: str | Path) -> PreTrainedTokenizerBase:
    """Load the model's tokenizer."""
    return AutoTokenizer.from_pretrained(str(model_dir))


def wikitext2_test(tokenizer: PreTrainedTokenizerBase, model_tag: str, seqlen: int = 2048) -> torch.Tensor:
    """WikiText-2 raw test split as consecutive non-overlapping windows, LongTensor [n_windows, seqlen]."""
    cache = DATA_DIR / f"{model_tag}_wikitext2_test_{seqlen}.pt"
    if cache.exists():
        return torch.load(cache)
    path = download(WIKITEXT2_TEST_URL, RAW_DIR / "wikitext-2-raw-v1_test.parquet")
    rows = pq.read_table(path).column("text").to_pylist()
    text = "".join(rows)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n = len(ids) // seqlen
    windows = torch.tensor(ids[: n * seqlen], dtype=torch.long).view(n, seqlen)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(windows, cache)
    return windows


def c4_documents() -> list[str]:
    """All documents of the C4 validation shard 0, in shard order."""
    path = download(C4_VALIDATION_URL, RAW_DIR / "c4-validation.00000-of-00008.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def c4_calibration(
    tokenizer: PreTrainedTokenizerBase, model_tag: str, n: int = 128, seqlen: int = 2048, seed: int = 0
) -> torch.Tensor:
    """n calibration windows of seqlen tokens from distinct C4 validation documents, LongTensor [n, seqlen].

    Documents are visited in a seeded random permutation of the shard order; a document with at least
    seqlen tokens contributes one window at a seeded random offset, shorter documents are skipped.
    """
    cache = DATA_DIR / f"{model_tag}_c4_calib_{n}x{seqlen}_seed{seed}.pt"
    if cache.exists():
        return torch.load(cache)
    docs = c4_documents()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    windows: list[torch.Tensor] = []
    for i in order:
        # No character-count pre-filter: byte-level BPE can emit several tokens per (multi-byte) character,
        # so only the token count decides (tokenizing the whole shard takes seconds with the fast tokenizer).
        ids = tokenizer(docs[i], add_special_tokens=False)["input_ids"]
        if len(ids) < seqlen:
            continue
        start = int(rng.integers(0, len(ids) - seqlen + 1))
        windows.append(torch.tensor(ids[start : start + seqlen], dtype=torch.long))
        if len(windows) == n:
            break
    if len(windows) < n:
        raise RuntimeError(f"only {len(windows)} C4 documents with >= {seqlen} tokens, need {n}")
    out = torch.stack(windows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out
