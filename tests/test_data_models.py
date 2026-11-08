"""Tests for wcomp.data and wcomp.models on the real Qwen3.5-0.8B (tokenizer + meta-device model)."""
from collections import Counter

import pytest
import torch

from wcomp.data import DATA_DIR, c4_calibration, load_tokenizer, wikitext2_test
from wcomp.models import MODELS, block_linears, layer_index, load_model, weight_type

TAG = "qwen3.5-0.8b"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(MODELS[TAG])


@pytest.fixture(scope="module")
def meta_model():
    model, _ = load_model(TAG, device="meta")
    return model


def test_weight_type_and_layer_index_names():
    qwen = {
        "model.layers.0.linear_attn.in_proj_qkv": ("in_proj_qkv", 0),
        "model.layers.0.linear_attn.in_proj_z": ("in_proj_z", 0),
        "model.layers.0.linear_attn.in_proj_b": ("in_proj_ab", 0),
        "model.layers.0.linear_attn.in_proj_a": ("in_proj_ab", 0),
        "model.layers.0.linear_attn.out_proj": ("out_proj", 0),
        "model.layers.3.self_attn.q_proj": ("q", 3),
        "model.layers.3.self_attn.k_proj": ("k", 3),
        "model.layers.3.self_attn.v_proj": ("v", 3),
        "model.layers.3.self_attn.o_proj": ("o", 3),
        "model.layers.23.mlp.gate_proj": ("gate", 23),
        "model.layers.23.mlp.up_proj": ("up", 23),
        "model.layers.23.mlp.down_proj": ("down", 23),
    }
    minicpm = {
        "model.layers.41.self_attn.q_proj": ("q", 41),
        "model.layers.41.self_attn.k_proj.weight": ("k", 41),
        "model.layers.10.mlp.down_proj.weight": ("down", 10),
    }
    for name, (wt, li) in {**qwen, **minicpm}.items():
        assert weight_type(name) == wt, name
        assert layer_index(name) == li, name
    with pytest.raises(ValueError):
        weight_type("lm_head")
    with pytest.raises(ValueError):
        weight_type("model.embed_tokens")
    with pytest.raises(ValueError):
        layer_index("lm_head")


def test_block_linears_qwen3_5_0_8b(meta_model):
    linears = block_linears(meta_model)
    names = [n for n, _ in linears]
    assert len(linears) == 186
    assert all(isinstance(m, torch.nn.Linear) for _, m in linears)
    assert not any("lm_head" in n or "embed_tokens" in n for n in names)
    assert len(set(names)) == 186

    counts = Counter(weight_type(n) for n in names)
    assert counts == {
        "in_proj_qkv": 18, "in_proj_z": 18, "in_proj_ab": 36, "out_proj": 18,
        "q": 6, "k": 6, "v": 6, "o": 6, "gate": 24, "up": 24, "down": 24,
    }

    # Forward order: layers non-decreasing, call order inside each layer.
    idx = [layer_index(n) for n in names]
    assert idx == sorted(idx)
    assert names[:8] == [
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.0.linear_attn.in_proj_z",
        "model.layers.0.linear_attn.in_proj_b",
        "model.layers.0.linear_attn.in_proj_a",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    ]
    layer3 = [n.split(".", 3)[-1] for n in names if layer_index(n) == 3]
    assert layer3 == [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    shapes = {n.split(".", 3)[-1]: tuple(m.weight.shape) for n, m in linears if layer_index(n) in (0, 3)}
    assert shapes["linear_attn.in_proj_qkv"] == (6144, 1024)
    assert shapes["linear_attn.in_proj_z"] == (2048, 1024)
    assert shapes["linear_attn.in_proj_a"] == (16, 1024)
    assert shapes["linear_attn.out_proj"] == (1024, 2048)
    assert shapes["self_attn.q_proj"] == (4096, 1024)
    assert shapes["self_attn.k_proj"] == (512, 1024)
    assert shapes["self_attn.o_proj"] == (1024, 2048)
    assert shapes["mlp.down_proj"] == (1024, 3584)


def test_wikitext2_test_windows(tokenizer):
    ids = wikitext2_test(tokenizer, TAG)
    assert ids.dtype == torch.long
    assert ids.dim() == 2 and ids.shape[1] == 2048
    assert ids.shape[0] >= 100
    assert (DATA_DIR / f"{TAG}_wikitext2_test_2048.pt").exists()
    text = tokenizer.decode(ids[0])
    assert text.startswith(" = Robert Boulter = ")
    # Cached call returns the same windows.
    assert torch.equal(wikitext2_test(tokenizer, TAG), ids)


def test_c4_calibration_windows(tokenizer):
    ids = c4_calibration(tokenizer, TAG, n=128, seqlen=2048, seed=0)
    assert ids.dtype == torch.long
    assert ids.shape == (128, 2048)
    assert (DATA_DIR / f"{TAG}_c4_calib_128x2048_seed0.pt").exists()
    assert len({tuple(r[:64].tolist()) for r in ids}) == 128  # distinct documents
    assert torch.equal(c4_calibration(tokenizer, TAG, n=128, seqlen=2048, seed=0), ids)
    other = c4_calibration(tokenizer, TAG, n=8, seqlen=2048, seed=1)
    assert other.shape == (8, 2048)
    assert not torch.equal(other, ids[:8])


class _ByteTokenizer:
    """Fake byte-level tokenizer: one token per UTF-8 byte, so a multi-byte character is several tokens."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode("utf-8"))}


def test_c4_calibration_counts_tokens_not_characters(monkeypatch, tmp_path):
    """DESIGN.md 4: documents shorter than seqlen *tokens* are skipped. A document of 40 four-byte characters has
    40 characters but 160 tokens and must be eligible for seqlen 128; the old character pre-filter dropped it."""
    import wcomp.data as data

    docs = ["ab" * 10, "𐐀" * 40, "x" * 200]
    monkeypatch.setattr(data, "c4_documents", lambda: docs)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    ids = c4_calibration(_ByteTokenizer(), "fake", n=2, seqlen=128, seed=0)
    assert ids.shape == (2, 128)
    assert any(0xF0 in r.tolist() for r in ids)  # the multi-byte document (lead byte 0xF0) contributed a window
    assert (tmp_path / "fake_c4_calib_2x128_seed0.pt").exists()
    with pytest.raises(RuntimeError):
        c4_calibration(_ByteTokenizer(), "fake", n=3, seqlen=128, seed=0)


def test_c4_calibration_excludes_documents_and_reports_them(monkeypatch, tmp_path):
    """Phase 3 damage set: the documents of the calibration seeds are excluded by shard index (not only by seed); the
    document ids of a set are reproducible from an existing Phase 1 window cache, and the exclusion names its own cache."""
    import wcomp.data as data
    from wcomp.data import c4_calibration_documents

    docs = [f"{chr(97 + i)}" * (150 + 7 * i) for i in range(12)]  # twelve eligible one-byte documents
    monkeypatch.setattr(data, "c4_documents", lambda: docs)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    tok = _ByteTokenizer()
    base = c4_calibration(tok, "fake", n=4, seqlen=128, seed=0)
    (tmp_path / "fake_c4_calib_4x128_seed0_docs.json").unlink()  # a Phase 1 cache has no document list
    d0 = c4_calibration_documents(tok, "fake", n=4, seqlen=128, seed=0)  # reproduced and checked against the cache
    assert len(set(d0)) == 4 and all(docs[i].encode()[0] == int(base[k, 0]) for k, i in enumerate(d0))
    assert (tmp_path / "fake_c4_calib_4x128_seed0_docs.json").exists()
    d1 = c4_calibration_documents(tok, "fake", n=4, seqlen=128, seed=1)
    excl = sorted(set(d0) | set(d1))
    other = c4_calibration(tok, "fake", n=3, seqlen=128, seed=2, exclude_docs=excl)
    d2 = c4_calibration_documents(tok, "fake", n=3, seqlen=128, seed=2, exclude_docs=excl)
    assert not (set(d2) & set(excl)) and other.shape == (3, 128)
    assert all(docs[i].encode()[0] == int(other[k, 0]) for k, i in enumerate(d2))
    plain = c4_calibration_documents(tok, "fake", n=3, seqlen=128, seed=2)  # the same seed without the exclusion
    assert plain != d2 or not (set(plain) & set(excl))
    caches = sorted(p.name for p in tmp_path.glob("fake_c4_calib_3x128_seed2*.pt"))
    assert len(caches) == 2 and any("_excl" in c for c in caches) and "fake_c4_calib_3x128_seed2.pt" in caches
    with pytest.raises(RuntimeError):  # excluding too much leaves fewer than n documents
        c4_calibration(tok, "fake", n=3, seqlen=128, seed=2, exclude_docs=list(range(10)))
