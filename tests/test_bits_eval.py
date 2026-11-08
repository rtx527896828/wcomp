"""Tests for wcomp.bits and wcomp.eval (synthetic model on CPU, BitAccount arithmetic, real 0.8B on CUDA)."""
import math
import os
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from wcomp.bits import BitAccount
from wcomp.eval import evaluate, reference_hidden

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen3.5-0.8B")


def _tiny_model() -> nn.Module:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).eval()


def _brute_force_ppl(model: nn.Module, ids: torch.Tensor) -> float:
    """Mean cross-entropy over all windows from full f32 logits, no chunking."""
    with torch.no_grad():
        logits = model(input_ids=ids).logits.float()  # [n, T, V]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    return math.exp(loss.item())


def test_evaluate_tiny_model_self():
    model = _tiny_model()
    torch.manual_seed(1)
    ids = torch.randint(0, 1000, (3, 700))  # T=700 forces chunk boundaries at 512 and a partial last chunk
    r = evaluate(model, model, ids, kl_windows=2)
    assert r["n_tokens"] == 3 * 699
    assert r["kl_tokens"] == 2 * 699
    assert r["kl"] == 0.0
    assert r["top1_agree"] == 1.0
    ppl_ref = _brute_force_ppl(model, ids)
    assert abs(r["ppl"] - ppl_ref) / ppl_ref < 1e-4
    assert abs(math.log(r["ppl"]) - r["nll"]) < 1e-9
    r0 = evaluate(model, None, ids)
    assert r0["ppl"] == r["ppl"] and r0["kl"] is None and r0["kl_tokens"] == 0


def test_evaluate_tiny_model_perturbed():
    """A perturbed copy must have kl > 0 and top1_agree < 1 and kl must be the brute-force value."""
    model = _tiny_model()
    other = _tiny_model()
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.05 * torch.randn_like(p))
    torch.manual_seed(2)
    ids = torch.randint(0, 1000, (2, 300))
    r = evaluate(other, model, ids, kl_windows=1)
    with torch.no_grad():
        lp_ref = F.log_softmax(model(input_ids=ids[:1]).logits.float()[0, :-1], -1)
        lp_q = F.log_softmax(other(input_ids=ids[:1]).logits.float()[0, :-1], -1)
    kl = (lp_ref.exp() * (lp_ref - lp_q)).sum(-1).mean().item()
    agree = (lp_ref.argmax(-1) == lp_q.argmax(-1)).float().mean().item()
    assert r["kl"] > 0 and abs(r["kl"] - kl) < 1e-5 * max(1.0, kl)
    assert abs(r["top1_agree"] - agree) < 1e-9 and r["top1_agree"] < 1.0


def test_evaluate_with_cached_reference_hidden_matches():
    """evaluate(..., ref_hidden=reference_hidden(...)) must equal the resident-reference path exactly."""
    model = _tiny_model()
    other = _tiny_model()
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.05 * torch.randn_like(p))
    torch.manual_seed(3)
    ids = torch.randint(0, 1000, (3, 300))
    r_resident = evaluate(other, model, ids, kl_windows=2)
    h = reference_hidden(model, ids, kl_windows=2)
    assert h.shape == (2, 300, 64) and h.device.type == "cpu"
    r_cached = evaluate(other, model, ids, kl_windows=2, ref_hidden=h)
    assert r_cached == r_resident
    r_self = evaluate(model, model, ids, kl_windows=2, ref_hidden=h)
    assert r_self["kl"] == 0.0 and r_self["top1_agree"] == 1.0
    with pytest.raises(ValueError):
        evaluate(other, model, ids, kl_windows=3, ref_hidden=h)


def test_bit_account_rtn_sym_g128_4bit():
    N, K, bits, group = 1024, 1024, 4, 128
    numel = N * K
    acc = BitAccount()
    acc.add("w", numel, index_bits=bits * numel, aux_bits=16 * (numel // group), note="rtn sym g128 f16 scale")
    assert acc.bits_blocks() == pytest.approx(4.125, abs=1e-12)

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Linear(K, N, bias=False)
            self.other = nn.Parameter(torch.zeros(1_000_000))

    m = M()
    expected = (4.125 * numel + 16 * 1_000_000) / (numel + 1_000_000)
    assert acc.bits_all(m) == pytest.approx(expected, rel=1e-12)
    s = acc.summary()
    assert s["n_tensors"] == 1 and s["numel"] == numel and s["index_bits"] == 4 * numel
    assert s["aux_bits"] == 16 * 8192 and s["bits_blocks"] == pytest.approx(4.125)
    with pytest.raises(ValueError):
        acc.add("w", 1, 1, 0)


@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(MODEL_DIR), reason="needs CUDA and Qwen3.5-0.8B")
def test_evaluate_real_0p8b_self():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
    text = (
        "The Industrial Revolution began in Great Britain in the late eighteenth century and spread to continental "
        "Europe and North America over the following decades. Mechanized textile production, steam power and iron "
        "making transformed manufacturing, transport and agriculture, and drew large populations into cities. "
    )
    ids = tok(text * 400, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_win = 4
    assert ids.numel() >= n_win * 2048
    eval_ids = ids[: n_win * 2048].view(n_win, 2048)
    try:
        evaluate(model, model, eval_ids[:1], kl_windows=1)  # warm-up (reference kernels compile on first call)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = evaluate(model, model, eval_ids, kl_windows=n_win)
        torch.cuda.synchronize()
        per_window = (time.perf_counter() - t0) / n_win
        print(f"\n0.8B self-eval: {r}  {per_window:.3f} s/window (two forwards per window)  "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        assert r["n_tokens"] == n_win * 2047 and r["kl_tokens"] == n_win * 2047
        assert r["kl"] == 0.0 and r["top1_agree"] == 1.0
        assert 1.0 < r["ppl"] < 100.0
        assert per_window < 1.0
    finally:
        del model
        torch.cuda.empty_cache()


def _tied_tiny_model(storage_tie: bool) -> nn.Module:
    """Qwen2 tiny model with lm_head tied to the embedding, either by Parameter identity (what transformers'
    tie_weights does) or by a second Parameter over the same storage."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=True,
    )
    m = Qwen2ForCausalLM(cfg).eval()
    assert m.lm_head.weight is m.model.embed_tokens.weight
    if storage_tie:
        m.lm_head.weight = nn.Parameter(m.model.embed_tokens.weight.data)
        assert m.lm_head.weight is not m.model.embed_tokens.weight
        assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    return m


def test_bits_all_counts_tied_embedding_once():
    from wcomp.bits import count_params
    from wcomp.models import block_linears

    ident, shared = _tied_tiny_model(False), _tied_tiny_model(True)
    n_ident = sum(count_params(ident).values())
    assert n_ident == sum(p.numel() for p in ident.parameters())  # identity ties are already deduplicated
    assert sum(count_params(shared).values()) == n_ident  # a storage tie must not double count the embedding
    assert sum(p.numel() for p in shared.parameters()) == n_ident + 1000 * 64

    def account_for(m):
        acc = BitAccount()
        for name, lin in block_linears(m):
            n = lin.weight.numel()
            acc.add(name, n, 4 * n, 16 * (n // 64))
        return acc

    assert account_for(ident).bits_all(ident) == pytest.approx(account_for(shared).bits_all(shared), rel=1e-12)
    # a meta model (every data_ptr 0) falls back to identity counting rather than collapsing to one storage
    with torch.device("meta"):
        meta = nn.Sequential(nn.Linear(8, 4, bias=False), nn.Linear(8, 4, bias=False))
    assert sum(count_params(meta).values()) == 64
