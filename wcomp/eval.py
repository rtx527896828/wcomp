"""Perplexity, KL(P_ref || P_q) and top-1 agreement (DESIGN.md, protocol decisions 2 and 3)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CHUNK = 512  # positions per log_softmax chunk; 512 x 248320 f32 = 0.5 GB


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def _hidden(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Final hidden states [T, H] (bf16) of a HF causal LM for one window `ids` [1, T]."""
    out = model.model(input_ids=ids, use_cache=False)
    return out.last_hidden_state[0]


@torch.no_grad()
def reference_hidden(ref_model: nn.Module, eval_ids: torch.Tensor, kl_windows: int = 32) -> torch.Tensor:
    """Final hidden states of `ref_model` on the first `kl_windows` windows, [kl_windows, T, H] bf16 on the CPU.

    Passed to evaluate(..., ref_hidden=...) so the KL can be computed with the reference's decoder layers
    off the GPU (only ref_model.lm_head must stay on the model's device). For the 2B models two resident
    bf16 copies plus the f32 log-softmax chunks exceed the ~10 GB available (DESIGN.md environment).
    """
    ref_model.eval()
    dev = _device(ref_model)
    n = min(kl_windows, eval_ids.shape[0])
    return torch.stack([_hidden(ref_model, eval_ids[w : w + 1].to(dev)).cpu() for w in range(n)])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    ref_model: nn.Module | None,
    eval_ids: torch.Tensor,
    kl_windows: int = 32,
    batch_size: int = 1,
    ref_hidden: torch.Tensor | None = None,
) -> dict:
    """Evaluate `model` on eval_ids [n_windows, T]; windows are independent sequences.

    ppl = exp(mean NLL over positions 1..T-1 of every window), NLL in f32 from the model's bf16 logits.
    KL and top-1 agreement are measured against `ref_model` on the first `kl_windows` windows. With
    `ref_hidden` (from reference_hidden) the reference's final hidden states are taken from that tensor
    instead of running ref_model.model, and only ref_model.lm_head is used (it must be on model's device).
    Windows run one at a time (`batch_size` is accepted for the contract and ignored).
    """
    model.eval()
    if ref_model is not None:
        ref_model.eval()
    dev = _device(model)
    eval_ids = eval_ids.to(dev)
    n_windows, T = eval_ids.shape
    kl_windows = min(kl_windows, n_windows) if ref_model is not None else 0
    if ref_hidden is not None and ref_hidden.shape[0] < kl_windows:
        raise ValueError(f"ref_hidden has {ref_hidden.shape[0]} windows, need {kl_windows}")

    nll_sum = torch.zeros((), dtype=torch.float64, device=dev)
    kl_sum = torch.zeros((), dtype=torch.float64, device=dev)
    agree_sum = torch.zeros((), dtype=torch.float64, device=dev)
    n_tokens = 0
    kl_tokens = 0

    for w in range(n_windows):
        ids = eval_ids[w : w + 1]
        h = _hidden(model, ids)
        with_kl = w < kl_windows
        h_ref = None
        if with_kl:
            h_ref = ref_hidden[w].to(dev) if ref_hidden is not None else _hidden(ref_model, ids)
        targets = ids[0, 1:]
        # logits at positions [s, e) predict tokens [s+1, e+1); the last position predicts nothing.
        for s in range(0, T - 1, CHUNK):
            e = min(s + CHUNK, T - 1)
            logp = F.log_softmax(model.lm_head(h[s:e]).float(), dim=-1)
            tgt = targets[s:e]
            nll_sum += -logp.gather(1, tgt[:, None]).double().sum()
            n_tokens += e - s
            if with_kl:
                logp_ref = F.log_softmax(ref_model.lm_head(h_ref[s:e]).float(), dim=-1)
                kl_sum += (logp_ref.exp() * (logp_ref - logp)).sum(-1).double().sum()
                agree_sum += (logp_ref.argmax(-1) == logp.argmax(-1)).double().sum()
                kl_tokens += e - s
                del logp_ref
            del logp

    nll = nll_sum.item() / n_tokens
    out = {"ppl": math.exp(nll), "nll": nll, "n_tokens": n_tokens, "kl": None, "top1_agree": None, "kl_tokens": 0}
    if kl_tokens:
        out["kl"] = kl_sum.item() / kl_tokens
        out["top1_agree"] = agree_sum.item() / kl_tokens
        out["kl_tokens"] = kl_tokens
    return out
