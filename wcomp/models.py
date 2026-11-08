"""Model registry, loading and block-linear enumeration (DESIGN.md, module contracts)."""
from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerBase

from wcomp.data import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]

MODELS: dict[str, str] = {
    "qwen3.5-0.8b": str(ROOT / "models" / "Qwen3.5-0.8B"),
    "qwen3.5-2b": r"D:\repos\webyIQ\models\hf\Qwen3.5-2B",
    "minicpm5-2b": r"D:\repos\webyIQ\models\hf\MiniCPM5-2B",
}

# Linear module name (last component) -> weight type.
_WEIGHT_TYPES: dict[str, str] = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
    "in_proj_qkv": "in_proj_qkv",
    "in_proj_z": "in_proj_z",
    "in_proj_a": "in_proj_ab",
    "in_proj_b": "in_proj_ab",
    "out_proj": "out_proj",
}

# Order in which the projections are applied inside one decoder layer (from the HF forward code;
# nn.Module registration order differs: Qwen3.5's DeltaNet registers out_proj first).
_FORWARD_ORDER: dict[str, int] = {
    name: i
    for i, name in enumerate(
        ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
         "q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj"]
    )
}

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def load_model(tag: str, device: str = "cuda") -> tuple[nn.Module, PreTrainedTokenizerBase]:
    """Load MODELS[tag] in bf16 eval mode on device, plus its tokenizer.

    device="meta" builds the architecture from the config without weights (for name / shape checks).
    """
    path = MODELS[tag]
    if device == "meta":
        config = AutoConfig.from_pretrained(path)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, load_tokenizer(path)


def layer_index(name: str) -> int:
    """Decoder layer index of a parameter / module name such as 'model.layers.7.mlp.up_proj'."""
    m = _LAYER_RE.search(name)
    if m is None:
        raise ValueError(f"no layer index in {name!r}")
    return int(m.group(1))


def weight_type(name: str) -> str:
    """Weight type of a block linear name (Qwen3.5 and MiniCPM5 / Llama names)."""
    leaf = name.split(".")[-1]
    if leaf == "weight" and "." in name:
        leaf = name.split(".")[-2]
    if leaf not in _WEIGHT_TYPES:
        raise ValueError(f"unknown block linear {name!r}")
    return _WEIGHT_TYPES[leaf]


def block_linears(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Every nn.Linear under model.layers.*, in forward order (layer, then call order inside the layer)."""
    found = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and _LAYER_RE.search(name) is not None
    ]
    for name, _ in found:
        weight_type(name)  # raise on an unexpected linear instead of silently misordering it
    found.sort(key=lambda nm: (layer_index(nm[0]), _FORWARD_ORDER[nm[0].split(".")[-1]]))
    return found
