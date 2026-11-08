"""Environment smoke test: load Qwen3.5-0.8B on CUDA, time a 2048-token forward pass."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = "models/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(p)
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map="cuda").eval()
print(type(m).__name__, f"load {time.time()-t0:.1f}s", sum(q.numel() for q in m.parameters()) / 1e6, "M params")
lin = [(n, tuple(mod.weight.shape)) for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)]
print(len(lin), "linears; layer 0:", [x for x in lin if ".layers.0." in x[0]])
print("layer 3:", [x for x in lin if ".layers.3." in x[0]])
print("last:", lin[-3:])
emb = [(n, tuple(q.shape)) for n, q in m.named_parameters() if q.dim() == 2 and "layers" not in n]
print("non-layer 2d:", emb)
ids = tok("The quick brown fox jumps over the lazy dog. " * 300, return_tensors="pt").input_ids[:, :2048].cuda()
with torch.no_grad():
    for i in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        out = m(ids, labels=ids)
        torch.cuda.synchronize()
        print(f"fwd {ids.shape[1]} tok: {time.time()-t0:.2f}s loss {out.loss.item():.3f}")
print("mem", torch.cuda.max_memory_allocated() / 2**30, "GiB")
