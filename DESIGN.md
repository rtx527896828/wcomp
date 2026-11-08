# wcomp: design and contracts

Lossy compression of LLM weights, used as a probe for structure in the weights. Phase 1 builds the
evaluation framework and the baselines. Everything later is measured with the code described here, so
these contracts are binding: a module that deviates must say so in NOTES.md.

## Environment (checked 2026-11-07)

- Windows 11, RTX 5080 16 GB (about 10 GB free: other applications hold ~5.5 GB), 61.6 GB RAM, 8-core CPU.
- Project `D:\repos\wcomp`, Python 3.12 via `uv`; run everything as `uv run python ...` from that directory.
  torch 2.11.0+cu128 (CUDA works, sm_120), transformers 5.22.2, numpy, scipy, matplotlib, pyarrow, safetensors, accelerate.
- One GPU: experiment runs are sequential. Unit tests on synthetic tensors run on CPU/GPU freely.
- Models (local safetensors, bf16):
  - `models/Qwen3.5-0.8B` (primary, fast iteration): `Qwen3_5ForCausalLM`, 752 M params, 24 layers (18 Gated
    DeltaNet "linear_attention" + 6 full attention, every 4th layer), hidden 1024, intermediate 3584, vocab
    248320, tied embedding / lm_head. The `Qwen3.5-0.6B` named in the task does not exist on the Hub; 0.8B is
    the smallest Qwen3.5 and is used instead.
  - `D:\repos\webyIQ\models\hf\Qwen3.5-2B` (same architecture, hidden 2048, intermediate 6144).
  - `D:\repos\webyIQ\models\hf\MiniCPM5-2B` (plain Llama architecture, 42 layers, hidden 2048, intermediate
    6144, vocab 130560, untied lm_head). Different architecture family: use it to check that conclusions are
    not DeltaNet-specific.
  - Load with `AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")`.
    Qwen3.5 falls back to reference PyTorch kernels for conv1d / gated delta rule (fla and causal_conv1d are
    not installed): the first forward pass takes ~15 s (warm-up), then 2048 tokens take ~0.2 s on the 0.8B.
    The forward with `labels=` materializes f32 logits over the 248K vocabulary (2 GB per 2048 tokens):
    evaluation code must compute logits/log-probs in position chunks, never the full [2048, V] f32 tensor
    twice.
- Linear modules per Qwen3.5 layer (names relative to `model.layers.N.`):
  - DeltaNet layers: `linear_attn.in_proj_qkv` [6144, 1024], `linear_attn.in_proj_z` [2048, 1024],
    `linear_attn.in_proj_b` [16, 1024], `linear_attn.in_proj_a` [16, 1024], `linear_attn.out_proj` [1024, 2048].
  - Attention layers: `self_attn.q_proj` [4096, 1024] (q and its output gate, interleaved per head),
    `self_attn.k_proj` [512, 1024], `self_attn.v_proj` [512, 1024], `self_attn.o_proj` [1024, 2048].
  - MLP (all layers): `mlp.gate_proj` [3584, 1024], `mlp.up_proj` [3584, 1024], `mlp.down_proj` [1024, 3584].
  - Outside layers: `model.embed_tokens` [248320, 1024] (tied with `lm_head`), norms, DeltaNet conv1d
    weights, `A_log`, `dt_bias`.
  - MiniCPM5: `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `model.embed_tokens`, `lm_head`.

## Fixed protocol decisions

1. **Evaluation data**: WikiText-2 raw, *test* split, from
   `https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet`,
   text = `"".join(rows)` (the rows already end with newlines), tokenized once without special tokens,
   split into consecutive non-overlapping windows of 2048 tokens (drop the remainder). Each window is an
   independent sequence (no carried context). Cached as `data/<model>_wikitext2_test_2048.pt`.
2. **Perplexity**: `exp(mean over all windows and positions 1..2047 of -log p(x_t | x_<t))`; the first token
   of each window is not predicted. Computed in f32 from bf16 logits. Report with 4 significant digits.
3. **KL divergence**: `KL(P_ref || P_q)` per predicted position, averaged over positions, on the first
   32 windows (65,504 predicted tokens), both models evaluated on the same input ids. Also report top-1
   agreement (fraction of positions where argmax matches). Use `log_softmax` in f32, chunked over positions.
   Reference logits may be cached on disk in bf16 for the KL windows only if there is room; recomputing
   with a resident reference model is also fine.
4. **Calibration data** (for GPTQ and any data-dependent method): C4 *validation*, shard
   `https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz`.
   Documents are sampled with a fixed seed, tokenized individually, and windows of 2048 tokens are cut from
   documents that are long enough (documents shorter than 2048 tokens are skipped), 128 windows by default.
   Cached as `data/<model>_c4_calib_<n>x2048_seed<s>.pt`. Never use WikiText for calibration.
5. **What is quantized**: every `nn.Linear` inside `model.layers.*` (attention, MLP and DeltaNet projections,
   including the tiny `in_proj_a` / `in_proj_b`). Embedding / lm_head, norms, conv1d, `A_log`, `dt_bias` are
   kept in bf16 in Phase 1 (they are identical across all methods, so they do not affect the comparison).
   Both scopes are always reported, see bits below. The MTP head is not loaded by `AutoModelForCausalLM`.
6. **Effective bits**: two-part code. For each quantized tensor, count the index bits plus all side
   information at its stored width: f16 scales (16 bits each), zero points at their width, codebooks,
   permutation indices, low-rank factors, rotation seeds (0 bits for a deterministic Hadamard, 0 bits for a
   seeded random matrix, but any *stored* matrix counts in full). Report:
   - `bits_blocks` = total bits of all quantized block linears / their parameter count (the x-axis of the
     rate-distortion plots);
   - `bits_all` = (bits of quantized linears + 16 × every other parameter) / total parameters.
   Example: RTN symmetric, 4 bit, group 128, f16 scale: 4 + 16/128 = 4.125 bits_blocks.
7. **Seeds**: everything seeded; default seed 0; GPTQ (calibration sampling) also run with seed 1 at 3 bits.
8. **No training / distillation**: only closed-form or local error compensation (GPTQ style).

## Module contracts (package `wcomp/`, importable from the project root)

```
wcomp/data.py
  load_tokenizer(model_dir) -> tokenizer
  wikitext2_test(tokenizer, model_tag, seqlen=2048) -> LongTensor [n_windows, seqlen]   (cached)
  c4_calibration(tokenizer, model_tag, n=128, seqlen=2048, seed=0) -> LongTensor [n, seqlen]  (cached)

wcomp/models.py
  MODELS = {"qwen3.5-0.8b": path, "qwen3.5-2b": path, "minicpm5-2b": path}
  load_model(tag, device="cuda") -> (model: nn.Module in bf16 eval mode, tokenizer)
  block_linears(model) -> list[(name, nn.Linear)]   # every nn.Linear under model.layers.*, in forward order
  weight_type(name) -> one of "q","k","v","o","gate","up","down","in_proj_qkv","in_proj_z","in_proj_ab","out_proj"
  layer_index(name) -> int

wcomp/bits.py
  @dataclass TensorBits: name, numel, index_bits, aux_bits, note
  class BitAccount:  add(name, numel, index_bits, aux_bits, note=""); bits_blocks(); bits_all(model); summary() -> dict
  # bits_all needs the model's total parameter count and the set of quantized tensor names.

wcomp/eval.py
  evaluate(model, ref_model, eval_ids, kl_windows=32, batch_size=1) -> dict:
     {"ppl": float, "nll": float, "n_tokens": int, "kl": float, "top1_agree": float, "kl_tokens": int}
  # ref_model may be None (then only ppl). Models stay where they are; eval_ids is moved to the model device.
  # Must run the 0.8B over all WikiText-2 windows in well under 2 minutes after warm-up.

wcomp/quant/rtn.py
  quantize_rtn(W: Tensor[N,K] f32, bits, group=128, sym=True) -> (W_hat f32, info dict)
  # sym: scale = amax/|qmin| per group, q in [-(2^(b-1)), 2^(b-1)-1]; asym: scale/zero from min/max,
  # q in [0, 2^b-1], zero stored as a b-bit integer. info = {"index_bits": bits*N*K, "aux_bits": ..., "note": ...}
  apply_rtn(model, bits, group, sym) -> BitAccount   # replaces each block linear's weight in place (bf16 of W_hat)

wcomp/quant/rotate.py
  hadamard(n) -> Tensor[n,n] orthonormal, built as H_m ⊗ H_{2^k} with m in {1, 12, 20, 28, 40, ...} from a
     table; raises if n has no such factorization (then the caller uses random_orthogonal(n, seed) and records it).
  random_orthogonal(n, seed) -> Tensor[n,n]
  rotate_inputs(model, which=all block linears) -> dict[name -> R]
  # exact, function preserving: W <- W @ R, and the module's forward computes W (R^T x); implement by
  # registering a forward_pre_hook that multiplies the input by R^T (keep R in bf16 on the device and verify
  # on a test that the PPL with rotation and no quantization equals the baseline within 0.1%).

wcomp/quant/gptq.py
  gptq_model(model, calib_ids, bits, group=128, sym=True, block=128, damp=0.01, act_order=False) -> BitAccount
  # Sequential, layer by layer: capture layer inputs with hooks, accumulate H = mean(x x^T) in f64 per
  # linear over the calibration windows, quantize all linears of the layer with GPTQ (Frantar et al. 2022:
  # error feedback through the inverse Cholesky factor, group grids computed when the group starts, same
  # quantization grid as quantize_rtn so the bits are identical), then run the quantized layer to produce the
  # next layer's inputs. For the DeltaNet layers hook the sub-modules' inputs, not the layer's input, so each
  # linear's Hessian is over its own input. Keep calibration activations on the GPU in bf16 if they fit
  # (128 × 2048 × 1024 × 2 bytes = 512 MB for the 0.8B), else on the CPU.

wcomp/methods.py
  METHODS: dict name -> callable(model, ctx) -> BitAccount, where ctx has calib_ids, seed, tokenizer.
  Names used in Phase 1: "fp16", "rtn-sym-g128-{8,4,3,2}", "rtn-asym-g128-{4,3,2}", "had-rtn-sym-g128-{4,3,2}",
  "gptq-sym-g128-{4,3,2}" (seed in ctx), "had-gptq-sym-g128-{3}" if time permits.

scripts/run_baselines.py  --model TAG --methods a,b,c [--seed 0]
  loads the model once per method (fresh weights), applies the method, evaluates against a resident
  reference copy (bf16) and appends one JSON line per run to results/<model>/runs.jsonl:
  {"model", "method", "seed", "bits_blocks", "bits_all", "ppl", "kl", "top1_agree", "n_tokens",
   "kl_tokens", "time_s", "note", "git_or_code_version", "timestamp"}
scripts/plot_rd.py  --model TAG  -> results/rd_<model>.png (x: bits_blocks, y: ppl - ppl_fp16 on a log
  scale, and a second panel with KL), one line per method family, points labelled with the bit width.
```

Tests live in `tests/` and run with `uv run python -m pytest -q` (add pytest to the dev dependencies).

## Phase 1 experiment list

Qwen3.5-0.8B: fp16; rtn-sym 8/4/3/2; rtn-asym 4/3/2; had-rtn-sym 4/3/2; gptq-sym 4/3/2 (3 with seeds 0 and 1).
Qwen3.5-2B: fp16; rtn-sym 4/3/2; gptq-sym 4/3. MiniCPM5-2B: fp16; rtn-sym 4/3; gptq-sym 4/3.

Sanity expectations (to catch bugs, not to tune towards): fp16 PPL of Qwen3.5-0.8B on this protocol is
unknown but should be in the 15–35 range (the 2B measured 17.1 on 8×512-token windows); rtn-sym-g128-4 should
cost a few percent, 3-bit RTN noticeably more (tens of percent on a small model), 2-bit symmetric RTN
should be near-destroyed (PPL in the hundreds or thousands); GPTQ should beat RTN clearly at 3 and 2 bits;
Hadamard should help RTN at 3/2 bits; `kl` must be 0 and `top1_agree` 1.0 for fp16 vs itself.
