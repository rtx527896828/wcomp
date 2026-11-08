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

## Phase 1 outcome (2026-11-08, see NOTES.md)

The "4-bit RTN costs a few percent" expectation above was wrong by an order of magnitude for g128 symmetric RTN
(+29% / +11% / +27%); GPTQ brings it to +3..6%. Measured calibration-seed spread (GPTQ 3-bit, 0.8B): 4.4% PPL.
Differences below that are not read anywhere in this project without seed repeats.

## Phase 2 contract: structure probing (written 2026-11-08 before any Phase 2 code)

Goal: for each weight type, measure candidate structure, compare it with a **shuffled control**, and convert it into
an **estimated extra bits per parameter** that a compressor could gain. Nothing here changes Phase 1 code paths
that produced numbers (rtn.py grid, eval.py, data.py, bits.py); extensions are additive.

### Common definitions

- Weight types (from `wcomp.models.weight_type`): q, k, v, o, gate, up, down (both models); in_proj_qkv, in_proj_z,
  in_proj_ab, out_proj (Qwen3.5 DeltaNet layers). Embedding / lm_head are probed by P1 and P2 only (marginal and
  rank), never quantized in Phase 2.
- **Shuffled control**: `shuffle_within_tensor(W, seed)` = a seeded random permutation of all elements of W. It
  keeps the marginal distribution exactly and destroys every intra-tensor arrangement (rows, columns, groups,
  rank). A claim about rows/columns/rank/inter-layer structure passes the shuffle test only if the statistic on W
  is outside what the control gives; always report both numbers. For claims about the *marginal itself* (P1) the
  shuffle control is vacuous; the control there is an iid Gaussian with the same per-group std (same grid applied),
  and that is stated in the record.
- **Gain in bits/param** (the common currency). High-resolution quantization: distortion D is proportional to
  2^(-2b), so a structure that lets a quantizer reach the same MSE with a factor rho less error is worth
  `0.5 * log2(rho)` bits per parameter; an entropy coder is worth `b - H(index)` bits. Every estimate subtracts its
  own side-information cost (masks, ranks, rotation matrices, tables) at the stored width, and reports
  `gain_bits = gain_structure - overhead`, plus the same quantity computed on the control (`gain_bits_control`).
  The claimed structure value is `gain_bits - gain_bits_control`. These estimates are MSE proxies; where a direct
  PPL/KL measurement is cheap (P0, P3, P7) it is run and reported next to the estimate, and the estimate is
  **not** trusted where they disagree.
- Data: all activation statistics use C4 calibration windows (`c4_calibration`, seeds 0 and 2 for a split-half
  check). WikiText-2 test is only ever touched by `evaluate`. Weight-only probes use no data.
- Seeds: control permutations use seed 0; a second control with seed 1 is computed for P2 and P3 so the control's
  own spread is visible.
- Output: `results/probe/<model>/<probe>.json` with
  `{"model", "probe", "code_version", "timestamp", "params": {...}, "records": [...], "aggregate": {...}}`;
  each record `{"tensor": full module name, "layer": int, "type": weight_type, "stat": {...}, "control": {...},
  "gain_bits": float|null, "gain_bits_control": float|null, "note": str}`; `aggregate` maps weight type to the
  numel-weighted mean of every numeric stat and gain. Ablation rows go to `results/probe/<model>/ablation.jsonl`
  in the exact `run_baselines.py` row schema (same fields, `method` carrying the parametric name).

### Probes

P0. **Direct ablation (RTN, measured)**. Parametric method names, resolved by `wcomp.methods.resolve(name)`:
  - `rtn-sym-g{32,64,128}-{4,3}`: group-size sweep (scale cost 16/G bits).
  - `rtn-sym-g128-{4,3}@only=<type>`: quantize only that weight type, everything else bf16 (bits reported
    honestly: bits_blocks mixes 16-bit and b-bit tensors).
  - `rtn-sym-g128-{4,3}@except=<type>`: all block linears except that type.
  - `rtn-sym-g128-3@layer=<i>`: only the block linears of decoder layer i.
  - `rtn-sym-g128-{4,3}+keepcols=<p>`: input columns whose L2 norm is in the top p percent of their tensor are
    kept in bf16 (per tensor, p in {0.5, 1, 2}); cost = 16 bits per kept weight plus a K-bit column mask per
    tensor, and the remaining columns are quantized with the usual grid over groups that skip the kept columns
    (the group is still the 128 consecutive columns; the kept ones simply do not participate in amax and are
    restored afterwards).
  - `rtn-sym-g128-{4,3}+keepsuper=<path>`: the weights listed in a P7 JSON (at most 64 (tensor,row,col)
    triples) are restored to bf16 after quantization; cost 16 + ceil(log2(N*K)) bits each.
  - `gptq-sym-g128-{4,3}` is unchanged in behaviour; see the speed item below.
  Additivity check: sum of `@only` KLs over all types vs the all-types KL at the same bits.

P1. **Marginals and entropy** (weights only, plus embedding). Per tensor: std, excess kurtosis, fraction beyond
  4 sigma and 6 sigma, best of {Gaussian, Laplace, Student-t (nu fitted)} by NLL. Index entropy: for the Phase 1
  grid (sym g128 at b = 4, 3, 2) the empirical entropy H_b of the pooled index distribution per tensor, and
  H_b^gauss for the Gaussian control with matched per-group std and the same grid.
  `gain_bits = b - H_b - (2^b * 16) / numel` (one 16-bit frequency table per tensor); `gain_bits_control` likewise
  on the Gaussian control. bf16 lossless bound: joint entropy of the 16-bit word per tensor (65536 symbols) and
  the sign/exponent/mantissa field entropies; report `16 - H_word` as the lossless bits saved, with the Gaussian
  control's value.

P2. **Low rank**. Per tensor (f32 SVD on GPU): energy fraction at rank r for r/min(N,K) in {0.01, 0.05, 0.10,
  0.25}, rank for 90% and 99% energy, stable rank. Control: shuffled tensor (seeds 0 and 1). Gain for a
  low-rank-plus-quantized-residual code at rank r: residual energy fraction e_r gives `0.5*log2(1/e_r)` bits saved
  on the residual, overhead `16*r*(N+K)/(N*K)` for U, V in bf16; report the maximum over the r grid and the r that
  attains it, for W and for the control. Also P2b (needs activations): the same for `W @ L` where `H = L L^T` is the
  Cholesky factor of the calibration Hessian of that linear (activation-weighted rank), the quantity that matters
  for output error; control: W shuffled, same L.

P3. **Outlier rows/columns**. Per tensor: column (input) and row (output) L2 norms: max/median, count above 5x
  and 10x median, kurtosis of the norm vector. Cross-layer consistency for hidden-input linears: top-16 outlier
  input channels per layer; mean pairwise Jaccard across layers vs the control (independently shuffled channel
  ids). Gain: `rho = MSE(RTN-4 on W) / MSE(RTN-4 on W with top-p% columns excluded from the grid)` for p in
  {0.5, 1, 2}, `overhead = p/100 * 12 + 1/N` (bf16 columns and the mask), on W and on the control (where the
  selection effect alone gives rho slightly above 1; that is what the control subtracts). Direct check: P0
  `+keepcols`.

P4. **Inter-layer correlation**. For every weight type and every layer pair (i, j): cosine of the flattened
  tensors (raw); after orthogonal Procrustes on the output side (R = argmin ||W_j - R W_i||, via SVD of
  W_j W_i^T) and on the input side; for gate/up/down additionally after the best neuron permutation on the
  intermediate axis (scipy linear_sum_assignment on the cosine matrix, 3584 x 3584). Report residual energy
  e = ||W_j - align(W_i)||^2 / ||W_j||^2 for adjacent pairs and the best pair. Gain for differential coding:
  `0.5*log2(1/e) - overhead(R)` with R at bf16 (`16*N*N/(N*K)`) or the permutation (`log2(M!)/(N*K)`). Control:
  independently shuffled W_i and W_j, same alignment (alignment always fits something; that is the point).

P5. **Activation sparsity and neuron usage** (C4, 32 windows x 2048 tokens, seeds 0 and 2 as split halves).
  Hooks on the input of every block linear accumulate `sum x_j^2` and `count |x_j| > tau` per input channel j in
  f32 (tau = 0.05 * rms of the tensor's inputs). This is `diag(H)` of GPTQ, stored for P6 and P2b too. For
  down_proj (input = act_fn(gate) * up) and o_proj / out_proj (input = attention / DeltaNet output): Gini of
  per-channel energy, share of channels carrying 50% / 90% of the energy, fraction of channels active on < 1% of
  tokens. Split-half check: Spearman of the per-channel energy between the two C4 halves (must be > 0.9 to call
  the usage pattern stable; otherwise it is noise). Control for Gini: a permutation of the channel energies has
  the same Gini, so the control here is the split-half agreement, stated as such. Gain: see P6 (column
  granularity).

P6. **Sensitivity**. (a) Hessian-diagonal proxy, every block linear, all three models: the output-error weight of
  column j is `H_jj`; per tensor `s_t = sum_j H_jj * sum_i (W_ij - Q(W)_ij)^2` with Q = RTN-4 (this is the
  `||dW X||^2` the GPTQ objective minimizes). Bit-allocation gain at granularity g (units = tensors, 128-column
  groups, or single columns), allocating b_u = b + 0.5*log2(s_u / s_mean) under a fixed average: saving =
  `0.5*log2( mean_u(s_u / n_u) / geomean_u(s_u / n_u) )` bits/param with n_u = numel of the unit, numel-weighted;
  overhead = allocation side info (ceil(log2(#levels)) bits per unit / n_u). This saving is permutation
  invariant, so the shuffle control is replaced by a **prediction check**: rank-correlate `s_t` per tensor with
  the measured P0 `@only` KL per type/layer, and state the Spearman. (b) Fisher diagonal, 0.8B only: squared
  gradients of the NLL over 32 C4 windows of 1024 tokens (bf16 forward, f32 accumulators for block linears,
  no optimizer step, nothing written back; this is measurement, not training). Per tensor: total Fisher, Gini
  across elements, energy captured by the best rank-1 factorization F_ij ~ r_i c_j (if > 70%, per-row plus
  per-column side info suffices), and the allocation saving at element / row / column / tensor granularity.
  Control: Fisher shuffled within tensor (rank-1 fit energy of the control shows the floor).

P7. **Super weights** (Yu et al. 2024 recipe). One C4 window; record per layer the max |input| and max |output|
  of down_proj; the layers whose output spike (> 20x the median layer max) co-occurs with an input spike give
  candidates W[row = argmax out channel, col = argmax in channel]; also the global top-16 by |w| across block
  linears. Tests on 16 WikiText-2 windows (eval protocol, kl_windows 16): zero one candidate -> dPPL and KL;
  controls: zero the single largest-|w| weight of the same tensor that is not the candidate, and zero one random
  weight (seed 0). Record all in `superweight.json` (candidates with tensor, row, col, value, dPPL, KL).

### GPTQ speed (needed for Phase 3, validated here)

Form each window's `x^T x` in f32 and add it into the f64 accumulator (`hess += (x32.T @ x32).double()`).
Acceptance: `gptq-sym-g128-3` on the 0.8B, seed 0, reproduces PPL 24.45 within 0.5% and KL 0.3131 within 2%;
otherwise keep f64 and record the discrepancy. Expected: 2B GPTQ from ~23 min to a few minutes.

### Pre-registered hypotheses (copied verbatim into NOTES.md before the runs; results recorded against them)

- H-P0 (concentration). The 4-bit RTN loss is carried by few weight types: the three largest `@only` KLs sum to
  more than 60% of the all-types KL on both Qwen models; candidates down, in_proj_ab (tiny, feeds the decay
  gates), v/o. Also: `@only` KLs are roughly additive (sum within 50% of the joint KL). g32 vs g128 at 4 bits
  buys more than its 0.375 extra bits are worth at the GPTQ RD slope (that is, g32 RTN beats g128 RTN by a
  factor > 2 in KL); otherwise group size is not the lever.
- H-P1 (heavy tails, known). Most tensors have excess kurtosis > 1; index entropy coding of 4-bit RTN saves
  0.3-0.6 bits/param, of which the Gaussian control already explains most (so the *shape beyond Gaussian* is
  worth < 0.15 bits/param). bf16 lossless bound: 16 - H_word in 2-4 bits (exponent field is low-entropy).
- H-P2 (not low rank). For q/o/gate/up/down the gain after overhead is < 0.05 bits/param above the control at
  every r; k and v (GQA, N = 512) may show real rank deficiency. Activation-weighted (P2b) rank is lower than
  plain rank for down/o/out_proj. Expected negative result for method B.
- H-P3 (outlier channels, known). < 1% of hidden input channels have norms > 5x median, the same channels across
  most layers (Jaccard > 0.3 vs control < 0.05); keepcols=1% recovers > 25% of the RTN-4 KL on the 0.8B at a
  cost of about 0.13 bits/param. Reproduction of LLM.int8 / AWQ.
- H-P4 (no inter-layer redundancy). Residual energy after alignment > 0.9 for every type and pair, control
  within 0.03 of it; differential coding gain negative after overhead. Expected negative result for method C.
- H-P5 (uneven neuron usage). Gini of intermediate-channel energy > 0.5, 20% of channels carry > 60% of the
  energy, split-half Spearman > 0.9 (so it is a property of the model, not of the sample).
- H-P6 (sensitivity spread is the big lever). Per-tensor `s_t / n_t` spans > 2 orders of magnitude; the
  allocation saving is > 0.3 bits/param at tensor granularity and > 0.5 at column granularity; Spearman between
  `s_t` and the measured `@only` KL > 0.7 (so the proxy predicts). Fisher rank-1 fit > 70%.
- H-P7 (super weight: open). A super weight exists in an early down_proj of Qwen3.5-0.8B (zeroing it: dPPL > 100%,
  the magnitude control < 2%). Genuinely uncertain for a DeltaNet-heavy model; MiniCPM5 (Llama) is the positive
  control where the literature says it exists.

### Phase 2 run list (sequential on the one GPU)

qwen3.5-0.8b: probe_weights (P1-P4, all block linears + embedding); probe_acts (P5, P6a, P6b, P7, P2b); ablation:
  rtn-sym-g{32,64}-{4,3}; @only for every type at 4 and 3 bits; @except for every type at 4 bits; @layer=i at 3
  bits for all 24 layers; +keepcols={0.5,1,2} at 4 and 3 bits; +keepsuper at 4 and 3 bits (if P7 finds any);
  GPTQ speed validation (gptq-sym-g128-3 seed 0) then gptq-sym-g128-4 seed 1, had-fp16, had-gptq-sym-g128-3.
qwen3.5-2b: probe_weights; probe_acts (P5, P6a, P7, P2b; no Fisher); ablation: @only every type at 4 bits;
  @layer=i at 3 bits; +keepcols=1 at 4 bits; +keepsuper at 4 bits; gptq-sym-g128-4 seed 1.
minicpm5-2b: probe_weights; probe_acts (P5, P6a, P7); ablation: @only every type at 4 bits; +keepcols=1 at 4
  bits; +keepsuper at 4 bits.
Everything appends; rerunning a finished item is never needed. Budget: stop and report if the total passes 4 h.

## Phase 2 outcome (2026-11-08, see NOTES.md "Phase 2")

Measured levers: type-level bit allocation from measured damage (0.15-0.24 bits/param, RTN, three models),
Hadamard + GPTQ at 3 bits (0.23-0.26 bits/param, 0.8B, one seed), entropy coding of indices (~0.5 bits/param,
0.44 of it generic). Refuted or negative: low rank (B), inter-layer coding (C), outlier-column protection by weight
norm, super-weight protection, usage-frequency (F), Hessian-diagonal sensitivity proxy (anti-correlated with the
measured damage; the Fisher diagonal predicts it, +0.96, post hoc). Noise: 4.4% PPL / 5% KL seed spread.

## Phase 3 contract: methods (written 2026-11-08 before any Phase 3 code)

Letters are the task statement's: A allocation, B low rank, C inter-layer, D rotation, E codebook + entropy coding,
F usage-aware, G own ideas. B, C, F are dropped on Phase 2 evidence and get one line each in REPORT.md.

### Rules that bind every Phase 3 row

- No training, no distillation, no gradient step on model weights. Allowed: GPTQ error feedback, closed-form
  codebook fitting on weights (Lloyd-Max iterations), measuring gradients (Fisher) on C4, water-filling on
  calibration-measured damage. A gradient-trained rotation ("learned rotation" of the task's D) is therefore NOT
  run; D is restricted to zero-bit rotations and one honestly-costed data-dependent rotation (see D).
- Calibration / allocation signals come from C4 only (`c4_calibration`, seed = run seed; a disjoint seed-2 set for
  damage measurement). WikiText-2 test is touched only by `evaluate` on the final model. Phase 2's `@only` KLs were
  measured on WikiText and are therefore NOT used to set bits; they are re-measured on C4 (A1 below).
- GPTQ in Phase 3 uses the f32 per-window Hessian product (`--hessian-f32`, explicit on every run; the env override
  `WCOMP_HESSIAN_F32` is removed). The Phase 3 baseline is therefore re-run with the same path and two seeds, so
  every comparison is like-for-like. Phase 1 rows are not compared against directly.
- Every row records three bit figures for its block linears: `bits_blocks` (fixed-length indices, as before),
  `bits_blocks_huff` (indices coded with a per-tensor Huffman code built from the tensor's exact symbol counts,
  code-length table counted at `b * 2^b` bits per tensor; the best of single-symbol and pair-symbol (alphabet
  2^(2b), table `2b * 2^(2b)` bits) codes per tensor, choice costs 1 bit per tensor), and `bits_blocks_shannon`
  (the entropy bound, for the gap). All side information (scales, zero points, codebooks, allocation tables, masks,
  rotation seeds = 0 bits, stored matrices at bf16) is in all three. `bits_all` as before, on the fixed figure.
  Indices are recovered exactly from (W_hat, scale[, codebook]); the code asserts `W_hat == dequant(idx)`.
- Reference curve per model: `gptq32-sym-g128-{4,3,2}` seeds 0 and 1 (MiniCPM: seed 0, plus seed 1 at 3 bits).
  Between the seed-mean points, log(KL) and log(PPL - PPL_fp16) are interpolated linearly in `bits_blocks`.
  A method **beats the baseline** at its bits if its KL is below the interpolated curve by more than 5% and its
  dPPL is below by more than 4.4%, on both seeds where two exist. **Bits saved** = the curve's bits at the method's
  KL minus the method's bits (reported for the fixed and the Huffman figure; the curve itself is read at fixed
  bits, and a second curve with the baseline's own Huffman figure is drawn so coding gain is never double counted).
- Shuffle controls: A and G permute the computed allocation across units (two permutations, seeds 1 and 2,
  rebalanced to the same `bits_blocks` within 0.02); a claimed allocation gain needs the real assignment to beat
  both permutations by > 5% KL. E's entropy gain is permutation invariant by construction (stated; its control is
  the Gaussian index entropy of P1, the "generic" part). D's control is the random orthogonal rotation.
- Seeds: 0.8B key rows at seeds 0 and 1 (calibration and rotation seed = run seed); 2B seeds 0 and 1 for the
  baseline, A and the combination; MiniCPM seed 0. Output: `results/phase3/<model>/runs.jsonl` (run_baselines row
  schema plus `bits_blocks_huff`, `bits_blocks_shannon`, `alloc` (per-type or per-tensor bits as a short string or
  path), `rotation`), plots `results/rd3_<model>.png`, table `results/phase3/summary.md`.
- Budget: ~2.5 min per 0.8B GPTQ row, ~6 min per 2B row, ~8 min per MiniCPM row; the run list below is ~5 h. If a
  model's list passes 3 h, finish the current row and report.

### Methods

**A. Sensitivity-driven bit allocation.**
A1 damage measurement on C4 (calibration-only): `rtn-sym-g128-4@only=<type>` for every type, evaluated with
  `evaluate(model, ref, c4_ids_seed2[:32], kl_windows=32)` (PPL and KL on the 32 seed-2 C4 windows; never
  WikiText), all three models; and per tensor where cheap (0.8B: 186 rows at ~8 s each on 32 windows = 25 min; do
  it). Stored in `results/phase3/<model>/damage_c4.jsonl`. d_u = KL_u / n_u (nats per parameter).
A2 allocation: minimize sum_u n_u d_u 2^(-2 (b_u - 4)) subject to the numel-weighted mean of b_u = target, b_u in
  {2, 3, 4, 5, 6, 8}, by greedy marginal gain (exact for this convex objective); side information 3 bits per unit.
  Units: weight types (11 / 7) from A1-type, or tensors (186 / ...) from A1-tensor (0.8B) or from the Fisher
  per-tensor total (`fisher.json`, 0.8B; for the 2B compute per-tensor Fisher totals only, with 512-token windows
  and scalar accumulators, if it fits in memory; else types only).
A3 rows: `alloc-type-gptq32-avg{2.5,3,3.5,4}` (0.8B both seeds, 2B both seeds, MiniCPM avg 3 and 3.5 seed 0);
  `alloc-tensor-gptq32-avg{3,3.5}` (0.8B; 2B if Fisher totals exist); `alloc-type-rtn-avg{3,3.5}` (0.8B, the RTN
  gate that connects to Phase 2); shuffle controls `alloc-type-shuf{1,2}-gptq32-avg3` (0.8B seed 0, 2B seed 0).
  "avg" is the target mean index width; the row reports the realized `bits_blocks`.

**D. Rotation (zero-bit).** `had-gptq32-sym-g128-{4,3,2}` (0.8B both seeds, 2B both seeds, MiniCPM 3 seed 0);
  `rnd-gptq32-sym-g128-3` (seeded random orthogonal per input width, 0 bits; 0.8B seeds 0 and 1);
  `hadsign-gptq32-sym-g128-3` (Hadamard times a seeded random sign diagonal; 0.8B seed 0). One honestly costed
  data-dependent rotation as an upper-bound probe: `eig-gptq32-sym-g128-3` (0.8B seed 0): per block linear the
  input rotation R = eigenvectors of its calibration Hessian, R stored at bf16 and counted (16 K / N bits/param
  per tensor, shared by linears with the same input where applicable); expected to lose on bits by a wide margin;
  it is run once to put the number on the table and to show what a free data-dependent rotation would be worth.

**E. Non-uniform codebook + entropy coding.**
E1 `+huff` accounting on every row (rule above); the baseline's own Huffman figure defines the coded reference curve.
E2 `lloyd-gptq32-g128-{4,3,2}`: per-tensor codebook of 2^b levels fitted by Lloyd-Max (k-means in 1-D, 50
  iterations, init at Gaussian quantiles) on the group-normalized weights w / s_g with s_g = group RMS (f16, 16/128
  bits); quantization in GPTQ = nearest codebook level; codebook cost 2^b * 16 bits per tensor. 0.8B both seeds,
  2B seed 0 (3 bits), MiniCPM seed 0 (3 bits).
E3 `nf-gptq32-g128-{4,3}`: fixed codebook = Gaussian quantiles (NF4-style, 0 table bits), same normalization; 0.8B
  seed 0. Shows how much of E2 a fixed codebook captures.
E4 RD-bound table: for every Phase 3 row, per weight type, the MSE of the group-normalized weights versus the
  Gaussian rate-distortion bound R(D) = 0.5 log2(1 / D) at the row's three bit figures: "bits above the 6.02 dB/bit
  bound" (fixed, Huffman, Shannon). Written to `results/phase3/<model>/rd_bound.jsonl` by the runner.

**G. Own idea: within-tensor row allocation by Fisher row energy.** Gate first on RTN (0.8B): rows of each tensor
  get b-1 / b / b+1 bits by water-filling on the Fisher row energy (`fisher.json` row sums r_i, d_i = r_i / K)
  at a fixed per-tensor mean of 4 bits; 2 bits side information per row. `rowalloc-fisher-rtn-avg4` vs
  `rtn-sym-g128-4` and the shuffled-row controls `rowalloc-shuf{1,2}-rtn-avg4` (seed 0). Gate: beats uniform by
  > 10% KL and both controls by > 5%. If passed: `rowalloc-fisher-gptq32-avg{4,3}` (0.8B both seeds) and, combined
  with A, in the combination rows. If failed: recorded as a negative result, no GPTQ rows.
  Second own observation to test cheaply (0.8B, RTN, C4 eval): is damage per parameter ~ 1 / n_t because small
  tensors have larger output scale? Correlate d_t with ||W_t X||^2 / n_t and with the output RMS; one table.

**Combination.** `combo-gptq32-avg{2.5,3,3.5}` = Hadamard (if D beats at 3 bits) + type-level allocation (A) +
  Lloyd codebook (only if E2 beat the uniform grid at 3 bits by > 5% KL on both seeds) + GPTQ + Huffman accounting;
  G rows included only if its gate passed. 0.8B seeds 0 and 1; 2B seeds 0 and 1; MiniCPM avg 3 seed 0. The
  combination's components are decided by the rules above after the single-method rows exist, and the decision is
  written to NOTES.md before the combo rows run.

### Pre-registered hypotheses (copied verbatim into NOTES.md before the runs)

- H3-A. Type-level allocation from C4-measured damage beats uniform GPTQ by 15-30% KL at equal `bits_blocks` at
  avg 3.5 and 3 on both Qwen models, both seeds; both shuffled allocations are no better than uniform (within 5%).
  Tensor-level allocation adds < 10% KL on top of type-level. The C4-measured per-type damage ranks the types the
  way the Phase 2 WikiText measurement did (Spearman > 0.9), so the allocation does not depend on the eval set.
- H3-D. Hadamard + GPTQ beats GPTQ by > 20% KL at 3 bits on all three models; at 4 bits the gain is < 10% (noise
  edge); at 2 bits it is the largest (> 40%). Random orthogonal equals Hadamard within 5% KL (the gain is generic
  incoherence, not model-specific structure); the eigenbasis rotation costs > 1 bit/param stored and loses.
- H3-E. Huffman coding of GPTQ indices saves 0.40-0.50 bits/param at 4 bits, 0.25-0.40 at 3, < 0.20 at 2 with
  single-symbol codes (pair codes recover part of the 2-bit loss); the gap to the Shannon figure is < 0.05 bits at
  4 bits. Lloyd-Max lowers KL by 10-25% at 3-4 bits for < 0.01 bits of table; the fixed Gaussian codebook captures
  at least two thirds of that. All coders sit 1.0-1.5 bits above the Gaussian RD bound at equal MSE at 4 bits, and
  the gap shrinks with fewer bits.
- H3-G. Uncertain (the only tested proxy failed in Phase 2). Prior: the Fisher row split beats uniform RTN-4 by
  10-20% KL at equal bits and the shuffled-row controls do not; if it fails, within-tensor allocation is dropped.
- H3-combo. At avg 3.0 (fixed `bits_blocks` ~3.1, Huffman ~2.7-2.8) the combination reaches the KL of uniform
  GPTQ-4 (fixed 4.125) on the 0.8B and the 2B, i.e. about 1.0-1.3 bits/param saved at equal quality against the
  Phase 1 best coder, of which ~0.4 is entropy coding, ~0.25 rotation, ~0.2 allocation, the rest interaction.
  At avg 2.5 the combination stays below +50% PPL (Phase 1's best 2-bit row was +655%).

### Phase 3 run list (sequential on the one GPU, from `D:\repos\wcomp`)

qwen3.5-0.8b: A1 damage (types and tensors, C4); baseline gptq32 4/3/2 seeds 0,1; D rows; E2/E3 rows; A3 rows
  (incl. RTN gate and shuffles); G gate (RTN) and, if passed, G GPTQ rows; combo decision written to NOTES.md;
  combo rows seeds 0,1; plots and summary.
qwen3.5-2b: A1 damage (types); Fisher per-tensor totals if feasible; baseline 4/3/2 seeds 0,1; had 4/3/2 seeds 0,1;
  lloyd 3 seed 0; alloc-type avg 2.5/3/3.5/4 seeds 0,1 (+ tensor-level if Fisher); shuffles avg 3 seed 0; combo
  avg 2.5/3/3.5 seeds 0,1.
minicpm5-2b: A1 damage (types); baseline 4/3 seed 0 and 3 seed 1; had 3; lloyd 3; alloc-type avg 3 and 3.5; combo
  avg 3.
