# NOTES

Running lab notebook. Every experiment: hypothesis, success criterion, result (also failures). Dates are
2026. Protocol and contracts: DESIGN.md.

## 2026-11-07 Setup

- Environment: RTX 5080 16 GB (≈10 GB free), 61.6 GB RAM, torch 2.11.0+cu128, transformers 5.22.2 (uv, Python 3.12).
- Models: Qwen3.5-0.8B (primary; the task said 0.6B, which does not exist), Qwen3.5-2B, MiniCPM5-2B (Llama
  architecture, as an architecture-independent check).
- Qwen3.5 runs on reference PyTorch kernels (no fla / causal_conv1d on Windows): 0.2 s per 2048-token forward on
  the 0.8B after a 15 s warm-up.

## Phase 1: evaluation framework and baselines

Build 2026-11-07 (afternoon to 21:59 local), full sweeps 2026-11-07 22:17 to 2026-11-08 00:59 local (runs.jsonl
timestamps are UTC, +5 h), audit 2026-11-08. All numbers below come from `results/<model>/runs.jsonl`; nothing was
hand-copied from logs.

### 2026-11-07 What was built

- `wcomp/{data,models,bits,eval,methods}.py`, `wcomp/quant/{rtn,rotate,gptq}.py`, `scripts/run_baselines.py`,
  `scripts/plot_rd.py`, `tests/` (five files), all per the DESIGN.md contracts; no deviations from the contract.
  pytest added to the `dev` dependency group (`pyproject.toml`, `uv.lock`), the only tracked changes.
- Code version that produced every Phase 1 number: `0afd0d8+dirty/d2cfe18180` (sha256 over `wcomp/**/*.py`,
  `scripts/*.py`, `uv.lock`); `uv run python -m pytest -q` on that hash: 122 passed (123 s).
- Memory: the reference copy keeps only its final hidden states + lm_head resident during KL, so even MiniCPM5-2B
  (two bf16 copies, 5.03 GB each) fit in the ~10.6 GB free; the "reference on CPU" fallback was never needed.
- Rotation (`had-*`): the stored bf16 Hadamard is only approximately orthonormal, so the weight is folded with the
  exact inverse of the *stored* matrix (f64). Rotation alone still costs about +0.07% PPL on the 0.8B (bf16
  rounding of `x R` and of the folded weight); inside the 0.1% contract but systematic. A `had-fp16` row exists
  for this but was not run in the sweeps (no had-* result is near 0.07%, so it does not matter in Phase 1).

### 2026-11-07 Smoke runs (what was rerun before the sweeps)

Smoke protocol `--limit-windows 8 --kl-windows 4|8 --calib-n 16`, rows kept apart in `results/smoke/` and marked
in the note field. Four smoke rounds on the 0.8B (`results/smoke/qwen3.5-0.8b/runs.jsonl`):

- Round 1 (21:00 local, no code hash yet): fp16 14.947, rtn-sym-4 19.008, gptq-sym-4 **15.948**.
- Round 2 (`0060cbb3c2`, 21:35, after a GPTQ fix in the build step; the module docstring now pins the grid to be
  literally `quantize_rtn`'s `group_grid`, f16-rounded scale included, and `gptq_quantize(W, H=I)` must equal
  RTN exactly, which is tested): gptq-sym-4 **16.248**; fp16 and rtn unchanged. The +0.3 PPL shift on 8 windows
  is the only metric that moved during the build. Its exact cause is not written in the logs I have; treat the
  round-1 value as pre-fix and discard it.
- Rounds 3 (`0f0e1b59fd`) and 4 (`d2cfe18180`, 22:13, the final hash, kl-windows 8): fp16 14.946913,
  rtn-sym-4 19.007506, gptq-sym-4 16.248178 reproduced to all printed digits, so the 21:57-21:59 edits
  (run_baselines, plot_rd, bits, methods, rtn, tests) moved no metric.
- Provenance gap (audit finding, low): the last pytest log before the sweeps (`results/pytest_round2.log`,
  21:41, 115 passed) predates those final edits. Closed after the fact by running the suite on `d2cfe18180`
  (122 passed). Rule from now on: log a pytest run after the last edit and before launching a sweep.

### 2026-11-07/08 Protocol as actually run

- Eval: WikiText-2 raw test, 2048-token windows, all of them: 145 windows / 296,815 predicted tokens for both
  Qwen3.5 tokenizers, 139 windows / 284,533 tokens for MiniCPM5. First 4096 cached tokens are byte-identical to
  webyIQ's independently produced WikiText windows (same join and tokenization).
- KL: first 32 windows = 65,504 predicted tokens on every row, `KL(P_ref || P_q)` in f32, chunked.
- Calibration: C4 validation, 128 x 2048 tokens, seed 0 (seed 1 for the 3-bit GPTQ repeat on the 0.8B);
  caches `data/<model>_c4_calib_128x2048_seed{0,1}.pt`.
- All 26 runs on the DESIGN.md experiment list exited 0, one process per sweep, sequential on one GPU; no smoke
  markers in any full row. The optional `had-gptq-sym-g128-3` was not run. Total 161 min wall, 160 min summed
  `time_s`, of which GPTQ 142 min (9 runs). Logs `results/<model>/run.log` end in `EXIT_SEED0 0`
  (+ `EXIT_SEED1 0` on the 0.8B) and `SWEEP_DONE`.

### Hypotheses, criteria, results

**E1. Framework sanity.** Hypothesis: the evaluator and bit accounting are correct. Criteria (DESIGN.md):
fp16 vs itself gives kl = 0 and top1 = 1.0 exactly; 0.8B fp16 PPL in 15-35; bits_blocks of rtn-sym-g128-4 = 4.125.
Result: met on all three models (kl 0.0, top1 1.0; the reference is a separate model copy, and rtn-8 against the
same reference gives a nonzero 0.0015, so the zero is not trivial). 0.8B fp16 PPL 17.57. All 26 bits_blocks /
bits_all values recomputed by hand from meta models match to 1e-9 (block fractions 0.661 / 0.730 / 0.787 of the
752 M / 1,882 M / 2,517 M parameters; every K divisible by 128). Cross-check against webyIQ: on the same first
8 x 512 tokens the bf16 evaluator gives 17.148 (Qwen3.5-2B; webyIQ f32 17.14) and 21.232 (MiniCPM5-2B;
webyIQ 21.23). The 2048-window protocol over all windows gives 11.81 / 13.78, lower by the expected
context-length effect, not by an eval error.

**E2. RTN symmetric bit sweep.** Hypothesis (DESIGN.md sanity list): 4-bit costs "a few percent", 3-bit tens of
percent, 2-bit near-destroyed (PPL hundreds to thousands). Result: 3- and 2-bit as expected or worse (3-bit
+383% / +195% / +611%; 2-bit PPL 1.0e6 / 2.0e4, top1 0.008 / 0.047, i.e. confidently wrong, above the vocabulary
size). **4-bit did not meet the expectation**: +29.0% (0.8B), +11.2% (2B), +26.6% (MiniCPM5), an order of
magnitude above "a few percent". Audit finding (low): the run report's "all sanity expectations met" was wrong on
this point; the numbers themselves are believed. Evidence that it is real and not an eval bug: KL on 32 windows
(0.205 / 0.133 / 0.268 nats) tracks ln(ppl_q / ppl_fp16) over all windows (0.255 / 0.106 / 0.236); asym and
Hadamard at the same width move it only a little while GPTQ removes most of it (+2.8% / +5.6% / +5.8%), the
signature of RTN's per-group grid; webyIQ's own g32 RTN figures (+4.8% Qwen3.5-2B, +11.4% MiniCPM5-2B with int8
activations) scale to these g128 numbers by the same 2.3x on both models; and `rtn.py` documents that the
symmetric grid clips the positive extreme of about half the groups. Resolution: no rerun; +11..29% is the
Phase 1 4-bit RTN baseline and the "few percent" line in DESIGN.md should be read as wrong for g128 symmetric RTN
on these models. 8-bit RTN (0.8B only): -0.05% PPL, KL 0.0015, top1 0.979, i.e. within noise of bf16.

**E3. Asymmetric vs symmetric RTN (0.8B).** Hypothesis: the b-bit zero point buys a small gain for ~0.03 extra
bits. Result: 4-bit 21.85 vs 22.67 (+24.4% vs +29.0%), 3-bit 72.1 vs 84.8, 2-bit 5.1e5 vs 1.0e6; KL and top1
move the same way. Direction consistent at all widths, but the 4-bit gap (3.6% of PPL) is smaller than the
seed-to-seed spread measured in E5 (4.4%), so it is not established at 4 bits; the 3-bit gap (15%) is.

**E4. Hadamard input rotation + RTN (0.8B).** Hypothesis (DESIGN.md): rotation helps RTN at 3 and 2 bits.
Criterion: lower PPL/KL than rtn-sym at equal bits (rotation adds 0 bits). Result: met at 3 bits (44.5 vs 84.8,
+153% vs +383%) and 2 bits (2.5e4 vs 1.0e6, KL 7.3 vs 11.1), both far outside the spread. At 4 bits 21.54 vs
22.67 (5.0% of PPL) is at the edge of the spread; not claimed. Rotation stays far from GPTQ at every width.

**E5. GPTQ vs RTN, and calibration-seed spread.** Hypothesis: GPTQ beats RTN clearly at 3 and 2 bits.
Criterion: lower PPL, KL and higher top1 at equal bits on every model. Result: met everywhere, in all three
measures: 3-bit 24.45 vs 84.77 (0.8B), 15.67 vs 34.80 (2B), 18.34 vs 97.96 (MiniCPM5); 2-bit 132.6 vs 1.0e6
(0.8B); 4-bit 18.06 vs 22.67, 12.47 vs 13.13, 14.58 vs 17.45. Seed repeat (gptq-sym-g128-3, 0.8B, calibration
seed 0 vs 1): PPL 24.45 vs 23.40 (4.4%), KL 0.313 vs 0.299 (4.6%), top1 0.7375 vs 0.7399, all in the same
direction. **This 4-5% (6 points of dPPL%) is the resolution limit used throughout these notes**; it was measured
for one method on one model only, so it is a lower bound on what to require of a difference, not a general
error bar. GPTQ at 3 bits lands at +33..39% (0.8B, two seeds), +32.6% (2B), +33.1% (MiniCPM5): no model
difference is resolvable at 3 bits. GPTQ 2-bit is still +655% on the 0.8B: usable 2-bit needs more than per-group
error feedback.

**E6. Architecture independence (MiniCPM5-2B, Llama).** Hypothesis: the ordering RTN < had-RTN < GPTQ and the
collapse pattern are not DeltaNet-specific. Result: ordering holds (only RTN and GPTQ were run on MiniCPM5).
MiniCPM5 is more RTN-sensitive than the Qwen 2B at 3 bits (+611% vs +195%) yet both reach the same +33% with
GPTQ, consistent with RTN loss being dominated by a few tensors that error feedback can compensate. Hypothesis
only; see Phase 2 questions.

**Distortion measures.** ln(ppl_q / ppl_fp16) over all windows and KL over the first 32 agree within 5% for every
row with dCE > 0.9 nats (2-bit rows: 10.95 vs 11.06, 7.41 vs 7.39) and within ~25% at 3-4 bits, except
gptq-sym-g128-4 on the 0.8B: KL 0.0712 vs dCE 0.0275 (ratio 2.6; all other rows 0.78-1.37). Audit finding (low):
not a wrong number; the absolute gap (0.044 nats) is the same size as on the other 4-bit rows, the ratio is large
only because dCE is tiny, and KL (expectation under P_ref) and dCE (under the data) decouple near zero. Left as
is. Consequence: at 4 bits, where dPPL is a few percent, use KL and top1 as the primary axis (the rtn-8 row
already shows dPPL -0.05% at KL 0.0015).

### Results: qwen3.5-0.8b (145 windows, 296,815 tokens; `results/qwen3.5-0.8b/runs.jsonl`)

| method | seed | bits_blocks | bits_all | ppl | dPPL % | kl | top1_agree |
|---|---|---|---|---|---|---|---|
| fp16 | 0 | 16.0000 | 16.0000 | 17.57 | +0.00 | 0.00000 | 1.0000 |
| rtn-sym-g128-8 | 0 | 8.1250 | 10.7917 | 17.56 | -0.05 | 0.00151 | 0.9791 |
| rtn-sym-g128-4 | 0 | 4.1250 | 8.1462 | 22.67 | +29.02 | 0.20514 | 0.7799 |
| rtn-sym-g128-3 | 0 | 3.1250 | 7.4848 | 84.77 | +382.52 | 1.50528 | 0.4596 |
| rtn-sym-g128-2 | 0 | 2.1250 | 6.8234 | 1.003e+06 | +5.7e6 | 11.06255 | 0.0076 |
| rtn-asym-g128-4 | 0 | 4.1562 | 8.1668 | 21.85 | +24.37 | 0.16925 | 0.7962 |
| rtn-asym-g128-3 | 0 | 3.1484 | 7.5003 | 72.13 | +310.57 | 1.30765 | 0.4823 |
| rtn-asym-g128-2 | 0 | 2.1406 | 6.8337 | 5.12e+05 | +2.9e6 | 10.47527 | 0.0047 |
| had-rtn-sym-g128-4 | 0 | 4.1250 | 8.1462 | 21.54 | +22.59 | 0.16707 | 0.7988 |
| had-rtn-sym-g128-3 | 0 | 3.1250 | 7.4848 | 44.52 | +153.44 | 0.92780 | 0.5745 |
| had-rtn-sym-g128-2 | 0 | 2.1250 | 6.8234 | 2.521e+04 | +1.4e5 | 7.33447 | 0.0250 |
| gptq-sym-g128-4 | 0 | 4.1250 | 8.1462 | 18.06 | +2.79 | 0.07125 | 0.8641 |
| gptq-sym-g128-3 | 0 | 3.1250 | 7.4848 | 24.45 | +39.17 | 0.31305 | 0.7375 |
| gptq-sym-g128-3 | 1 | 3.1250 | 7.4848 | 23.40 | +33.20 | 0.29937 | 0.7399 |
| gptq-sym-g128-2 | 0 | 2.1250 | 6.8234 | 132.6 | +654.64 | 1.91236 | 0.3817 |

Wall 44.4 min; fp16/RTN rows ~30 s each, GPTQ rows ~575 s each.

### Results: qwen3.5-2b (145 windows, 296,815 tokens; `results/qwen3.5-2b/runs.jsonl`)

| method | seed | bits_blocks | bits_all | ppl | dPPL % | kl | top1_agree |
|---|---|---|---|---|---|---|---|
| fp16 | 0 | 16.0000 | 16.0000 | 11.81 | +0.00 | 0.00000 | 1.0000 |
| rtn-sym-g128-4 | 0 | 4.1250 | 7.3377 | 13.13 | +11.16 | 0.13308 | 0.8241 |
| rtn-sym-g128-3 | 0 | 3.1250 | 6.6082 | 34.80 | +194.65 | 1.03017 | 0.5651 |
| rtn-sym-g128-2 | 0 | 2.1250 | 5.8787 | 1.957e+04 | +1.7e5 | 7.39226 | 0.0469 |
| gptq-sym-g128-4 | 0 | 4.1250 | 7.3377 | 12.47 | +5.61 | 0.05908 | 0.8821 |
| gptq-sym-g128-3 | 0 | 3.1250 | 6.6082 | 15.67 | +32.64 | 0.27190 | 0.7615 |

Wall 48.6 min; GPTQ rows 23.5 and 22.3 min.

### Results: minicpm5-2b (139 windows, 284,533 tokens; `results/minicpm5-2b/runs.jsonl`)

| method | seed | bits_blocks | bits_all | ppl | dPPL % | kl | top1_agree |
|---|---|---|---|---|---|---|---|
| fp16 | 0 | 16.0000 | 16.0000 | 13.78 | +0.00 | 0.00000 | 1.0000 |
| rtn-sym-g128-4 | 0 | 4.1250 | 6.6491 | 17.45 | +26.63 | 0.26816 | 0.7570 |
| rtn-sym-g128-3 | 0 | 3.1250 | 5.8616 | 97.96 | +610.80 | 2.07439 | 0.3902 |
| gptq-sym-g128-4 | 0 | 4.1250 | 6.6491 | 14.58 | +5.80 | 0.07708 | 0.8657 |
| gptq-sym-g128-3 | 0 | 3.1250 | 5.8616 | 18.34 | +33.07 | 0.34255 | 0.7299 |

Wall 68.1 min; GPTQ rows 32.6 and 32.2 min (42 layers at ~45 s each), inside the 40-minute limit but close.

bits_all differs between models at equal bits_blocks because the unquantized share (embedding / lm_head, tied on
Qwen3.5) is 34% / 27% / 21% of the parameters; compare methods on bits_blocks, as DESIGN.md says.

### Plots

`results/rd_qwen3.5-0.8b.png`, `results/rd_qwen3.5-2b.png`, `results/rd_minicpm5-2b.png` (x = bits_blocks;
left panel ppl - ppl_fp16 on a log axis with a right-hand twin axis in absolute PPL; right panel KL on a log
axis; one line per family through default-seed rows; points labelled with the bit width). The 8-bit RTN point on
the 0.8B has ppl - ppl_fp16 = -0.00914 < 0 and is drawn hollow at an artificial floor (half the smallest positive
delta, 0.245) with the measured value in its label and a footer line; the script prints the same warning to
stderr. The plots were regenerated once after the sweeps (2026-11-08 ~01:0x local) without touching the jsonl.

Plot audit (all applied to `scripts/plot_rd.py`; data files unchanged; the 27 plot tests pass):
- medium: bit-width labels of different families overprinted at shared x ("2 2", "3 3 3", and the seed-1 "3 s1 c1"
  on top of seed 0). Fixed: labels in the family color, per-family quadrant slot, light backing box, non-default
  seed labels mirrored.
- medium: the hollow point carried no number. Fixed: label "8 (ppl-fp16 = -0.00914)" and a footer.
- medium: no fp16 reference or absolute PPL readable. Fixed: legend entry for fp16 (y = 0, off the log axis) and
  the absolute-PPL twin axis.
- low: the GPTQ line was drawn through both 3-bit seeds (a vertical segment read as a rate step). Fixed: lines
  through default-seed rows only; other seeds detached markers.
- low: axis labels did not name the jsonl fields and the figure did not cite its source. Fixed (x-label names
  bits_blocks, titles carry n_tokens, footer cites `results/<model>/runs.jsonl`).
- low, left as is: the single 8-bit point stretches the 0.8B x-axis to 8.1, leaving the 2-4 bit region in ~35% of
  the width. DESIGN.md names bits_blocks as the axis without saying linear; kept linear so the three figures
  compare alike. Option: log2 x-axis or an inset.

### Other audit findings

- low (`wcomp/quant/gptq.py` ~line 260): GPTQ wall time is dominated by the f64 Hessian accumulation on the GPU
  (36-57 s of each 45-61 s layer; consumer Blackwell runs f64 FMA at 1/64 of f32 rate), which is why MiniCPM5 came
  within 8 minutes of the 40-minute limit. Results unaffected. Left for Phase 2: form each window's `x.T @ x` in
  f32 and add it into the f64 accumulator; verify on the 0.8B that gptq-sym-g128-3 reproduces 24.45 within the
  seed spread before relying on it.
- Checked and OK: every table cell above matches runs.jsonl; monotone in bits within every family on every model
  in ppl, kl and top1; one code version and env string on all 26 rows; no code changed during or after the
  sweeps (all source mtimes <= 21:59:42, first sweep 22:17:38); nothing committed; DESIGN.md untouched.

### Open questions for Phase 2

1. Which tensors carry the RTN loss? The 4-bit RTN cost (+11..29%) is far above expectation and GPTQ removes most
   of it; a per-weight-type RTN sensitivity ablation (quantize one type at a time; `in_proj_a` / `in_proj_b` are
   16 x 1024 and feed the DeltaNet decay gates) would say whether a few tensors dominate, and whether the
   MiniCPM5-vs-Qwen 2B sensitivity gap (+611% vs +195% at 3 bits) is the same story.
2. Noise model. One seed pair (4.4% PPL at 3 bits, 0.8B) is the only spread measured. Phase 2 needs at least
   seed repeats of GPTQ at 4 bits and on one 2B model before method differences of a few percent are read, and a
   decision whether KL/top1 (less noisy near zero) replaces dPPL as the primary axis at 4 bits.
3. GPTQ throughput (f32 per-window products into the f64 Hessian) decides how many data-dependent Phase 2 runs
   fit; must be validated against 24.45 (3-bit, 0.8B) first.
4. `had-gptq-sym-g128-3` and `had-fp16` were not run; rotation + error feedback is the obvious next baseline, and
   the rotation-only +0.07% needs a row of its own once had-* differences approach that size.
5. Group size: webyIQ's g32 figures are 2.3x smaller than g128 on both 2B models; the bits-vs-distortion trade of
   g32 / g64 (0.5 / 0.25 extra bits for f16 scales) is unmeasured here.
6. 2 bits is not reached by any Phase 1 method (best +655%, GPTQ on the 0.8B); whatever structure Phase 2 probes
   has to show up as a 2-3 bit result, where the measures are far outside the noise.

## Phase 2: structure probing

### Pre-registered hypotheses (verbatim from DESIGN.md)

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

2026-11-08: pre-registered before any Phase 2 run; code `72c7eff+dirty/68c5deb59e`.

### Phase 2 run list (exact commands, sequential on the one GPU, from `D:\repos\wcomp`)

qwen3.5-0.8b:

```
uv run python scripts/probe_weights.py --model qwen3.5-0.8b --include-embedding
uv run python scripts/probe_acts.py --model qwen3.5-0.8b
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g32-4,rtn-sym-g32-3,rtn-sym-g64-4,rtn-sym-g64-3
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4@only=q,rtn-sym-g128-4@only=k,rtn-sym-g128-4@only=v,rtn-sym-g128-4@only=o,rtn-sym-g128-4@only=gate,rtn-sym-g128-4@only=up,rtn-sym-g128-4@only=down,rtn-sym-g128-4@only=in_proj_qkv,rtn-sym-g128-4@only=in_proj_z,rtn-sym-g128-4@only=in_proj_ab,rtn-sym-g128-4@only=out_proj
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-3@only=q,rtn-sym-g128-3@only=k,rtn-sym-g128-3@only=v,rtn-sym-g128-3@only=o,rtn-sym-g128-3@only=gate,rtn-sym-g128-3@only=up,rtn-sym-g128-3@only=down,rtn-sym-g128-3@only=in_proj_qkv,rtn-sym-g128-3@only=in_proj_z,rtn-sym-g128-3@only=in_proj_ab,rtn-sym-g128-3@only=out_proj
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4@except=q,rtn-sym-g128-4@except=k,rtn-sym-g128-4@except=v,rtn-sym-g128-4@except=o,rtn-sym-g128-4@except=gate,rtn-sym-g128-4@except=up,rtn-sym-g128-4@except=down,rtn-sym-g128-4@except=in_proj_qkv,rtn-sym-g128-4@except=in_proj_z,rtn-sym-g128-4@except=in_proj_ab,rtn-sym-g128-4@except=out_proj
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-3@layer=0,rtn-sym-g128-3@layer=1,rtn-sym-g128-3@layer=2,rtn-sym-g128-3@layer=3,rtn-sym-g128-3@layer=4,rtn-sym-g128-3@layer=5,rtn-sym-g128-3@layer=6,rtn-sym-g128-3@layer=7,rtn-sym-g128-3@layer=8,rtn-sym-g128-3@layer=9,rtn-sym-g128-3@layer=10,rtn-sym-g128-3@layer=11,rtn-sym-g128-3@layer=12,rtn-sym-g128-3@layer=13,rtn-sym-g128-3@layer=14,rtn-sym-g128-3@layer=15,rtn-sym-g128-3@layer=16,rtn-sym-g128-3@layer=17,rtn-sym-g128-3@layer=18,rtn-sym-g128-3@layer=19,rtn-sym-g128-3@layer=20,rtn-sym-g128-3@layer=21,rtn-sym-g128-3@layer=22,rtn-sym-g128-3@layer=23
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4+keepcols=0.5,rtn-sym-g128-4+keepcols=1,rtn-sym-g128-4+keepcols=2,rtn-sym-g128-3+keepcols=0.5,rtn-sym-g128-3+keepcols=1,rtn-sym-g128-3+keepcols=2
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4+keepsuper=results/probe/qwen3.5-0.8b/superweight.json,rtn-sym-g128-3+keepsuper=results/probe/qwen3.5-0.8b/superweight.json
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods gptq-sym-g128-3
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods gptq-sym-g128-4 --seed 1
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/probe --out-name ablation.jsonl --methods had-fp16,had-gptq-sym-g128-3
```

qwen3.5-2b:

```
uv run python scripts/probe_weights.py --model qwen3.5-2b --include-embedding
uv run python scripts/probe_acts.py --model qwen3.5-2b --probes p5,p6a,p7,p2b
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4@only=q,rtn-sym-g128-4@only=k,rtn-sym-g128-4@only=v,rtn-sym-g128-4@only=o,rtn-sym-g128-4@only=gate,rtn-sym-g128-4@only=up,rtn-sym-g128-4@only=down,rtn-sym-g128-4@only=in_proj_qkv,rtn-sym-g128-4@only=in_proj_z,rtn-sym-g128-4@only=in_proj_ab,rtn-sym-g128-4@only=out_proj
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-3@layer=0,rtn-sym-g128-3@layer=1,rtn-sym-g128-3@layer=2,rtn-sym-g128-3@layer=3,rtn-sym-g128-3@layer=4,rtn-sym-g128-3@layer=5,rtn-sym-g128-3@layer=6,rtn-sym-g128-3@layer=7,rtn-sym-g128-3@layer=8,rtn-sym-g128-3@layer=9,rtn-sym-g128-3@layer=10,rtn-sym-g128-3@layer=11,rtn-sym-g128-3@layer=12,rtn-sym-g128-3@layer=13,rtn-sym-g128-3@layer=14,rtn-sym-g128-3@layer=15,rtn-sym-g128-3@layer=16,rtn-sym-g128-3@layer=17,rtn-sym-g128-3@layer=18,rtn-sym-g128-3@layer=19,rtn-sym-g128-3@layer=20,rtn-sym-g128-3@layer=21,rtn-sym-g128-3@layer=22,rtn-sym-g128-3@layer=23
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4+keepcols=1
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4+keepsuper=results/probe/qwen3.5-2b/superweight.json
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/probe --out-name ablation.jsonl --methods gptq-sym-g128-4 --seed 1
```

minicpm5-2b:

```
uv run python scripts/probe_weights.py --model minicpm5-2b --include-embedding
uv run python scripts/probe_acts.py --model minicpm5-2b --probes p5,p6a,p7
uv run python scripts/run_baselines.py --model minicpm5-2b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4@only=q,rtn-sym-g128-4@only=k,rtn-sym-g128-4@only=v,rtn-sym-g128-4@only=o,rtn-sym-g128-4@only=gate,rtn-sym-g128-4@only=up,rtn-sym-g128-4@only=down
uv run python scripts/run_baselines.py --model minicpm5-2b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4+keepcols=1
uv run python scripts/run_baselines.py --model minicpm5-2b --results-dir results/probe --out-name ablation.jsonl --methods rtn-sym-g128-4+keepsuper=results/probe/minicpm5-2b/superweight.json
```

### 2026-11-08 Runs: provenance and what was run

- Every Phase 2 number (24 probe JSONs, 113 ablation rows, 4 appended `runs.jsonl` rows) carries code
  `72c7eff+dirty/9662cc2cfa`, not the `68c5deb59e` of the pre-registration line above: a fix round
  (wcomp/methods.py, wcomp/quant/gptq.py, scripts/run_baselines.py, wcomp/probe/*.py, scripts/probe_*.py,
  scripts/plot_probe.py, scripts/structure_table.py, tests) landed at 09:13Z, between the pre-registration
  (07:58Z) and the first run (09:21Z). The hypotheses text was not changed. The run step itself changed no code
  (no source mtime after 09:18Z; `code_version()` recomputed after the runs = `9662cc2cfa`); `uv run python -m
  pytest -q` on that hash: 220 passed. Phase 1 rows are untouched (git diff of the three `runs.jsonl` is
  append-only: 3 / 1 / 0 rows; the 15 / 6 / 5 Phase 1 rows keep `0afd0d8+dirty/d2cfe18180`), and the Phase 1
  regression values reproduce at the new hash (smoke fp16 14.946913 / rtn-sym-4 19.007506 / gptq-sym-4
  16.248178, identical digits).
- After the runs, the plot audit edited `scripts/plot_probe.py`, `scripts/plot_rd.py` and
  `scripts/structure_table.py` (presentation only, see "Audit findings" below; 58 plot tests pass). Those
  scripts are inside the hash, so the next `code_version()` will differ from `9662cc2cfa` without any
  number-producing code having changed. Nothing committed; DESIGN.md untouched.
- Runs, sequential on the one GPU, from `D:\repos\wcomp`, logs `results/probe/<model>/run.log` (every
  `EXIT_* 0`): 0.8B 09:21-10:27Z (65 min wall, ~59 min GPU), 2B 10:30-11:35Z (65 min), MiniCPM5 11:37-12:26Z
  (49 min). Total 3 h, inside the 4 h budget. GPU left at 275 MiB used after each list.
- Deviations from the run list above: the GPTQ acceptance row needed `--hessian-f32` (the runner refuses a GPTQ
  row into `results/probe` without an explicit Hessian flag); the three 0.8B step-5 rows and the 2B GPTQ seed-1
  row went to `results/<model>/runs.jsonl` with `--hessian-f64` (not to `ablation.jsonl`); 2B `probe_acts` ran
  with `--layer-chunk 4` (peak 5.27 GB); `probe_acts --probes p6a` was rerun on every model after its ablation
  rows existed, to fill the P6 prediction check (the only probe file rewritten); the `+keepsuper` rows on both
  Qwen models were issued and skipped by the runner (P7 found no candidate with dPPL > 10%); the MiniCPM
  `+keepsuper` row ran. One extra diagnostic, `scripts/hessian_paths.py --layer 0`
  (`results/qwen3.5-0.8b/hessian_paths.json`), C4 only, ~30 s.
- Concurrency: `results/smoke/fix3_smoke.sh` (the fix-round verification) overlapped the 0.8B `probe_weights`
  step (weight-only, deterministic; timings only). Its pytest log had 6 plot-CLI subprocess failures
  (`STATUS_DLL_INIT_FAILED` under load); rerun alone: 6 passed (`results/probe/qwen3.5-0.8b/pytest_plot_recheck.log`).
- Evaluation protocol of every ablation row equals the Phase 1 rows it is compared with (296,815 / 65,504
  tokens on Qwen, 284,533 / 65,504 on MiniCPM; seed 0; dPPL against the Phase 1 fp16 rows 17.57 / 11.81 /
  13.78). bits_blocks of every `@only` / `@except` row reproduces `(n_t b + (N - n_t) 16) / N` to 4 decimals.
- Files: `results/probe/<model>/{marginal,lowrank,outliers,interlayer,acts,sensitivity,superweight}.json`
  (+ `fisher.json`, `lowrank_weighted.json` on the 0.8B, `lowrank_weighted.json` on the 2B), `ablation.jsonl`
  (68 / 36 / 9 rows), `results/structure_table.md` (generated by the audit; the run step had forgotten it),
  figures `results/probe_<model>_<probe>.png` and `results/rd_<model>.png` (regenerated with `--extra
  ablation.jsonl`).

Reading rule used throughout: nothing below the Phase 1 seed spread (4.4% PPL, ~5% KL) is read as a difference;
an MSE-proxy estimate that the direct ablation contradicts is flagged, not averaged with it; a shuffle-test
"pass" with `gain_bits <= 0` means "not white noise", not a compression lever (the probe's own overhead already
eats the structure). The verdict rule of `structure_table.md`: |gain_bits - gain_bits_control| > 0.02 bits/param
and > the two-seed control spread.

### P0 direct ablation (measured), all three models

Hypothesis H-P0: three largest `@only` KLs > 60% of the joint KL on both Qwen models (candidates down,
in_proj_ab, v/o); `@only` roughly additive (sum within 50% of joint); g32 beats g128 by > 2x in KL at 4 bits.

**qwen3.5-0.8b** (`results/probe/qwen3.5-0.8b/ablation.jsonl`, 68 rows; joint Phase 1 rtn-sym-g128-4 KL 0.20514
/ PPL 22.67, rtn-sym-g128-3 KL 1.50528 / PPL 84.77; dPPL vs 17.57):

| method | bits_blocks | ppl | dPPL % | kl | top1 |
|---|---|---|---|---|---|
| gptq-sym-g128-3 (f32 Hessian, acceptance) | 3.1250 | 23.32 | +32.75 | 0.29988 | 0.7391 |
| rtn-sym-g32-4 | 4.5000 | 22.00 | +25.25 | 0.14744 | 0.8128 |
| rtn-sym-g32-3 | 3.5000 | 32.68 | +86.04 | 0.65434 | 0.6295 |
| rtn-sym-g64-4 | 4.2500 | 22.13 | +25.98 | 0.16432 | 0.8020 |
| rtn-sym-g64-3 | 3.2500 | 56.11 | +219.39 | 1.11756 | 0.5430 |
| rtn-sym-g128-4@only=q | 15.3994 | 17.68 | +0.65 | 0.00637 | 0.9562 |
| rtn-sym-g128-4@only=k | 15.9249 | 17.54 | -0.16 | 0.00308 | 0.9691 |
| rtn-sym-g128-4@only=v | 15.9249 | 17.50 | -0.36 | 0.01017 | 0.9467 |
| rtn-sym-g128-4@only=o | 15.6997 | 17.78 | +1.19 | 0.02524 | 0.9207 |
| rtn-sym-g128-4@only=gate | 13.8981 | 17.89 | +1.86 | 0.01428 | 0.9361 |
| rtn-sym-g128-4@only=up | 13.8981 | 18.35 | +4.44 | 0.02911 | 0.9075 |
| rtn-sym-g128-4@only=down | 13.8981 | 18.61 | +5.92 | 0.03200 | 0.9031 |
| rtn-sym-g128-4@only=in_proj_qkv | 13.2975 | 17.80 | +1.30 | 0.01731 | 0.9317 |
| rtn-sym-g128-4@only=in_proj_z | 15.0992 | 17.84 | +1.56 | 0.00886 | 0.9511 |
| rtn-sym-g128-4@only=in_proj_ab | 15.9859 | 17.67 | +0.59 | 0.00191 | 0.9758 |
| rtn-sym-g128-4@only=out_proj | 15.0992 | 19.40 | +10.46 | 0.04739 | 0.8912 |
| rtn-sym-g128-3@only=q | 15.3489 | 18.04 | +2.71 | 0.02504 | 0.9138 |
| rtn-sym-g128-3@only=k | 15.9186 | 17.56 | -0.04 | 0.01042 | 0.9459 |
| rtn-sym-g128-3@only=v | 15.9186 | 18.11 | +3.11 | 0.03660 | 0.9059 |
| rtn-sym-g128-3@only=o | 15.6744 | 18.95 | +7.89 | 0.09687 | 0.8408 |
| rtn-sym-g128-3@only=gate | 13.7211 | 17.85 | +1.61 | 0.06421 | 0.8689 |
| rtn-sym-g128-3@only=up | 13.7211 | 19.73 | +12.31 | 0.11202 | 0.8275 |
| rtn-sym-g128-3@only=down | 13.7211 | 20.39 | +16.07 | 0.12005 | 0.8274 |
| rtn-sym-g128-3@only=in_proj_qkv | 13.0699 | 18.20 | +3.59 | 0.07097 | 0.8635 |
| rtn-sym-g128-3@only=in_proj_z | 15.0233 | 18.36 | +4.51 | 0.03564 | 0.9045 |
| rtn-sym-g128-3@only=in_proj_ab | 15.9847 | 17.60 | +0.18 | 0.00497 | 0.9625 |
| rtn-sym-g128-3@only=out_proj | 15.0233 | 22.37 | +27.36 | 0.18785 | 0.7838 |
| rtn-sym-g128-4@except=q | 4.7256 | 22.64 | +28.86 | 0.19853 | 0.7838 |
| rtn-sym-g128-4@except=k | 4.2001 | 22.68 | +29.11 | 0.20424 | 0.7795 |
| rtn-sym-g128-4@except=v | 4.2001 | 22.74 | +29.44 | 0.19676 | 0.7846 |
| rtn-sym-g128-4@except=o | 4.4253 | 22.46 | +27.85 | 0.18106 | 0.7904 |
| rtn-sym-g128-4@except=gate | 6.2269 | 22.43 | +27.66 | 0.19167 | 0.7867 |
| rtn-sym-g128-4@except=up | 6.2269 | 21.73 | +23.69 | 0.16717 | 0.7996 |
| rtn-sym-g128-4@except=down | 6.2269 | 21.87 | +24.48 | 0.17170 | 0.7976 |
| rtn-sym-g128-4@except=in_proj_qkv | 6.8275 | 22.50 | +28.05 | 0.18814 | 0.7883 |
| rtn-sym-g128-4@except=in_proj_z | 5.0258 | 22.07 | +25.65 | 0.19091 | 0.7875 |
| rtn-sym-g128-4@except=in_proj_ab | 4.1391 | 22.55 | +28.36 | 0.20272 | 0.7812 |
| rtn-sym-g128-4@except=out_proj | 5.0258 | 20.30 | +15.56 | 0.14597 | 0.8081 |
| rtn-sym-g128-3@layer=0 | 15.4430 | 19.49 | +10.93 | 0.09883 | 0.8551 |
| rtn-sym-g128-3@layer=1 | 15.4430 | 17.74 | +1.00 | 0.02820 | 0.9172 |
| rtn-sym-g128-3@layer=2 | 15.4430 | 19.11 | +8.79 | 0.03882 | 0.8997 |
| rtn-sym-g128-3@layer=3 | 15.5252 | 17.79 | +1.28 | 0.03655 | 0.9012 |
| rtn-sym-g128-3@layer=4 | 15.4430 | 18.29 | +4.12 | 0.02101 | 0.9239 |
| rtn-sym-g128-3@layer=5 | 15.4430 | 17.24 | -1.89 | 0.02279 | 0.9205 |
| rtn-sym-g128-3@layer=6 | 15.4430 | 18.11 | +3.06 | 0.02704 | 0.9124 |
| rtn-sym-g128-3@layer=7 | 15.5252 | 18.61 | +5.94 | 0.03375 | 0.9026 |
| rtn-sym-g128-3@layer=8 | 15.4430 | 17.57 | -0.00 | 0.01333 | 0.9369 |
| rtn-sym-g128-3@layer=9 | 15.4430 | 17.28 | -1.63 | 0.01434 | 0.9348 |
| rtn-sym-g128-3@layer=10 | 15.4430 | 18.13 | +3.19 | 0.01735 | 0.9295 |
| rtn-sym-g128-3@layer=11 | 15.5252 | 17.90 | +1.88 | 0.02503 | 0.9162 |
| rtn-sym-g128-3@layer=12 | 15.4430 | 17.76 | +1.07 | 0.01227 | 0.9399 |
| rtn-sym-g128-3@layer=13 | 15.4430 | 17.77 | +1.13 | 0.01538 | 0.9348 |
| rtn-sym-g128-3@layer=14 | 15.4430 | 18.06 | +2.81 | 0.03639 | 0.9025 |
| rtn-sym-g128-3@layer=15 | 15.5252 | 18.21 | +3.67 | 0.04146 | 0.8951 |
| rtn-sym-g128-3@layer=16 | 15.4430 | 18.01 | +2.51 | 0.01882 | 0.9292 |
| rtn-sym-g128-3@layer=17 | 15.4430 | 17.86 | +1.65 | 0.01644 | 0.9351 |
| rtn-sym-g128-3@layer=18 | 15.4430 | 17.86 | +1.64 | 0.01859 | 0.9319 |
| rtn-sym-g128-3@layer=19 | 15.5252 | 18.34 | +4.42 | 0.04508 | 0.9036 |
| rtn-sym-g128-3@layer=20 | 15.4430 | 17.87 | +1.70 | 0.01709 | 0.9395 |
| rtn-sym-g128-3@layer=21 | 15.4430 | 17.95 | +2.15 | 0.01719 | 0.9373 |
| rtn-sym-g128-3@layer=22 | 15.4430 | 17.91 | +1.94 | 0.02214 | 0.9247 |
| rtn-sym-g128-3@layer=23 | 15.5252 | 19.45 | +10.70 | 0.08330 | 0.8411 |
| rtn-sym-g128-4+keepcols=0.5 | 4.1844 | 22.84 | +30.01 | 0.20247 | 0.7828 |
| rtn-sym-g128-4+keepcols=1 | 4.2433 | 22.66 | +28.98 | 0.19855 | 0.7851 |
| rtn-sym-g128-4+keepcols=2 | 4.3617 | 22.19 | +26.33 | 0.18819 | 0.7902 |
| rtn-sym-g128-3+keepcols=0.5 | 3.1894 | 71.13 | +304.90 | 1.32539 | 0.4883 |
| rtn-sym-g128-3+keepcols=1 | 3.2532 | 71.21 | +305.34 | 1.32072 | 0.4952 |
| rtn-sym-g128-3+keepcols=2 | 3.3814 | 64.93 | +269.62 | 1.24488 | 0.5094 |

At 4 bits the `@only` dPPL of single types (-0.4..+10%) is mostly inside the 4.4% spread; KL and top1 (0.003-0.047,
0.89-0.97) are the readable axes, as decided after Phase 1. The three negative dPPL (k, v at 4 bits, k at 3 bits)
are noise around zero.

**qwen3.5-2b** (`results/probe/qwen3.5-2b/ablation.jsonl`, 36 rows; joint rtn-sym-g128-4 KL 0.13308 / PPL
13.13, rtn-sym-g128-3 KL 1.03017; dPPL vs 11.81):

| method | bits_blocks | ppl | dPPL % | kl | top1 |
|---|---|---|---|---|---|
| rtn-sym-g128-4@only=q | 15.5646 | 11.87 | +0.50 | 0.00484 | 0.9638 |
| rtn-sym-g128-4@only=k | 15.9456 | 11.80 | -0.06 | 0.00250 | 0.9740 |
| rtn-sym-g128-4@only=v | 15.9456 | 11.91 | +0.86 | 0.00839 | 0.9561 |
| rtn-sym-g128-4@only=o | 15.7823 | 11.90 | +0.78 | 0.01159 | 0.9453 |
| rtn-sym-g128-4@only=gate | 13.3876 | 11.88 | +0.58 | 0.01284 | 0.9428 |
| rtn-sym-g128-4@only=up | 13.3876 | 12.18 | +3.17 | 0.02291 | 0.9212 |
| rtn-sym-g128-4@only=down | 13.3876 | 11.94 | +1.12 | 0.02342 | 0.9223 |
| rtn-sym-g128-4@only=in_proj_qkv | 14.0407 | 12.02 | +1.81 | 0.01432 | 0.9401 |
| rtn-sym-g128-4@only=in_proj_z | 15.3469 | 11.82 | +0.07 | 0.00871 | 0.9549 |
| rtn-sym-g128-4@only=in_proj_ab | 15.9898 | 11.83 | +0.14 | 0.00166 | 0.9792 |
| rtn-sym-g128-4@only=out_proj | 15.3469 | 11.94 | +1.14 | 0.02721 | 0.9214 |
| rtn-sym-g128-3@layer=0 | 15.4486 | 12.67 | +7.32 | 0.08288 | 0.8719 |
| rtn-sym-g128-3@layer=1 | 15.4486 | 11.88 | +0.58 | 0.02437 | 0.9302 |
| rtn-sym-g128-3@layer=2 | 15.4486 | 12.55 | +6.24 | 0.03452 | 0.9137 |
| rtn-sym-g128-3@layer=3 | 15.5083 | 12.20 | +3.31 | 0.02388 | 0.9242 |
| rtn-sym-g128-3@layer=4 | 15.4486 | 11.81 | -0.05 | 0.02131 | 0.9295 |
| rtn-sym-g128-3@layer=5 | 15.4486 | 12.00 | +1.63 | 0.02540 | 0.9199 |
| rtn-sym-g128-3@layer=6 | 15.4486 | 11.90 | +0.75 | 0.03055 | 0.9141 |
| rtn-sym-g128-3@layer=7 | 15.5083 | 12.45 | +5.44 | 0.03434 | 0.9067 |
| rtn-sym-g128-3@layer=8 | 15.4486 | 11.90 | +0.79 | 0.01201 | 0.9441 |
| rtn-sym-g128-3@layer=9 | 15.4486 | 11.86 | +0.38 | 0.01248 | 0.9419 |
| rtn-sym-g128-3@layer=10 | 15.4486 | 12.18 | +3.15 | 0.01893 | 0.9287 |
| rtn-sym-g128-3@layer=11 | 15.5083 | 12.06 | +2.07 | 0.01973 | 0.9298 |
| rtn-sym-g128-3@layer=12 | 15.4486 | 11.98 | +1.42 | 0.01075 | 0.9472 |
| rtn-sym-g128-3@layer=13 | 15.4486 | 11.94 | +1.09 | 0.01143 | 0.9453 |
| rtn-sym-g128-3@layer=14 | 15.4486 | 12.05 | +1.99 | 0.02198 | 0.9276 |
| rtn-sym-g128-3@layer=15 | 15.5083 | 11.98 | +1.47 | 0.02441 | 0.9241 |
| rtn-sym-g128-3@layer=16 | 15.4486 | 12.03 | +1.84 | 0.01342 | 0.9456 |
| rtn-sym-g128-3@layer=17 | 15.4486 | 11.94 | +1.08 | 0.01089 | 0.9505 |
| rtn-sym-g128-3@layer=18 | 15.4486 | 11.95 | +1.14 | 0.01189 | 0.9494 |
| rtn-sym-g128-3@layer=19 | 15.5083 | 12.03 | +1.88 | 0.02125 | 0.9352 |
| rtn-sym-g128-3@layer=20 | 15.4486 | 11.98 | +1.42 | 0.01107 | 0.9530 |
| rtn-sym-g128-3@layer=21 | 15.4486 | 11.97 | +1.37 | 0.01146 | 0.9515 |
| rtn-sym-g128-3@layer=22 | 15.4486 | 12.02 | +1.80 | 0.01456 | 0.9433 |
| rtn-sym-g128-3@layer=23 | 15.5083 | 12.55 | +6.24 | 0.05805 | 0.8688 |
| rtn-sym-g128-4+keepcols=1 | 4.2430 | 12.96 | +9.73 | 0.12265 | 0.8304 |

**minicpm5-2b** (`results/probe/minicpm5-2b/ablation.jsonl`, 9 rows; joint rtn-sym-g128-4 KL 0.26816 / PPL
17.45; dPPL vs 13.78):

| method | bits_blocks | ppl | dPPL % | kl | top1 |
|---|---|---|---|---|---|
| rtn-sym-g128-4@only=q | 14.9444 | 14.00 | +1.55 | 0.02365 | 0.9243 |
| rtn-sym-g128-4@only=k | 15.8681 | 13.87 | +0.64 | 0.01171 | 0.9453 |
| rtn-sym-g128-4@only=v | 15.8681 | 14.28 | +3.64 | 0.03176 | 0.9132 |
| rtn-sym-g128-4@only=o | 14.9444 | 14.21 | +3.12 | 0.03552 | 0.9051 |
| rtn-sym-g128-4@only=gate | 12.8333 | 14.12 | +2.48 | 0.03056 | 0.9123 |
| rtn-sym-g128-4@only=up | 12.8333 | 14.25 | +3.38 | 0.04610 | 0.8896 |
| rtn-sym-g128-4@only=down | 12.8333 | 14.81 | +7.48 | 0.08193 | 0.8567 |
| rtn-sym-g128-4+keepcols=1 | 4.2431 | 16.59 | +20.38 | 0.22265 | 0.7767 |
| rtn-sym-g128-4+keepsuper=results/probe/minicpm5-2b/superweight.json | 4.1250 | 17.32 | +25.69 | 0.26742 | 0.7578 |

**Additivity check** (sum of `@only` KL vs joint KL at the same bits):

| model | bits | sum @only | joint | sum / joint | top-3 @only (share of joint) |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 4 | 0.1957 | 0.2051 | 0.95 | out_proj 0.0474, down 0.0320, up 0.0291 (53%) |
| qwen3.5-0.8b | 3 | 0.7646 | 1.5053 | 0.51 | out_proj 0.1878, down 0.1200, up 0.1120 (28%) |
| qwen3.5-2b | 4 | 0.1384 | 0.1331 | 1.04 | out_proj 0.0272, down 0.0234, up 0.0229 (55%) |
| minicpm5-2b | 4 | 0.2612 | 0.2682 | 0.97 | down 0.0819, up 0.0461, o 0.0355 (61%) |

The `@except` rows close the loop from the other side on the 0.8B: joint - `@except[t]` agrees with `@only[t]`
within 0.012 for all 11 types (largest gaps out_proj 0.059 vs 0.047, up 0.038 vs 0.029, in_proj_z 0.014 vs
0.009). Per layer at 3 bits: `@layer` sums 0.721 (0.8B) / 0.562 (2B) vs joint 1.505 / 1.030 (ratios 0.48 /
0.55), largest layers 0 (0.099 / 0.083) and 23 (0.083 / 0.058), then 19, 15, 2 (0.8B) and 2, 7, 6 (2B);
smallest layer 12 (0.012 / 0.011). The per-type KL(3 bit) / KL(4 bit) ratios on the 0.8B are 3.4-4.5 for 10 of
11 types (in_proj_ab 2.6), i.e. D ~ 2^-2b holds per type between 3 and 4 bits; the joint curve is steeper
(7.3x per bit) because of the super-additivity at 3 bits.

Verdict H-P0: **concentration refuted** on both Qwen models (53% / 55%, < 60%; MiniCPM 61% clears it) and the
candidates were wrong: in_proj_ab and v carry 0.9% / 5.0% (0.8B) and 1.3% / 6.3% (2B) of the joint KL; the loss
sits in the output-writing types out_proj / down / up (/ o on MiniCPM), 53-61%. **Additivity supported at 4
bits** on all three models (0.95 / 1.04 / 0.97), **not at 3 bits** (0.51 and 0.48 by layer: the joint loss is
twice the sum, super-additive). **Group size is not the lever at 4 bits** (refuted clause): see next item.

**g32 / g64 / keepcols against the Phase 1 RD curve (0.8B).** The RTN g128 curve runs from KL 0.2051 at 4.125
bits to 1.5053 at 3.125 bits, a factor 7.3 per bit; that slope is the exchange rate a Phase 2 variant has to beat
to be a better use of its extra bits than index bits are.

| variant | bits_blocks | extra bits | KL | KL ratio vs g128 | curve's own factor for the extra bits | reading |
|---|---|---|---|---|---|---|
| rtn-sym-g64-4 | 4.250 | +0.125 | 0.1643 | 1.25 | 1.28 | on the curve |
| rtn-sym-g32-4 | 4.500 | +0.375 | 0.1474 | 1.39 | 2.11 | below the curve (H-P0 asked > 2) |
| rtn-sym-g64-3 | 3.250 | +0.125 | 1.1176 | 1.35 | 1.28 | on the curve |
| rtn-sym-g32-3 | 3.500 | +0.375 | 0.6543 | 2.30 | 2.11 | on the curve (> 2, but no better than index bits) |
| +keepcols=0.5 / 1 / 2 at 4 bits | 4.184 / 4.243 / 4.362 | +0.059 / 0.118 / 0.237 | 0.2025 / 0.1986 / 0.1882 | 1.01 / 1.03 / 1.09 | 1.12 / 1.27 / 1.60 | far below |
| +keepcols=0.5 / 1 / 2 at 3 bits | 3.189 / 3.253 / 3.381 | +0.064 / 0.128 / 0.256 | 1.3254 / 1.3207 / 1.2449 | 1.14 / 1.14 / 1.21 | 1.14 / 1.29 / 1.67 | 0.5 on the curve, 1 and 2 below |

The same comparison with GPTQ: g32-4 at 4.5 bits (0.1474) is still 2.1x the KL of GPTQ-4 at 4.125 (0.0713), and
g32-3 at 3.5 bits (0.654) 2.1x GPTQ-3 at 3.125 (0.313). On the 2B keepcols=1 gives 1.085 for +0.118 bits
(curve 7.7x/bit: 1.27); on MiniCPM 1.20 for +0.118 (curve 1.27), the only model where keepcols is near
break-even. The audit's "bits-equivalent" figures (0.5 log2 of the KL ratio: g32-4 0.238, g64-4 0.160,
keepcols=1 0.024 bits) assume the flatter D ~ 2^-2b exchange and are upper bounds; against the measured curve
every keepcols point and g32-4 lose. In the RD plots (`rd_qwen3.5-0.8b.png`) these 11 points sit on or above the
RTN line between 3.1 and 4.5 bits, far from the GPTQ line.

### GPTQ speed acceptance and the new seed-1 rows

Acceptance (contract): form `x^T x` in f32, accumulate in f64; gptq-sym-g128-3 seed 0 on the 0.8B must
reproduce PPL 24.45 within 0.5% and KL 0.3131 within 2%. Result (`ablation.jsonl` row 1, `--hessian-f32`): PPL
23.32 (-4.6%), KL 0.29988 (-4.2%), top1 0.7391; 110 s instead of 575 s (Hessian 1.0 s/layer vs ~16 s).
**Fails the pre-registered rule**; f64 (`--hessian-f64`) was used for every GPTQ row after it, as the rule says.
Diagnostic `scripts/hessian_paths.py --layer 0` (`results/qwen3.5-0.8b/hessian_paths.json`): the two Hessians
agree to relative Frobenius 1.3e-7..1.8e-6, max relative diagonal difference <= 5.9e-6, GPTQ objective ratio
0.99996-1.00002, yet 0.003-0.18% of the weights round differently per linear. The -4.6% is therefore rounding
chaos propagated through 24 sequentially quantized layers, the same size as the measured seed spread (seed 1 of
the f64 path: 23.40, -4.4%), not a precision loss. Audit reading, recorded: the 0.5% / 2% tolerance was
mis-specified (9x below the method's own 4.4% seed spread, so no outcome could separate precision from rounding
chaos); the right test would have been "within the seed spread" or the Hessian-level relF check. Decision: keep
f64 for every row that is compared against a Phase 1 number; f32 may be used for new Phase 3 rows after a
two-seed check of its own. Latent hazard (audit, low, left): `wcomp/quant/gptq.py` reads `WCOMP_HESSIAN_F32`
from the environment as the default, so a Phase 1 GPTQ name would silently take the f32 path if that variable is
set; not triggered here (every GPTQ row after the acceptance passed `--hessian-f64` and shows the f64 Hessian
time, 14-20 s/layer on the 0.8B, 37-43 s/layer on the 2B); Phase 3 should drop the env override or make the flag
mandatory.

New full-model rows (`results/<model>/runs.jsonl`, f64 Hessian, code `9662cc2cfa`):

| model | method | seed | bits_blocks | ppl | dPPL % | kl | top1 | time_s |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | gptq-sym-g128-4 | 1 (calib seed 1) | 4.1250 | 18.44 | +4.99 | 0.06763 | 0.8689 | 478 |
| qwen3.5-0.8b | had-fp16 | 0 | 16.0000 | 17.58 | +0.07 | 0.00078 | 0.9855 | 27 |
| qwen3.5-0.8b | had-gptq-sym-g128-3 | 0 | 3.1250 | 20.87 | +18.79 | 0.21401 | 0.7753 | 483 |
| qwen3.5-2b | gptq-sym-g128-4 | 1 (calib seed 1) | 4.1250 | 12.36 | +4.61 | 0.05747 | 0.8837 | 1166 |

Noise model update (Phase 1 open question 2): there are now three calibration-seed pairs for GPTQ. 3 bits, 0.8B:
24.45 vs 23.40 (4.4% PPL, 4.6% KL). 4 bits, 0.8B: 18.06 vs 18.44 (2.1% PPL, 5.2% KL: 0.0713 vs 0.0676). 4 bits,
2B: 12.47 vs 12.36 (0.9% PPL, 2.7% KL: 0.0591 vs 0.0575). Top1 moves by 0.002-0.005. The 4.4% PPL / ~5% KL
threshold stays (the largest observed; still a lower bound, one method). The new rows also settle Phase 1 open
question 4: rotation alone costs +0.07% PPL / KL 0.0008 (inside the 0.1% contract), and **Hadamard + GPTQ at 3
bits beats GPTQ at 3 bits by 15% PPL and 32% KL at identical bits** (20.87 / 0.214 vs 24.45 / 0.313; vs the
seed-1 GPTQ-3 23.40 / 0.299: 11% / 28%), well outside the spread. Along GPTQ's own curve on the 0.8B (KL 0.313 at
3.125 -> 0.071 at 4.125, 4.4x per bit) that is worth 0.23-0.26 bits/param at 3 bits. One model, one seed.

### Structure table (trimmed from `results/structure_table.md`)

Columns: W = the statistic on the real tensor, control = the same on the shuffled control (Gaussian control for
P1; split-half Spearman for P5), gain_bits / gain_control in bits/param after the probe's own side information,
difference = gain_bits - gain_control (the "claimed" structure value), verdict per the 0.02-bit / control-spread
rule. Values are numel-weighted over the tensors of a type. The full table has the key-statistic text, the
control-seed spread and the notes.

`qwen3.5-0.8b`

| probe | type | stat | W | control | gain_bits | gain_control | difference | verdict | extra |
|---|---|---|---|---|---|---|---|---|---|
| marginal | q | H_4 | 3.491 | 3.561 | +0.509 | +0.439 | +0.069 | pass |  |
| marginal | k | H_4 | 3.460 | 3.560 | +0.540 | +0.440 | +0.100 | pass |  |
| marginal | v | H_4 | 3.395 | 3.560 | +0.605 | +0.440 | +0.165 | pass |  |
| marginal | o | H_4 | 3.398 | 3.562 | +0.601 | +0.438 | +0.163 | pass |  |
| marginal | gate | H_4 | 3.520 | 3.561 | +0.480 | +0.439 | +0.040 | pass |  |
| marginal | up | H_4 | 3.514 | 3.561 | +0.486 | +0.439 | +0.047 | pass |  |
| marginal | down | H_4 | 3.491 | 3.561 | +0.509 | +0.439 | +0.070 | pass |  |
| marginal | in_proj_qkv | H_4 | 3.512 | 3.561 | +0.488 | +0.439 | +0.049 | pass |  |
| marginal | in_proj_z | H_4 | 3.519 | 3.562 | +0.481 | +0.438 | +0.043 | pass |  |
| marginal | in_proj_ab | H_4 | 3.474 | 3.550 | +0.510 | +0.434 | +0.076 | pass |  |
| marginal | out_proj | H_4 | 3.443 | 3.562 | +0.556 | +0.438 | +0.118 | pass |  |
| marginal | embed | H_4 | 3.497 | 3.561 | +0.503 | +0.439 | +0.064 | pass |  |
| lowrank | q | energy@10% | 0.466 | 0.194 | -0.031 | -0.180 | +0.148 | pass | best r 0.01 |
| lowrank | k | energy@10% | 0.466 | 0.241 | -0.168 | -0.214 | +0.046 | pass | best r 0.01 |
| lowrank | v | energy@10% | 0.335 | 0.242 | -0.181 | -0.214 | +0.033 | pass | best r 0.01 |
| lowrank | o | energy@10% | 0.395 | 0.241 | -0.175 | -0.214 | +0.039 | pass | best r 0.01 |
| lowrank | gate | energy@10% | 0.317 | 0.202 | -0.141 | -0.185 | +0.044 | pass | best r 0.01 |
| lowrank | up | energy@10% | 0.255 | 0.202 | -0.170 | -0.185 | +0.015 | fail | best r 0.01 |
| lowrank | down | energy@10% | 0.268 | 0.202 | -0.169 | -0.185 | +0.015 | fail | best r 0.01 |
| lowrank | in_proj_qkv | energy@10% | 0.349 | 0.175 | -0.101 | -0.169 | +0.068 | pass | best r 0.01 |
| lowrank | in_proj_z | energy@10% | 0.426 | 0.241 | -0.126 | -0.214 | +0.089 | pass | best r 0.01 |
| lowrank | in_proj_ab | energy@10% | 0.563 | 0.154 | -0.657 | -0.956 | +0.299 | pass | best r 0.05 |
| lowrank | out_proj | energy@10% | 0.350 | 0.241 | -0.177 | -0.214 | +0.038 | pass | best r 0.01 |
| lowrank | embed | energy@10% | 0.360 | 0.110 | +0.039 | -0.149 | +0.188 | pass | best r 0.01 |
| lowrank_weighted | q | energy@10% (WL) | 0.970 | 0.676 | +0.067 | -0.161 | +0.228 | pass | best r 0.01 |
| lowrank_weighted | k | energy@10% (WL) | 0.772 | 0.632 | -0.156 | -0.211 | +0.055 | pass | best r 0.01 |
| lowrank_weighted | v | energy@10% (WL) | 0.725 | 0.629 | -0.179 | -0.210 | +0.030 | pass | best r 0.01 |
| lowrank_weighted | o | energy@10% (WL) | 0.758 | 0.868 | -0.216 | -0.203 | -0.013 | fail | best r 0.01 |
| lowrank_weighted | gate | energy@10% (WL) | 0.902 | 0.661 | -0.080 | -0.167 | +0.087 | pass | best r 0.01 |
| lowrank_weighted | up | energy@10% (WL) | 0.674 | 0.661 | -0.166 | -0.171 | +0.006 | fail | best r 0.01 |
| lowrank_weighted | down | energy@10% (WL) | 0.541 | 0.520 | -0.152 | -0.182 | +0.030 | pass | best r 0.01 |
| lowrank_weighted | in_proj_qkv | energy@10% (WL) | 0.918 | 0.670 | -0.057 | -0.148 | +0.091 | pass | best r 0.01 |
| lowrank_weighted | in_proj_z | energy@10% (WL) | 0.947 | 0.687 | -0.029 | -0.195 | +0.166 | pass | best r 0.01 |
| lowrank_weighted | in_proj_ab | energy@10% (WL) | 0.905 | 0.435 | -0.591 | -0.950 | +0.359 | pass | best r 0.05 |
| lowrank_weighted | out_proj | energy@10% (WL) | 0.698 | 0.786 | -0.198 | -0.192 | -0.006 | fail | best r 0.01 |
| outliers | q | colnorm max/med | 2.01 | 1.06 | -0.034 | -0.053 | +0.019 | fail | Jaccard 0.130 / 0.004 |
| outliers | k | colnorm max/med | 2.22 | 1.18 | -0.033 | -0.051 | +0.018 | fail | Jaccard 0.185 / 0.004 |
| outliers | v | colnorm max/med | 1.74 | 1.30 | -0.044 | -0.048 | +0.004 | fail | Jaccard 0.083 / 0.004 |
| outliers | o | colnorm max/med | 2.50 | 1.16 | -0.032 | -0.052 | +0.020 | pass | Jaccard 0.070 / 0.008 |
| outliers | gate | colnorm max/med | 1.42 | 1.07 | -0.049 | -0.053 | +0.004 | fail | Jaccard 0.193 / 0.008 |
| outliers | up | colnorm max/med | 1.20 | 1.04 | -0.052 | -0.054 | +0.002 | fail | Jaccard 0.077 / 0.008 |
| outliers | down | colnorm max/med | 1.52 | 1.27 | -0.048 | -0.049 | +0.001 | fail | Jaccard 0.070 / 0.008 |
| outliers | in_proj_qkv | colnorm max/med | 1.36 | 1.05 | -0.048 | -0.053 | +0.005 | fail | Jaccard 0.133 / 0.008 |
| outliers | in_proj_z | colnorm max/med | 1.68 | 1.09 | -0.041 | -0.053 | +0.012 | fail | Jaccard 0.324 / 0.008 |
| outliers | in_proj_ab | colnorm max/med | 4.60 | 2.43 | +0.005 | -0.053 | +0.058 | pass | Jaccard 0.294 / 0.008 |
| outliers | out_proj | colnorm max/med | 1.82 | 1.23 | -0.042 | -0.047 | +0.005 | fail | Jaccard 0.070 / 0.008 |
| interlayer | q | min e, adjacent | 0.464 | 0.152 | -0.502 | -0.533 | +0.031 | pass |  |
| interlayer | k | min e, adjacent | 0.647 | 0.273 | -0.534 | -0.532 | -0.001 | fail |  |
| interlayer | v | min e, adjacent | 0.484 | 0.254 | -0.453 | -0.453 | +0.001 | fail |  |
| interlayer | o | min e, adjacent | 0.440 | 0.268 | -0.489 | -0.489 | -0.000 | fail |  |
| interlayer | gate | min e, adjacent | 0.202 | 0.146 | -0.335 | -0.416 | +0.080 | pass |  |
| interlayer | up | min e, adjacent | 0.142 | 0.142 | -0.394 | -0.420 | +0.026 | pass |  |
| interlayer | down | min e, adjacent | 0.146 | 0.144 | -0.376 | -0.428 | +0.051 | pass |  |
| interlayer | in_proj_qkv | min e, adjacent | 0.194 | 0.088 | -0.494 | -0.494 | -0.000 | fail |  |
| interlayer | in_proj_z | min e, adjacent | 0.377 | 0.253 | -0.494 | -0.500 | +0.006 | fail |  |
| interlayer | in_proj_ab | min e, adjacent | 0.332 | 0.023 | -0.109 | -0.509 | +0.400 | pass |  |
| interlayer | out_proj | min e, adjacent | 0.299 | 0.253 | -0.501 | -0.501 | +0.000 | fail |  |
| acts | o | Gini | 0.743 | split-half rho 0.967 | - | - | - | stable | ch. for 50/90 % energy 0.043/0.401 |
| acts | down | Gini | 0.436 | split-half rho 0.928 | - | - | - | stable | ch. for 50/90 % energy 0.194/0.716 |
| acts | out_proj | Gini | 0.756 | split-half rho 0.966 | - | - | - | stable | ch. for 50/90 % energy 0.031/0.397 |
| sensitivity | q | s_t/n_t | 5.5e-06 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0064 |
| sensitivity | k | s_t/n_t | 2.97e-06 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0031 |
| sensitivity | v | s_t/n_t | 5.87e-06 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0102 |
| sensitivity | o | s_t/n_t | 8.12e-08 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0252 |
| sensitivity | gate | s_t/n_t | 3.88e-06 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0143 |
| sensitivity | up | s_t/n_t | 1.17e-06 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0291 |
| sensitivity | down | s_t/n_t | 1.22e-08 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0320 |
| sensitivity | in_proj_qkv | s_t/n_t | 1.02e-05 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0173 |
| sensitivity | in_proj_z | s_t/n_t | 7.71e-06 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0089 |
| sensitivity | in_proj_ab | s_t/n_t | 2.24e-05 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0019 |
| sensitivity | out_proj | s_t/n_t | 1.54e-08 | - | - | - | - | no prediction (rho -0.33) | @only KL4 0.0474 |
| fisher | q | rank-1 fit | 0.988 | 0.133 | - | - | - | rank-1 fits (> 0.7) | Gini 0.774 |
| fisher | k | rank-1 fit | 0.968 | 0.190 | - | - | - | rank-1 fits (> 0.7) | Gini 0.709 |
| fisher | v | rank-1 fit | 0.943 | 0.148 | - | - | - | rank-1 fits (> 0.7) | Gini 0.649 |
| fisher | o | rank-1 fit | 0.985 | 0.034 | - | - | - | rank-1 fits (> 0.7) | Gini 0.795 |
| fisher | gate | rank-1 fit | 0.930 | 0.141 | - | - | - | rank-1 fits (> 0.7) | Gini 0.639 |
| fisher | up | rank-1 fit | 0.960 | 0.148 | - | - | - | rank-1 fits (> 0.7) | Gini 0.592 |
| fisher | down | rank-1 fit | 0.933 | 0.186 | - | - | - | rank-1 fits (> 0.7) | Gini 0.486 |
| fisher | in_proj_qkv | rank-1 fit | 0.898 | 0.121 | - | - | - | rank-1 fits (> 0.7) | Gini 0.804 |
| fisher | in_proj_z | rank-1 fit | 0.956 | 0.186 | - | - | - | rank-1 fits (> 0.7) | Gini 0.801 |
| fisher | in_proj_ab | rank-1 fit | 0.943 | 0.312 | - | - | - | rank-1 fits (> 0.7) | Gini 0.707 |
| fisher | out_proj | rank-1 fit | 0.990 | 0.032 | - | - | - | rank-1 fits (> 0.7) | Gini 0.781 |
| model.layers.23.mlp.down_proj | 23 | model.layers.23.mlp.down_proj | [0, 254] | -0.0181 | -0.00 | 0.0000 | 0.12 | max_spike_below_threshold |  |
| model.layers.20.linear_attn.in_proj_qkv | 20 | model.layers.20.linear_attn.in_proj_qkv | [1682, 74] | 0.582 | 0.00 | 0.0004 | 0.01 | top16 |  |
| model.layers.23.self_attn.v_proj | 23 | model.layers.23.self_attn.v_proj | [184, 123] | 0.539 | -0.19 | 0.0012 | 0.03 | top16 |  |
| model.layers.23.mlp.gate_proj | 23 | model.layers.23.mlp.gate_proj | [1608, 531] | -0.512 | 0.01 | 0.0001 | 0.00 | top16 |  |
| model.layers.21.linear_attn.in_proj_qkv | 21 | model.layers.21.linear_attn.in_proj_qkv | [1969, 66] | 0.496 | -0.02 | 0.0003 | 0.01 | top16 |  |
| model.layers.20.linear_attn.in_proj_qkv | 20 | model.layers.20.linear_attn.in_proj_qkv | [4883, 24] | 0.475 | 0.01 | 0.0004 | 0.00 | top16 |  |
| model.layers.19.mlp.gate_proj | 19 | model.layers.19.mlp.gate_proj | [1045, 807] | -0.471 | 0.02 | 0.0004 | 0.00 | top16 |  |
| model.layers.0.linear_attn.in_proj_z | 0 | model.layers.0.linear_attn.in_proj_z | [542, 111] | -0.461 | 0.01 | 0.0007 | 0.02 | top16 |  |
| model.layers.23.mlp.gate_proj | 23 | model.layers.23.mlp.gate_proj | [252, 460] | 0.441 | 0.00 | 0.0001 | 0.01 | top16 |  |
| model.layers.14.linear_attn.in_proj_qkv | 14 | model.layers.14.linear_attn.in_proj_qkv | [2347, 16] | 0.441 | 0.04 | 0.0005 | 0.02 | top16 |  |
| model.layers.23.mlp.gate_proj | 23 | model.layers.23.mlp.gate_proj | [1608, 807] | -0.438 | 0.00 | 0.0001 | 0.01 | top16 |  |
| model.layers.20.linear_attn.in_proj_qkv | 20 | model.layers.20.linear_attn.in_proj_qkv | [1682, 166] | 0.436 | -0.01 | 0.0004 | 0.00 | top16 |  |
| model.layers.23.mlp.gate_proj | 23 | model.layers.23.mlp.gate_proj | [252, 679] | 0.41 | -0.01 | 0.0001 | 0.01 | top16 |  |
| model.layers.20.linear_attn.in_proj_qkv | 20 | model.layers.20.linear_attn.in_proj_qkv | [3811, 74] | -0.41 | 0.03 | 0.0004 | 0.00 | top16 |  |
| model.layers.22.mlp.gate_proj | 22 | model.layers.22.mlp.gate_proj | [1334, 425] | 0.408 | -0.01 | 0.0002 | 0.01 | top16 |  |
| model.layers.22.linear_attn.in_proj_qkv | 22 | model.layers.22.linear_attn.in_proj_qkv | [4598, 807] | 0.406 | 0.00 | 0.0002 | -0.02 | top16 |  |
| model.layers.23.mlp.gate_proj | 23 | model.layers.23.mlp.gate_proj | [757, 523] | 0.402 | 0.01 | 0.0001 | 0.01 | top16 |  |

`qwen3.5-2b`

| probe | type | stat | W | control | gain_bits | gain_control | difference | verdict | extra |
|---|---|---|---|---|---|---|---|---|---|
| marginal | q | H_4 | 3.457 | 3.561 | +0.543 | +0.439 | +0.104 | pass |  |
| marginal | k | H_4 | 3.326 | 3.563 | +0.674 | +0.437 | +0.238 | pass |  |
| marginal | v | H_4 | 3.241 | 3.563 | +0.759 | +0.437 | +0.322 | pass |  |
| marginal | o | H_4 | 3.428 | 3.561 | +0.572 | +0.439 | +0.133 | pass |  |
| marginal | gate | H_4 | 3.508 | 3.561 | +0.492 | +0.439 | +0.053 | pass |  |
| marginal | up | H_4 | 3.509 | 3.561 | +0.491 | +0.439 | +0.052 | pass |  |
| marginal | down | H_4 | 3.490 | 3.561 | +0.510 | +0.439 | +0.070 | pass |  |
| marginal | in_proj_qkv | H_4 | 3.485 | 3.561 | +0.515 | +0.439 | +0.076 | pass |  |
| marginal | in_proj_z | H_4 | 3.497 | 3.561 | +0.503 | +0.439 | +0.064 | pass |  |
| marginal | in_proj_ab | H_4 | 3.223 | 3.560 | +0.769 | +0.433 | +0.337 | pass |  |
| marginal | out_proj | H_4 | 3.465 | 3.561 | +0.535 | +0.439 | +0.096 | pass |  |
| marginal | embed | H_4 | 3.447 | 3.561 | +0.553 | +0.439 | +0.113 | pass |  |
| lowrank | q | energy@10% | 0.507 | 0.242 | -0.090 | -0.214 | +0.124 | pass | best r 0.01 |
| lowrank | k | energy@10% | 0.399 | 0.195 | -0.145 | -0.180 | +0.035 | pass | best r 0.01 |
| lowrank | v | energy@10% | 0.280 | 0.195 | -0.153 | -0.180 | +0.027 | pass | best r 0.01 |
| lowrank | o | energy@10% | 0.479 | 0.314 | -0.238 | -0.285 | +0.048 | pass | best r 0.01 |
| lowrank | gate | energy@10% | 0.308 | 0.212 | -0.156 | -0.191 | +0.036 | pass | best r 0.01 |
| lowrank | up | energy@10% | 0.254 | 0.212 | -0.178 | -0.191 | +0.013 | fail | best r 0.01 |
| lowrank | down | energy@10% | 0.265 | 0.212 | -0.177 | -0.191 | +0.014 | fail | best r 0.01 |
| lowrank | in_proj_qkv | energy@10% | 0.372 | 0.212 | -0.129 | -0.191 | +0.063 | pass | best r 0.01 |
| lowrank | in_proj_z | energy@10% | 0.483 | 0.314 | -0.196 | -0.285 | +0.089 | pass | best r 0.01 |
| lowrank | in_proj_ab | energy@10% | 0.496 | 0.146 | -0.715 | -0.952 | +0.238 | pass | best r 0.05 |
| lowrank | out_proj | energy@10% | 0.410 | 0.314 | -0.240 | -0.285 | +0.045 | pass | best r 0.01 |
| lowrank | embed | energy@10% | 0.357 | 0.115 | +0.040 | -0.149 | +0.189 | pass | best r 0.01 |
| lowrank_weighted | q | energy@10% (WL) | 0.968 | 0.737 | -0.035 | -0.206 | +0.170 | pass | best r 0.01 |
| lowrank_weighted | k | energy@10% (WL) | 0.749 | 0.614 | -0.114 | -0.166 | +0.052 | pass | best r 0.01 |
| lowrank_weighted | v | energy@10% (WL) | 0.664 | 0.613 | -0.131 | -0.175 | +0.044 | pass | best r 0.01 |
| lowrank_weighted | o | energy@10% (WL) | 0.812 | 0.888 | -0.273 | -0.287 | +0.013 | fail | best r 0.01 |
| lowrank_weighted | gate | energy@10% (WL) | 0.896 | 0.681 | -0.112 | -0.187 | +0.074 | pass | best r 0.01 |
| lowrank_weighted | up | energy@10% (WL) | 0.691 | 0.680 | -0.181 | -0.188 | +0.007 | fail | best r 0.01 |
| lowrank_weighted | down | energy@10% (WL) | 0.545 | 0.519 | -0.178 | -0.191 | +0.013 | fail | best r 0.01 |
| lowrank_weighted | in_proj_qkv | energy@10% (WL) | 0.925 | 0.721 | -0.048 | -0.181 | +0.133 | pass | best r 0.01 |
| lowrank_weighted | in_proj_z | energy@10% (WL) | 0.961 | 0.749 | -0.045 | -0.278 | +0.233 | pass | best r 0.01 |
| lowrank_weighted | in_proj_ab | energy@10% (WL) | 0.895 | 0.438 | -0.664 | -0.942 | +0.279 | pass | best r 0.05 |
| lowrank_weighted | out_proj | energy@10% (WL) | 0.775 | 0.804 | -0.262 | -0.270 | +0.009 | fail | best r 0.01 |
| outliers | q | colnorm max/med | 1.98 | 1.06 | -0.032 | -0.053 | +0.021 | pass | Jaccard 0.204 / 0.006 |
| outliers | k | colnorm max/med | 2.78 | 1.22 | +0.005 | -0.050 | +0.055 | pass | Jaccard 0.294 / 0.006 |
| outliers | v | colnorm max/med | 2.16 | 1.33 | -0.034 | -0.048 | +0.014 | fail | Jaccard 0.071 / 0.006 |
| outliers | o | colnorm max/med | 1.93 | 1.13 | -0.042 | -0.051 | +0.009 | fail | Jaccard 0.080 / 0.005 |
| outliers | gate | colnorm max/med | 1.53 | 1.05 | -0.046 | -0.053 | +0.007 | fail | Jaccard 0.246 / 0.005 |
| outliers | up | colnorm max/med | 1.15 | 1.03 | -0.052 | -0.054 | +0.002 | fail | Jaccard 0.060 / 0.005 |
| outliers | down | colnorm max/med | 1.42 | 1.30 | -0.052 | -0.051 | -0.000 | fail | Jaccard 0.080 / 0.005 |
| outliers | in_proj_qkv | colnorm max/med | 1.38 | 1.05 | -0.043 | -0.053 | +0.010 | fail | Jaccard 0.227 / 0.004 |
| outliers | in_proj_z | colnorm max/med | 1.85 | 1.08 | -0.036 | -0.053 | +0.017 | fail | Jaccard 0.327 / 0.004 |
| outliers | in_proj_ab | colnorm max/med | 6.41 | 2.94 | +0.192 | -0.039 | +0.231 | pass | Jaccard 0.387 / 0.004 |
| outliers | out_proj | colnorm max/med | 1.68 | 1.23 | -0.043 | -0.047 | +0.004 | fail | Jaccard 0.080 / 0.005 |
| interlayer | q | min e, adjacent | 0.556 | 0.274 | -0.511 | -0.525 | +0.014 | fail |  |
| interlayer | k | min e, adjacent | 0.491 | 0.134 | -0.529 | -0.527 | -0.001 | fail |  |
| interlayer | v | min e, adjacent | 0.363 | 0.137 | -0.433 | -0.434 | +0.001 | fail |  |
| interlayer | o | min e, adjacent | 0.631 | 0.503 | -0.474 | -0.474 | -0.000 | fail |  |
| interlayer | gate | min e, adjacent | 0.206 | 0.168 | -0.364 | -0.433 | +0.070 | pass |  |
| interlayer | up | min e, adjacent | 0.157 | 0.165 | -0.412 | -0.435 | +0.024 | pass |  |
| interlayer | down | min e, adjacent | 0.158 | 0.166 | -0.396 | -0.442 | +0.046 | pass |  |
| interlayer | in_proj_qkv | min e, adjacent | 0.269 | 0.169 | -0.496 | -0.496 | -0.000 | fail |  |
| interlayer | in_proj_z | min e, adjacent | 0.570 | 0.500 | -0.492 | -0.498 | +0.005 | fail |  |
| interlayer | in_proj_ab | min e, adjacent | 0.301 | 0.021 | -0.126 | -0.511 | +0.384 | pass |  |
| interlayer | out_proj | min e, adjacent | 0.509 | 0.501 | -0.498 | -0.499 | +0.000 | fail |  |
| acts | o | Gini | 0.694 | split-half rho 0.956 | - | - | - | stable | ch. for 50/90 % energy 0.058/0.479 |
| acts | down | Gini | 0.427 | split-half rho 0.889 | - | - | - | noise | ch. for 50/90 % energy 0.203/0.715 |
| acts | out_proj | Gini | 0.686 | split-half rho 0.957 | - | - | - | stable | ch. for 50/90 % energy 0.059/0.483 |
| sensitivity | q | s_t/n_t | 3.36e-06 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0048 |
| sensitivity | k | s_t/n_t | 3.12e-06 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0025 |
| sensitivity | v | s_t/n_t | 8.19e-06 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0084 |
| sensitivity | o | s_t/n_t | 2.74e-07 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0116 |
| sensitivity | gate | s_t/n_t | 2.06e-06 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0128 |
| sensitivity | up | s_t/n_t | 1.11e-06 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0229 |
| sensitivity | down | s_t/n_t | 3.96e-08 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0234 |
| sensitivity | in_proj_qkv | s_t/n_t | 6.19e-06 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0143 |
| sensitivity | in_proj_z | s_t/n_t | 4e-06 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0087 |
| sensitivity | in_proj_ab | s_t/n_t | 1.07e-05 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0017 |
| sensitivity | out_proj | s_t/n_t | 4.22e-08 | - | - | - | - | no prediction (rho -0.05) | @only KL4 0.0272 |
| model.layers.23.mlp.down_proj | 23 | model.layers.23.mlp.down_proj | [1201, 5567] | -0.0308 | -0.01 | 0.0001 | -0.00 | spike_no_input |  |
| model.layers.23.mlp.down_proj | 23 | model.layers.23.mlp.down_proj | [1201, 945] | 0.914 | -0.00 | 0.0005 | -0.01 | top16 |  |
| model.layers.22.linear_attn.out_proj | 22 | model.layers.22.linear_attn.out_proj | [1201, 1203] | -0.551 | -0.01 | 0.0003 | -0.03 | top16 |  |
| model.layers.19.self_attn.v_proj | 19 | model.layers.19.self_attn.v_proj | [495, 1192] | 0.547 | -0.03 | 0.0004 | 0.02 | top16 |  |
| model.layers.19.self_attn.v_proj | 19 | model.layers.19.self_attn.v_proj | [431, 1360] | 0.512 | -0.02 | 0.0004 | -0.03 | top16 |  |
| model.layers.19.self_attn.v_proj | 19 | model.layers.19.self_attn.v_proj | [282, 525] | -0.512 | 0.02 | 0.0004 | -0.03 | top16 |  |
| model.layers.1.linear_attn.out_proj | 1 | model.layers.1.linear_attn.out_proj | [1201, 215] | -0.455 | -0.05 | 0.0006 | 0.61 | top16 |  |
| model.layers.19.self_attn.v_proj | 19 | model.layers.19.self_attn.v_proj | [446, 1789] | -0.451 | -0.01 | 0.0004 | -0.03 | top16 |  |
| model.layers.19.self_attn.v_proj | 19 | model.layers.19.self_attn.v_proj | [283, 493] | 0.451 | -0.01 | 0.0004 | -0.03 | top16 |  |
| model.layers.22.linear_attn.out_proj | 22 | model.layers.22.linear_attn.out_proj | [1201, 311] | -0.441 | -0.03 | 0.0005 | -0.01 | top16 |  |
| model.layers.19.self_attn.v_proj | 19 | model.layers.19.self_attn.v_proj | [288, 203] | 0.441 | -0.01 | 0.0004 | -0.03 | top16 |  |
| model.layers.1.linear_attn.out_proj | 1 | model.layers.1.linear_attn.out_proj | [1201, 1399] | 0.434 | 0.61 | 0.0011 | -0.05 | top16 |  |
| model.layers.23.mlp.gate_proj | 23 | model.layers.23.mlp.gate_proj | [3841, 1963] | 0.43 | -0.01 | 0.0001 | -0.00 | top16 |  |
| model.layers.23.mlp.gate_proj | 23 | model.layers.23.mlp.gate_proj | [3484, 1963] | 0.43 | -0.00 | 0.0000 | -0.01 | top16 |  |
| model.layers.22.linear_attn.in_proj_qkv | 22 | model.layers.22.linear_attn.in_proj_qkv | [4598, 1840] | 0.43 | -0.00 | 0.0002 | -0.01 | top16 |  |
| model.layers.19.self_attn.v_proj | 19 | model.layers.19.self_attn.v_proj | [428, 1344] | 0.43 | -0.01 | 0.0004 | -0.03 | top16 |  |
| model.layers.14.linear_attn.out_proj | 14 | model.layers.14.linear_attn.out_proj | [1201, 1872] | -0.422 | 0.03 | 0.0006 | -0.06 | top16 |  |

`minicpm5-2b`

| probe | type | stat | W | control | gain_bits | gain_control | difference | verdict | extra |
|---|---|---|---|---|---|---|---|---|---|
| marginal | q | H_4 | 3.508 | 3.561 | +0.491 | +0.439 | +0.052 | pass |  |
| marginal | k | H_4 | 3.430 | 3.560 | +0.569 | +0.440 | +0.130 | pass |  |
| marginal | v | H_4 | 3.374 | 3.560 | +0.626 | +0.440 | +0.186 | pass |  |
| marginal | o | H_4 | 3.561 | 3.561 | +0.439 | +0.439 | +0.000 | fail |  |
| marginal | gate | H_4 | 3.527 | 3.561 | +0.473 | +0.439 | +0.034 | pass |  |
| marginal | up | H_4 | 3.530 | 3.561 | +0.470 | +0.439 | +0.031 | pass |  |
| marginal | down | H_4 | 3.505 | 3.561 | +0.495 | +0.439 | +0.056 | pass |  |
| marginal | embed | H_4 | 3.559 | 3.556 | +0.441 | +0.444 | -0.003 | fail |  |
| marginal | lm_head | H_4 | 3.472 | 3.561 | +0.528 | +0.439 | +0.089 | pass |  |
| lowrank | q | energy@10% | 0.521 | 0.314 | -0.217 | -0.285 | +0.068 | pass | best r 0.01 |
| lowrank | k | energy@10% | 0.293 | 0.168 | -0.170 | -0.195 | +0.026 | pass | best r 0.01 |
| lowrank | v | energy@10% | 0.187 | 0.167 | -0.191 | -0.196 | +0.004 | fail | best r 0.01 |
| lowrank | o | energy@10% | 0.444 | 0.314 | -0.255 | -0.285 | +0.030 | pass | best r 0.01 |
| lowrank | gate | energy@10% | 0.258 | 0.212 | -0.179 | -0.191 | +0.012 | fail | best r 0.01 |
| lowrank | up | energy@10% | 0.235 | 0.212 | -0.184 | -0.191 | +0.007 | fail | best r 0.01 |
| lowrank | down | energy@10% | 0.254 | 0.212 | -0.178 | -0.191 | +0.013 | fail | best r 0.01 |
| lowrank | embed | energy@10% | 0.281 | 0.121 | -0.104 | -0.150 | +0.046 | pass | best r 0.01 |
| lowrank | lm_head | energy@10% | 0.268 | 0.121 | -0.058 | -0.150 | +0.092 | pass | best r 0.01 |
| outliers | q | colnorm max/med | 1.88 | 1.11 | -0.023 | -0.053 | +0.029 | pass | Jaccard 0.221 / 0.004 |
| outliers | k | colnorm max/med | 2.82 | 1.41 | +0.012 | -0.047 | +0.058 | pass | Jaccard 0.241 / 0.004 |
| outliers | v | colnorm max/med | 2.26 | 1.23 | -0.033 | -0.054 | +0.020 | pass | Jaccard 0.057 / 0.004 |
| outliers | o | colnorm max/med | 1.37 | 1.07 | -0.052 | -0.053 | +0.001 | fail | Jaccard 0.150 / 0.004 |
| outliers | gate | colnorm max/med | 2.34 | 1.20 | -0.033 | -0.051 | +0.018 | fail | Jaccard 0.292 / 0.004 |
| outliers | up | colnorm max/med | 2.30 | 1.18 | -0.029 | -0.051 | +0.022 | pass | Jaccard 0.427 / 0.004 |
| outliers | down | colnorm max/med | 1.85 | 1.62 | -0.039 | -0.049 | +0.010 | fail | Jaccard 0.150 / 0.004 |
| interlayer | q | min e, adjacent | 0.587 | 0.511 | -0.493 | -0.493 | -0.000 | fail |  |
| interlayer | k | min e, adjacent | 0.276 | 0.083 | -0.503 | -0.503 | +0.000 | fail |  |
| interlayer | v | min e, adjacent | 0.128 | 0.077 | -0.488 | -0.487 | -0.000 | fail |  |
| interlayer | o | min e, adjacent | 0.579 | 0.497 | -0.487 | -0.487 | -0.000 | fail |  |
| interlayer | gate | min e, adjacent | 0.186 | 0.167 | -0.405 | -0.443 | +0.038 | pass |  |
| interlayer | up | min e, adjacent | 0.160 | 0.167 | -0.429 | -0.446 | +0.017 | fail |  |
| interlayer | down | min e, adjacent | 0.161 | 0.168 | -0.412 | -0.442 | +0.030 | pass |  |
| acts | o | Gini | 0.367 | split-half rho 0.969 | - | - | - | stable | ch. for 50/90 % energy 0.253/0.718 |
| acts | down | Gini | 0.460 | split-half rho 0.902 | - | - | - | stable | ch. for 50/90 % energy 0.184/0.687 |
| sensitivity | q | s_t/n_t | 3.64e-06 | - | - | - | - | no prediction (rho 0.25) | @only KL4 0.0236 |
| sensitivity | k | s_t/n_t | 8.03e-06 | - | - | - | - | no prediction (rho 0.25) | @only KL4 0.0117 |
| sensitivity | v | s_t/n_t | 6.96e-06 | - | - | - | - | no prediction (rho 0.25) | @only KL4 0.0318 |
| sensitivity | o | s_t/n_t | 3.64e-07 | - | - | - | - | no prediction (rho 0.25) | @only KL4 0.0355 |
| sensitivity | gate | s_t/n_t | 2.02e-06 | - | - | - | - | no prediction (rho 0.25) | @only KL4 0.0306 |
| sensitivity | up | s_t/n_t | 2.14e-06 | - | - | - | - | no prediction (rho 0.25) | @only KL4 0.0461 |
| sensitivity | down | s_t/n_t | 1.25e-06 | - | - | - | - | no prediction (rho 0.25) | @only KL4 0.0819 |
| model.layers.7.mlp.down_proj | 7 | model.layers.7.mlp.down_proj | [106, 1460] | 1.05 | 205.59 | 1.2415 | 0.08 | spike |  |
| model.layers.40.mlp.down_proj | 40 | model.layers.40.mlp.down_proj | [106, 271] | 0.973 | -0.14 | 0.0014 | -0.13 | spike |  |
| model.layers.41.mlp.down_proj | 41 | model.layers.41.mlp.down_proj | [106, 4621] | -0.824 | -0.01 | 0.0002 | -0.08 | spike_no_input |  |
| model.layers.40.mlp.gate_proj | 40 | model.layers.40.mlp.gate_proj | [1331, 106] | 2.19 | 0.02 | 0.0009 | -0.20 | top16 |  |
| model.layers.40.mlp.up_proj | 40 | model.layers.40.mlp.up_proj | [271, 750] | 1.98 | 0.25 | 0.0027 | 0.06 | top16 |  |
| model.layers.7.mlp.gate_proj | 7 | model.layers.7.mlp.gate_proj | [4426, 1108] | -1.84 | 1.23 | 0.0118 | 0.11 | top16 |  |
| model.layers.41.mlp.gate_proj | 41 | model.layers.41.mlp.gate_proj | [5497, 106] | -1.84 | -0.03 | 0.0002 | 1.26 | top16 |  |
| model.layers.31.mlp.gate_proj | 31 | model.layers.31.mlp.gate_proj | [4411, 106] | -1.83 | 0.33 | 0.0008 | 0.01 | top16 |  |
| model.layers.41.mlp.up_proj | 41 | model.layers.41.mlp.up_proj | [5497, 106] | 1.8 | 0.05 | 0.0002 | -0.08 | top16 |  |
| model.layers.40.mlp.up_proj | 40 | model.layers.40.mlp.up_proj | [1331, 106] | 1.74 | 0.06 | 0.0008 | 0.25 | top16 |  |
| model.layers.40.mlp.up_proj | 40 | model.layers.40.mlp.up_proj | [1290, 106] | 1.73 | 3.02 | 0.0100 | 0.25 | top16 |  |
| model.layers.40.mlp.gate_proj | 40 | model.layers.40.mlp.gate_proj | [271, 106] | 1.71 | -0.20 | 0.0014 | 0.02 | top16 |  |
| model.layers.7.mlp.gate_proj | 7 | model.layers.7.mlp.gate_proj | [3516, 1108] | -1.68 | 0.11 | 0.0019 | 1.23 | top16 |  |
| model.layers.7.mlp.gate_proj | 7 | model.layers.7.mlp.gate_proj | [3919, 1108] | -1.62 | 0.11 | 0.0012 | 1.23 | top16 |  |
| model.layers.28.mlp.down_proj | 28 | model.layers.28.mlp.down_proj | [106, 4293] | 1.62 | 0.01 | 0.0007 | 0.03 | top16 |  |
| model.layers.22.mlp.up_proj | 22 | model.layers.22.mlp.up_proj | [3256, 106] | -1.6 | 0.06 | 0.0008 | 0.00 | top16 |  |
| model.layers.14.mlp.down_proj | 14 | model.layers.14.mlp.down_proj | [106, 1564] | 1.56 | 0.15 | 0.0008 | 0.01 | top16 |  |
| model.layers.7.mlp.up_proj | 7 | model.layers.7.mlp.up_proj | [3127, 1108] | 1.53 | 0.02 | 0.0011 | 0.06 | top16 |  |
| model.layers.36.mlp.down_proj | 36 | model.layers.36.mlp.down_proj | [106, 3611] | -1.52 | 0.07 | 0.0006 | -0.00 | top16 |  |

Row counts behind the aggregates: 0.8B 187 tensors (186 block linears + tied embedding), 2B 187, MiniCPM5 296
(294 + embed + lm_head); P3 adds 10 / 10 / 6 cross-layer records; P7 17 / 17 / 19 candidates; P6b 186 (0.8B).

### P1 marginals and entropy (weights only; `marginal.json`)

Hypothesis: most tensors excess kurtosis > 1; index entropy coding of 4-bit RTN saves 0.3-0.6 bits/param, shape
beyond Gaussian < 0.15; bf16 lossless bound 16 - H_word in 2-4 bits.
Run: every block linear + embedding (+ lm_head on MiniCPM), all three models, 15 / 74 / 34 s, no data.
Result: kurtosis > 1 on 151/187, 135/187, 192/296 tensors; Student-t best by NLL on 185/187, 187/187, 296/296
(nu 5-28). Type means: v 4.4 / 6.3 / 1.7, k 2.7 / 3.5 / 6.9, in_proj_ab 2.4 / 4.2, q 2.7 / 2.1 / 4.7, down
1.3 / 1.4 / 2.9, up 0.5 / 0.4 / 2.6, o 1.8 / 1.9 / 0.8; embedding 0.45 / 0.76 / 0.13. Entropy-coding gain at 4
bits (one 16-bit table per tensor) numel-weighted over the block linears 0.500 / 0.508 / 0.479 bits/param, per
type 0.44-0.77; the Gaussian control (same per-group std, same grid) gives 0.433-0.444 everywhere. Difference
"shape beyond Gaussian": v +0.165 / +0.322 / +0.186, in_proj_ab +0.076 / +0.337, k +0.100 / +0.238 / +0.130,
o +0.163 / +0.133 / +0.000, out_proj +0.118 / +0.096, big MLP types +0.03..+0.07; numel-weighted over all block
linears +0.060 / +0.069 / +0.040. The 11 / 42 / 24 tensors above 0.6 bits hold 1.95% / 1.06% / 0.65% of the
block parameters. bf16 lossless bound 16 - H_word: 5.30-5.61 / 5.31-5.52 / 5.04-5.53 bits, Gaussian control
5.13-5.59; H_exp 2.55-3.02 (the 8-bit exponent field carries ~2.6 bits), H_sign + H_exp + H_mant = 10.63 vs
H_word 10.58 on record 0 (fields nearly independent).
Control: Gaussian with matched per-group std. Its measured excess kurtosis is 0.16-2.4, not the 6/(g-1) = 0.05
the record note attributes to 128-sample std estimation: the control keeps W's between-group scale
heterogeneity (that is what a mixture of Gaussians with different scales looks like), so the run reports'
explanation of the control's kurtosis was wrong (audit); the control itself is the intended one.
Audit GPU check (not a probe file): rescaling the input columns of a tensor to equal norm before the grid removes
most of the beyond-Gaussian gain for o / out_proj / k / in_proj_qkv (layers.3.o_proj H_4 3.421 -> 3.509, 62% of
the 0.141 gain over the control; layers.7.o_proj 0.109 -> 0.054) and almost none for v, down, gate (v 3.490 ->
3.492). So the P1 gain splits into (i) Gaussian + per-group scale, 0.44, the control; (ii) within-group column
scale heterogeneity, recoverable instead by per-column scaling or the P6 column allocation (do not add the two);
(iii) a true heavy tail of 0.03-0.3 on v, in_proj_ab, k, down, gate.
Verdict: **supported** for kurtosis > 1 and for the 0.3-0.6 type-level range; **refuted** for "< 0.15 beyond
Gaussian" on v (all three models), k and in_proj_ab (2B), o (0.8B), though these are 1-2% of the parameters and
the model-level excess is 0.04-0.07; **refuted** for the 2-4 bit lossless range: it is ~5.4 bits and the control
gives the same, so bf16 lossless coding has no structure beyond the marginal to use. Nothing rerun.

### P2 low rank, plain and activation-weighted (`lowrank.json`, `lowrank_weighted.json`)

Hypothesis: for q/o/gate/up/down the gain after overhead is < 0.05 above the control at every r; k/v may show
rank deficiency; P2b rank lower than plain rank for down/o/out_proj; negative result for method B.
Run: P2 all tensors, three models (103 / 182 / 133 s; f32 svdvals, f64 Gram eigenvalues for the embeddings);
controls seeds 0 and 1. P2b 0.8B and 2B (38 / 133 s), W @ L with L the Cholesky factor of the C4 Hessian from
P5's hooks; control W shuffled, same L. P2b gain = measured RTN-4 output-error ratio of the quantized residual
(not the e_r formula; see note).
Result (plain): energy at 10% rank q 0.47 / 0.51 / 0.52, in_proj_z 0.43 / 0.48, o 0.39 / 0.48 / 0.44, down
0.27 / 0.27 / 0.25, up 0.25 / 0.25 / 0.24 (control 0.12-0.31). Best gain after bf16 U, V is **negative for
every block type on every model**, best r always the smallest (1%): q -0.031 / -0.090 / -0.217, o -0.175 /
-0.238 / -0.255, gate -0.141 / -0.156 / -0.179, up -0.170 / -0.178 / -0.184, down -0.169 / -0.177 / -0.178,
k -0.168 / -0.145 / -0.170, v -0.181 / -0.153 / -0.191, in_proj_ab -0.657 / -0.715; controls -0.18..-0.95.
Numel-weighted over block linears -0.140 / -0.167 / -0.190. Structure above control: q +0.148 / +0.124 / +0.068,
in_proj_z +0.089 / +0.089, in_proj_qkv +0.068 / +0.063, k +0.046 / +0.035 / +0.025, v +0.033 / +0.027 / +0.004,
MLP types +0.007..+0.044, growing with r (q +0.44..+0.48 at r = 25%) while both absolute gains fall further
below zero. Positive absolute gains in 4 / 2 / 2 tensors only (layer-0 in_proj_qkv +0.050, two late q_proj,
MiniCPM layers.0.q_proj +0.311) and the embedding: +0.039 / +0.040 (Qwen, control -0.149), MiniCPM embed -0.104,
lm_head -0.058. Control-seed spread 3e-5..5e-3 bits.
Result (P2b): Hessian-weighted stable rank collapses to 1.2-1.6 for q / in_proj_z / in_proj_ab / gate /
in_proj_qkv (plain 12-37 / 21-65), 4-11 for out_proj / up / down (plain 77-290), **but the shuffled control
under the same L collapses too** (0.8B down 16.8 / o 2.2 / out_proj 3.4 vs W 9.1 / 3.7 / 4.7): the collapse is a
property of the activation Hessian, not of W. The contract's e_r formula would say +1.74 (q) / +1.38 (in_proj_z)
/ +1.10 (in_proj_qkv); the measured RTN-4 error of the residual is 1.03-1.85x *worse* than RTN-4 on W (the
residual is not what the grid sees), so the honest gain is q +0.067 / -0.035 (0.8B / 2B; control -0.161 /
-0.206), in_proj_z -0.029 / -0.045, in_proj_qkv -0.057 / -0.048, gate -0.080 / -0.112, down -0.152 / -0.178,
o -0.216 / -0.273, out_proj -0.198 / -0.262. The only positive is q on the 0.8B (11 positive records, all
q_proj / in_proj_z); q is 5.1% of block params and 3% of the joint KL, so model-wide < 0.004 bits/param.
Audit GPU check: the plain-SVD formula matches the measured RTN-4 MSE ratio within 0.02 bits on six q_proj and
one down_proj; for the embedding (rank 10) the measured ratio is 1.40x, +0.087 bits/param measured vs +0.039 by
the formula, so the embedding gain is real and slightly larger than estimated (it was never quantized here).
Verdict: **supported** in conclusion (method B is dead for block linears on every model), but the numeric
"< 0.05 above control at every r" clause was mis-specified and fails for q on all three models and for
o/gate/down/up at r >= 5-10% (74 / 43 / 36 / 24 / 20 records on the 0.8B); "k/v rank deficiency" **refuted**
(+0.03..+0.05, no more than q); "P2b rank lower for down/o/out_proj" holds numerically (9 / 4 / 5 vs 131 / 81 /
77) but is **not structure in W** (control equally low) and buys nothing; the formula gain is **contradicted by
the measured residual error and is not trusted** (contract rule). Nothing rerun.

### P3 outlier columns (`outliers.json`, direct check `+keepcols`)

Hypothesis: < 1% of hidden input channels have norms > 5x median, the same channels across layers (Jaccard >
0.3 vs control < 0.05); keepcols=1% recovers > 25% of the RTN-4 KL on the 0.8B for ~0.13 bits/param.
Run: all tensors, three models, 6 / 7 / 10 s; keepcols 0.5 / 1 / 2 at 4 and 3 bits on the 0.8B, 1 at 4 bits on
the 2B and MiniCPM (full protocol).
Result: columns > 5x median per tensor: 0 for every type except in_proj_ab 0.47 (0.8B), in_proj_ab 3.25 and k
0.67 (2B), k 1.14 and q 0.31 (MiniCPM); > 10x only MiniCPM k (0.33). Totals 17 / 121 / 61 columns over 138 /
138 / 210 hidden-input tensors. Column max/median 1.2-2.8 (in_proj_ab 4.6 / 6.4) vs control 1.03-1.6. Column-
norm kurtosis is high (7-26 Qwen, gate/up 480-490 on MiniCPM vs control < 1 (MiniCPM 58-69)): the norm vector
is heavy-tailed, but its extremes are 2-3x, not 5x. Cross-layer Jaccard of the top-16 channels: in_proj_a/b
0.26-0.39, in_proj_z 0.32-0.33, k 0.19-0.29, gate 0.19-0.29, MiniCPM up 0.43, pooled over all hidden-input
linears 0.070 / 0.080 / 0.150; control 0.004-0.008. MSE-proxy keepcols gain negative for every type except
in_proj_ab (+0.005 / +0.192) and k on the 2B / MiniCPM (+0.005 / +0.012), control -0.04..-0.05; the selection
effect alone (control rho 1.01-1.03) is what the control subtracts.
Direct check: keepcols=1% recovers **3.2% / 7.8% / 17.0%** of the RTN-4 KL (0.2051 -> 0.1986, 0.1331 ->
0.1227, 0.2682 -> 0.2227) for +0.118 bits/param; 0.5 / 2% on the 0.8B 1.3% / 8.3%; at 3 bits 12.0 / 12.3 /
17.3% for +0.064 / +0.128 / +0.256. Against the RD curve (table above) every point except keepcols=0.5 at 3
bits is a worse use of bits than uniform index bits, and MiniCPM is the only model near break-even. The MSE
proxy is confirmed on the 0.8B (estimate -0.09, measured net -0.095 bits/param at D ~ 2^-2b) and underestimates
where real outlier columns exist (MiniCPM: estimate -0.08, measured +0.016).
Verdict: **refuted** on the direct check (3% not > 25%); "< 1% channels > 5x" holds trivially (almost none
exist: the LLM.int8 / AWQ picture of a few huge input channels does not transfer to these models' weights at
g128); Jaccard > 0.3 holds only for the DeltaNet a / b / z gates (16- and 2048-row tensors) and MiniCPM up_proj,
pooled 0.07-0.15 (> control, < 0.3). The AWQ-style "known" gain does not reproduce with weight-norm column
selection; if columns are to be protected in Phase 3 they should be selected by activation energy (diag H) and
judged against the group-size curve, not against RTN g128. Nothing rerun; nothing failed.

### P4 inter-layer correlation (`interlayer.json`)

Hypothesis: residual energy after alignment > 0.9 for every type and pair, control within 0.03; differential
coding gain negative after overhead (method C dead).
Run: all types, adjacent pairs and best pair, raw / Procrustes out / Procrustes in / permutation (gate, up,
down), three models (220 / 683 / 1636 s; the 6144 x 6144 assignment takes ~410 s per MLP type on MiniCPM).
Result: raw |cos| between layers < 0.05 (in_proj_ab 0.20-0.27), e_raw 1.84-2.11 with the control within 0.04
(in_proj_ab 1.50 vs 2.04). Output-side Procrustes e_out 0.14-0.27 for gate / up / in_proj_qkv, but the control
gives 0.09-0.17 (R has N x N free parameters for an N x K tensor with N > K: it fits anything), and the bf16 R
costs 16 N/K bits/param, gain_out -4..-95. Permutation (the only cheap alignment, log2(M!)/(NK)): e_perm gate
1.58 / 1.65 / 1.74, up 1.70 / 1.76 / 1.80, down 1.63 / 1.72 / 1.76 vs control 1.76-1.84, i.e. e > 1 everywhere
(the aligned W_i is worse than predicting zero). Best gain per type -0.34..-0.53 (0.8B), -0.36..-0.53 (2B),
-0.41..-0.50 (MiniCPM); controls -0.42..-0.53; structure above control at most +0.08 (gate, via the
permutation). Positive-gain records 10 / 11 / 0, all in_proj_a / in_proj_b pairs (+0.07..+0.16 bits on 16K
parameters). Audit GPU check: the matched-neuron cosine of W is 0.17-0.22 vs 0.11-0.12 for shuffles, so the
neuron correspondence between adjacent MLPs is real but worthless for coding (needs e < 1).
Verdict: **supported** in conclusion (method C dead on every model, best -0.34 bits/param); the two numeric
sub-clauses fail literally (e_out, e_in are far below 0.9 and far from the control) because the rotated side is
the larger dimension on both W and the control; read "> 0.9 / within 0.03" against e_raw and e_perm only, where
it holds. Nothing rerun.

### P5 activation sparsity and neuron usage (`acts.json`)

Hypothesis: Gini of intermediate-channel energy > 0.5, 20% of channels carry > 60%, split-half Spearman > 0.9.
Run: C4, 32 windows x 2048 tokens seed 0, split half seed 2, hooks on every block linear input (46 / 88 / 162 s,
peak 2.5 / 5.3 / 5.8 GB).
Result, down_proj input: Gini 0.436 / 0.427 / 0.460; 19.4 / 20.3 / 18.4% of channels carry 50%, 71.6 / 71.5 /
68.7% carry 90%; top-20% share 0.524 / 0.516 / 0.540; split-half Spearman mean 0.928 / 0.889 / 0.902, min 0.812
(layer 3) / 0.783 (layers 2-3) / 0.767 (layer 36), **7/24, 13/24, 16/42 layers below 0.9** (counted from the
records; the 2B run report's "8 of 24" was wrong and the 0.8B report omitted its 7). Per-layer down Gini rises
with depth (0.8B 0.23 at layer 3 to 0.86 at layer 23; MiniCPM 0.29-0.53 except layers 1 / 7 / 40 / 41 at
0.83-1.00, the massive-activation layers). o_proj / out_proj input on Qwen: Gini 0.743 / 0.756 (0.8B), 0.694 /
0.686 (2B), top-20% share 0.72-0.78, Spearman 0.956-0.967, 0-1 layers < 0.9; MiniCPM o Gini 0.367, share 0.44,
Spearman 0.969. Hidden-state inputs Gini 0.47-0.58 (q/k/v on MiniCPM 0.54, gate/up 0.29), Spearman 0.96-0.99.
Channels active on < 1% of tokens: 0 to 4e-4 everywhere (no dead channels, nothing sparse to code).
Control: split-half agreement, as stated in the contract (Gini is permutation invariant); Gini of half 1 and
half 2 agree to 0.01 everywhere.
Verdict: **refuted for down_proj** on all three models (Gini 0.43-0.46 < 0.5, share 0.52-0.54 < 0.6, usage
pattern not stable in a third to half of the layers), **supported for the attention / DeltaNet output inputs on
Qwen** (o / out_proj: Gini 0.69-0.76, stable), not on MiniCPM o (0.37). Nothing here converts to bits except
through the P6 column allocation, which is untested (below). Nothing rerun.

### P6 sensitivity: Hessian-diagonal proxy (a) and Fisher (b) (`sensitivity.json`, `fisher.json`)

Hypothesis: s_t / n_t spans > 2 decades; allocation saving > 0.3 bits/param at tensor and > 0.5 at column
granularity; Spearman(s_t, measured `@only` KL) > 0.7; Fisher rank-1 fit > 70%.
Run: (a) all block linears, three models, s_t = sum_j H_jj sum_i (W - Q(W))^2_ij from the P5 hooks and RTN-4;
prediction check filled by a `--probes p6a` rerun after the ablation rows existed (68 / 36 / 9 rows seen,
reproduced with scipy by the audit). (b) 0.8B, 32 C4 windows x 1024 tokens, bf16 forward, f32 squared gradients
of the summed window NLL per position (position-wise K-FAC form; a batched per-window variant is stored next to
it), nothing written back (`checksum_unchanged` true), 36 s, peak 12.6 GB.
Result (a): s_t / n_t spans 4.6 / 4.5 / 4.7 decades. Headline allocation saving tensor 1.10 / 0.99 / 0.69,
128-group 1.27 / 1.06 / 0.75, column 1.37 / 1.33 / 0.93 bits/param, but with 21% / 13% / 2% of the weights
floored at 0 bits; capped at b_u >= 2 bits: tensor 0.59 / 0.59 / 0.54, column 0.83 / 0.86 / 0.71; within-type
only 0.19 / 0.32 / 0.58; on the relative output error s_t / ||W X||^2 0.30 / 0.29 / 0.30 (no flooring).
**Prediction check: Spearman(s_t per type, `@only` KL at 4 bits) = -0.33 / -0.05 / +0.25** (11 / 11 / 7 types;
-0.33 at 3 bits too), per layer -0.28 / -0.51 (24 layers; the 2B value is significantly *inverted*, p = 0.01: s
grows toward layers 16-23 while the measured `@layer` damage peaks at layers 0 and 23). s_t ranks in_proj_qkv >
gate > in_proj_z > q > ... > down > o > out_proj, the measurement ranks out_proj > down > up > o > ...: the proxy
follows the input RMS of the linear (1.0-1.4 for hidden-state inputs, 0.09-0.23 for down / o / out_proj) and
misses how a linear's output error reaches the loss. The implied allocation would give down -4.2, out_proj -4.0,
o -2.8 bits relative to the mean on the 0.8B (-2.9 / -2.8 / -1.5 on the 2B): it starves exactly the three types
with the largest measured damage. s_rel (relative output error) is 0.011-0.026 for every type (a factor 2.4), so
the 4.5-decade spread is input scale, not sensitivity.
Result (b): rank-1 fit F ~ r c^T 0.90-0.99 for every type (min single tensor 0.661, 1 of 186 below 0.7) vs
shuffled control 0.03-0.31; Gini across elements 0.49 (down) to 0.80 (in_proj_qkv / in_proj_z). The fit is not
a few-token artifact (audit: top position 0.6-4%, top 1% of positions 10-22% of the trace, position 0 <= 2.8%),
but F = sum_t (d_t^2)(x_t^2)^T is a sum of non-negative rank-1 terms, so row x column separability is the
expected K-FAC form, not new structure. Tensor-level Fisher allocation gain 0.478 (12 levels, 0% floored);
element-level saving 0.34-1.20 but overhead 4.3-5 bits (negative); row granularity large for hidden-input types
(in_proj_qkv +0.90, in_proj_z +0.77, q +0.64 vs control 0.02-0.04), column granularity for o / out_proj
(+1.05 / +0.94 vs 0.02-0.05), i.e. the output-row sensitivity varies where the input-column one does not and
vice versa. **Post hoc (audit, not pre-registered): Spearman(Fisher s per type, `@only` KL at 4 bits) = +0.96
on the 0.8B** (+0.89 for the Fisher total): the gradient-weighted proxy ranks the types the way the measurement
does, out_proj > down > up > o > ...
Audit-derived allocation from the measurement itself (scratchpad computation, not a results file; to be
re-derived in Phase 3 code): with d_t = `@only` KL_t / n_t and D ~ 2^-2b (per-type KL(3)/KL(4) = 3.4-4.5
supports it), reverse water-filling across the 11 / 11 / 7 types gives 0.236 (0.8B 4 bit), 0.217 (0.8B 3 bit),
0.190 (2B), 0.148 (MiniCPM) bits/param, i.e. KL x 0.72-0.81 at equal bits, with b_t 5.2-5.9 for v / in_proj_ab
/ o and 3.3-3.4 for gate / in_proj_qkv around a 4-bit mean. Damage per parameter correlates with 1 / n_t at
Spearman 0.80 / 0.91 / 0.85: small tensors are damage-dense (in_proj_ab 3.2e-9, v 3.2e-9, o 2.0e-9 vs gate
1.6e-10, in_proj_qkv 1.5e-10 nats per parameter on the 0.8B), which no probe predicted.
Verdict: spread **supported**; allocation thresholds met only on the capped numbers (0.54-0.59 tensor, 0.71-0.86
column), and the headline 1.10 / 1.37 is **refuted as a lever** because it rests on a between-type ranking that
the measurement contradicts (flagged, not averaged, per the contract rule); "proxy predicts" **refuted on all
three models** for H_jj and **supported post hoc for the Fisher diagonal** (0.8B only, one check); Fisher rank-1
> 70% **supported** (but expected from the construction). The within-tensor row / column numbers (0.3-1.05) are
proxy only: no pre-registered ablation tests them (`+keepcols` selects by weight norm, a different criterion).
The measured type-level allocation lever is 0.15-0.24 bits/param. Rerun: the p6a recheck on every model (the
first pass had no ablation rows to compare against); nothing failed.

### P7 super weights (`superweight.json`, direct check `+keepsuper`)

Hypothesis: a super weight exists in an early down_proj of Qwen3.5-0.8B (zeroing: dPPL > 100%, magnitude
control < 2%); MiniCPM5 is the positive control.
Run: one C4 window for the down_proj input / output spikes (20x median rule), global top-16 |w|, zeroing tests
on 16 WikiText-2 windows with magnitude and random controls (118 / 117 / 267 s); `+keepsuper` rows as listed.
Result, 0.8B: no output spike (layer max 0.46-1.24, best 11.8x median at layer 23, threshold 20x), no
co-occurring candidate; 17 candidates zeroed: dPPL -0.19..+0.04%, KL <= 0.0012; magnitude controls <= +0.12%,
random <= +0.04%. 2B: down_proj output channel 1201 is the max-output channel in all 24 layers (massive-
activation channel), layer 23 out_max 33.5 = 20.6x median but no input spike (46.75 vs 122.5 at layer 14),
`n_spike_candidates` 0; best of 17: layers.1.out_proj[1201,1399] dPPL +0.61%, KL 0.0011; magnitude controls <=
0.61%, random <= 0.01%. MiniCPM5: channel 106 is the max-output channel in layers 7-41; layer 7 down_proj
out_max 4640 (273x median) with in_max 4320 (165x), both on token 0 (Yu et al. recipe): **layers.7.mlp.down_proj
[106,1460] = 1.055; zeroed: dPPL +205.6%, KL 1.2415, top1 0.512; magnitude control +0.08%, random +0.01%**. A
second co-occurring spike at layer 40 ([106,271], dPPL -0.14%) and an output-only spike at layer 41 are not super
weights; the 16 largest |w| (1.5-2.2, all in gate/up/down rows or columns of channel 106 or 1108) give dPPL
-0.2..+3.0%. `+keepsuper` (18 weights restored, bits +3.5e-7): KL 0.26742 vs 0.26816, **0.3% of the RTN-4 KL**,
PPL 17.32 vs 17.45 (inside noise). The super weight is already represented by its group's 4-bit grid (error at
most the amax clip, 12.5%, if it is the group max, else <= scale/2), so protecting it buys nothing at 4 bits; it
is a pruning / zeroing hazard, not an RTN lever.
Verdict: **refuted for Qwen3.5** (no super weight in either DeltaNet-heavy model; the 2B has a persistent
massive-activation channel without a single critical weight), **positive control passed on MiniCPM5**, and the
conversion to bits is **negative** (0.3%). The Qwen `+keepsuper` rows were correctly skipped (no candidate over
dPPL 10%). Nothing failed.

### Classification of the findings

Reproduced known results: heavy-tailed weights, Student-t best, entropy coding ~0.5 bits/param of which 0.44 is
the Gaussian + group-scale part (P1); no usable low rank in block linears (P2); no inter-layer redundancy (P4);
GPTQ >> RTN and Hadamard + GPTQ > GPTQ at 3 bits (QuaRot / SpinQuant-type result, here +0.23-0.26 bits at 3
bits on one model); a super weight in an early down_proj of the Llama-architecture model with +206% PPL when
zeroed, and a massive-activation channel that persists across layers (MiniCPM 106, Qwen 2B 1201); Fisher-diag
row x column separability (K-FAC form); GPTQ calibration-seed spread of 1-5%.
Small refinements: the 4-bit RTN loss is spread over types, with the output-writing types out_proj / down / up
(o on MiniCPM) at 53-61% of the joint KL and the pre-registered candidates in_proj_ab and v at 1-6%; `@only`
losses are additive at 4 bits (0.95-1.04) and super-additive at 3 bits (0.51, by layer 0.48-0.55); the first and
last layers carry the largest 3-bit damage; g64 / g32 sit on the RTN g128 RD curve (group size is not a lever at
4 bits, factor 1.39 for +0.375 bits); the P2b weighted-rank collapse is an activation property; the embedding
has a small real low-rank gain (+0.04 formula, +0.09 measured MSE) if it is ever quantized; no dead or sparse
channels anywhere.
New / unexpected: (1) the LLM.int8 / AWQ outlier-channel picture does not show in these weights (no column > 5x
median outside in_proj_ab / k) and weight-norm keepcols buys 3-8% of the KL on Qwen and 17% on MiniCPM for
0.118 bits, worse than uniform index bits; (2) the GPTQ-objective Hessian-diagonal sensitivity proxy is
uncorrelated or anti-correlated with the measured per-type and per-layer damage on all three models (-0.33 /
-0.05 / +0.25; per layer -0.28 / -0.51), because it is input-RMS dominated, while the Fisher diagonal predicts it
(+0.96, 0.8B, post hoc); (3) damage per parameter tracks 1 / n_t (Spearman 0.80-0.91): the small tensors v,
in_proj_ab, o, out_proj, k are the damage-dense ones and the big MLP / in_proj_qkv tensors the cheap ones, so the
measured type-level allocation lever points the opposite way from the H_jj proxy; (4) Qwen3.5 has no super
weight, and even the textbook MiniCPM super weight is worth 0.3% of the RTN-4 KL when protected; (5) down_proj
neuron usage is only mildly uneven and partly sample-dependent; (6) the bf16 lossless bound is ~5.4 bits and
entirely marginal.

Negative results (all with controls, all at or below the noise floor where applicable): P2 low-rank + residual
on every block type and model; P2b weighted low rank (all but q on the 0.8B, < 0.004 bits model-wide); P3
keepcols at 0.5-2% on the 0.8B (4 and 3 bits), 1% on the 2B and MiniCPM; P3 MSE proxy for every type but
in_proj_ab and k; P4 differential coding at every alignment; P5 for down_proj (Gini, share, stability) and
MiniCPM o; P6a prediction on three models; P6a headline allocation; P7 on both Qwen models; `+keepsuper` on
MiniCPM (0.3%); H-P0 concentration on Qwen; g32 at 4 bits (1.39x, below the curve); the GPTQ f32-Hessian
acceptance (-4.6%, outside the 0.5% rule; attributed to rounding chaos); bf16 lossless bound beyond the marginal
(none); P1 shape beyond Gaussian for the big MLP types (0.03-0.07).

**Correction (2026-11-08, REPORT.md fact-check).** New / unexpected (3), "damage per parameter tracks 1 / n_t", is
**retracted**: d_t = KL_t / n_t shares the denominator with 1 / n_t, and a KL independent of n gives the same
correlation. With KL permuted across types (5000 permutations, numpy rng seed 1; `scripts/report_numbers.py` section
(7)), Spearman(d, 1/n) on the WikiText-2 `rtn-sym-g128-4@only` KL is 0.802 / 0.908 / 0.850 against null medians 0.820 /
0.876 / 0.945, p(null >= observed) 0.582 / 0.312 / 0.853 (C4 damage rows: 0.624 / 0.322 / 0.767), i.e. it does not pass
a shuffle test. Per-type KL instead rises with n (Spearman +0.636 / +0.802 / +0.605, log-log slope 0.44 / 0.40 / 0.29):
only "KL is sub-linear in n" remains, a descriptive fact. The paragraph in P6 above and item (3) are left as written.
Item (4), "Qwen3.5 has no super weight", is classified in REPORT.md as consistent with known work, not new: both Qwen3.5
configs set `attn_output_gate: true`, and SDPA output gating is reported to suppress massive activations (Qiu et al.
2025, Gated Attention), which super weights produce (Yu et al. 2024); the DeltaNet attribution was a prior, never tested.

### Audit findings and how each was resolved

Hypothesis audit (verdicts recorded above as stated):
- H-P0 concentration refuted / additivity supported at 4 bits / group size not a lever (medium): recorded;
  candidates replaced by out_proj / down / up in the Phase 3 reading.
- H-P1 partly refuted (< 0.15 beyond Gaussian fails on v, k, in_proj_ab, o; lossless bound ~5.4 not 2-4)
  (medium): recorded with the parameter shares (1-2%).
- H-P2 numeric clause mis-specified, k/v deficiency refuted, P2b collapse is an activation property, formula gain
  contradicted by the measured residual error (medium): recorded; formula numbers not trusted.
- H-P3 refuted on the direct check (medium): recorded; keepcols compared against the RD curve.
- H-P4 supported in conclusion, sub-clauses fail literally for e_out / e_in (low): recorded with the N > K
  explanation.
- H-P5 refuted for down_proj; the 2B report's "8 of 24 layers < 0.9" is wrong (13 of 24), the 0.8B report
  omitted 7 of 24 (low): corrected here from the records (7 / 13 / 16).
- H-P6 prediction refuted, allocation headline not trustworthy (medium): recorded; the cap-2-bit and within-type
  figures are quoted next to the headline, and the headline is flagged.
- Fisher predicts the measured damage, Spearman +0.96 (low): added to the P6 verdict as post hoc, not
  pre-registered; Phase 3 to re-derive allocation from the Fisher or from the `@only` rows, not from H_jj.
- H-P7 refuted on Qwen, positive control passed, +keepsuper 0.3% (medium): recorded.
- 2B run-report prose error on the split-half count (low): corrected above.
Structure-table audit:
- P6a allocation savings do not survive; the `@only`-implied type allocation is 0.15-0.24 bits/param and points
  the other way (high): recorded as the only measured allocation lever; the P6a model-wide and within-type
  numbers are recorded as refuted levers; the 0.3 "relative" variant has no prediction check either.
- P6 within-tensor row / column savings are proxy numbers without an ablation; the rank-1 fit is the expected
  K-FAC form (medium): labelled "proxy, untested" above; one pre-registered ablation needed before trusting any
  of the 0.3-1.0 figures.
- P1 shape-beyond-Gaussian partly column-scale heterogeneity; the std-estimation explanation of the control's
  kurtosis was wrong (medium): recorded with the three-way split; P1 and P6-column gains are not to be added.
- Shuffle-test passes with gain_bits <= 0 are "not white noise", not levers; in_proj_ab passes every probe but is
  0.12% / 0.09% of the parameters and 1% of the joint KL (medium): reading rule stated at the top of this
  section; in_proj_ab's passes are reported with its share. Left: no "realizable" column added to
  `structure_table.py` (presentation; the rule in NOTES suffices for now).
- P3 proxy confirmed on the 0.8B, underestimates on MiniCPM; keepcols dominated by smaller groups (medium):
  recorded; Phase 3 column protection, if any, by activation energy and judged against the group-size curve.
- P2 embedding gain survives and is larger when measured (+0.087) (low): recorded; applies only to the
  embedding (34% / 27% of the parameters, unquantized here); verify with a PPL run only if Phase 3 quantizes it.
- GPTQ acceptance tolerance below the seed spread (low): recorded as mis-specified; f64 kept for Phase-1-
  comparable rows, f32 allowed for new rows after a two-seed check.
- P5 is a property of o / out_proj inputs only (low): recorded.
Provenance audit:
- Pre-registration hash stale (68c5deb59e vs 9662cc2cfa) (low): the provenance paragraph at the top of this
  section records it; the hypotheses text is unchanged.
- `WCOMP_HESSIAN_F32` env default (low): recorded under the acceptance item; not triggered; left for Phase 3.
- Checked and OK: no code changed by the run step; pytest 220 passed; one code version on every row / JSON;
  Phase 1 rows untouched and reproduced; all deviations as reported; GPU freed.
Plot / table audit (applied to `scripts/plot_probe.py`, `scripts/plot_rd.py`, `scripts/structure_table.py`;
data files unchanged; 58 plot tests pass; figures regenerated):
- `results/structure_table.md` had never been generated for the real runs, only the smoke fixture (high):
  generated; the table should be a step of every run list from now on.
- P4 panels clipped the y axis at 1.02, hiding e_raw and e_perm (high): y limits follow the data; legend states
  e = 1 and e = 2 reference levels.
- P6b titles overprinted, control lines unlabelled (medium): wrapped titles, legend entries.
- Lowrank / outliers / interlayer legends covered the q panel (medium): legends outside the axes.
- Symlog axes without readable ticks on P1 / P4 / P7 panels (medium): explicit 1-2-5 ticks; P4 gain panel
  linear; P7 limits from the data with the 100% line labelled.
- P2 gain-vs-r panels on a 0..-7 linear axis hid the maximum at r = 1% (medium): symlog with an in-panel
  "max over r: W x / control y" annotation.
- P0 dPPL panel silently dropped bars <= 0 (k, v at 4 bits, k at 3 bits on the 0.8B; k on the 2B) (medium):
  listed in the panel legend.
- The f32-Hessian acceptance point was drawn on the GPTQ line labelled "3" (medium): detached "3 H32" marker.
- P7 dPPL printed with one decimal (low): two decimals. Jaccard controls rounded to 0.00 (low): three decimals.
  Additivity label did not say "super-additive" at 0.51 (low): direction appended. "Skipped" section did not say
  the files were never scheduled (low): says so now. RD legend on the had-fp16 line (low): data-aware placement.
  Allocation-panel legends over bars, clipped labels (low): moved. P5 "Lorenz curves" are three-point polylines
  (low): footer says so.
- Left as is: label pile-up between 3.1 and 4.5 bits on `rd_qwen3.5-0.8b.png` (would need label repulsion or an
  inset; every value is in `ablation.jsonl`); the `constrained_layout` warning in the degraded-fixture test
  `test_figures_tolerate_missing_stats` (no real figure affected).

### Phase 3 shortlist

DESIGN.md names only B (low-rank + quantized residual) and C (inter-layer differential coding) by letter. The
mapping used for the others here: A = entropy / non-uniform coding of the indices (P1), D = outlier-column
protection (P3), E = sensitivity-driven bit allocation across tensors / columns (P5, P6), F = super-weight
protection (P7), G = rotation + error feedback and group size (P0 g32 / g64, had-GPTQ). Estimates are bits/param
at 3-4 bits on top of the current best coder (GPTQ sym g128), conservative; "measured" means a PPL / KL row
exists, "proxy" means an MSE estimate only.

Ranked, supported by the measured structure:
1. **E, type-level allocation from the measured `@only` damage** (measured on RTN, 3 models): 0.15-0.24 bits/param
   (KL x 0.72-0.81 at equal bits); more bits to v / k / o / out_proj / in_proj_ab, fewer to gate / up /
   in_proj_qkv. Risk: derived from RTN at 4 bits where losses are additive; at 3 bits they are super-additive
   (0.51) and GPTQ already removes 56-71% of the RTN-4 KL (0.205 -> 0.071, 0.133 -> 0.059, 0.268 -> 0.077), so the transfer to GPTQ is unmeasured. Pre-register
   one mixed-width run (e.g. 3 / 5 bits by type at ~4.1 average) against gptq-sym-g128-4 on the 0.8B and the 2B
   before building more. Do not use H_jj for it (inverted ranking); the Fisher diagonal (+0.96) is the proxy to
   extend it below type granularity.
2. **G, Hadamard rotation + GPTQ** (measured, 0.8B, one seed): 0.23-0.26 bits/param at 3 bits (had-gptq-3 KL
   0.214 vs 0.313 / 0.299), rotation cost 0 bits and +0.07% PPL at fp16. Risk: one model, one seed; the 2B and
   MiniCPM rows (and 4-bit, where the Phase 1 had-RTN gap was at the noise edge) are missing; the inference
   kernel must apply the Hadamard (webyIQ cost). Group size is not part of it: g32 / g64 sit on the RTN curve.
3. **A, index coding** (proxy, all tensors, zero quality loss by construction): 0.49-0.51 bits/param with a
   16-bit table per tensor, of which 0.44 is generic (Gaussian + group scale, i.e. what a fixed non-uniform
   4-bit codebook such as NF4 captures without a variable-length decoder) and 0.04-0.07 is model-specific tail.
   Risk: variable-length decode in a WGSL kernel; part of the per-type excess (o / out_proj / k) is column-scale
   heterogeneity that E would already take, so A and a column scheme are not additive; measured only as entropy,
   not as a running decoder.
4. **E, within-tensor row / column allocation** (proxy only: P6a column 0.26-0.90, Fisher rows +0.64..+0.90 for
   hidden-input types, columns +0.94..+1.05 for o / out_proj): potentially the largest number on the table and
   the least trusted, because no ablation tests it and `+keepcols` (a different criterion) lost. Risk: side
   information and kernel complexity; the P6a proxy already failed its only prediction test. Gate it on one
   pre-registered direct ablation (per-row or per-column 3 / 5-bit split by Fisher row energy or diag H at fixed
   average bits, 0.8B) before any estimate is quoted.

Dropped, with the number that drops them:
- **B, low rank + residual**: negative on every block type and model (-0.03..-0.72 after bf16 U, V; weighted
  variant's only positive, q on the 0.8B, < 0.004 bits model-wide). Keep one line for the embedding (+0.04..
  +0.09) if the embedding is ever quantized.
- **C, inter-layer differential coding**: -0.34..-0.53 bits/param, e > 1 at the only cheap alignment; no model
  shows anything coding can use.
- **D, outlier-column protection by weight norm**: 3% / 8% / 17% of the RTN-4 KL for 0.118 bits, below the RD
  curve on every model; no columns > 5x median outside in_proj_ab / k. Survives only as a variant of E
  (selection by activation or Fisher column energy), subject to the same ablation gate.
- **F, super-weight protection**: no super weight on Qwen3.5; on MiniCPM it is worth 0.3% of the KL at 4 bits.
  Keep as a free guard (18 weights) in any 2-bit, clipping or pruning method, where zeroing it costs +206%.
- **Group size** (part of G as a knob): g32 / g64 are on the RTN RD curve at 3 and 4 bits (factor 1.39 at 4 bits
  for +0.375); not a lever on its own, and 2.1x behind GPTQ at equal bits.
- **bf16 lossless coding**: 5.4 bits/param, all of it marginal (the control matches); the 4-bit coders make it
  moot.

What the Phase 2 numbers cannot say: anything about 2 bits (no Phase 2 row below 3.125 bits; Phase 1's best is
GPTQ +655%), anything about GPTQ-based allocation (all `@only` rows are RTN), and anything below the 4.4% PPL /
5% KL spread without a second seed.

### Correction to the shortlist's letter mapping (2026-11-08, project owner)

The shortlist above uses its own letters. The task statement's letters are: **A** sensitivity-driven bit
allocation (average 3 / 3.5 bits), **B** low-rank + quantized residual, **C** inter-layer differential coding,
**D** learned rotation minimizing output KL, **E** non-uniform codebook + entropy-coded indices against the
6.02 dB/bit bound, **F** usage-frequency-aware quantization, **G** own ideas. Read the shortlist as: item 1 and
item 4 = **A** (type-level, then row/column level), item 2 = **D** (rotation; only zero-bit rotations are
admissible without training, see the Phase 3 contract), item 3 = **E**, "D outlier-column protection" and
"F super-weight protection" and group size = dropped sub-variants, **B** and **C** dropped, **F** (usage-aware)
dropped on P5/P7. Phase 3 and REPORT.md use the task statement's letters.

## Phase 3: methods

### Pre-registered hypotheses (verbatim from DESIGN.md "Phase 3 contract: methods")

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

### Decisions fixed at integration, before any run

- "gptq32" is GPTQ with the f32 per-window Hessian product, selected by the method name alone (`gptq_model(...,
  hessian_f32=True)`); the Phase 1 `gptq-sym-g128-*` names keep the f64 path and the `WCOMP_HESSIAN_F32` environment
  override is gone (the module flag is False at import whatever the environment says; `tests/test_quant_phase3.py`
  asserts it). Regression after integration (`results/smoke/phase3/qwen3.5-0.8b/regress.jsonl`): fp16 / rtn-sym-g128-4
  with `--limit-windows 8 --kl-windows 8` = 14.946912850877544 / 19.007505611738555, gptq-sym-g128-4 with
  `--limit-windows 8 --kl-windows 4 --calib-n 16 --hessian-f64` = 16.248178182081652 (all three identical to Phase 1 / 2).
- E3's fixed codebook (`nf-gptq32-g128-*`) is the Lloyd-Max optimum of N(0, 1) (kind "nf", computed from the density,
  0 table bits; 4 levels +-0.4528 / +-1.510, 16 levels max 2.733), not the literal equiprobable Gaussian quantiles
  Phi^-1((i+0.5)/L): under RMS normalization the quantile codebook's outer level sits at 1.86 sigma at 16 levels, it
  clips the tails and its MSE on Gaussian weights is 2x the uniform amax grid's at 4 bits (8.91e-6 vs 4.48e-6) and 1.2x
  at 3 bits (2.235e-5 vs 1.814e-5), so an E3 row built on it would lose by construction. Measured MSEs (unit variance):
  4 bits lloyd 3.81e-6, nf 3.76e-6, uniform 4.48e-6, nfq 8.91e-6; 3 bits lloyd 1.361e-5, nf 1.363e-5, uniform 1.814e-5,
  nfq 2.235e-5. The quantile codebook is kept as kind "nfq" (the Lloyd-Max initialisation E2 names) and is run next to
  nf at seed 0 (`nfq-gptq32-g128-{4,3}`, added to the run list at the review before the runs), so the contract's E3 as
  written has its own number beside the substitute the H3-E "fixed codebook captures >= 2/3" clause is read on.
- Lloyd-Max (E2): 50 iterations from the Gaussian-quantile initialisation on the RMS-normalized weights of the whole
  tensor (the original weights, before any GPTQ update), one codebook per tensor (per distinct width under row
  allocation), stored f16, 2^b * 16 bits in the tensor's aux_bits; group scale = f16 RMS (16/128 bits); a zero-RMS
  group stores scale 0 and dequantizes to 0. Indices are recovered exactly on every path (`W_hat == dequant(idx)` is
  asserted in f32, uniform grids recover the indices from W_hat and the scales after the GPTQ loop).
- Bit figures: `bits_blocks` (fixed), `bits_blocks_huff` (per tensor the best of a single-symbol canonical Huffman
  code with a b * 2^b table, a pair-symbol code with a 2b * 2^(2b) table and the plain b-bit code with no table, plus
  2 choice bits, so huff <= fixed + 2 bits per tensor; DESIGN.md's two-way choice with 1 bit had no table-free option
  and charged small 8-bit tensors (36 x 16384 params: the 2048-bit table is 0.125 b/param) and near-uniform 3-bit
  tensors above their fixed width in the smoke, 1.6e-4 bits/param on the combo row; amended at the review before the
  runs), `bits_blocks_shannon` (the entropy bound of the code chosen for that tensor: the single-symbol entropy for the
  single-symbol and the plain code, the pair-stream entropy per symbol when the pair code won, so the Shannon figure
  never exceeds the table-free part of the Huffman one and huff - shannon is the coder's loss plus its tables), and
  `bits_blocks_entropy0` (the plain order-0 entropy figure, the contract's "Shannon figure" as pre-registered; the
  summary prints huff - entropy0 beside huff - shannon in H3-E); all side information in all of them. A row-allocated tensor
  (method G, rows at different widths) is coded per width class (the decoder knows every row's width from the 2
  bits/row side information): one Huffman table and one entropy per distinct width; coding the pooled stream on the
  2^max alphabet would charge the width mixture a second time and exceeded the fixed figure in the smoke. The E4
  table (`rd_bound.jsonl`) is per weight type from the group-normalized MSE against R(D) = 0.5 log2(1/D); for a
  rotated row the reference weight is folded into the rotated input basis (the module's `wcomp_input_fold`) before
  the MSE, i.e. the distortion is measured on the tensor the quantizer was given.
- Allocation (A): d_u = KL_u / n_u from `results/phase3/<model>/damage_c4.jsonl` (32 seed-2 C4 windows, rtn-sym-g128-4
  on the unit alone); the widths minimize sum n_u d_u 2^(-2(b_u-4)) over {2,3,4,5,6,8} with the numel-weighted mean
  <= target by an exact multiple-choice-knapsack DP over the integer budget in units of gcd(numel)
  (`wcomp.alloc.allocate`; amended at the review before the runs: DESIGN.md's "greedy marginal gain (exact for this
  convex objective)" is not exact for this problem, with unequal numel and the 6 -> 8 double step it returns a 6%
  worse objective at the same realized mean at avg 2.5 and 3.5 on the 0.8B type damage, e.g. avg 2.5 DP
  {in_proj_z:2, out_proj:3, up:3, k:3, v:3} vs greedy {in_proj_z:3, out_proj:4, up:2, k:5, v:5}, and the same
  allocation at avg 3 and 4; `tests/test_alloc.py` checks the DP against brute force); 3 bits per unit charged on the
  unit's first tensor. Tensor units take the per-tensor damage rows (0.8B: required, `wcomp.methods.TENSOR_DAMAGE_MODELS`,
  a file without them is an error) and, for the other models, the Fisher per-tensor totals
  (`results/probe/<model>/fisher.json`, or the "totals" of `fisher_rows.pt` from `scripts/fisher_rows.py`, 512-token
  windows on the 2B as A2 allows); the note, the alloc JSON ("signal") and the summary's tensor-vs-type line say which.
  Shuffle controls: seeded derangement of the real assignment (no unit keeps its own slot, the largest unit frozen at
  its permuted width), rebalanced to the real mean within 0.02 by single-level moves; a rebalanced draw equal to the
  real allocation or leaving more than half of the parameters at their real width is not a control and is rejected
  (next draw; on the 2B type shape, 7 types of which six share one width, the first seed-1 draw was walked back to
  the real allocation by the rebalancer), an all-one-width allocation has no control (error); the alloc JSON records
  the share of the parameters at their real width and the summary prints it beside the real / shuffle ratio, marking
  a control identical to the real allocation invalid.
  The chosen widths are written next to the runs (`<results-dir>/<model>/alloc_<method>_seed<s>.json`, i.e.
  `results/phase3/<model>/` for protocol runs) and into the row's `alloc` field.
- Row allocation (G): rows at b-1 / b / b+1 by the P6b Fisher row sums (`results/probe/<model>/fisher_rows.pt`, 32 C4
  windows x 1024 tokens, seed 0, no weight gradient), per-tensor mean exactly b, 2 bits per row charged by the
  quantizer; the shuffled-row controls permute each tensor's widths over its rows (mean unchanged). The 0.8B file exists
  (totals agree with fisher.json to 5.1e-7 relative). A tensor at 2 or 8 bits is not split (no room below / above) and
  carries no row side information, which the decoder infers from the width; a tensor at an interior width keeps its
  row table and its 2 bits/row even when water-filling leaves every row at the tensor's width (Fisher row energies
  within a factor of 4), since the decoder expects a table for every interior-width tensor (review fix; no smoke or
  protocol row had such a tensor).
- Rotations (D): "hadamard" as in Phase 1 (random orthogonal where no Hadamard order exists), "hadsign" = Hadamard
  times a seeded random sign diagonal, "rnd" = seeded random orthogonal per input width, all 0 bits; "eig" = per
  shared-input group (in_proj_*, q/k/v, gate/up share one input) the eigenvectors of the calibration Hessian, stored
  bf16 and charged 16 K K bits per distinct matrix in aux_bits (collected 6 decoder layers at a time through the
  already rotated layers).
- Combination: `combo-gptq32-avg{x}` reads `results/phase3/<model>/combo.json` {"had", "lloyd", "rowalloc"} (booleans,
  written after the decision rules below are applied to the single-method rows; the file does not exist before that
  and the method refuses to run without it) and composes had() + type allocation at avg x + row allocation around each
  type's width (if rowalloc) + Lloyd codebook (if lloyd) + GPTQ f32.
- Smoke evidence for the pipeline (off-protocol, `--limit-windows 4 --kl-windows 4 --calib-n 8`, never mixed with
  protocol rows): `results/smoke/phase3/qwen3.5-0.8b/runs.jsonl`, `rd_bound.jsonl`, `damage_c4.jsonl` (2 windows),
  `results/smoke/phase3/rd3_qwen3.5-0.8b_smoke.png`, `results/smoke/phase3/summary_smoke.md`, logs in
  `results/smoke/phase3/`.
- The dPPL half of the "beats" rule (review before the runs): DESIGN.md's 4.4% is applied to dPPL = PPL - PPL_fp16
  as written (`scripts/phase3_common.py compare()`, unchanged), but the 4.4% was measured as a spread of PPL (24.45 vs
  23.40 at 3 bits); in dPPL the same seed pair differs by 18% at 3 bits (6.88 vs 5.83) and 77% at 4 bits (0.49 vs
  0.87), while the -4.4% dPPL threshold is a PPL move of 1.2% at 3.125 bits and 0.16% at 4.125 bits. The dPPL margin
  is therefore recorded as non-binding: any row that clears the 5% KL margin on every seed clears it automatically,
  the verdict is carried by the KL margin plus the every-seed requirement, and summary.md (reference table: the
  measured seed spread of KL / PPL / dPPL per width, footnote) and the rd3 plot footer say so. The margin is not
  re-tuned after the fact.
- Presentation (review before the runs): a row within 0.01 bits beyond the top reference point (G's 2 bits/row put
  avg-4 rows at 4.1266) is not flagged as extrapolated; a method with a seed row that has no readable comparison is
  "n/a (k of n seeds readable)", not "yes (1 seed)"; RTN-quantized rows get "rtn gate" instead of a verdict against
  the gptq32 curve (their gates are the uniform RTN curve and rtn-sym-g128-4 with the shuffled-row controls); the
  H3-A Spearman line reads damage_c4.jsonl through the allocation methods' checked reader; plot labels avoid other
  rows' markers, move at most two steps without a leader line and get one beyond that.

2026-11-08: pre-registered before any Phase 3 run; code `bf5b4e7+dirty/8270c1e57d` (wcomp.version.code_version after the integration and the two review-fix rounds; pytest 329 passed on it, `results/smoke/phase3/pytest_fixes2.log`).

### Phase 3 run list (exact commands, sequential on the one GPU, from `D:\repos\wcomp`)

Decision rules applied between the blocks (DESIGN.md): a method beats the baseline at its bits if its KL is > 5% below
the interpolated gptq32 curve and its dPPL > 4.4% below, on both seeds where two exist; a claimed allocation gain needs
the real assignment to beat both shuffles by > 5% KL; G gate = beats uniform rtn-sym-g128-4 by > 10% KL and both
shuffled-row controls by > 5%; combo components: had iff had-gptq32 beats at 3 bits, lloyd iff lloyd-gptq32-g128-3
beats the uniform grid by > 5% KL on both seeds, rowalloc iff the G gate passed. The decisions are written to NOTES.md
and `results/phase3/<model>/combo.json` before the combo rows run. If a model's list passes 3 h, finish the current row
and report.

qwen3.5-0.8b:

```
uv run python scripts/damage_c4.py --model qwen3.5-0.8b --units type,tensor
uv run python scripts/fisher_rows.py --model qwen3.5-0.8b
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods fp16,rtn-sym-g128-4,gptq32-sym-g128-4,gptq32-sym-g128-3,gptq32-sym-g128-2 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods gptq32-sym-g128-4,gptq32-sym-g128-3,gptq32-sym-g128-2 --seed 1
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods had-gptq32-sym-g128-4,had-gptq32-sym-g128-3,had-gptq32-sym-g128-2,rnd-gptq32-sym-g128-3,hadsign-gptq32-sym-g128-3,eig-gptq32-sym-g128-3 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods had-gptq32-sym-g128-4,had-gptq32-sym-g128-3,had-gptq32-sym-g128-2,rnd-gptq32-sym-g128-3 --seed 1
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods lloyd-gptq32-g128-4,lloyd-gptq32-g128-3,lloyd-gptq32-g128-2,nf-gptq32-g128-4,nf-gptq32-g128-3,nfq-gptq32-g128-4,nfq-gptq32-g128-3 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods lloyd-gptq32-g128-4,lloyd-gptq32-g128-3,lloyd-gptq32-g128-2 --seed 1
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods alloc-type-gptq32-avg2.5,alloc-type-gptq32-avg3,alloc-type-gptq32-avg3.5,alloc-type-gptq32-avg4,alloc-tensor-gptq32-avg3,alloc-tensor-gptq32-avg3.5,alloc-type-rtn-avg3,alloc-type-rtn-avg3.5,alloc-type-shuf1-gptq32-avg3,alloc-type-shuf2-gptq32-avg3 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods alloc-type-gptq32-avg2.5,alloc-type-gptq32-avg3,alloc-type-gptq32-avg3.5,alloc-type-gptq32-avg4,alloc-tensor-gptq32-avg3,alloc-tensor-gptq32-avg3.5 --seed 1
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods rowalloc-fisher-rtn-avg4,rowalloc-shuf1-rtn-avg4,rowalloc-shuf2-rtn-avg4 --seed 0
# G gate decision written to NOTES.md; the next two commands only if it passed
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods rowalloc-fisher-gptq32-avg4,rowalloc-fisher-gptq32-avg3 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods rowalloc-fisher-gptq32-avg4,rowalloc-fisher-gptq32-avg3 --seed 1
# combo decision written to NOTES.md and results/phase3/qwen3.5-0.8b/combo.json, then
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods combo-gptq32-avg2.5,combo-gptq32-avg3,combo-gptq32-avg3.5 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --results-dir results/phase3 --methods combo-gptq32-avg2.5,combo-gptq32-avg3,combo-gptq32-avg3.5 --seed 1
uv run python scripts/plot_rd3.py --model qwen3.5-0.8b
```

qwen3.5-2b:

```
uv run python scripts/damage_c4.py --model qwen3.5-2b --units type
uv run python scripts/fisher_rows.py --model qwen3.5-2b --tokens 512
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/phase3 --methods fp16,gptq32-sym-g128-4,gptq32-sym-g128-3,gptq32-sym-g128-2 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/phase3 --methods gptq32-sym-g128-4,gptq32-sym-g128-3,gptq32-sym-g128-2 --seed 1
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/phase3 --methods had-gptq32-sym-g128-4,had-gptq32-sym-g128-3,had-gptq32-sym-g128-2,lloyd-gptq32-g128-3 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/phase3 --methods had-gptq32-sym-g128-4,had-gptq32-sym-g128-3,had-gptq32-sym-g128-2 --seed 1
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/phase3 --methods alloc-type-gptq32-avg2.5,alloc-type-gptq32-avg3,alloc-type-gptq32-avg3.5,alloc-type-gptq32-avg4,alloc-type-shuf1-gptq32-avg3,alloc-type-shuf2-gptq32-avg3 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/phase3 --methods alloc-type-gptq32-avg2.5,alloc-type-gptq32-avg3,alloc-type-gptq32-avg3.5,alloc-type-gptq32-avg4 --seed 1
# only if fisher_rows.py fit in memory (results/probe/qwen3.5-2b/fisher_rows.pt exists)
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/phase3 --methods alloc-tensor-gptq32-avg3,alloc-tensor-gptq32-avg3.5 --seed 0
# combo decision written to NOTES.md and results/phase3/qwen3.5-2b/combo.json, then
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/phase3 --methods combo-gptq32-avg2.5,combo-gptq32-avg3,combo-gptq32-avg3.5 --seed 0
uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/phase3 --methods combo-gptq32-avg2.5,combo-gptq32-avg3,combo-gptq32-avg3.5 --seed 1
uv run python scripts/plot_rd3.py --model qwen3.5-2b
```

minicpm5-2b:

```
uv run python scripts/damage_c4.py --model minicpm5-2b --units type
uv run python scripts/run_baselines.py --model minicpm5-2b --results-dir results/phase3 --methods fp16,gptq32-sym-g128-4,gptq32-sym-g128-3,had-gptq32-sym-g128-3,lloyd-gptq32-g128-3,alloc-type-gptq32-avg3,alloc-type-gptq32-avg3.5 --seed 0
uv run python scripts/run_baselines.py --model minicpm5-2b --results-dir results/phase3 --methods gptq32-sym-g128-3 --seed 1
# combo decision written to NOTES.md and results/phase3/minicpm5-2b/combo.json (a combo with rowalloc needs
# uv run python scripts/fisher_rows.py --model minicpm5-2b --tokens 512 first), then
uv run python scripts/run_baselines.py --model minicpm5-2b --results-dir results/phase3 --methods combo-gptq32-avg3 --seed 0
uv run python scripts/plot_rd3.py --model minicpm5-2b
uv run python scripts/phase3_summary.py --models qwen3.5-0.8b,qwen3.5-2b,minicpm5-2b
```

### Verify round 3 close-out (2026-11-08, project owner)

The round-3 fix agent returned before its own check script finished (results/smoke/phase3/fixes3_smoke.sh). Checked by hand:
the Phase 1 regression reproduces fp16 14.946913 and rtn-sym-g128-4 19.007506 exactly; the 3-row Phase 3 smoke, the plot and
the summary ran with exit 0. The full pytest reported 11 failures, all subprocess launches returning 0xC0000142 (DLL init
failed) with empty stdout/stderr while the test process held ~46 GB RAM; rerunning those test files alone gave 80 passed.
Classified as an environment failure, not a code defect. From the Run phase on, subagents run on Opus 5.5 (model switch only;
contract, code and protocol unchanged).

### Combination decision (0.8B), 2026-11-08

Applied after all single-method rows existed (`results/phase3/qwen3.5-0.8b/runs.jsonl`, 51 rows, code `bf5b4e7+dirty/afb6534a49`;
reference curve = seed means of gptq32-sym-g128-{4,3,2}: KL 0.06943 / 0.30173 / 1.83255 at 4.125 / 3.125 / 2.125 bits),
before any combo row; written to `results/phase3/qwen3.5-0.8b/combo.json` = `{"had": true, "lloyd": false, "rowalloc": true}`.
- had = true: had-gptq32-sym-g128-3 beats the baseline on both seeds: KL 0.21136 / 0.21445 vs curve 0.30173 (ratio 0.701 /
  0.711, needs < 0.95), dPPL ratio 0.531 / 0.559 (needs < 0.956).
- lloyd = false: lloyd-gptq32-g128-3 is worse than the uniform grid at 3 bits on both seeds: KL 0.34142 / 0.35874 vs curve
  0.30171 (ratio 1.132 / 1.189; vs the same-seed gptq32-sym-g128-3 0.29988 / 0.30357: +13.9% / +18.2%), needs > 5% below.
- rowalloc = true: the G gate passed (this is also the G gate decision of the run list): rowalloc-fisher-rtn-avg4 KL 0.17509
  vs rtn-sym-g128-4 0.20514 (-14.7%, needs > 10%) and vs the shuffled-row controls 0.26623 / 0.24154 (-34.2% / -27.5%, each
  needs > 5%).
The combo rows are therefore had + type allocation + Fisher row allocation around each type's width + uniform grid + GPTQ f32.

### Combination decision (2B), 2026-11-08

Applied after all single-method rows existed (`results/phase3/qwen3.5-2b/runs.jsonl`, 26 rows, code `bf5b4e7+dirty/afb6534a49`;
reference curve = seed means of gptq32-sym-g128-{4,3,2}: KL 0.05810 / 0.25334 / 1.92254 at 4.125 / 3.125 / 2.125 bits),
before any combo row; the 2B's own had / lloyd rows agree with the 0.8B decision, so it is kept; written to
`results/phase3/qwen3.5-2b/combo.json` = `{"had": true, "lloyd": false, "rowalloc": true}`.
- had = true: had-gptq32-sym-g128-3 beats the baseline on both seeds: KL 0.16275 / 0.16383 vs curve 0.25334 (ratio 0.642 /
  0.647, needs < 0.95), dPPL ratio 0.556 / 0.686 (needs < 0.956).
- lloyd = false: lloyd-gptq32-g128-3 (seed 0, the only 2B seed the contract runs) is worse than the uniform grid at 3 bits:
  KL 0.62069 vs curve 0.25333 (ratio 2.450; vs the same-seed gptq32-sym-g128-3 0.25460: +143.8%), needs > 5% below.
- rowalloc = true: the G gate is measured on the 0.8B only (contract) and passed there (see the 0.8B decision); the 2B row
  allocation uses `results/probe/qwen3.5-2b/fisher_rows.pt` (32 C4 windows x 512 tokens, 16,352 tokens, peak 8.63 GB).
The combo rows are therefore had + type allocation + Fisher row allocation around each type's width + uniform grid + GPTQ f32.

### Combination decision (MiniCPM), 2026-11-08

Applied after all single-method rows existed (`results/phase3/minicpm5-2b/runs.jsonl`, 8 rows, code `bf5b4e7+dirty/afb6534a49`;
reference curve = seed means of gptq32-sym-g128-{4,3} (4 bits seed 0 only, 3 bits seeds 0 and 1, as the contract runs
MiniCPM): KL 0.07713 / 0.34362 at 4.125 / 3.125 bits), before any combo row; the MiniCPM had / lloyd rows agree with the
0.8B and 2B decisions, so it is kept; written to `results/phase3/minicpm5-2b/combo.json` = `{"had": true, "lloyd": false, "rowalloc": true}`.
- had = true: had-gptq32-sym-g128-3 (seed 0, the only MiniCPM seed the contract runs) beats the baseline: KL 0.24102 vs
  curve 0.34362 (ratio 0.701, needs < 0.95), dPPL ratio 0.634 (needs < 0.956).
- lloyd = false: lloyd-gptq32-g128-3 (seed 0) is worse than the uniform grid at 3 bits: KL 1.26109 vs curve 0.34361
  (ratio 3.670; vs the same-seed gptq32-sym-g128-3 0.34211: +268.6%), needs > 5% below.
- rowalloc = true: the G gate is measured on the 0.8B only (contract) and passed there (see the 0.8B decision); the MiniCPM
  row allocation uses `results/probe/minicpm5-2b/fisher_rows.pt` (32 C4 windows x 512 tokens, 16,352 tokens, 294 linears,
  peak 9.04 GB), written by the run list's `fisher_rows.py --model minicpm5-2b --tokens 512` before the combo row.
- Note: the type allocation at avg 3 is uniform on MiniCPM (all 7 types at 3 bits; alloc-type-gptq32-avg3 reproduces
  gptq32-sym-g128-3 seed 0 exactly, KL 0.34211, bits_blocks 3.1250).
The combo row is therefore had + type allocation + Fisher row allocation around each type's width + uniform grid + GPTQ f32.

### Results (2026-11-08)

Sources: `results/phase3/<model>/runs.jsonl` (0.8B 57 rows, 2B 32, MiniCPM 9), `damage_c4.jsonl`, `rd_bound.jsonl`,
`combo.json`, `results/phase3/summary.md`, plots `results/rd3_<model>.png`. Every number below is read from those files
or recomputed from them; the audit recomputations are named where they go beyond summary.md. Rule used throughout
(DESIGN.md): "beats" = KL more than 5% and dPPL more than 4.4% below the seed-mean gptq32 curve (log-linear in
`bits_blocks`), on every seed that exists; "saved" = curve bits at the row's KL minus the row's bits (fixed against
the fixed curve, huff against the baseline's own Huffman curve); `*` = read on an extended end segment (outside the
measured curve), `-15.036` for eig is read far outside it.

#### Provenance, change points, failures

- Run order and times: 0.8B 12:02-14:26 (51 + 6 combo rows), 2B 14:28-16:34, MiniCPM 16:36-17:31. The (method, seed)
  sets equal the run list exactly; every group exited 0; no failure, no rerun, no code edit during the runs.
  combo.json was written before the combo rows on every model (0.8B 14:12:36 before 14:12:49; 2B 16:08:48; MiniCPM
  17:23:03), and the G gate decision is the rowalloc paragraph of the 0.8B combination decision above.
- Code: every Phase 3 row (98 of 98) carries `bf5b4e7+dirty/afb6534a49`, not the pre-registration line's
  `8270c1e57d`. `afb6534a49` is the code after the round-3 fixes accepted in the round-3 close-out (last source mtime
  11:41:03); its regression rows (`results/smoke/phase3/qwen3.5-0.8b/regress.jsonl`, 11:55) reproduce 14.946913 /
  19.007506. The close-out did not name the hash; this line does.
- The 11 per-type damage rows of the 0.8B and the fp16 / `@all` entries of its `damage_c4_ref.json` were measured at
  `8270c1e57d` and reused by the resume logic; the 186 per-tensor rows at `afb6534a49`. `damage_c4.py` restamps the
  ref file's version on every run, so the ref file shows `afb6534a49` for values measured earlier. Audit re-measure
  of all 11 types on the current code: KL / PPL / top1 identical (max relative difference 0.0), ref fp16 PPL 20.0389 and
  `@all` KL 0.17074 identical. No number is affected. Open (low): stamp a reused ref entry with its own version.
- `results/phase3/qwen3.5-0.8b/stale_pre_exclusion/` holds a damage file from 08:33 (integration, before the
  calibration-document exclusion); the alloc JSONs name the sha of the current damage file, so it is not used.
- `fisher_rows.py` overwrote `results/probe/qwen3.5-0.8b/fisher_rows.pt` (32,736 tokens, totals within 5.14e-7 of
  `fisher.json`); the 2B (16,352 tokens at 512-token windows, peak 8.63 GB) and MiniCPM (16,352 tokens, 294 linears,
  peak 9.04 GB) files are new and untracked.
- Phase 1 / 2 untouched: `git diff --quiet HEAD -- results/` clean; Phase 3 fp16 rows equal Phase 1 bit-for-bit
  (17.567525 / 11.810692 / 13.781604), 0.8B rtn-sym-g128-4 KL 0.2051447 equals Phase 1, gptq32-3 seed 0 (23.32 /
  0.29988) equals the Phase 2 acceptance row.
- Presentation change after the runs (change point `bf5b4e7+dirty/5c0601e1e3`): during the post-run audit, the
  presentation-audit subagent edited `scripts/plot_rd3.py` (17:37:48), `scripts/phase3_common.py` (17:38:13, only the
  `DPPL_RULE_NOTE` string and its comment) and `scripts/phase3_summary.py` (17:38:39), and regenerated
  `results/phase3/summary.md` and the three `results/rd3_*.png` (17:39:40-17:39:51). Why: the footnote's "dPPL margin
  carried by the KL margin" claim was false on the data (next section), the summary had no dPPL ratio column, and the
  plot footers were cut off and labels overlapped. No runner, damage or Fisher script imports these modules; no
  runs.jsonl / damage / rd_bound file changed after 17:30:30; the first 14 columns of every summary row, the reference
  tables, hypotheses and RD tables are unchanged (diff with the added columns stripped). Regression smoke under
  `5c0601e1e3`: 14.946913 / 19.007506. `tests/test_phase3_presentation.py` alone: 21 passed; full pytest on
  `5c0601e1e3`: 329 passed, 10 failed in the full run (all 10 are subprocess launches returning 0xC0000142, DLL init failed, with empty output, the same environment failure as the round-3 close-out); the four affected files rerun alone: 114 passed. Any row run from now on carries `5c0601e1e3` or later; the 98 protocol rows carry
  `afb6534a49`.

#### Correction to the pre-registration: the dPPL margin binds

The "Decisions fixed at integration" paragraph above records the dPPL margin as non-binding ("any row that clears the
5% KL margin on every seed clears it automatically"). That is false on the protocol data. The margin was not re-tuned;
the verdicts below are the rule as written. Eleven rows (seven method / width verdicts) clear the KL margin and fail
on dPPL alone; MiniCPM has none:

| model | method | KL ratio s0 / s1 | dPPL ratio s0 / s1 | verdict (rule) | KL-only verdict |
|---|---|---|---|---|---|
| 0.8B | had-gptq32-sym-g128-4 | 0.755 / 0.703 | 2.306 / 1.820 | no | yes |
| 0.8B | alloc-type-gptq32-avg3.5 | 0.930 / 0.914 | 1.500 / 1.326 | no | yes |
| 0.8B | alloc-tensor-gptq32-avg3.5 | 0.804 / 0.840 | 0.963 / 1.383 | no | yes |
| 0.8B | rowalloc-fisher-gptq32-avg4 | 0.914 / 0.925 | 2.272 / 2.873 | no | yes |
| 0.8B | rowalloc-fisher-gptq32-avg3 | 0.919 / 0.908 | 1.087 / 0.935 | no (1 of 2 seeds) | yes |
| 0.8B | combo-gptq32-avg3 | 0.884 / 0.908 | 0.854 / 0.963 | no (1 of 2 seeds) | yes |
| 2B | combo-gptq32-avg3.5 | 0.725 / 0.740 | 0.636 / 0.970 | no (1 of 2 seeds) | yes |

(alloc-tensor-gptq32-avg3.5 was missed by the run reports; summary.md lists it.) Cause: the dPPL half rests on the
4-bit end of the curve, where the reference seed pair is 0.49 vs 0.91 dPPL on the 0.8B (113% spread) and the 4.4%
threshold is a PPL move of 0.16%; at 4 bits dPPL is noise (2B had-4 seed 1: PPL 11.862 vs fp16 11.811, dPPL ratio
0.075 at KL ratio 0.727). Every KL-only "yes" above is reported as such; none is promoted to a "beats".

#### Interpolation (chord) bias between curve points (audit)

log(KL) is convex in bits on both Qwen curves (quadratic through the three seed-mean points: +0.167 / bit^2 on the
0.8B, +0.277 / bit^2 on the 2B), so the log-linear chord overstates the curve's KL by about 4.3% (0.8B) and 7.2% (2B) at
a segment midpoint, and log(dPPL) by about 9% (0.8B, +0.346 / bit^2). Rows at 3.125 / 4.125 / 2.125 bits (all D, E, G
rows, alloc-tensor avg 3, combo avg 3, alloc-type avg 3 / 4) are read on measured points and are unaffected. Rows at
2.62 / 3.62 bits (alloc-type avg 2.5 / 3.5, alloc-tensor avg 3.5, combo avg 2.5 / 3.5) carry this bias in their favour;
a mid-segment ratio above ~0.91 (0.8B) or ~0.88 (2B) is not resolvable. Applied below where it changes a verdict.

#### Reference curve and the noise model with all seed pairs

| model | method | seed | bits_blocks | huff | shannon | ppl | dPPL% | kl | top1 | KL curve @bits | KL ratio | dPPL ratio | beats | saved fixed | saved huff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8B | gptq32-sym-g128-4 | 0 | 4.1250 | 3.6423 | 3.6257 | 18.00 | +2.4% | 0.07206 | 0.8659 | 0.06943 | — | — | ref | — | — |
| 0.8B | gptq32-sym-g128-4 | 1 | 4.1250 | 3.6423 | 3.6257 | 18.48 | +5.2% | 0.06679 | 0.8727 | 0.06943 | — | — | ref | — | — |
| 0.8B | gptq32-sym-g128-3 | 0 | 3.1250 | 2.6570 | 2.6395 | 23.32 | +32.8% | 0.29988 | 0.7391 | 0.30173 | — | — | ref | — | — |
| 0.8B | gptq32-sym-g128-3 | 1 | 3.1250 | 2.6570 | 2.6396 | 23.63 | +34.5% | 0.30357 | 0.7365 | 0.30173 | — | — | ref | — | — |
| 0.8B | gptq32-sym-g128-2 | 0 | 2.1250 | 1.6809 | 1.6683 | 127.26 | +624.4% | 1.88628 | 0.3824 | 1.83255 | — | — | ref | — | — |
| 0.8B | gptq32-sym-g128-2 | 1 | 2.1250 | 1.6809 | 1.6683 | 116.17 | +561.3% | 1.77881 | 0.4029 | 1.83255 | — | — | ref | — | — |
| 0.8B | rtn-sym-g128-4 | 0 | 4.1250 | 3.6412 | 3.6248 | 22.67 | +29.0% | 0.20514 | 0.7799 | 0.06943 | 2.955 | 7.617 | rtn gate | -0.737 | -0.725 |
| 2B | gptq32-sym-g128-4 | 0 | 4.1250 | 3.6338 | 3.6172 | 12.46 | +5.5% | 0.05931 | 0.8795 | 0.05810 | — | — | ref | — | — |
| 2B | gptq32-sym-g128-4 | 1 | 4.1250 | 3.6338 | 3.6173 | 12.53 | +6.1% | 0.05689 | 0.8856 | 0.05810 | — | — | ref | — | — |
| 2B | gptq32-sym-g128-3 | 0 | 3.1250 | 2.6485 | 2.6314 | 15.11 | +28.0% | 0.25460 | 0.7707 | 0.25334 | — | — | ref | — | — |
| 2B | gptq32-sym-g128-3 | 1 | 3.1250 | 2.6486 | 2.6314 | 15.06 | +27.5% | 0.25208 | 0.7692 | 0.25334 | — | — | ref | — | — |
| 2B | gptq32-sym-g128-2 | 0 | 2.1250 | 1.6738 | 1.6609 | 85.85 | +626.9% | 1.94470 | 0.4103 | 1.92254 | — | — | ref | — | — |
| 2B | gptq32-sym-g128-2 | 1 | 2.1250 | 1.6739 | 1.6609 | 87.56 | +641.4% | 1.90039 | 0.4124 | 1.92254 | — | — | ref | — | — |
| MiniCPM | gptq32-sym-g128-4 | 0 | 4.1250 | 3.6610 | 3.6457 | 14.58 | +5.8% | 0.07713 | 0.8678 | 0.07713 | — | — | ref | — | — |
| MiniCPM | gptq32-sym-g128-3 | 0 | 3.1250 | 2.6756 | 2.6573 | 18.23 | +32.3% | 0.34211 | 0.7337 | 0.34362 | — | — | ref | — | — |
| MiniCPM | gptq32-sym-g128-3 | 1 | 3.1250 | 2.6756 | 2.6573 | 18.37 | +33.3% | 0.34513 | 0.7278 | 0.34362 | — | — | ref | — | — |

Seed spreads (max / min - 1) over every two-seed pair, GPTQ calibration seed only (rotation seed = run seed where
used):

| model | pairs | KL spread: median / max | KL pairs above 5% | PPL spread max | dPPL spread at ~4.1 bits |
|---|---|---|---|---|---|
| 0.8B | 21 | 1.8% / 7.9% (gptq32-4) | 5 (gptq32-4 7.9, had-4 7.4, gptq32-2 6.0, lloyd-4 5.2, lloyd-3 5.1) | 9.5% (gptq32-2) | 7-113% |
| 2B | 13 | 2.1% / 12.3% (had-4) | 1 (had-4 12.3) | 3.9% | 3-713% |
| MiniCPM | 1 | 0.9% (gptq32-3) | 0 | 0.8% | n/a |

Reading: at 3.125 bits the KL spread is 0.7-3.9% for every family except Lloyd (5.1%), so the 5% KL margin is
adequate there; at 4.125 and 2.125 bits KL spreads reach 6-12%, so a KL ratio between ~0.88 and 0.95 at those widths
is not resolvable with two seeds; dPPL at ~4 bits is noise-dominated (above). Not sampled at all: noise of the
allocation signal (every seed uses the same damage file and the same Fisher file; the seed-0 and seed-1 alloc JSONs
have identical widths and row splits in all 18 pairs checked) and noise of the eval windows (all rows share the same
32 WikiText KL windows, i.e. the comparisons are paired). The old Phase 2 figure (4.4% PPL / ~5% KL, three pairs) is
consistent with the 3-bit pairs and too small for the 4-bit and 2-bit pairs.

#### A: sensitivity-driven bit allocation (H3-A)

Pre-registered: type-level allocation from C4-measured damage beats uniform GPTQ by 15-30% KL at equal `bits_blocks`
at avg 3.5 and 3 on both Qwen models, both seeds; both shuffled allocations are no better than uniform (within 5%);
tensor-level adds < 10% KL on top of type-level; C4 damage ranks the types as the Phase 2 WikiText measurement did
(Spearman > 0.9).

Run: damage on 32 seed-2 C4 windows (calibration documents of seeds 0, 1 removed: 242 / 242 / 241 documents),
rtn-sym-g128-4 on one unit at a time, types on all three models and 186 tensors on the 0.8B (7.4 s per unit);
alloc-type avg 2.5 / 3 / 3.5 / 4 on both Qwens seeds 0 and 1, MiniCPM avg 3 / 3.5 seed 0; alloc-tensor avg 3 / 3.5
(0.8B both seeds, damage signal; 2B seed 0, Fisher-totals signal); alloc-type-rtn avg 3 / 3.5 (0.8B seed 0); shuffles
avg 3 seed 0 on both Qwens.

C4 damage per type (KL of rtn-sym-g128-4 on that type alone; d = KL / numel) against the Phase 2 WikiText `@only` KL:

| model | type | numel (M) | C4 KL | d (nats/param) | WikiText @only KL | C4 rank / WT rank |
|---|---|---|---|---|---|---|
| 0.8B | out_proj | 37.75 | 0.03739 | 9.91e-10 | 0.04739 | 1 / 1 |
| 0.8B | down | 88.08 | 0.02693 | 3.06e-10 | 0.03200 | 2 / 2 |
| 0.8B | up | 88.08 | 0.02592 | 2.94e-10 | 0.02911 | 3 / 3 |
| 0.8B | o | 12.58 | 0.02020 | 1.61e-09 | 0.02524 | 4 / 4 |
| 0.8B | in_proj_qkv | 113.25 | 0.01442 | 1.27e-10 | 0.01731 | 5 / 5 |
| 0.8B | gate | 88.08 | 0.01343 | 1.52e-10 | 0.01428 | 6 / 6 |
| 0.8B | v | 3.15 | 0.00835 | 2.66e-09 | 0.01017 | 7 / 7 |
| 0.8B | in_proj_z | 37.75 | 0.00763 | 2.02e-10 | 0.00886 | 8 / 8 |
| 0.8B | q | 25.17 | 0.00479 | 1.90e-10 | 0.00637 | 9 / 9 |
| 0.8B | k | 3.15 | 0.00242 | 7.70e-10 | 0.00308 | 10 / 10 |
| 0.8B | in_proj_ab | 0.59 | 0.00167 | 2.83e-09 | 0.00191 | 11 / 11 |
| 2B | out_proj | 75.50 | 0.02355 | 3.12e-10 | 0.02721 | 1 / 1 |
| 2B | down | 301.99 | 0.02133 | 7.06e-11 | 0.02342 | 2 / 2 |
| 2B | up | 301.99 | 0.01981 | 6.56e-11 | 0.02291 | 3 / 3 |
| 2B | in_proj_qkv | 226.49 | 0.01165 | 5.14e-11 | 0.01432 | 4 / 4 |
| 2B | gate | 301.99 | 0.01113 | 3.69e-11 | 0.01284 | 5 / 5 |
| 2B | o | 25.17 | 0.01014 | 4.03e-10 | 0.01159 | 6 / 6 |
| 2B | v | 6.29 | 0.00746 | 1.19e-09 | 0.00839 | 7 / 8 |
| 2B | in_proj_z | 75.50 | 0.00732 | 9.70e-11 | 0.00871 | 8 / 7 |
| 2B | q | 50.33 | 0.00388 | 7.71e-11 | 0.00484 | 9 / 9 |
| 2B | k | 6.29 | 0.00199 | 3.16e-10 | 0.00250 | 10 / 10 |
| 2B | in_proj_ab | 1.18 | 0.00169 | 1.43e-09 | 0.00166 | 11 / 11 |
| MiniCPM | down | 528.48 | 0.07591 | 1.44e-10 | 0.08193 | 1 / 1 |
| MiniCPM | up | 528.48 | 0.03950 | 7.47e-11 | 0.04610 | 2 / 2 |
| MiniCPM | v | 22.02 | 0.02870 | 1.30e-09 | 0.03176 | 3 / 4 |
| MiniCPM | o | 176.16 | 0.02844 | 1.61e-10 | 0.03552 | 4 / 3 |
| MiniCPM | gate | 528.48 | 0.02542 | 4.81e-11 | 0.03056 | 5 / 5 |
| MiniCPM | q | 176.16 | 0.02194 | 1.25e-10 | 0.02365 | 6 / 6 |
| MiniCPM | k | 22.02 | 0.00943 | 4.28e-10 | 0.01171 | 7 / 7 |

Spearman(C4, WikiText) on raw KL 1.000 / 0.991 / 0.964, on d 0.991 / 1.000 / 1.000 (0.8B / 2B / MiniCPM). C4 KL is
0.75-1.02x the WikiText KL per type (C4 fp16 PPL 20.04 / 15.58 / 20.62). The sum of the type KLs is 0.163 / 0.120 /
0.229 against 0.171 / 0.120 / 0.245 with every type quantized at once (near-additive at 4 bits). Per tensor (0.8B,
186 rows): d spans 1.58e-10 to 5.45e-08 (2.5 decades; the top six are layer 0-4 in_proj_a / in_proj_b, 16,384 params
each); within a type d varies by 1.3x (q, k) up to 11.7x (out_proj); layer-0 out_proj alone has KL 0.01541, 41% of
the whole out_proj type's 0.03739; the per-tensor KLs sum to 0.280, 1.7x the type sum (0.163), i.e. damage is
sub-additive across the tensors of a type (each tensor measured alone overstates its share of the joint damage).

Widths (alloc JSONs next to the runs; identical for seeds 0 and 1):
- 0.8B avg 3: in_proj_qkv 2, in_proj_z 3, in_proj_ab 8, out_proj 5, gate 3, up 3, down 3, q 3, k 4, v 5, o 5;
  tensor-level avg 3: 7 tensors at 2 bits, 117 at 3, 22 at 4, 16 at 5, 24 at 6.
- 2B avg 3: in_proj_qkv 3, in_proj_z 4, in_proj_ab 8, out_proj 5, gate 2, up 3, down 3, q 3, k 4, v 5, o 5;
  avg 3.5: 3, 4, 8, 5, 3, 3, 4, 4, 6, 8, 5; tensor-level (Fisher) avg 3: 2b:33, 3b:96, 4b:41, 5b:12, 6b:4.
- MiniCPM avg 3: all seven types at 3 (the exact optimum for these damages, cost 0.91733 = uniform; the cheapest
  trade, v to 4 and k to 2, costs more than it saves), so alloc-type-gptq32-avg3 is gptq32-sym-g128-3 seed 0 to the
  last digit; avg 3.5: q 4, k 5, v 6, o 4, gate 3, up 3, down 4.
- The audit's own DP reproduces every type allocation (widths and objective), and bits_blocks = mean width + 0.125 +
  3 bits/unit matches every row to 1e-5.

| model | method | seed | bits_blocks | huff | shannon | ppl | dPPL% | kl | top1 | KL curve @bits | KL ratio | dPPL ratio | beats | saved fixed | saved huff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8B | alloc-type-gptq32-avg4 | 0 | 4.1234 | 3.6445 | 3.6255 | 18.53 | +5.5% | 0.07249 | 0.8640 | 0.06959 | 1.042 | 1.432 | no | -0.028 | -0.031 |
| 0.8B | alloc-type-gptq32-avg4 | 1 | 4.1234 | 3.6445 | 3.6255 | 18.91 | +7.6% | 0.07280 | 0.8638 | 0.06959 | 1.046 | 1.999 | no | -0.031 | -0.034 |
| 0.8B | alloc-type-gptq32-avg3.5 | 0 | 3.6240 | 3.1487 | 3.1312 | 20.56 | +17.0% | 0.13482 | 0.8183 | 0.14494 | 0.930 | 1.500 | no (dPPL decides) | +0.049 | +0.049 |
| 0.8B | alloc-type-gptq32-avg3.5 | 1 | 3.6240 | 3.1487 | 3.1312 | 20.21 | +15.0% | 0.13249 | 0.8218 | 0.14494 | 0.914 | 1.326 | no (dPPL decides) | +0.061 | +0.060 |
| 0.8B | alloc-type-gptq32-avg3 | 0 | 3.1246 | 2.6582 | 2.6418 | 24.61 | +40.1% | 0.31921 | 0.7288 | 0.30194 | 1.057 | 1.191 | no | -0.031 | -0.032 |
| 0.8B | alloc-type-gptq32-avg3 | 1 | 3.1246 | 2.6583 | 2.6418 | 24.24 | +38.0% | 0.31491 | 0.7306 | 0.30194 | 1.043 | 1.128 | no | -0.023 | -0.024 |
| 0.8B | alloc-type-gptq32-avg2.5 | 0 | 2.6228 | 2.1645 | 2.1507 | 35.02 | +99.4% | 0.69544 | 0.6163 | 0.74651 | 0.932 | 0.699 | yes | +0.039 | +0.041 |
| 0.8B | alloc-type-gptq32-avg2.5 | 1 | 2.6228 | 2.1646 | 2.1508 | 37.14 | +111.4% | 0.69757 | 0.6124 | 0.74651 | 0.934 | 0.784 | yes | +0.038 | +0.039 |
| 0.8B | alloc-type-rtn-avg3.5 | 0 | 3.6240 | 3.1468 | 3.1292 | 25.21 | +43.5% | 0.35249 | 0.7174 | 0.14494 | 2.432 | 3.836 | rtn gate | -0.585 | -0.574 |
| 0.8B | alloc-type-rtn-avg3 | 0 | 3.1246 | 2.6544 | 2.6376 | 48.42 | +175.6% | 0.94535 | 0.5516 | 0.30194 | 3.131 | 5.215 | rtn gate | -0.633 | -0.615 |
| 0.8B | alloc-tensor-gptq32-avg3.5 | 0 | 3.6250 | 3.1499 | 3.1324 | 19.48 | +10.9% | 0.11637 | 0.8296 | 0.14473 | 0.804 | 0.963 | no (dPPL decides) | +0.148 | +0.146 |
| 0.8B | alloc-tensor-gptq32-avg3.5 | 1 | 3.6250 | 3.1499 | 3.1324 | 20.32 | +15.7% | 0.12159 | 0.8251 | 0.14473 | 0.840 | 1.383 | no (dPPL decides) | +0.119 | +0.117 |
| 0.8B | alloc-tensor-gptq32-avg3 | 0 | 3.1250 | 2.6572 | 2.6402 | 22.65 | +29.0% | 0.24545 | 0.7603 | 0.30173 | 0.813 | 0.861 | yes | +0.141 | +0.138 |
| 0.8B | alloc-tensor-gptq32-avg3 | 1 | 3.1250 | 2.6572 | 2.6403 | 22.63 | +28.8% | 0.24099 | 0.7673 | 0.30173 | 0.799 | 0.857 | yes | +0.153 | +0.151 |
| 0.8B | alloc-type-shuf1-gptq32-avg3 | 0 | 3.1135 | 2.6481 | 2.6325 | 30.07 | +71.2% | 0.54256 | 0.6602 | 0.30803 | 1.761 | 2.048 | no | -0.314 | -0.309 |
| 0.8B | alloc-type-shuf2-gptq32-avg3 | 0 | 3.1274 | 2.6673 | 2.6526 | 57.44 | +227.0% | 1.12708 | 0.5209 | 0.30068 | 3.748 | 6.782 | no | -0.733 | -0.723 |
| 2B | alloc-type-gptq32-avg4 | 0 | 4.1239 | 3.6352 | 3.6174 | 12.60 | +6.6% | 0.06452 | 0.8763 | 0.05820 | 1.109 | 1.146 | no | -0.070 | -0.071 |
| 2B | alloc-type-gptq32-avg4 | 1 | 4.1239 | 3.6352 | 3.6174 | 12.62 | +6.8% | 0.06466 | 0.8778 | 0.05820 | 1.111 | 1.179 | no | -0.072 | -0.073 |
| 2B | alloc-type-gptq32-avg3.5 | 0 | 3.6243 | 3.1411 | 3.1237 | 13.35 | +13.0% | 0.12170 | 0.8347 | 0.12145 | 1.002 | 1.025 | no | -0.001 | -0.002 |
| 2B | alloc-type-gptq32-avg3.5 | 1 | 3.6243 | 3.1412 | 3.1237 | 13.22 | +12.0% | 0.12025 | 0.8356 | 0.12145 | 0.990 | 0.943 | no | +0.007 | +0.006 |
| 2B | alloc-type-gptq32-avg3 | 0 | 3.1247 | 2.6496 | 2.6336 | 16.15 | +36.8% | 0.30352 | 0.7500 | 0.25349 | 1.197 | 1.323 | no | -0.089 | -0.088 |
| 2B | alloc-type-gptq32-avg3 | 1 | 3.1247 | 2.6496 | 2.6337 | 15.65 | +32.5% | 0.29309 | 0.7551 | 0.25349 | 1.156 | 1.171 | no | -0.072 | -0.071 |
| 2B | alloc-type-gptq32-avg2.5 | 0 | 2.6234 | 2.1591 | 2.1450 | 22.95 | +94.3% | 0.64756 | 0.6444 | 0.70013 | 0.925 | 0.708 | yes | +0.039 | +0.038 |
| 2B | alloc-type-gptq32-avg2.5 | 1 | 2.6234 | 2.1591 | 2.1450 | 22.79 | +92.9% | 0.64251 | 0.6415 | 0.70013 | 0.918 | 0.697 | yes | +0.042 | +0.042 |
| 2B | alloc-tensor-gptq32-avg3.5 | 0 | 3.6250 | 3.1423 | 3.1256 | 13.33 | +12.9% | 0.11551 | 0.8419 | 0.12132 | 0.952 | 1.016 | no | +0.033 | +0.032 |
| 2B | alloc-tensor-gptq32-avg3 | 0 | 3.1250 | 2.6510 | 2.6348 | 15.51 | +31.3% | 0.27580 | 0.7601 | 0.25334 | 1.089 | 1.128 | no | -0.042 | -0.043 |
| 2B | alloc-type-shuf1-gptq32-avg3 | 0 | 3.1396 | 2.6664 | 2.6503 | 20.30 | +71.9% | 0.55421 | 0.6711 | 0.24795 | 2.235 | 2.649 | no | -0.401 | -0.394 |
| 2B | alloc-type-shuf2-gptq32-avg3 | 0 | 3.1267 | 2.6550 | 2.6396 | 24.56 | +107.9% | 0.70729 | 0.6273 | 0.25270 | 2.799 | 3.899 | no | -0.508 | -0.500 |
| MiniCPM | alloc-type-gptq32-avg3.5 | 0 | 3.6250 | 3.1683 | 3.1510 | 15.64 | +13.5% | 0.16066 | 0.8108 | 0.16280 | 0.987 | 0.979 | no | +0.009 | +0.009 |
| MiniCPM | alloc-type-gptq32-avg3 | 0 | 3.1250 | 2.6756 | 2.6573 | 18.23 | +32.3% | 0.34211 | 0.7337 | 0.34362 | 0.996 | 0.985 | no | +0.003 | +0.003 |

Controls. Shuffles (bits within 0.015 of the real allocation; 10-22% of the parameters left at their real width):
KL ratio vs the curve 1.761 / 3.748 (0.8B) and 2.235 / 2.799 (2B); the real allocation's KL is 0.588 / 0.283 and
0.548 / 0.429 of the shuffles'. RTN gate (0.8B): alloc-type-rtn avg 3 / 3.5 KL 0.94535 / 0.35249 = 0.628 / 0.633 of
the uniform RTN curve at the same bits, i.e. the Phase 2 RTN-derived lever reproduces on RTN, larger than Phase 2's
estimate (KL x 0.72-0.81).

Verdict per clause:
- Type-level beats uniform GPTQ by 15-30% at avg 3.5 and 3, both Qwens, both seeds: **refuted**. 0.8B avg 3.5
  -7.0% / -8.6% (KL-only clears 5%, dPPL decides "no"; mid-segment, inside the ~4% chord bias plus margin), avg 3
  +5.7% / +4.3% (worse); 2B avg 3.5 +0.2% / -1.0%, avg 3 +19.7% / +15.6% (worse). Also worse at avg 4: +4.2% /
  +4.6% (0.8B, inside noise), +10.9% / +11.1% (2B). MiniCPM avg 3.5 -1.3% (no). The only rule "yes" is avg 2.5
  (0.932 / 0.934 on the 0.8B, 0.925 / 0.918 on the 2B, +0.04 bits), which sits mid-segment: against the quadratic
  curve it is 0.971 / 0.974 and 0.991 / 0.984, so **not established**. At every width that lies on a measured curve
  point, type allocation is equal to or worse than uniform.
- Shuffles no better than uniform: **supported** (they are 1.8-3.7x worse, not "within 5%"); real beats both shuffles
  by > 5%: true, but there is no gain over uniform to attribute.
- Tensor-level adds < 10% on top of type-level: **refuted on the 0.8B** (tensor / type KL 0.769 / 0.765 at avg 3,
  0.863 / 0.918 at avg 3.5), **holds on the 2B** (0.909 at avg 3, 0.949 at avg 3.5, Fisher signal, seed 0 only).
  alloc-tensor-gptq32-avg3 on the 0.8B is the one positive A row: 0.813 / 0.799 of the curve, dPPL 0.861 / 0.857,
  +0.141 / +0.153 bits fixed, on a measured curve point, both seeds. Its allocation-signal noise is untested (one
  damage draw). The 2B tensor rows do not beat (1.089 at avg 3, 0.952 at avg 3.5 with dPPL 1.016).
- Spearman > 0.9: **supported** on all three models.
- Mechanism for the GPTQ failure (proposed, not tested): the allocator assumes KL x 4 per bit removed (2^(-2 b)) and is
  fitted on RTN-4 damage; the measured uniform GPTQ curve multiplies KL by 4.35 / 4.36 from 4 to 3 bits but by 6.07 /
  7.59 from 3 to 2 bits (0.8B / 2B), and avg 3 sends the largest types to 2 bits (in_proj_qkv, 23% of the 0.8B's block
  parameters; gate, 22% of the 2B's). GPTQ also removes 56-71% of the RTN-4 KL (Phase 2), and not necessarily evenly across types.

#### D: zero-bit rotation (H3-D)

Pre-registered: Hadamard + GPTQ beats GPTQ by > 20% KL at 3 bits on all three models; at 4 bits the gain is < 10%;
at 2 bits it is the largest (> 40%); random orthogonal equals Hadamard within 5% KL; the eigenbasis rotation costs
> 1 bit/param stored and loses.

Run: had-gptq32 4 / 3 / 2 on both Qwens seeds 0 and 1, MiniCPM 3 seed 0; rnd-3 0.8B seeds 0, 1; hadsign-3 and eig-3
0.8B seed 0.

| model | method | seed | bits_blocks | huff | shannon | ppl | dPPL% | kl | top1 | KL curve @bits | KL ratio | dPPL ratio | beats | saved fixed | saved huff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8B | had-gptq32-sym-g128-4 | 0 | 4.1250 | 3.6992 | 3.6858 | 19.11 | +8.8% | 0.05244 | 0.8838 | 0.06943 | 0.755 | 2.306 | no (dPPL decides) | +0.191* | +0.131* |
| 0.8B | had-gptq32-sym-g128-4 | 1 | 4.1250 | 3.6992 | 3.6858 | 18.79 | +6.9% | 0.04884 | 0.8868 | 0.06943 | 0.703 | 1.820 | no (dPPL decides) | +0.239* | +0.179* |
| 0.8B | had-gptq32-sym-g128-3 | 0 | 3.1250 | 2.7138 | 2.6942 | 20.71 | +17.9% | 0.21136 | 0.7775 | 0.30173 | 0.701 | 0.531 | yes | +0.242 | +0.182 |
| 0.8B | had-gptq32-sym-g128-3 | 1 | 3.1250 | 2.7138 | 2.6942 | 20.87 | +18.8% | 0.21445 | 0.7767 | 0.30173 | 0.711 | 0.559 | yes | +0.232 | +0.172 |
| 0.8B | had-gptq32-sym-g128-2 | 0 | 2.1250 | 1.7152 | 1.7080 | 69.80 | +297.3% | 1.28951 | 0.4936 | 1.83255 | 0.704 | 0.502 | yes | +0.195 | +0.156 |
| 0.8B | had-gptq32-sym-g128-2 | 1 | 2.1250 | 1.7152 | 1.7079 | 67.16 | +282.3% | 1.29795 | 0.4951 | 1.83255 | 0.708 | 0.476 | yes | +0.191 | +0.152 |
| 0.8B | rnd-gptq32-sym-g128-3 | 0 | 3.1250 | 2.7142 | 2.6946 | 22.09 | +25.7% | 0.21271 | 0.7753 | 0.30173 | 0.705 | 0.765 | yes | +0.238 | +0.177 |
| 0.8B | rnd-gptq32-sym-g128-3 | 1 | 3.1250 | 2.7141 | 2.6945 | 21.38 | +21.7% | 0.20470 | 0.7798 | 0.30173 | 0.678 | 0.644 | yes | +0.264 | +0.203 |
| 0.8B | hadsign-gptq32-sym-g128-3 | 0 | 3.1250 | 2.7142 | 2.6946 | 21.46 | +22.1% | 0.21220 | 0.7742 | 0.30173 | 0.703 | 0.658 | yes (1 seed) | +0.240 | +0.179 |
| 0.8B | eig-gptq32-sym-g128-3 | 0 | 17.8923 | 17.4618 | 17.4430 | 29.67 | +68.9% | 0.48990 | 0.6692 | —* | — | — | no | -15.036 | -15.067 |
| 2B | had-gptq32-sym-g128-4 | 0 | 4.1250 | 3.6988 | 3.6857 | 12.23 | +3.5% | 0.03764 | 0.9043 | 0.05810 | 0.648 | 0.610 | yes | +0.295* | +0.225* |
| 2B | had-gptq32-sym-g128-4 | 1 | 4.1250 | 3.6988 | 3.6857 | 11.86 | +0.4% | 0.04226 | 0.9004 | 0.05810 | 0.727 | 0.075 | yes | +0.216* | +0.148* |
| 2B | had-gptq32-sym-g128-3 | 0 | 3.1250 | 2.7135 | 2.6940 | 13.63 | +15.4% | 0.16275 | 0.8092 | 0.25334 | 0.642 | 0.556 | yes | +0.301 | +0.231 |
| 2B | had-gptq32-sym-g128-3 | 1 | 3.1250 | 2.7135 | 2.6940 | 14.06 | +19.0% | 0.16383 | 0.8099 | 0.25334 | 0.647 | 0.686 | yes | +0.296 | +0.227 |
| 2B | had-gptq32-sym-g128-2 | 0 | 2.1250 | 1.7187 | 1.7105 | 32.76 | +177.4% | 1.01444 | 0.5608 | 1.92254 | 0.528 | 0.280 | yes | +0.315 | +0.263 |
| 2B | had-gptq32-sym-g128-2 | 1 | 2.1250 | 1.7187 | 1.7105 | 34.01 | +188.0% | 1.00553 | 0.5684 | 1.92254 | 0.523 | 0.296 | yes | +0.320 | +0.267 |
| MiniCPM | had-gptq32-sym-g128-3 | 0 | 3.1250 | 2.7137 | 2.6942 | 16.64 | +20.8% | 0.24102 | 0.7757 | 0.34362 | 0.701 | 0.634 | yes (1 seed) | +0.237 | +0.196 |

Control: rnd / had (same seed) KL 1.006 / 0.955, hadsign / had 1.004; on PPL, rnd is 6.7% / 2.4% worse than had
(seed 0 beyond the 4.4% PPL spread, seed 1 inside; the rnd pair itself spreads 3.3%), not a resolved difference.

Verdict per clause:
- > 20% at 3 bits on all three models: **supported** (0.8B -29.9% / -28.9%, 2B -35.8% / -35.3%, MiniCPM -29.9% one
  seed); beats on KL and dPPL; +0.23 to +0.30 bits fixed, +0.17 to +0.23 Huffman.
- < 10% at 4 bits: **refuted** on both Qwens. 0.8B -24.5% / -29.7% (verdict "no" by dPPL alone, 2.31 / 1.82; the
  KL-only reading beats), 2B -35.2% / -27.3% (beats). The 2B saved figures +0.295 / +0.216 are read outside the curve
  (KL below its lowest point; +0.400 / +0.285 on the quadratic): **extrapolated, not measured**.
- Largest at 2 bits, > 40%: **refuted on the 0.8B** (-29.6% / -29.2%, not larger than at 3 bits), **supported on the
  2B** (-47.2% / -47.7%, the largest of its three widths).
- rnd within 5% of had: **supported** (generic incoherence; seed 1 at -4.5% is near the edge).
- eig costs > 1 bit/param and loses: **supported**: 14.767 bits/param of stored bf16 rotations (re-derived from the
  shapes), saved -15.04; its KL 0.48990 is also 1.63x plain gptq32-3 (seed 0) before any bits are counted (the eigenbasis
  makes GPTQ worse, not better; mechanism not examined).
- Side observations: rotation costs coding gain (had-3 Huffman saving 0.411 vs 0.468 for the baseline at 3 bits, i.e.
  -0.057 bits), so had's Huffman saving is ~0.06 bits below its fixed saving; rotation equalizes the RD gap across
  types (MiniCPM k 1.88 -> 1.14 fixed bits above the bound at 3 bits; all-types 1.27 -> 1.13 on the 0.8B at 4 bits).
- Equal-quality point against GPTQ-4 (interpolated inside the measured had segment, log-linear; not a measured row):
  the had curve reaches the gptq32-4 seed-mean KL at 3.905 bits on the 0.8B and 3.859 on the 2B, i.e. 0.22 / 0.27
  bits fixed saved; with Huffman (3.483 / 3.437) that is 0.64 / 0.69 bits below gptq32-4's fixed 4.125 and 0.16 / 0.20
  below gptq32-4's own Huffman figure. The chord bias, if had's curve is convex like the baseline's, makes these
  savings lower bounds.

#### E: codebook and entropy coding (H3-E)

Pre-registered: Huffman coding of GPTQ indices saves 0.40-0.50 bits/param at 4 bits, 0.25-0.40 at 3, < 0.20 at 2 with
single-symbol codes (pair codes recover part of the 2-bit loss); the gap to the Shannon figure is < 0.05 bits at 4
bits; Lloyd-Max lowers KL by 10-25% at 3-4 bits for < 0.01 bits of table; the fixed Gaussian codebook captures at
least two thirds of that; all coders sit 1.0-1.5 bits above the Gaussian RD bound at equal MSE at 4 bits, and the gap
shrinks with fewer bits.

Run: Huffman / Shannon / entropy0 figures on every row; lloyd-gptq32 4 / 3 / 2 on the 0.8B seeds 0, 1, lloyd-3 on the
2B and MiniCPM seed 0; nf-4 / 3 and nfq-4 / 3 on the 0.8B seed 0; rd_bound.jsonl for every row (616 / 341 / 56 rows).

Huffman on the gptq32 baseline (fixed - huff; identical on both seeds to 3 decimals):

| model | 4 bits | 3 bits | 2 bits | huff - entropy0 at 4 bits |
|---|---|---|---|---|
| 0.8B | 0.483 | 0.468 | 0.444 | 0.016 |
| 2B | 0.491 | 0.476 | 0.451 | 0.016 |
| MiniCPM | 0.464 | 0.449 | n/a | 0.015 |

An independent Huffman implementation (audit, 8 rows incl. combo, row-allocated and RTN rows) reproduces
`bits_blocks_huff` and entropy0 to full float precision.

| model | method | seed | bits_blocks | huff | shannon | ppl | dPPL% | kl | top1 | KL curve @bits | KL ratio | dPPL ratio | beats | saved fixed | saved huff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8B | lloyd-gptq32-g128-4 | 0 | 4.1251 | 3.9297 | 3.9179 | 19.58 | +11.4% | 0.10666 | 0.8320 | 0.06942 | 1.536 | 3.005 | no | -0.292 | -0.575 |
| 0.8B | lloyd-gptq32-g128-4 | 1 | 4.1251 | 3.9297 | 3.9179 | 19.44 | +10.7% | 0.10141 | 0.8398 | 0.06942 | 1.461 | 2.797 | no | -0.258 | -0.542 |
| 0.8B | lloyd-gptq32-g128-3 | 0 | 3.1250 | 2.9388 | 2.9248 | 24.28 | +38.2% | 0.34142 | 0.6881 | 0.30171 | 1.132 | 1.136 | no | -0.069 | -0.349 |
| 0.8B | lloyd-gptq32-g128-3 | 1 | 3.1250 | 2.9388 | 2.9247 | 23.97 | +36.4% | 0.35874 | 0.6827 | 0.30171 | 1.189 | 1.083 | no | -0.096 | -0.375 |
| 0.8B | lloyd-gptq32-g128-2 | 0 | 2.1250 | 2.0424 | 2.0232 | 71.10 | +304.7% | 1.32661 | 0.4671 | 1.83247 | 0.724 | 0.514 | yes | +0.179 | -0.187 |
| 0.8B | lloyd-gptq32-g128-2 | 1 | 2.1250 | 2.0424 | 2.0232 | 71.81 | +308.8% | 1.35011 | 0.4665 | 1.83247 | 0.737 | 0.521 | yes | +0.169 | -0.196 |
| 0.8B | nf-gptq32-g128-4 | 0 | 4.1250 | 3.9014 | 3.8861 | 19.43 | +10.6% | 0.12637 | 0.8154 | 0.06943 | 1.820 | 2.785 | no | -0.408 | -0.661 |
| 0.8B | nf-gptq32-g128-3 | 0 | 3.1250 | 2.9569 | 2.9439 | 25.60 | +45.7% | 0.39999 | 0.6705 | 0.30173 | 1.326 | 1.359 | no | -0.156 | -0.452 |
| 0.8B | nfq-gptq32-g128-4 | 0 | 4.1250 | 4.1250 | 4.1233 | 28.88 | +64.4% | 0.49311 | 0.6530 | 0.06943 | 7.103 | 16.908 | no | -1.272 | -1.734 |
| 0.8B | nfq-gptq32-g128-3 | 0 | 3.1250 | 3.1250 | 3.1216 | 53.90 | +206.8% | 1.05740 | 0.5252 | 0.30173 | 3.504 | 6.148 | no | -0.695 | -1.147 |
| 2B | lloyd-gptq32-g128-3 | 0 | 3.1250 | 2.9354 | 2.9208 | 21.83 | +84.8% | 0.62069 | 0.6423 | 0.25333 | 2.450 | 3.055 | no | -0.442 | -0.718 |
| MiniCPM | lloyd-gptq32-g128-3 | 0 | 3.1250 | 2.9477 | 2.9346 | 44.76 | +224.8% | 1.26109 | 0.5258 | 0.34361 | 3.670 | 6.860 | no | -0.870* | -1.130* |

Lloyd-Max mechanism (audit probe, 15 tensors of 0.8B layers 3 and 12; not a pre-registered measurement): Lloyd's
group-normalized MSE is lower than the uniform grid's (0.01721 vs 0.01923 at 4 bits, 0.0579 vs 0.0777 at 3), but its
3-bit outer level sits at 2.2-2.4 sigma while max |w / rms_g| reaches 10.4: it clips 2.8% of the weights, which carry
22.7% of the squared error (0.66% / 17.9% at 4 bits); the amax uniform grid never clips. The KL loss is tail
clipping under RMS normalization (MSE-optimal, KL-harmful), not a broken path.

RD gap (bits above R(D) = 0.5 log2(1/D) on the group-normalized weight MSE, i.e. the distance to 6.02 dB/bit), gptq32
numel-weighted over types, fixed / Huffman / Shannon:

| model | 4 bits | 3 bits | 2 bits |
|---|---|---|---|
| 0.8B | 1.27 / 0.79 / 0.77 | 1.28 / 0.81 / 0.79 | 1.32 / 0.87 / 0.86 |
| 2B | 1.24 / 0.75 / 0.73 | 1.25 / 0.77 / 0.75 | 1.29 / 0.84 / 0.83 |
| MiniCPM | 1.21 / 0.74 / 0.73 | 1.21 / 0.76 / 0.74 | n/a |

Per type, median over every row of the model's rd_bound.jsonl (fixed / Huffman / Shannon):

| type | 0.8B (56 runs) | 2B (31 runs) | MiniCPM (8 runs) |
|---|---|---|---|
| in_proj_qkv | 1.25 / 0.84 / 0.82 | 1.26 / 0.77 / 0.76 | - |
| in_proj_z | 1.25 / 0.81 / 0.79 | 1.24 / 0.78 / 0.76 | - |
| in_proj_ab | 1.52 / 1.12 / 1.00 | 1.66 / 0.94 / 0.87 | - |
| out_proj | 1.32 / 0.88 / 0.86 | 1.34 / 0.83 / 0.82 | - |
| gate | 1.22 / 0.79 / 0.77 | 1.21 / 0.75 / 0.74 | 1.18 / 0.73 / 0.72 |
| up | 1.19 / 0.76 / 0.75 | 1.19 / 0.75 / 0.74 | 1.17 / 0.73 / 0.71 |
| down | 1.31 / 0.86 / 0.85 | 1.26 / 0.77 / 0.75 | 1.19 / 0.72 / 0.70 |
| q | 1.33 / 0.86 / 0.84 | 1.32 / 0.81 / 0.80 | 1.35 / 0.90 / 0.88 |
| k | 1.42 / 0.95 / 0.93 | 1.49 / 0.84 / 0.82 | 1.86 / 1.33 / 1.31 |
| v | 1.54 / 0.99 / 0.96 | 1.60 / 0.91 / 0.89 | 1.46 / 0.87 / 0.85 |
| o | 1.37 / 0.89 / 0.87 | 1.38 / 0.84 / 0.83 | 1.24 / 0.83 / 0.81 |

All 1013 rd_bound rows satisfy gap = bits - 0.5 log2(1/D) exactly, and numel-weighting the per-type rows reproduces
each run's bits figures.

**Proxy flagged, not averaged:** the weight-MSE RD gap is contradicted by KL in four families. RTN-4 sits at 0.96
bits above the bound against GPTQ-4's 1.27 while its KL is 2.95x higher; Lloyd lowers the gap (1.07 vs 1.28 at 3 bits)
while raising KL 13-19%; row allocation raises it (1.49-1.51 vs 1.27-1.28) while lowering KL 7.5-9.2%; the combo sits at
1.37 at avg 3.5 with KL 0.66x the curve. GPTQ trades weight MSE for output error by design, so the weight-domain
gap measures distance from a bound on the wrong distortion; it is reported as pre-registered but carries no
quality claim.

Verdict per clause:
- 0.40-0.50 at 4 bits: **supported** (0.464-0.491).
- 0.25-0.40 at 3 bits: **refuted** (0.449-0.476, above the range).
- < 0.20 at 2 bits: **refuted** (0.444 / 0.451); the pair-code clause is moot (no 2-bit loss to recover).
- Shannon gap < 0.05 at 4 bits: **supported** (huff - entropy0 0.015-0.016).
- Lloyd-Max 10-25% lower KL at 3-4 bits for < 0.01 bits of table: table cost holds (<= 0.0001 bits), the KL clause
  is **refuted**: +53.6% / +46.1% at 4 bits and +13.2% / +18.9% at 3 bits on the 0.8B, +145% on the 2B and +267% on
  MiniCPM at 3 bits. At 2 bits Lloyd beats on the fixed curve (0.724 / 0.737, +0.18 / +0.17 bits) but loses on the
  coded curve (-0.19 / -0.20 bits: its indices are near-equiprobable, 2.04 of 2.125 bits after Huffman against 1.68
  for the uniform grid) and against Hadamard (1.025 / 1.044 of the had curve): **no net win**.
- Fixed Gaussian codebook captures >= 2/3 of the Lloyd gain: **refuted** (no gain to share; nf is itself worse than
  Lloyd, seed 0: 1.820 vs 1.536 at 4 bits, 1.326 vs 1.132 at 3; nfq as written in DESIGN.md is 7.10 / 3.50).
- 1.0-1.5 bits above the bound at 4 bits: **supported for the fixed code** (1.21-1.27); the entropy-coded figures are
  0.74-0.79, below the range.
- Gap shrinks with fewer bits: **refuted** (fixed gap grows by 0.05 bits from 4 to 2 on both Qwens, the Huffman gap by
  0.08-0.09; flat from 4 to 3 on MiniCPM, which has no 2-bit row).

#### G: within-tensor row allocation by Fisher row energy (H3-G)

Pre-registered: uncertain; prior: the Fisher row split beats uniform RTN-4 by 10-20% KL at equal bits and the
shuffled-row controls do not; if it fails, within-tensor allocation is dropped.

Run: rowalloc-fisher-rtn-avg4 and two shuffled-row controls (0.8B seed 0); gate passed, so rowalloc-fisher-gptq32
avg 4 / 3 (0.8B seeds 0, 1). Rows at b-1 / b / b+1, 2 bits per row of side information (+0.0016 bits/param).

| model | method | seed | bits_blocks | huff | shannon | ppl | dPPL% | kl | top1 | KL curve @bits | KL ratio | dPPL ratio | beats | saved fixed | saved huff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8B | rowalloc-shuf1-rtn-avg4 | 0 | 4.1266 | 3.6452 | 3.6273 | 24.20 | +37.7% | 0.26623 | 0.7535 | 0.06926 | 3.844 | 9.938 | rtn gate | -0.916 | -0.904 |
| 0.8B | rowalloc-shuf2-rtn-avg4 | 0 | 4.1266 | 3.6453 | 3.6274 | 23.37 | +33.0% | 0.24154 | 0.7570 | 0.06926 | 3.487 | 8.693 | rtn gate | -0.850 | -0.839 |
| 0.8B | rowalloc-fisher-gptq32-avg4 | 0 | 4.1266 | 3.6463 | 3.6282 | 19.08 | +8.6% | 0.06328 | 0.8714 | 0.06926 | 0.914 | 2.272 | no (dPPL decides) | +0.061* | +0.058* |
| 0.8B | rowalloc-fisher-gptq32-avg4 | 1 | 4.1266 | 3.6463 | 3.6282 | 19.48 | +10.9% | 0.06408 | 0.8732 | 0.06926 | 0.925 | 2.873 | no (dPPL decides) | +0.053* | +0.050* |
| 0.8B | rowalloc-fisher-gptq32-avg3 | 0 | 3.1266 | 2.6593 | 2.6428 | 23.97 | +36.4% | 0.27663 | 0.7489 | 0.30102 | 0.919 | 1.087 | no (1 of 2 seeds) (dPPL decides) | +0.058 | +0.056 |
| 0.8B | rowalloc-fisher-gptq32-avg3 | 1 | 3.1266 | 2.6593 | 2.6428 | 23.07 | +31.3% | 0.27347 | 0.7482 | 0.30102 | 0.908 | 0.935 | no (1 of 2 seeds) | +0.065 | +0.064 |
| 0.8B | rowalloc-fisher-rtn-avg4 | 0 | 4.1266 | 3.6450 | 3.6270 | 21.13 | +20.3% | 0.17509 | 0.7949 | 0.06926 | 2.528 | 5.346 | rtn gate | -0.631 | -0.623 |

Gate: KL 0.17509 vs rtn-sym-g128-4 0.20514 = 0.853 (-14.7%); controls 0.26623 / 0.24154 are 1.30x / 1.18x worse than
uniform, real / control 0.658 / 0.725; all three at exactly 4.12661 bits.

Verdict: **supported** at the pre-registered clause (RTN gate, -14.7%, inside 10-20%; controls do not beat uniform).
With GPTQ the gain shrinks to -7.5% to -9.2% KL (+0.05 to +0.07 bits fixed) and does not pass the two-sided rule
(avg 4 "no", avg 3 "no (1 of 2 seeds)", dPPL decides every one of the four rows; KL-only all four pass). Within-tensor
allocation is therefore not dropped, but its GPTQ gain is not established. The cheap second observation of G
(d_t against ||W_t X||^2 / n_t and the output RMS) was not in the run list and was not run.
**Correction (2026-11-08, REPORT.md fact-check):** "dPPL decides every one of the four rows" is wrong. dPPL decides
three: avg 4 seeds 0 and 1 (dPPL ratio 2.272 / 2.873) and avg 3 seed 0 (1.087). Avg 3 seed 1 clears both margins (KL
0.908, dPPL 0.935, row beats = True, as in the table above); the avg-3 verdict is "no (1 of 2 seeds)" because seed 0
fails on dPPL. The conclusion (GPTQ gain not established) is unchanged.

#### Combination (H3-combo)

Pre-registered: at avg 3.0 (fixed ~3.1, Huffman ~2.7-2.8) the combination reaches the KL of uniform GPTQ-4 (fixed
4.125) on the 0.8B and the 2B, i.e. about 1.0-1.3 bits/param saved at equal quality against the Phase 1 best coder,
of which ~0.4 is entropy coding, ~0.25 rotation, ~0.2 allocation, the rest interaction; at avg 2.5 it stays below
+50% PPL.

Run: components from the decisions above (`combo.json` = had + rowalloc, no Lloyd, identical sha 8a50c79fd2 on all
three models): had + type allocation + Fisher row split around each type's width + uniform grid + GPTQ f32. Avg 2.5 /
3 / 3.5 on both Qwens seeds 0, 1; avg 3 on MiniCPM seed 0 (where the type allocation is uniform, so the combo is had +
row split).

| model | method | seed | bits_blocks | huff | shannon | ppl | dPPL% | kl | top1 | KL curve @bits | KL ratio | dPPL ratio | beats | saved fixed | saved huff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8B | combo-gptq32-avg3.5 | 0 | 3.6256 | 3.2063 | 3.1915 | 19.34 | +10.1% | 0.09473 | 0.8504 | 0.14460 | 0.655 | 0.893 | yes | +0.288 | +0.228 |
| 0.8B | combo-gptq32-avg3.5 | 1 | 3.6256 | 3.2063 | 3.1915 | 19.41 | +10.5% | 0.09741 | 0.8483 | 0.14460 | 0.674 | 0.929 | yes | +0.269 | +0.209 |
| 0.8B | combo-gptq32-avg3 | 0 | 3.1258 | 2.7122 | 2.6970 | 22.60 | +28.7% | 0.26646 | 0.7541 | 0.30139 | 0.884 | 0.854 | no (1 of 2 seeds) | +0.084 | +0.028 |
| 0.8B | combo-gptq32-avg3 | 1 | 3.1258 | 2.7122 | 2.6970 | 23.25 | +32.3% | 0.27357 | 0.7477 | 0.30139 | 0.908 | 0.963 | no (1 of 2 seeds) (dPPL decides) | +0.066 | +0.010 |
| 0.8B | combo-gptq32-avg2.5 | 0 | 2.6234 | 2.2123 | 2.1997 | 31.55 | +79.6% | 0.55650 | 0.6500 | 0.74575 | 0.746 | 0.561 | yes | +0.162 | +0.113 |
| 0.8B | combo-gptq32-avg2.5 | 1 | 2.6234 | 2.2122 | 2.1997 | 32.65 | +85.8% | 0.57858 | 0.6429 | 0.74575 | 0.776 | 0.605 | yes | +0.141 | +0.092 |
| 2B | combo-gptq32-avg3.5 | 0 | 3.6251 | 3.2071 | 3.1914 | 12.76 | +8.1% | 0.08790 | 0.8595 | 0.12130 | 0.725 | 0.636 | no (1 of 2 seeds) | +0.219 | +0.150 |
| 2B | combo-gptq32-avg3.5 | 1 | 3.6251 | 3.2071 | 3.1914 | 13.26 | +12.3% | 0.08970 | 0.8565 | 0.12130 | 0.740 | 0.970 | no (1 of 2 seeds) (dPPL decides) | +0.205 | +0.136 |
| 2B | combo-gptq32-avg3 | 0 | 3.1253 | 2.7124 | 2.6971 | 14.43 | +22.2% | 0.21294 | 0.7843 | 0.25322 | 0.841 | 0.799 | yes | +0.118 | +0.052 |
| 2B | combo-gptq32-avg3 | 1 | 3.1253 | 2.7124 | 2.6972 | 14.78 | +25.2% | 0.20839 | 0.7878 | 0.25322 | 0.823 | 0.907 | yes | +0.132 | +0.067 |
| 2B | combo-gptq32-avg2.5 | 0 | 2.6237 | 2.2140 | 2.2022 | 18.66 | +58.0% | 0.45086 | 0.6921 | 0.69978 | 0.644 | 0.435 | yes | +0.217 | +0.157 |
| 2B | combo-gptq32-avg2.5 | 1 | 2.6237 | 2.2140 | 2.2022 | 18.28 | +54.8% | 0.43439 | 0.6966 | 0.69978 | 0.621 | 0.411 | yes | +0.235 | +0.175 |
| MiniCPM | combo-gptq32-avg3 | 0 | 3.1258 | 2.7139 | 2.6956 | 16.56 | +20.2% | 0.23240 | 0.7766 | 0.34321 | 0.677 | 0.616 | yes (1 seed) | +0.261 | +0.220 |

KL / gptq32-4 seed-mean KL at avg 3: 3.84 / 3.94 (0.8B), 3.67 / 3.59 (2B), 3.01 (MiniCPM). Total bits saved at equal
KL against fixed-length uniform GPTQ (curve fixed bits at the combo's KL minus the combo's Huffman bits): 0.8B 0.497 /
0.479, 2B 0.531 / 0.545, MiniCPM 0.673 at avg 3; best case 0.8B avg 3.5 0.707 / 0.688 (mid-segment).

Decomposition at avg 3 (each part = that single-method row's own saved fixed at ~3.125 bits; coding gain = combo
fixed - combo Huffman):

| model | coding gain | had | type alloc | row alloc (gptq32 avg 3) | sum of parts (excl. coding) | combo saved fixed | interaction |
|---|---|---|---|---|---|---|---|
| 0.8B s0 / s1 | 0.414 | +0.242 / +0.232 | -0.031 / -0.023 | +0.058 / +0.065 | +0.269 / +0.274 | +0.084 / +0.066 | -0.185 / -0.209 |
| 2B s0 / s1 | 0.413 | +0.301 / +0.296 | -0.089 / -0.072 | not run | - | +0.118 / +0.132 | rowalloc + interaction -0.094 / -0.092 |
| MiniCPM s0 | 0.412 | +0.237 | +0.003 (uniform) | not run | - | +0.261 | rowalloc + interaction +0.024 (inside noise) |
| pre-registered | ~0.4 | ~0.25 | ~0.2 (allocation) | | | | rest |

At avg 3.5 the 0.8B parts are roughly additive (had ~+0.22 / +0.24 interpolated, type +0.049 / +0.061, combo +0.288 /
+0.269); on the 2B the interaction is -0.08 / -0.06; at avg 2.5 -0.07 to -0.09 on both.

Against Hadamard alone (combo KL / had-gptq32 curve, log-linear chord between the seed-mean had points; audit's
quadratic in brackets): 0.8B avg 2.5 1.057 / 1.099 (1.107 / 1.151), avg 3 1.253 / 1.286, avg 3.5 0.913 / 0.939 (0.956
/ 0.983); 2B avg 2.5 1.108 / 1.067 (1.166 / 1.124), avg 3 1.305 / 1.277, avg 3.5 1.089 / 1.111 (1.146 / 1.170);
MiniCPM avg 3 0.964 of had-3 (one seed, inside the noise). The combination never beats Hadamard alone by more than
the noise and is 25-31% worse at avg 3: its bits saved are Hadamard's, and type allocation plus the row split add
zero or less on top of the rotation.

Verdict per clause:
- Reaches gptq32-4's KL at avg 3: **refuted** (3.0-3.9x).
- 1.0-1.3 bits saved: **refuted** (0.48-0.55 on the Qwens, 0.67 on MiniCPM, coding included).
- Fixed ~3.1 / Huffman 2.7-2.8: holds (3.1258 / 2.712).
- Split ~0.4 coding / ~0.25 rotation / ~0.2 allocation: coding and rotation **match**, allocation **refuted**
  (-0.02 to -0.09 type, +0.06 row), interaction **negative** (-0.19 / -0.21 on the 0.8B at avg 3).
- Avg 2.5 below +50% PPL: **refuted** (+79.6% / +85.8% on the 0.8B, +58.0% / +54.8% on the 2B; Phase 1's best 2-bit
  row was +655%, so it is much better than that, but not below +50%).
- Rule verdicts against the gptq32 curve: 0.8B avg 2.5 yes, avg 3 no (1 of 2 seeds, dPPL), avg 3.5 yes; 2B avg 2.5
  yes, avg 3 yes, avg 3.5 no (1 of 2 seeds, dPPL); MiniCPM avg 3 yes (1 seed). The 0.8B avg 3.5 "yes" holds on KL
  under the quadratic (0.683 / 0.702) but its dPPL half does not (0.974 / 1.012): read it as **KL-only**. Combo avg 2.5
  passes both margins under the quadratic on both Qwens (0.8B KL 0.778 / 0.809, dPPL 0.612 / 0.660; 2B 0.690 / 0.665,
  0.529 / 0.500).

#### Classification of every result

| result | class |
|---|---|
| Hadamard + GPTQ -29% to -36% KL at 3 bits on three models, both seeds; random orthogonal and signed Hadamard equal it | reproduced known (incoherence processing; Phase 2 had one seed) |
| Hadamard at 4 bits -25% to -35% KL (pre-registered < 10%) | new or unexpected (against our prior; not new to the field) |
| Hadamard at 2 bits -29% (0.8B) vs -47% (2B) | small refinement (model-dependent, two seeds each) |
| Eigenbasis rotation: 14.8 bits/param stored, and KL 1.63x worse than no rotation | reproduced known (cost); new or unexpected (quality loss before bits) |
| Huffman saves 0.44-0.49 bits at 2-4 bits, 0.015-0.016 bits from entropy0 | reproduced known (P1 entropy proxy 0.49-0.51); small refinement (3- and 2-bit savings above the pre-registered ranges) |
| Rotation costs 0.057 bits of coding gain | small refinement |
| Lloyd-Max: lower weight MSE, higher KL (+13% to +267%) at 3-4 bits; tail clipping under RMS normalization | new or unexpected |
| Lloyd at 2 bits wins fixed, loses coded | small refinement |
| Type-level allocation works on RTN (KL x 0.63) | reproduced known (Phase 2 estimate x 0.72-0.81, larger here) |
| Type-level allocation does not transfer to GPTQ: worse at avg 3 and 4 on both Qwens, mid-segment wins inside the chord bias | new or unexpected |
| Tensor-level allocation -19% / -20% KL at avg 3 on the 0.8B, but not on the 2B (Fisher signal) | small refinement (one model, one damage draw) |
| Shuffled allocations 1.8-3.7x worse than uniform | reproduced known (the allocation is not arbitrary) |
| C4 vs WikiText damage Spearman 0.96-1.00 | reproduced known |
| Fisher row split -14.7% on RTN, controls worse than uniform; -8% to -9% with GPTQ (not passing) | small refinement (own idea; gate passed, transfer to GPTQ not established) |
| Combination sub-additive at avg 2.5 / 3, never better than Hadamard alone | new or unexpected |
| RD gap grows with fewer bits; weight-MSE gap contradicted by KL in four families | new or unexpected (proxy flagged) |
| dPPL margin binds (11 rows), dPPL at 4 bits noise-dominated | new or unexpected (process; contradicts the pre-registration's own note) |
| MiniCPM type allocation at avg 3 is uniform | reproduced known (follows from its damages) |

#### Negative results

- Type-level allocation from RTN-4 damage does not beat uniform GPTQ at avg 3, 3.5 or 4 on either Qwen; on the 2B it
  is 16-20% worse at avg 3.
- Lloyd-Max per-tensor codebooks under RMS normalization lose 13-267% KL at 3-4 bits; the fixed Gaussian Lloyd-Max
  codebook (nf) loses more; the Gaussian-quantile codebook (nfq) loses 3.5-7x.
- The eigenbasis rotation loses 15 bits/param and is worse than no rotation even before bits.
- The combination does not reach GPTQ-4 at avg 3 (3-4x its KL), saves 0.48-0.67 bits not 1.0-1.3, and is worse than
  Hadamard alone at avg 2.5 / 3.
- Fisher row allocation with GPTQ: KL -8% to -9%, not passing the two-sided rule.
- The 2B tensor-level allocation from Fisher totals does not beat the curve.
- The weight-MSE RD gap does not predict KL (see E).

#### Audit findings and their resolution

| finding (severity) | resolution |
|---|---|
| dPPL margin binds, pre-registration calls it non-binding; 11 rows / 7 verdicts; run reports missed one (medium / high) | recorded in "Correction" above with a KL-only column; summary footnote, rows-table columns and plot footer fixed in `5c0601e1e3`; margin not re-tuned |
| H3-A refuted, H3-D partly refuted, H3-E mostly refuted, H3-combo refuted, H3-G supported (medium / low) | recorded per clause above, as the audit's recomputation (which matches summary.md to the printed digits) |
| Mid-segment "beats" rest on a 4-7% chord bias: alloc-type avg 2.5 (high); 0.8B combo avg 3.5 dPPL half (medium) | alloc-type avg 2.5 recorded "not established"; combo avg 3.5 (0.8B) recorded KL-only; future: measure a mid-segment reference point |
| Combo bits saved are Hadamard's (medium) | combo reported against the had curve as well; allocation components credited zero or less |
| 2B had-4 saved figures are extrapolated (low) | labelled extrapolated; verdict (read at a curve point) kept |
| Lloyd-2 wins only on fixed bits (low) | recorded "no net win"; codebook rows to be judged on both curves |
| Two seeds repeat only GPTQ calibration noise, not allocation-signal noise (low) | recorded in the noise model; a second damage draw is a next step before claiming the tensor-level gain is robust |
| Presentation code edited after the runs, code version moved to `5c0601e1e3`, unrecorded (medium) | recorded as a change point in Provenance (who, why, what changed, regression smoke, pytest) |
| 0.8B damage file mixes two code versions; ref file restamped (low) | recorded; bit-for-bit reproduction at the current code; restamp fix in `damage_c4.py` left open |
| Run-phase code `afb6534a49` never named in NOTES (low) | named in Provenance |
| Plot footers cut off, hollow markers ambiguous, labels placed before tight_layout, MiniCPM seed legend, clipped labels (medium / low) | fixed in `5c0601e1e3`; remaining 0.8B density and the MiniCPM avg-3 coincidence left (data, not encoding) |

Checked and found correct by the audits (not repeated in detail): verdicts recomputed without project imports match
summary.md on all three models; bits_blocks hand-counted for all 95 non-fp16, non-eig rows (difference exactly 0);
Huffman and entropy0 re-implemented (full precision match); no calibration leak (damage rows only seed-2 C4 with the
calibration documents removed, WikiText read only by `evaluate`); controls at equal bits; type allocations optimal;
eig cost re-derived; combo component decisions follow the rules.

#### Next directions

1. **Codebook on rotated weights, clipping-aware.** Lloyd-Max lowers the group-normalized MSE by 25% at 3 bits (0.0579
   vs 0.0777) but clips 2.8% of the weights carrying 22.7% of the squared error, and KL rises 13-19%; Hadamard removes
   exactly those tails (k on MiniCPM 1.88 -> 1.14 bits above the bound; max |w / rms_g| up to 10.4 before rotation). A
   pre-registered `had-lloyd` row (and an amax-normalized or outer-level-pinned Lloyd) at 3 bits is the cheapest test
   of whether a non-uniform grid adds anything on top of the one lever that worked (-30% to -36% KL).
2. **Allocation measured on GPTQ, at the widths it assigns, on top of Hadamard.** Type allocation from RTN-4 damage
   loses 4-20% KL at avg 3, while the same allocation wins 37% on RTN and the 0.8B tensor-level one wins 19-20%; the
   per-bit KL factor of GPTQ is 4.4 from 4 to 3 bits and 6.1-7.6 from 3 to 2, against the 4 the allocator assumes. A
   damage measured with had-GPTQ at 2 and 3 bits per type / tensor (plus a second C4 draw as the signal-noise control)
   would show whether any allocation survives on the had curve, where the combo currently sits at 1.25-1.31x.
3. **Fix the yardstick before the next method round.** Measure mid-segment reference points (uniform half-layer 2/3 and
   3/4-bit splits at 2.625 / 3.625 bits) and a 5-bit anchor, so mid-segment rows are read without the 4-7% chord bias
   and 4-bit rows are not extrapolated; and pre-register a dPPL rule that respects the measured 4-bit dPPL spread
   (113% on the 0.8B), since 11 rows were decided by a margin inside the noise.

### REPORT.md written (2026-11-08)

REPORT.md (Chinese) summarizes Phases 1-3; figures results/report_rd.png and results/report_rd_ppl.png from
scripts/plot_report.py (test tests/test_report_plot.py); report numbers recomputed by scripts/report_numbers.py. The
report went through two review rounds (numbers, literature classification, requirement coverage, reader) with each
finding judged independently. Subagents were not allowed to write the report file, so the reviewed text was written to
disk by the main session unchanged except for the reproduction section's commit line.
