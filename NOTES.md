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
