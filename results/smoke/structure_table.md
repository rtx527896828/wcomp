# Phase 2 structure table

Generated 2026-11-08T09:26:51+00:00 by scripts/structure_table.py from `D:/repos/wcomp/results/smoke/probe/<model>/*.json` and `ablation.jsonl`.

Verdict rule (DESIGN.md Phase 2): a claim passes the shuffle test iff |gain_bits - gain_bits_control| exceeds both 0.02 bits/param and the spread between the two control seeds (where a second seed exists; otherwise marked "1 seed"). P1 is judged against its Gaussian control (the shuffle is vacuous for a marginal). P5: split-half Spearman > 0.9. P6: Spearman(s_t, measured @only KL) > 0.7. P7: dPPL > 100 % with the magnitude control < 2 %. `claimed` = gain_bits - gain_bits_control is the structure value in bits/param; `spread` = |control seed 0 - seed 1|. Values are numel-weighted means over the tensors of a type.


## qwen3.5-0.8b

### P0 additivity: sum of @only KL vs joint KL (rtn-sym-g128, same bits)

_no @only rows in ablation.jsonl_

### P6 prediction check and allocation saving

- Spearman(s_t per type, rtn-sym-g128-4@only KL): - over 0 types (contract > 0.7)
- Spearman(sum of s_t per layer, rtn-sym-g128-3@layer KL): - over 0 layers
- allocation saving, tensor granularity: 1.100 bits/param with 20.7% of the weights floored at 0 bits; b_u >= 2 bits: 0.590; unfloored units alone: 0.512
- allocation saving, group granularity: 1.269 bits/param with 22.6% of the weights floored at 0 bits; b_u >= 2 bits: 0.748; unfloored units alone: 0.656
- allocation saving, col granularity: 1.367 bits/param with 24.4% of the weights floored at 0 bits; b_u >= 2 bits: 0.833; unfloored units alone: 0.718
- tensor granularity restricted to within-type units: 0.199; on the relative output error s_t / ||W X||^2: 0.297 (floored 0.0%)
- s_t is in absolute output units: across types it tracks the input scale of the linear, so the model-wide saving is outside the D ~ 2^-2b model for the floored weights; the Spearman above is the test of the proxy

### Probes P1-P6 per weight type

| model | probe | type | key statistic | W | control | gain_bits | gain_bits_control | claimed | spread | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | marginal | gate | H_4 (index entropy, sym g128 4 bit) | 3.541 | 3.561 | +0.459 | +0.439 | +0.019 | - | fail (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | up | H_4 (index entropy, sym g128 4 bit) | 3.534 | 3.561 | +0.466 | +0.439 | +0.026 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | down | H_4 (index entropy, sym g128 4 bit) | 3.528 | 3.561 | +0.472 | +0.439 | +0.033 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | in_proj_qkv | H_4 (index entropy, sym g128 4 bit) | 3.517 | 3.560 | +0.483 | +0.440 | +0.043 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | in_proj_z | H_4 (index entropy, sym g128 4 bit) | 3.524 | 3.562 | +0.476 | +0.438 | +0.038 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | in_proj_ab | H_4 (index entropy, sym g128 4 bit) | 3.489 | 3.550 | +0.495 | +0.434 | +0.061 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | out_proj | H_4 (index entropy, sym g128 4 bit) | 3.484 | 3.562 | +0.516 | +0.438 | +0.078 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | embed | H_4 (index entropy, sym g128 4 bit) | 3.497 | 3.561 | +0.503 | +0.439 | +0.064 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | lowrank | gate | energy captured at r = 0.1 min(N, K) | 0.267 | 0.202 | -0.156 | -0.185 | +0.028 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.156); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | up | energy captured at r = 0.1 min(N, K) | 0.208 | 0.202 | -0.182 | -0.185 | +0.002 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.182); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | down | energy captured at r = 0.1 min(N, K) | 0.218 | 0.201 | -0.173 | -0.185 | +0.012 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.173); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | in_proj_qkv | energy captured at r = 0.1 min(N, K) | 0.398 | 0.175 | -0.035 | -0.169 | +0.133 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.035); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | in_proj_z | energy captured at r = 0.1 min(N, K) | 0.341 | 0.241 | -0.157 | -0.214 | +0.057 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.157); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | in_proj_ab | energy captured at r = 0.1 min(N, K) | 0.608 | 0.154 | -0.660 | -0.957 | +0.297 | 0.001 | pass | best aggregated r = 0.05 min(N, K) (gain there -0.660); gain_bits = mean of per-record maxima, per-record best r in {0.062} |
| qwen3.5-0.8b | lowrank | out_proj | energy captured at r = 0.1 min(N, K) | 0.268 | 0.241 | -0.180 | -0.214 | +0.035 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.180); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | embed | energy captured at r = 0.1 min(N, K) | 0.360 | 0.110 | +0.039 | -0.149 | +0.188 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there +0.039); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank_weighted | q | energy captured at r = 0.1 (W @ L) | 0.974 | 0.725 | +0.059 | -0.164 | +0.223 | 0.000 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.770 vs control +0.234: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there +0.059); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | k | energy captured at r = 0.1 (W @ L) | 0.794 | 0.673 | -0.170 | -0.200 | +0.030 | 0.006 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.239 vs control +0.127: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.170); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | v | energy captured at r = 0.1 (W @ L) | 0.750 | 0.670 | -0.179 | -0.202 | +0.024 | 0.014 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.407 vs control +0.116: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.179); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | o | energy captured at r = 0.1 (W @ L) | 0.817 | 0.902 | -0.215 | -0.214 | -0.002 | 0.012 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.325 vs control +0.668: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.215); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | gate | energy captured at r = 0.1 (W @ L) | 0.914 | 0.712 | -0.089 | -0.169 | +0.080 | 0.004 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.042 vs control +0.220: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.089); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | up | energy captured at r = 0.1 (W @ L) | 0.721 | 0.712 | -0.167 | -0.174 | +0.007 | 0.003 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.241 vs control +0.219: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.167); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | down | energy captured at r = 0.1 (W @ L) | 0.611 | 0.588 | -0.153 | -0.181 | +0.028 | 0.003 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.101 vs control +0.035: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.153); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | in_proj_qkv | energy captured at r = 0.1 (W @ L) | 0.928 | 0.717 | -0.061 | -0.151 | +0.089 | 0.001 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.120 vs control +0.245: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.061); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | in_proj_z | energy captured at r = 0.1 (W @ L) | 0.953 | 0.731 | -0.037 | -0.197 | +0.159 | 0.003 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.406 vs control +0.195: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.037); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | in_proj_ab | energy captured at r = 0.1 (W @ L) | 0.906 | 0.453 | -0.555 | -0.938 | +0.383 | 0.027 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.181 vs control -0.707: a statistic, not bits/param); best aggregated r = 0.05 min(N, K) (gain there -0.555); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | out_proj | energy captured at r = 0.1 (W @ L) | 0.758 | 0.825 | -0.199 | -0.198 | -0.001 | 0.005 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.245 vs control +0.400: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.199); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | outliers | gate | column-norm max / median | 1.35 | 1.04 | -0.051 | -0.054 | +0.003 | 0.000 | fail | top-16 channel Jaccard across layers 0.04 vs control 0.03 (all hidden-input linears) |
| qwen3.5-0.8b | outliers | up | column-norm max / median | 1.07 | 1.04 | -0.054 | -0.054 | +0.000 | 0.000 | fail | top-16 channel Jaccard across layers 0.04 vs control 0.03 (all hidden-input linears) |
| qwen3.5-0.8b | outliers | down | column-norm max / median | 1.10 | 1.41 | -0.055 | -0.044 | -0.011 | 0.001 | fail | top-16 channel Jaccard across layers 0.04 vs control 0.03 (all hidden-input linears) |
| qwen3.5-0.8b | outliers | in_proj_qkv | column-norm max / median | 1.65 | 1.06 | -0.041 | -0.052 | +0.011 | 0.000 | fail | top-16 channel Jaccard across layers 0.03 vs control 0.03 |
| qwen3.5-0.8b | outliers | in_proj_z | column-norm max / median | 1.55 | 1.11 | -0.044 | -0.052 | +0.008 | 0.000 | fail | top-16 channel Jaccard across layers 0.07 vs control 0.03 |
| qwen3.5-0.8b | outliers | in_proj_ab | column-norm max / median | 4.11 | 2.38 | -0.003 | -0.061 | +0.058 | 0.005 | pass | top-16 channel Jaccard across layers 0.03 vs control 0.03 |
| qwen3.5-0.8b | outliers | out_proj | column-norm max / median | 1.44 | 1.24 | -0.053 | -0.045 | -0.008 | 0.000 | fail | top-16 channel Jaccard across layers 0.04 vs control 0.03 (all hidden-input linears) |
| qwen3.5-0.8b | interlayer | in_proj_qkv | min residual energy over alignments, adjacent pairs | 0.426 | 0.083 | -0.497 | -0.497 | +0.000 | - | fail (1 seed) | control 0.083; gain at the best alignment |
| qwen3.5-0.8b | interlayer | in_proj_z | min residual energy over alignments, adjacent pairs | 0.361 | 0.236 | -0.431 | -0.430 | -0.001 | - | fail (1 seed) | control 0.236; gain at the best alignment |
| qwen3.5-0.8b | interlayer | in_proj_ab | min residual energy over alignments, adjacent pairs | 0.467 | 0.043 | -0.540 | -0.568 | +0.028 | - | pass (1 seed) | control 0.043; gain at the best alignment |
| qwen3.5-0.8b | acts | o | Gini of per-channel input energy | 0.749 | 0.865 | - | - | - | - | noise | control = split-half Spearman 0.865; channels for 50 % / 90 % of the energy: 0.042 / 0.390 |
| qwen3.5-0.8b | acts | down | Gini of per-channel input energy | 0.463 | 0.735 | - | - | - | - | noise | control = split-half Spearman 0.735; channels for 50 % / 90 % of the energy: 0.175 / 0.700 |
| qwen3.5-0.8b | acts | out_proj | Gini of per-channel input energy | 0.770 | 0.862 | - | - | - | - | noise | control = split-half Spearman 0.862; channels for 50 % / 90 % of the energy: 0.028 / 0.375 |
| qwen3.5-0.8b | sensitivity | q | s_t / n_t | 5.47e-06 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | k | s_t / n_t | 2.97e-06 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | v | s_t / n_t | 5.85e-06 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | o | s_t / n_t | 8.32e-08 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | gate | s_t / n_t | 3.8e-06 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | up | s_t / n_t | 1.15e-06 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | down | s_t / n_t | 1.27e-08 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | in_proj_qkv | s_t / n_t | 1.01e-05 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | in_proj_z | s_t / n_t | 7.61e-06 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | in_proj_ab | s_t / n_t | 2.24e-05 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | sensitivity | out_proj | s_t / n_t | 1.58e-08 | - | - | - | - | - | n/a (no ablation rows) | @only KL 4 bit - |
| qwen3.5-0.8b | fisher | q | rank-1 fit energy of F | 0.977 | 0.181 | - | - | - | - | rank-1 fits (> 0.7); control 0.181 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.807 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.966, Gini 0.879 |
| qwen3.5-0.8b | fisher | k | rank-1 fit energy of F | 0.977 | 0.178 | - | - | - | - | rank-1 fits (> 0.7); control 0.178 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.740 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.820, Gini 0.754 |
| qwen3.5-0.8b | fisher | v | rank-1 fit energy of F | 0.901 | 0.148 | - | - | - | - | rank-1 fits (> 0.7); control 0.148 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.686 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.963, Gini 0.848 |
| qwen3.5-0.8b | fisher | o | rank-1 fit energy of F | 0.977 | 0.074 | - | - | - | - | rank-1 fits (> 0.7); control 0.074 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.838 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.902, Gini 0.893 |
| qwen3.5-0.8b | fisher | gate | rank-1 fit energy of F | 0.876 | 0.120 | - | - | - | - | rank-1 fits (> 0.7); control 0.120 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.697 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.920, Gini 0.789 |
| qwen3.5-0.8b | fisher | up | rank-1 fit energy of F | 0.910 | 0.128 | - | - | - | - | rank-1 fits (> 0.7); control 0.128 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.632 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.931, Gini 0.759 |
| qwen3.5-0.8b | fisher | down | rank-1 fit energy of F | 0.848 | 0.123 | - | - | - | - | rank-1 fits (> 0.7); control 0.123 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.558 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.776, Gini 0.680 |
| qwen3.5-0.8b | fisher | in_proj_qkv | rank-1 fit energy of F | 0.887 | 0.081 | - | - | - | - | rank-1 fits (> 0.7); control 0.081 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.829 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.917, Gini 0.901 |
| qwen3.5-0.8b | fisher | in_proj_z | rank-1 fit energy of F | 0.937 | 0.209 | - | - | - | - | rank-1 fits (> 0.7); control 0.209 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.826 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.926, Gini 0.887 |
| qwen3.5-0.8b | fisher | in_proj_ab | rank-1 fit energy of F | 0.958 | 0.317 | - | - | - | - | rank-1 fits (> 0.7); control 0.317 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.740 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.917, Gini 0.816 |
| qwen3.5-0.8b | fisher | out_proj | rank-1 fit energy of F | 0.960 | 0.038 | - | - | - | - | rank-1 fits (> 0.7); control 0.038 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.816 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.890, Gini 0.877 |

### P7 super-weight candidates

| model | tensor | layer | type | [row, col] | value | dPPL % | KL | dPPL % magnitude control | dPPL % random | candidate | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | model.layers.23.mlp.down_proj | 23 | down | [0, 254] | -0.0181 | 0.0 | 0.0000 | 0.07 | 0.00 | max_spike_below_threshold | not a super weight |
| qwen3.5-0.8b | model.layers.20.linear_attn.in_proj_qkv | 20 | in_proj_qkv | [1682, 74] | 0.582 | 0.0 | 0.0004 | -0.00 | -0.03 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.self_attn.v_proj | 23 | v | [184, 123] | 0.539 | -0.1 | 0.0013 | 0.08 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [1608, 531] | -0.512 | -0.0 | 0.0001 | 0.01 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.21.linear_attn.in_proj_qkv | 21 | in_proj_qkv | [1969, 66] | 0.496 | -0.0 | 0.0003 | 0.02 | -0.02 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.20.linear_attn.in_proj_qkv | 20 | in_proj_qkv | [4883, 24] | 0.475 | -0.0 | 0.0004 | 0.04 | -0.03 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.19.mlp.gate_proj | 19 | gate | [1045, 807] | -0.471 | 0.0 | 0.0004 | 0.01 | -0.02 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.0.linear_attn.in_proj_z | 0 | in_proj_z | [542, 111] | -0.461 | -0.0 | 0.0007 | 0.02 | 0.12 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [252, 460] | 0.441 | 0.0 | 0.0001 | -0.01 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.14.linear_attn.in_proj_qkv | 14 | in_proj_qkv | [2347, 16] | 0.441 | 0.0 | 0.0005 | 0.00 | 0.03 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [1608, 807] | -0.438 | -0.0 | 0.0001 | -0.01 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.20.linear_attn.in_proj_qkv | 20 | in_proj_qkv | [1682, 166] | 0.436 | -0.0 | 0.0004 | 0.04 | -0.03 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [252, 679] | 0.41 | -0.0 | 0.0001 | -0.01 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.20.linear_attn.in_proj_qkv | 20 | in_proj_qkv | [3811, 74] | -0.41 | 0.0 | 0.0004 | 0.04 | -0.03 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.22.mlp.gate_proj | 22 | gate | [1334, 425] | 0.408 | -0.0 | 0.0002 | 0.04 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.22.linear_attn.in_proj_qkv | 22 | in_proj_qkv | [4598, 807] | 0.406 | -0.0 | 0.0002 | -0.02 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [757, 523] | 0.402 | 0.0 | 0.0001 | -0.01 | -0.01 | top16 | not a super weight |


## Skipped

- qwen3.5-0.8b: no ablation rows
