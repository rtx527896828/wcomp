# Phase 2 structure table

Generated 2026-11-08T12:35:30+00:00 by scripts/structure_table.py from `D:/repos/wcomp/results/probe/<model>/*.json` and `ablation.jsonl`.

Verdict rule (DESIGN.md Phase 2): a claim passes the shuffle test iff |gain_bits - gain_bits_control| exceeds both 0.02 bits/param and the spread between the two control seeds (where a second seed exists; otherwise marked "1 seed"). P1 is judged against its Gaussian control (the shuffle is vacuous for a marginal). P5: split-half Spearman > 0.9. P6: Spearman(s_t, measured @only KL) > 0.7. P7: dPPL > 100 % with the magnitude control < 2 %. `claimed` = gain_bits - gain_bits_control is the structure value in bits/param; `spread` = |control seed 0 - seed 1|. Values are numel-weighted means over the tensors of a type.


## qwen3.5-0.8b

### P0 additivity: sum of @only KL vs joint KL (rtn-sym-g128, same bits)

| bits | types | sum @only KL | joint KL | sum / joint | three largest @only KL | top-3 share of joint | H-P0 additivity |
|---|---|---|---|---|---|---|---|
| 4 | 11 | 0.1957 | 0.2051 | 0.95 | out_proj 0.0474, down 0.0320, up 0.0291 | 53% | additive (within 50 %) |
| 3 | 11 | 0.7646 | 1.5053 | 0.51 | out_proj 0.1878, down 0.1200, up 0.1120 | 28% | additive (within 50 %) (sum < joint: super-additive) |

### P0 pre-registered figures: H-P3 (+keepcols recovery) and H-P0 group size

- +keepcols=0.5 at 4 bit recovers 1% of the RTN KL (0.2051 -> 0.2025) at +0.059 bits/param: H-P3 not met
- +keepcols=1 at 4 bit recovers 3% of the RTN KL (0.2051 -> 0.1986) at +0.118 bits/param: H-P3 not met
- +keepcols=2 at 4 bit recovers 8% of the RTN KL (0.2051 -> 0.1882) at +0.237 bits/param: H-P3 not met
- KL(g128) / KL(g32) at 4 bit: 1.39 (0.2051 / 0.1474, +0.375 bits/param): H-P0 group size not met (group size is not the lever)
- +keepcols=0.5 at 3 bit recovers 12% of the RTN KL (1.5053 -> 1.3254) at +0.064 bits/param: H-P3 not met
- +keepcols=1 at 3 bit recovers 12% of the RTN KL (1.5053 -> 1.3207) at +0.128 bits/param: H-P3 not met
- +keepcols=2 at 3 bit recovers 17% of the RTN KL (1.5053 -> 1.2449) at +0.256 bits/param: H-P3 not met
- KL(g128) / KL(g32) at 3 bit: 2.30 (1.5053 / 0.6543, +0.375 bits/param): H-P0 group size met (> 2)

### P6 prediction check and allocation saving

- Spearman(s_t per type, rtn-sym-g128-4@only KL): -0.327 over 11 types (contract > 0.7)
- Spearman(sum of s_t per layer, rtn-sym-g128-3@layer KL): -0.282 over 24 layers
- allocation saving, tensor granularity: 1.102 bits/param with 20.7% of the weights floored at 0 bits; b_u >= 2 bits: 0.590; unfloored units alone: 0.514
- allocation saving, group granularity: 1.272 bits/param with 22.8% of the weights floored at 0 bits; b_u >= 2 bits: 0.750; unfloored units alone: 0.654
- allocation saving, col granularity: 1.368 bits/param with 24.4% of the weights floored at 0 bits; b_u >= 2 bits: 0.834; unfloored units alone: 0.720
- tensor granularity restricted to within-type units: 0.192; on the relative output error s_t / ||W X||^2: 0.300 (floored 0.0%)
- s_t is in absolute output units: across types it tracks the input scale of the linear, so the model-wide saving is outside the D ~ 2^-2b model for the floored weights; the Spearman above is the test of the proxy

### Probes P1-P6 per weight type

| model | probe | type | key statistic | W | control | gain_bits | gain_bits_control | claimed | spread | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | marginal | q | H_4 (index entropy, sym g128 4 bit) | 3.491 | 3.561 | +0.509 | +0.439 | +0.069 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | k | H_4 (index entropy, sym g128 4 bit) | 3.460 | 3.560 | +0.540 | +0.440 | +0.100 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | v | H_4 (index entropy, sym g128 4 bit) | 3.395 | 3.560 | +0.605 | +0.440 | +0.165 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | o | H_4 (index entropy, sym g128 4 bit) | 3.398 | 3.562 | +0.601 | +0.438 | +0.163 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | gate | H_4 (index entropy, sym g128 4 bit) | 3.520 | 3.561 | +0.480 | +0.439 | +0.040 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | up | H_4 (index entropy, sym g128 4 bit) | 3.514 | 3.561 | +0.486 | +0.439 | +0.047 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | down | H_4 (index entropy, sym g128 4 bit) | 3.491 | 3.561 | +0.509 | +0.439 | +0.070 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | in_proj_qkv | H_4 (index entropy, sym g128 4 bit) | 3.512 | 3.561 | +0.488 | +0.439 | +0.049 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | in_proj_z | H_4 (index entropy, sym g128 4 bit) | 3.519 | 3.562 | +0.481 | +0.438 | +0.043 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | in_proj_ab | H_4 (index entropy, sym g128 4 bit) | 3.474 | 3.550 | +0.510 | +0.434 | +0.076 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | out_proj | H_4 (index entropy, sym g128 4 bit) | 3.443 | 3.562 | +0.556 | +0.438 | +0.118 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | marginal | embed | H_4 (index entropy, sym g128 4 bit) | 3.497 | 3.561 | +0.503 | +0.439 | +0.064 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-0.8b | lowrank | q | energy captured at r = 0.1 min(N, K) | 0.466 | 0.194 | -0.031 | -0.180 | +0.148 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.031); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | k | energy captured at r = 0.1 min(N, K) | 0.466 | 0.241 | -0.168 | -0.214 | +0.046 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.168); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | v | energy captured at r = 0.1 min(N, K) | 0.335 | 0.242 | -0.181 | -0.214 | +0.033 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.181); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | o | energy captured at r = 0.1 min(N, K) | 0.395 | 0.241 | -0.175 | -0.214 | +0.039 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.175); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | gate | energy captured at r = 0.1 min(N, K) | 0.317 | 0.202 | -0.141 | -0.185 | +0.044 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.141); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | up | energy captured at r = 0.1 min(N, K) | 0.255 | 0.202 | -0.170 | -0.185 | +0.015 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.170); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | down | energy captured at r = 0.1 min(N, K) | 0.268 | 0.202 | -0.169 | -0.185 | +0.015 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.169); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | in_proj_qkv | energy captured at r = 0.1 min(N, K) | 0.349 | 0.175 | -0.101 | -0.169 | +0.068 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.101); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | in_proj_z | energy captured at r = 0.1 min(N, K) | 0.426 | 0.241 | -0.126 | -0.214 | +0.089 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.126); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | in_proj_ab | energy captured at r = 0.1 min(N, K) | 0.563 | 0.154 | -0.657 | -0.956 | +0.299 | 0.001 | pass | best aggregated r = 0.05 min(N, K) (gain there -0.657); gain_bits = mean of per-record maxima, per-record best r in {0.062} |
| qwen3.5-0.8b | lowrank | out_proj | energy captured at r = 0.1 min(N, K) | 0.350 | 0.241 | -0.177 | -0.214 | +0.038 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.177); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank | embed | energy captured at r = 0.1 min(N, K) | 0.360 | 0.110 | +0.039 | -0.149 | +0.188 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there +0.039); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-0.8b | lowrank_weighted | q | energy captured at r = 0.1 (W @ L) | 0.970 | 0.676 | +0.067 | -0.161 | +0.228 | 0.001 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.744 vs control +0.190: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there +0.067); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | k | energy captured at r = 0.1 (W @ L) | 0.772 | 0.632 | -0.156 | -0.211 | +0.055 | 0.012 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.205 vs control +0.101: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.156); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | v | energy captured at r = 0.1 (W @ L) | 0.725 | 0.629 | -0.179 | -0.210 | +0.030 | 0.003 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.394 vs control +0.093: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.179); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | o | energy captured at r = 0.1 (W @ L) | 0.758 | 0.868 | -0.216 | -0.203 | -0.013 | 0.001 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.232 vs control +0.548: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.216); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | gate | energy captured at r = 0.1 (W @ L) | 0.902 | 0.661 | -0.080 | -0.167 | +0.087 | 0.003 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.014 vs control +0.171: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.080); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | up | energy captured at r = 0.1 (W @ L) | 0.674 | 0.661 | -0.166 | -0.171 | +0.006 | 0.006 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.194 vs control +0.170: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.166); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | down | energy captured at r = 0.1 (W @ L) | 0.541 | 0.520 | -0.152 | -0.182 | +0.030 | 0.003 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.056 vs control -0.003: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.152); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | in_proj_qkv | energy captured at r = 0.1 (W @ L) | 0.918 | 0.670 | -0.057 | -0.148 | +0.091 | 0.001 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.101 vs control +0.206: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.057); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | in_proj_z | energy captured at r = 0.1 (W @ L) | 0.947 | 0.687 | -0.029 | -0.195 | +0.166 | 0.003 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.383 vs control +0.156: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.029); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | in_proj_ab | energy captured at r = 0.1 (W @ L) | 0.905 | 0.435 | -0.591 | -0.950 | +0.359 | 0.024 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.205 vs control -0.721: a statistic, not bits/param); best aggregated r = 0.05 min(N, K) (gain there -0.591); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | lowrank_weighted | out_proj | energy captured at r = 0.1 (W @ L) | 0.698 | 0.786 | -0.198 | -0.192 | -0.006 | 0.009 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.166 vs control +0.346: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.198); gain_bits = mean of per-record maxima |
| qwen3.5-0.8b | outliers | q | column-norm max / median | 2.01 | 1.06 | -0.034 | -0.053 | +0.019 | 0.000 | fail | top-16 channel Jaccard across layers 0.130 vs control 0.004 |
| qwen3.5-0.8b | outliers | k | column-norm max / median | 2.22 | 1.18 | -0.033 | -0.051 | +0.018 | 0.000 | fail | top-16 channel Jaccard across layers 0.185 vs control 0.004 |
| qwen3.5-0.8b | outliers | v | column-norm max / median | 1.74 | 1.30 | -0.044 | -0.048 | +0.004 | 0.001 | fail | top-16 channel Jaccard across layers 0.083 vs control 0.004 |
| qwen3.5-0.8b | outliers | o | column-norm max / median | 2.50 | 1.16 | -0.032 | -0.052 | +0.020 | 0.000 | pass | top-16 channel Jaccard across layers 0.070 vs control 0.008 (all hidden-input linears) |
| qwen3.5-0.8b | outliers | gate | column-norm max / median | 1.42 | 1.07 | -0.049 | -0.053 | +0.004 | 0.000 | fail | top-16 channel Jaccard across layers 0.193 vs control 0.008 |
| qwen3.5-0.8b | outliers | up | column-norm max / median | 1.20 | 1.04 | -0.052 | -0.054 | +0.002 | 0.000 | fail | top-16 channel Jaccard across layers 0.077 vs control 0.008 |
| qwen3.5-0.8b | outliers | down | column-norm max / median | 1.52 | 1.27 | -0.048 | -0.049 | +0.001 | 0.000 | fail | top-16 channel Jaccard across layers 0.070 vs control 0.008 (all hidden-input linears) |
| qwen3.5-0.8b | outliers | in_proj_qkv | column-norm max / median | 1.36 | 1.05 | -0.048 | -0.053 | +0.005 | 0.000 | fail | top-16 channel Jaccard across layers 0.133 vs control 0.008 |
| qwen3.5-0.8b | outliers | in_proj_z | column-norm max / median | 1.68 | 1.09 | -0.041 | -0.053 | +0.012 | 0.000 | fail | top-16 channel Jaccard across layers 0.324 vs control 0.008 |
| qwen3.5-0.8b | outliers | in_proj_ab | column-norm max / median | 4.60 | 2.43 | +0.005 | -0.053 | +0.058 | 0.001 | pass | top-16 channel Jaccard across layers 0.294 vs control 0.008 |
| qwen3.5-0.8b | outliers | out_proj | column-norm max / median | 1.82 | 1.23 | -0.042 | -0.047 | +0.005 | 0.001 | fail | top-16 channel Jaccard across layers 0.070 vs control 0.008 (all hidden-input linears) |
| qwen3.5-0.8b | interlayer | q | min residual energy over alignments, adjacent pairs | 0.464 | 0.152 | -0.502 | -0.533 | +0.031 | - | pass (1 seed) | control 0.152; gain at the best alignment |
| qwen3.5-0.8b | interlayer | k | min residual energy over alignments, adjacent pairs | 0.647 | 0.273 | -0.534 | -0.532 | -0.001 | - | fail (1 seed) | control 0.273; gain at the best alignment |
| qwen3.5-0.8b | interlayer | v | min residual energy over alignments, adjacent pairs | 0.484 | 0.254 | -0.453 | -0.453 | +0.001 | - | fail (1 seed) | control 0.254; gain at the best alignment |
| qwen3.5-0.8b | interlayer | o | min residual energy over alignments, adjacent pairs | 0.440 | 0.268 | -0.489 | -0.489 | -0.000 | - | fail (1 seed) | control 0.268; gain at the best alignment |
| qwen3.5-0.8b | interlayer | gate | min residual energy over alignments, adjacent pairs | 0.202 | 0.146 | -0.335 | -0.416 | +0.080 | - | pass (1 seed) | control 0.146; gain at the best alignment |
| qwen3.5-0.8b | interlayer | up | min residual energy over alignments, adjacent pairs | 0.142 | 0.142 | -0.394 | -0.420 | +0.026 | - | pass (1 seed) | control 0.142; gain at the best alignment |
| qwen3.5-0.8b | interlayer | down | min residual energy over alignments, adjacent pairs | 0.146 | 0.144 | -0.376 | -0.428 | +0.051 | - | pass (1 seed) | control 0.144; gain at the best alignment |
| qwen3.5-0.8b | interlayer | in_proj_qkv | min residual energy over alignments, adjacent pairs | 0.194 | 0.088 | -0.494 | -0.494 | -0.000 | - | fail (1 seed) | control 0.088; gain at the best alignment |
| qwen3.5-0.8b | interlayer | in_proj_z | min residual energy over alignments, adjacent pairs | 0.377 | 0.253 | -0.494 | -0.500 | +0.006 | - | fail (1 seed) | control 0.253; gain at the best alignment |
| qwen3.5-0.8b | interlayer | in_proj_ab | min residual energy over alignments, adjacent pairs | 0.332 | 0.023 | -0.109 | -0.509 | +0.400 | - | pass (1 seed) | control 0.023; gain at the best alignment |
| qwen3.5-0.8b | interlayer | out_proj | min residual energy over alignments, adjacent pairs | 0.299 | 0.253 | -0.501 | -0.501 | +0.000 | - | fail (1 seed) | control 0.253; gain at the best alignment |
| qwen3.5-0.8b | acts | o | Gini of per-channel input energy | 0.743 | 0.967 | - | - | - | - | stable | control = split-half Spearman 0.967; channels for 50 % / 90 % of the energy: 0.043 / 0.401 |
| qwen3.5-0.8b | acts | down | Gini of per-channel input energy | 0.436 | 0.928 | - | - | - | - | stable | control = split-half Spearman 0.928; channels for 50 % / 90 % of the energy: 0.194 / 0.716 |
| qwen3.5-0.8b | acts | out_proj | Gini of per-channel input energy | 0.756 | 0.966 | - | - | - | - | stable | control = split-half Spearman 0.966; channels for 50 % / 90 % of the energy: 0.031 / 0.397 |
| qwen3.5-0.8b | sensitivity | q | s_t / n_t | 5.5e-06 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0064 |
| qwen3.5-0.8b | sensitivity | k | s_t / n_t | 2.97e-06 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0031 |
| qwen3.5-0.8b | sensitivity | v | s_t / n_t | 5.87e-06 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0102 |
| qwen3.5-0.8b | sensitivity | o | s_t / n_t | 8.12e-08 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0252 |
| qwen3.5-0.8b | sensitivity | gate | s_t / n_t | 3.88e-06 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0143 |
| qwen3.5-0.8b | sensitivity | up | s_t / n_t | 1.17e-06 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0291 |
| qwen3.5-0.8b | sensitivity | down | s_t / n_t | 1.22e-08 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0320 |
| qwen3.5-0.8b | sensitivity | in_proj_qkv | s_t / n_t | 1.02e-05 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0173 |
| qwen3.5-0.8b | sensitivity | in_proj_z | s_t / n_t | 7.71e-06 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0089 |
| qwen3.5-0.8b | sensitivity | in_proj_ab | s_t / n_t | 2.24e-05 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0019 |
| qwen3.5-0.8b | sensitivity | out_proj | s_t / n_t | 1.54e-08 | - | - | - | - | - | no prediction (Spearman -0.33) | @only KL 4 bit 0.0474 |
| qwen3.5-0.8b | fisher | q | rank-1 fit energy of F | 0.988 | 0.133 | - | - | - | - | rank-1 fits (> 0.7); control 0.133 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.774 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.977, Gini 0.841 |
| qwen3.5-0.8b | fisher | k | rank-1 fit energy of F | 0.968 | 0.190 | - | - | - | - | rank-1 fits (> 0.7); control 0.190 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.709 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.843, Gini 0.707 |
| qwen3.5-0.8b | fisher | v | rank-1 fit energy of F | 0.943 | 0.148 | - | - | - | - | rank-1 fits (> 0.7); control 0.148 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.649 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.972, Gini 0.834 |
| qwen3.5-0.8b | fisher | o | rank-1 fit energy of F | 0.985 | 0.034 | - | - | - | - | rank-1 fits (> 0.7); control 0.034 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.795 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.985, Gini 0.857 |
| qwen3.5-0.8b | fisher | gate | rank-1 fit energy of F | 0.930 | 0.141 | - | - | - | - | rank-1 fits (> 0.7); control 0.141 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.639 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.942, Gini 0.726 |
| qwen3.5-0.8b | fisher | up | rank-1 fit energy of F | 0.960 | 0.148 | - | - | - | - | rank-1 fits (> 0.7); control 0.148 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.592 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.961, Gini 0.710 |
| qwen3.5-0.8b | fisher | down | rank-1 fit energy of F | 0.933 | 0.186 | - | - | - | - | rank-1 fits (> 0.7); control 0.186 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.486 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.882, Gini 0.569 |
| qwen3.5-0.8b | fisher | in_proj_qkv | rank-1 fit energy of F | 0.898 | 0.121 | - | - | - | - | rank-1 fits (> 0.7); control 0.121 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.804 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.934, Gini 0.889 |
| qwen3.5-0.8b | fisher | in_proj_z | rank-1 fit energy of F | 0.956 | 0.186 | - | - | - | - | rank-1 fits (> 0.7); control 0.186 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.801 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.939, Gini 0.869 |
| qwen3.5-0.8b | fisher | in_proj_ab | rank-1 fit energy of F | 0.943 | 0.312 | - | - | - | - | rank-1 fits (> 0.7); control 0.312 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.707 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.933, Gini 0.786 |
| qwen3.5-0.8b | fisher | out_proj | rank-1 fit energy of F | 0.990 | 0.032 | - | - | - | - | rank-1 fits (> 0.7); control 0.032 | position-wise F (d_t of the summed window NLL; not the per-token Fisher); Gini 0.781 vs control -; batched (per-window gradient squared, not used for the verdict): rank-1 0.963, Gini 0.846 |

### P7 super-weight candidates

| model | tensor | layer | type | [row, col] | value | dPPL % | KL | dPPL % magnitude control | dPPL % random | candidate | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | model.layers.23.mlp.down_proj | 23 | down | [0, 254] | -0.0181 | -0.00 | 0.0000 | 0.12 | -0.00 | max_spike_below_threshold | not a super weight |
| qwen3.5-0.8b | model.layers.20.linear_attn.in_proj_qkv | 20 | in_proj_qkv | [1682, 74] | 0.582 | 0.00 | 0.0004 | 0.01 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.self_attn.v_proj | 23 | v | [184, 123] | 0.539 | -0.19 | 0.0012 | 0.03 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [1608, 531] | -0.512 | 0.01 | 0.0001 | 0.00 | -0.00 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.21.linear_attn.in_proj_qkv | 21 | in_proj_qkv | [1969, 66] | 0.496 | -0.02 | 0.0003 | 0.01 | 0.00 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.20.linear_attn.in_proj_qkv | 20 | in_proj_qkv | [4883, 24] | 0.475 | 0.01 | 0.0004 | 0.00 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.19.mlp.gate_proj | 19 | gate | [1045, 807] | -0.471 | 0.02 | 0.0004 | 0.00 | 0.00 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.0.linear_attn.in_proj_z | 0 | in_proj_z | [542, 111] | -0.461 | 0.01 | 0.0007 | 0.02 | 0.04 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [252, 460] | 0.441 | 0.00 | 0.0001 | 0.01 | -0.00 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.14.linear_attn.in_proj_qkv | 14 | in_proj_qkv | [2347, 16] | 0.441 | 0.04 | 0.0005 | 0.02 | 0.03 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [1608, 807] | -0.438 | 0.00 | 0.0001 | 0.01 | -0.00 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.20.linear_attn.in_proj_qkv | 20 | in_proj_qkv | [1682, 166] | 0.436 | -0.01 | 0.0004 | 0.00 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [252, 679] | 0.41 | -0.01 | 0.0001 | 0.01 | -0.00 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.20.linear_attn.in_proj_qkv | 20 | in_proj_qkv | [3811, 74] | -0.41 | 0.03 | 0.0004 | 0.00 | -0.01 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.22.mlp.gate_proj | 22 | gate | [1334, 425] | 0.408 | -0.01 | 0.0002 | 0.01 | 0.00 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.22.linear_attn.in_proj_qkv | 22 | in_proj_qkv | [4598, 807] | 0.406 | 0.00 | 0.0002 | -0.02 | 0.00 | top16 | not a super weight |
| qwen3.5-0.8b | model.layers.23.mlp.gate_proj | 23 | gate | [757, 523] | 0.402 | 0.01 | 0.0001 | 0.01 | -0.00 | top16 | not a super weight |


## qwen3.5-2b

### P0 additivity: sum of @only KL vs joint KL (rtn-sym-g128, same bits)

| bits | types | sum @only KL | joint KL | sum / joint | three largest @only KL | top-3 share of joint | H-P0 additivity |
|---|---|---|---|---|---|---|---|
| 4 | 11 | 0.1384 | 0.1331 | 1.04 | out_proj 0.0272, down 0.0234, up 0.0229 | 55% | additive (within 50 %) |

### P0 pre-registered figures: H-P3 (+keepcols recovery) and H-P0 group size

- +keepcols=1 at 4 bit recovers 8% of the RTN KL (0.1331 -> 0.1226) at +0.118 bits/param: H-P3 not met

### P6 prediction check and allocation saving

- Spearman(s_t per type, rtn-sym-g128-4@only KL): -0.045 over 11 types (contract > 0.7)
- Spearman(sum of s_t per layer, rtn-sym-g128-3@layer KL): -0.512 over 24 layers
- allocation saving, tensor granularity: 0.988 bits/param with 13.1% of the weights floored at 0 bits; b_u >= 2 bits: 0.587; unfloored units alone: 0.634
- allocation saving, group granularity: 1.060 bits/param with 14.1% of the weights floored at 0 bits; b_u >= 2 bits: 0.647; unfloored units alone: 0.688
- allocation saving, col granularity: 1.329 bits/param with 16.0% of the weights floored at 0 bits; b_u >= 2 bits: 0.862; unfloored units alone: 0.946
- tensor granularity restricted to within-type units: 0.323; on the relative output error s_t / ||W X||^2: 0.293 (floored 0.0%)
- s_t is in absolute output units: across types it tracks the input scale of the linear, so the model-wide saving is outside the D ~ 2^-2b model for the floored weights; the Spearman above is the test of the proxy

### Probes P1-P6 per weight type

| model | probe | type | key statistic | W | control | gain_bits | gain_bits_control | claimed | spread | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | marginal | q | H_4 (index entropy, sym g128 4 bit) | 3.457 | 3.561 | +0.543 | +0.439 | +0.104 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | k | H_4 (index entropy, sym g128 4 bit) | 3.326 | 3.563 | +0.674 | +0.437 | +0.238 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | v | H_4 (index entropy, sym g128 4 bit) | 3.241 | 3.563 | +0.759 | +0.437 | +0.322 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | o | H_4 (index entropy, sym g128 4 bit) | 3.428 | 3.561 | +0.572 | +0.439 | +0.133 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | gate | H_4 (index entropy, sym g128 4 bit) | 3.508 | 3.561 | +0.492 | +0.439 | +0.053 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | up | H_4 (index entropy, sym g128 4 bit) | 3.509 | 3.561 | +0.491 | +0.439 | +0.052 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | down | H_4 (index entropy, sym g128 4 bit) | 3.490 | 3.561 | +0.510 | +0.439 | +0.070 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | in_proj_qkv | H_4 (index entropy, sym g128 4 bit) | 3.485 | 3.561 | +0.515 | +0.439 | +0.076 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | in_proj_z | H_4 (index entropy, sym g128 4 bit) | 3.497 | 3.561 | +0.503 | +0.439 | +0.064 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | in_proj_ab | H_4 (index entropy, sym g128 4 bit) | 3.223 | 3.560 | +0.769 | +0.433 | +0.337 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | out_proj | H_4 (index entropy, sym g128 4 bit) | 3.465 | 3.561 | +0.535 | +0.439 | +0.096 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | marginal | embed | H_4 (index entropy, sym g128 4 bit) | 3.447 | 3.561 | +0.553 | +0.439 | +0.113 | - | pass (1 seed) | Gaussian control, not a shuffle |
| qwen3.5-2b | lowrank | q | energy captured at r = 0.1 min(N, K) | 0.507 | 0.242 | -0.090 | -0.214 | +0.124 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.090); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | k | energy captured at r = 0.1 min(N, K) | 0.399 | 0.195 | -0.145 | -0.180 | +0.035 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.145); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | v | energy captured at r = 0.1 min(N, K) | 0.280 | 0.195 | -0.153 | -0.180 | +0.027 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.153); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | o | energy captured at r = 0.1 min(N, K) | 0.479 | 0.314 | -0.238 | -0.285 | +0.048 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.238); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | gate | energy captured at r = 0.1 min(N, K) | 0.308 | 0.212 | -0.156 | -0.191 | +0.036 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.156); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | up | energy captured at r = 0.1 min(N, K) | 0.254 | 0.212 | -0.178 | -0.191 | +0.013 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.178); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | down | energy captured at r = 0.1 min(N, K) | 0.265 | 0.212 | -0.177 | -0.191 | +0.014 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.177); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | in_proj_qkv | energy captured at r = 0.1 min(N, K) | 0.372 | 0.212 | -0.129 | -0.191 | +0.063 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.129); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | in_proj_z | energy captured at r = 0.1 min(N, K) | 0.483 | 0.314 | -0.196 | -0.285 | +0.089 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.196); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | in_proj_ab | energy captured at r = 0.1 min(N, K) | 0.496 | 0.146 | -0.715 | -0.952 | +0.238 | 0.000 | pass | best aggregated r = 0.05 min(N, K) (gain there -0.715); gain_bits = mean of per-record maxima, per-record best r in {0.062} |
| qwen3.5-2b | lowrank | out_proj | energy captured at r = 0.1 min(N, K) | 0.410 | 0.314 | -0.240 | -0.285 | +0.045 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.240); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank | embed | energy captured at r = 0.1 min(N, K) | 0.357 | 0.115 | +0.040 | -0.149 | +0.189 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there +0.040); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| qwen3.5-2b | lowrank_weighted | q | energy captured at r = 0.1 (W @ L) | 0.968 | 0.737 | -0.035 | -0.206 | +0.170 | 0.002 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.518 vs control +0.237: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.035); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | k | energy captured at r = 0.1 (W @ L) | 0.749 | 0.614 | -0.114 | -0.166 | +0.052 | 0.014 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.206 vs control +0.164: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.114); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | v | energy captured at r = 0.1 (W @ L) | 0.664 | 0.613 | -0.131 | -0.175 | +0.044 | 0.005 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.340 vs control +0.155: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.131); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | o | energy captured at r = 0.1 (W @ L) | 0.812 | 0.888 | -0.273 | -0.287 | +0.013 | 0.001 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.224 vs control +0.486: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.273); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | gate | energy captured at r = 0.1 (W @ L) | 0.896 | 0.681 | -0.112 | -0.187 | +0.074 | 0.000 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.961 vs control +0.193: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.112); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | up | energy captured at r = 0.1 (W @ L) | 0.691 | 0.680 | -0.181 | -0.188 | +0.007 | 0.001 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.217 vs control +0.193: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.181); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | down | energy captured at r = 0.1 (W @ L) | 0.545 | 0.519 | -0.178 | -0.191 | +0.013 | 0.001 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.056 vs control -0.012: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.178); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | in_proj_qkv | energy captured at r = 0.1 (W @ L) | 0.925 | 0.721 | -0.048 | -0.181 | +0.133 | 0.003 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.092 vs control +0.272: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.048); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | in_proj_z | energy captured at r = 0.1 (W @ L) | 0.961 | 0.749 | -0.045 | -0.278 | +0.233 | 0.002 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +1.415 vs control +0.173: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.045); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | in_proj_ab | energy captured at r = 0.1 (W @ L) | 0.895 | 0.438 | -0.664 | -0.942 | +0.279 | 0.009 | pass | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.160 vs control -0.659: a statistic, not bits/param); best aggregated r = 0.05 min(N, K) (gain there -0.664); gain_bits = mean of per-record maxima |
| qwen3.5-2b | lowrank_weighted | out_proj | energy captured at r = 0.1 (W @ L) | 0.775 | 0.804 | -0.262 | -0.270 | +0.009 | 0.003 | fail | gain = measured RTN-4 output-error ratio of the L-metric residual minus bf16 U, V (the contract's e_r energy formula would say +0.182 vs control +0.261: a statistic, not bits/param); best aggregated r = 0.01 min(N, K) (gain there -0.262); gain_bits = mean of per-record maxima |
| qwen3.5-2b | outliers | q | column-norm max / median | 1.98 | 1.06 | -0.032 | -0.053 | +0.021 | 0.000 | pass | top-16 channel Jaccard across layers 0.204 vs control 0.006 |
| qwen3.5-2b | outliers | k | column-norm max / median | 2.78 | 1.22 | +0.005 | -0.050 | +0.055 | 0.001 | pass | top-16 channel Jaccard across layers 0.294 vs control 0.006 |
| qwen3.5-2b | outliers | v | column-norm max / median | 2.16 | 1.33 | -0.034 | -0.048 | +0.014 | 0.000 | fail | top-16 channel Jaccard across layers 0.071 vs control 0.006 |
| qwen3.5-2b | outliers | o | column-norm max / median | 1.93 | 1.13 | -0.042 | -0.051 | +0.009 | 0.000 | fail | top-16 channel Jaccard across layers 0.080 vs control 0.005 (all hidden-input linears) |
| qwen3.5-2b | outliers | gate | column-norm max / median | 1.53 | 1.05 | -0.046 | -0.053 | +0.007 | 0.000 | fail | top-16 channel Jaccard across layers 0.246 vs control 0.005 |
| qwen3.5-2b | outliers | up | column-norm max / median | 1.15 | 1.03 | -0.052 | -0.054 | +0.002 | 0.000 | fail | top-16 channel Jaccard across layers 0.060 vs control 0.005 |
| qwen3.5-2b | outliers | down | column-norm max / median | 1.42 | 1.30 | -0.052 | -0.051 | -0.000 | 0.000 | fail | top-16 channel Jaccard across layers 0.080 vs control 0.005 (all hidden-input linears) |
| qwen3.5-2b | outliers | in_proj_qkv | column-norm max / median | 1.38 | 1.05 | -0.043 | -0.053 | +0.010 | 0.000 | fail | top-16 channel Jaccard across layers 0.227 vs control 0.004 |
| qwen3.5-2b | outliers | in_proj_z | column-norm max / median | 1.85 | 1.08 | -0.036 | -0.053 | +0.017 | 0.000 | fail | top-16 channel Jaccard across layers 0.327 vs control 0.004 |
| qwen3.5-2b | outliers | in_proj_ab | column-norm max / median | 6.41 | 2.94 | +0.192 | -0.039 | +0.231 | 0.003 | pass | top-16 channel Jaccard across layers 0.387 vs control 0.004 |
| qwen3.5-2b | outliers | out_proj | column-norm max / median | 1.68 | 1.23 | -0.043 | -0.047 | +0.004 | 0.000 | fail | top-16 channel Jaccard across layers 0.080 vs control 0.005 (all hidden-input linears) |
| qwen3.5-2b | interlayer | q | min residual energy over alignments, adjacent pairs | 0.556 | 0.274 | -0.511 | -0.525 | +0.014 | - | fail (1 seed) | control 0.274; gain at the best alignment |
| qwen3.5-2b | interlayer | k | min residual energy over alignments, adjacent pairs | 0.491 | 0.134 | -0.529 | -0.527 | -0.001 | - | fail (1 seed) | control 0.134; gain at the best alignment |
| qwen3.5-2b | interlayer | v | min residual energy over alignments, adjacent pairs | 0.363 | 0.137 | -0.433 | -0.434 | +0.001 | - | fail (1 seed) | control 0.137; gain at the best alignment |
| qwen3.5-2b | interlayer | o | min residual energy over alignments, adjacent pairs | 0.631 | 0.503 | -0.474 | -0.474 | -0.000 | - | fail (1 seed) | control 0.503; gain at the best alignment |
| qwen3.5-2b | interlayer | gate | min residual energy over alignments, adjacent pairs | 0.206 | 0.168 | -0.364 | -0.433 | +0.070 | - | pass (1 seed) | control 0.168; gain at the best alignment |
| qwen3.5-2b | interlayer | up | min residual energy over alignments, adjacent pairs | 0.157 | 0.165 | -0.412 | -0.435 | +0.024 | - | pass (1 seed) | control 0.165; gain at the best alignment |
| qwen3.5-2b | interlayer | down | min residual energy over alignments, adjacent pairs | 0.158 | 0.166 | -0.396 | -0.442 | +0.046 | - | pass (1 seed) | control 0.166; gain at the best alignment |
| qwen3.5-2b | interlayer | in_proj_qkv | min residual energy over alignments, adjacent pairs | 0.269 | 0.169 | -0.496 | -0.496 | -0.000 | - | fail (1 seed) | control 0.169; gain at the best alignment |
| qwen3.5-2b | interlayer | in_proj_z | min residual energy over alignments, adjacent pairs | 0.570 | 0.500 | -0.492 | -0.498 | +0.005 | - | fail (1 seed) | control 0.500; gain at the best alignment |
| qwen3.5-2b | interlayer | in_proj_ab | min residual energy over alignments, adjacent pairs | 0.301 | 0.021 | -0.126 | -0.511 | +0.384 | - | pass (1 seed) | control 0.021; gain at the best alignment |
| qwen3.5-2b | interlayer | out_proj | min residual energy over alignments, adjacent pairs | 0.509 | 0.501 | -0.498 | -0.499 | +0.000 | - | fail (1 seed) | control 0.501; gain at the best alignment |
| qwen3.5-2b | acts | o | Gini of per-channel input energy | 0.694 | 0.956 | - | - | - | - | stable | control = split-half Spearman 0.956; channels for 50 % / 90 % of the energy: 0.058 / 0.479 |
| qwen3.5-2b | acts | down | Gini of per-channel input energy | 0.427 | 0.889 | - | - | - | - | noise | control = split-half Spearman 0.889; channels for 50 % / 90 % of the energy: 0.203 / 0.715 |
| qwen3.5-2b | acts | out_proj | Gini of per-channel input energy | 0.686 | 0.957 | - | - | - | - | stable | control = split-half Spearman 0.957; channels for 50 % / 90 % of the energy: 0.059 / 0.483 |
| qwen3.5-2b | sensitivity | q | s_t / n_t | 3.36e-06 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0048 |
| qwen3.5-2b | sensitivity | k | s_t / n_t | 3.12e-06 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0025 |
| qwen3.5-2b | sensitivity | v | s_t / n_t | 8.19e-06 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0084 |
| qwen3.5-2b | sensitivity | o | s_t / n_t | 2.74e-07 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0116 |
| qwen3.5-2b | sensitivity | gate | s_t / n_t | 2.06e-06 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0128 |
| qwen3.5-2b | sensitivity | up | s_t / n_t | 1.11e-06 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0229 |
| qwen3.5-2b | sensitivity | down | s_t / n_t | 3.96e-08 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0234 |
| qwen3.5-2b | sensitivity | in_proj_qkv | s_t / n_t | 6.19e-06 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0143 |
| qwen3.5-2b | sensitivity | in_proj_z | s_t / n_t | 4e-06 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0087 |
| qwen3.5-2b | sensitivity | in_proj_ab | s_t / n_t | 1.07e-05 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0017 |
| qwen3.5-2b | sensitivity | out_proj | s_t / n_t | 4.22e-08 | - | - | - | - | - | no prediction (Spearman -0.05) | @only KL 4 bit 0.0272 |

### P7 super-weight candidates

| model | tensor | layer | type | [row, col] | value | dPPL % | KL | dPPL % magnitude control | dPPL % random | candidate | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | model.layers.23.mlp.down_proj | 23 | down | [1201, 5567] | -0.0308 | -0.01 | 0.0001 | -0.00 | 0.00 | spike_no_input | not a super weight |
| qwen3.5-2b | model.layers.23.mlp.down_proj | 23 | down | [1201, 945] | 0.914 | -0.00 | 0.0005 | -0.01 | 0.00 | top16 | not a super weight |
| qwen3.5-2b | model.layers.22.linear_attn.out_proj | 22 | out_proj | [1201, 1203] | -0.551 | -0.01 | 0.0003 | -0.03 | -0.00 | top16 | not a super weight |
| qwen3.5-2b | model.layers.19.self_attn.v_proj | 19 | v | [495, 1192] | 0.547 | -0.03 | 0.0004 | 0.02 | -0.01 | top16 | not a super weight |
| qwen3.5-2b | model.layers.19.self_attn.v_proj | 19 | v | [431, 1360] | 0.512 | -0.02 | 0.0004 | -0.03 | -0.01 | top16 | not a super weight |
| qwen3.5-2b | model.layers.19.self_attn.v_proj | 19 | v | [282, 525] | -0.512 | 0.02 | 0.0004 | -0.03 | -0.01 | top16 | not a super weight |
| qwen3.5-2b | model.layers.1.linear_attn.out_proj | 1 | out_proj | [1201, 215] | -0.455 | -0.05 | 0.0006 | 0.61 | -0.01 | top16 | not a super weight |
| qwen3.5-2b | model.layers.19.self_attn.v_proj | 19 | v | [446, 1789] | -0.451 | -0.01 | 0.0004 | -0.03 | -0.01 | top16 | not a super weight |
| qwen3.5-2b | model.layers.19.self_attn.v_proj | 19 | v | [283, 493] | 0.451 | -0.01 | 0.0004 | -0.03 | -0.01 | top16 | not a super weight |
| qwen3.5-2b | model.layers.22.linear_attn.out_proj | 22 | out_proj | [1201, 311] | -0.441 | -0.03 | 0.0005 | -0.01 | -0.00 | top16 | not a super weight |
| qwen3.5-2b | model.layers.19.self_attn.v_proj | 19 | v | [288, 203] | 0.441 | -0.01 | 0.0004 | -0.03 | -0.01 | top16 | not a super weight |
| qwen3.5-2b | model.layers.1.linear_attn.out_proj | 1 | out_proj | [1201, 1399] | 0.434 | 0.61 | 0.0011 | -0.05 | -0.01 | top16 | not a super weight |
| qwen3.5-2b | model.layers.23.mlp.gate_proj | 23 | gate | [3841, 1963] | 0.43 | -0.01 | 0.0001 | -0.00 | 0.00 | top16 | not a super weight |
| qwen3.5-2b | model.layers.23.mlp.gate_proj | 23 | gate | [3484, 1963] | 0.43 | -0.00 | 0.0000 | -0.01 | 0.00 | top16 | not a super weight |
| qwen3.5-2b | model.layers.22.linear_attn.in_proj_qkv | 22 | in_proj_qkv | [4598, 1840] | 0.43 | -0.00 | 0.0002 | -0.01 | 0.00 | top16 | not a super weight |
| qwen3.5-2b | model.layers.19.self_attn.v_proj | 19 | v | [428, 1344] | 0.43 | -0.01 | 0.0004 | -0.03 | -0.01 | top16 | not a super weight |
| qwen3.5-2b | model.layers.14.linear_attn.out_proj | 14 | out_proj | [1201, 1872] | -0.422 | 0.03 | 0.0006 | -0.06 | -0.00 | top16 | not a super weight |


## minicpm5-2b

### P0 additivity: sum of @only KL vs joint KL (rtn-sym-g128, same bits)

| bits | types | sum @only KL | joint KL | sum / joint | three largest @only KL | top-3 share of joint | H-P0 additivity |
|---|---|---|---|---|---|---|---|
| 4 | 7 | 0.2612 | 0.2682 | 0.97 | down 0.0819, up 0.0461, o 0.0355 | 61% | additive (within 50 %) |

### P0 pre-registered figures: H-P3 (+keepcols recovery) and H-P0 group size

- +keepcols=1 at 4 bit recovers 17% of the RTN KL (0.2682 -> 0.2227) at +0.118 bits/param: H-P3 not met

### P6 prediction check and allocation saving

- Spearman(s_t per type, rtn-sym-g128-4@only KL): 0.250 over 7 types (contract > 0.7)
- Spearman(sum of s_t per layer, rtn-sym-g128-3@layer KL): - over 0 layers
- allocation saving, tensor granularity: 0.689 bits/param with 2.1% of the weights floored at 0 bits; b_u >= 2 bits: 0.540; unfloored units alone: 0.633
- allocation saving, group granularity: 0.751 bits/param with 2.8% of the weights floored at 0 bits; b_u >= 2 bits: 0.581; unfloored units alone: 0.678
- allocation saving, col granularity: 0.927 bits/param with 3.3% of the weights floored at 0 bits; b_u >= 2 bits: 0.714; unfloored units alone: 0.846
- tensor granularity restricted to within-type units: 0.584; on the relative output error s_t / ||W X||^2: 0.297 (floored 0.0%)
- s_t is in absolute output units: across types it tracks the input scale of the linear, so the model-wide saving is outside the D ~ 2^-2b model for the floored weights; the Spearman above is the test of the proxy

### Probes P1-P6 per weight type

| model | probe | type | key statistic | W | control | gain_bits | gain_bits_control | claimed | spread | verdict | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| minicpm5-2b | marginal | q | H_4 (index entropy, sym g128 4 bit) | 3.508 | 3.561 | +0.491 | +0.439 | +0.052 | - | pass (1 seed) | Gaussian control, not a shuffle |
| minicpm5-2b | marginal | k | H_4 (index entropy, sym g128 4 bit) | 3.430 | 3.560 | +0.569 | +0.440 | +0.130 | - | pass (1 seed) | Gaussian control, not a shuffle |
| minicpm5-2b | marginal | v | H_4 (index entropy, sym g128 4 bit) | 3.374 | 3.560 | +0.626 | +0.440 | +0.186 | - | pass (1 seed) | Gaussian control, not a shuffle |
| minicpm5-2b | marginal | o | H_4 (index entropy, sym g128 4 bit) | 3.561 | 3.561 | +0.439 | +0.439 | +0.000 | - | fail (1 seed) | Gaussian control, not a shuffle |
| minicpm5-2b | marginal | gate | H_4 (index entropy, sym g128 4 bit) | 3.527 | 3.561 | +0.473 | +0.439 | +0.034 | - | pass (1 seed) | Gaussian control, not a shuffle |
| minicpm5-2b | marginal | up | H_4 (index entropy, sym g128 4 bit) | 3.530 | 3.561 | +0.470 | +0.439 | +0.031 | - | pass (1 seed) | Gaussian control, not a shuffle |
| minicpm5-2b | marginal | down | H_4 (index entropy, sym g128 4 bit) | 3.505 | 3.561 | +0.495 | +0.439 | +0.056 | - | pass (1 seed) | Gaussian control, not a shuffle |
| minicpm5-2b | marginal | embed | H_4 (index entropy, sym g128 4 bit) | 3.559 | 3.556 | +0.441 | +0.444 | -0.003 | - | fail (1 seed) | Gaussian control, not a shuffle |
| minicpm5-2b | marginal | lm_head | H_4 (index entropy, sym g128 4 bit) | 3.472 | 3.561 | +0.528 | +0.439 | +0.089 | - | pass (1 seed) | Gaussian control, not a shuffle |
| minicpm5-2b | lowrank | q | energy captured at r = 0.1 min(N, K) | 0.521 | 0.314 | -0.217 | -0.285 | +0.068 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.217); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| minicpm5-2b | lowrank | k | energy captured at r = 0.1 min(N, K) | 0.293 | 0.168 | -0.170 | -0.195 | +0.026 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.170); gain_bits = mean of per-record maxima, per-record best r in {0.012, 0.1} |
| minicpm5-2b | lowrank | v | energy captured at r = 0.1 min(N, K) | 0.187 | 0.167 | -0.191 | -0.196 | +0.004 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.191); gain_bits = mean of per-record maxima, per-record best r in {0.012} |
| minicpm5-2b | lowrank | o | energy captured at r = 0.1 min(N, K) | 0.444 | 0.314 | -0.255 | -0.285 | +0.030 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.255); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| minicpm5-2b | lowrank | gate | energy captured at r = 0.1 min(N, K) | 0.258 | 0.212 | -0.179 | -0.191 | +0.012 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.179); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| minicpm5-2b | lowrank | up | energy captured at r = 0.1 min(N, K) | 0.235 | 0.212 | -0.184 | -0.191 | +0.007 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.184); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| minicpm5-2b | lowrank | down | energy captured at r = 0.1 min(N, K) | 0.254 | 0.212 | -0.178 | -0.191 | +0.013 | 0.000 | fail | best aggregated r = 0.01 min(N, K) (gain there -0.178); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| minicpm5-2b | lowrank | embed | energy captured at r = 0.1 min(N, K) | 0.281 | 0.121 | -0.104 | -0.150 | +0.046 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.104); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| minicpm5-2b | lowrank | lm_head | energy captured at r = 0.1 min(N, K) | 0.268 | 0.121 | -0.058 | -0.150 | +0.092 | 0.000 | pass | best aggregated r = 0.01 min(N, K) (gain there -0.058); gain_bits = mean of per-record maxima, per-record best r in {0.0098} |
| minicpm5-2b | outliers | q | column-norm max / median | 1.88 | 1.11 | -0.023 | -0.053 | +0.029 | 0.000 | pass | top-16 channel Jaccard across layers 0.221 vs control 0.004 |
| minicpm5-2b | outliers | k | column-norm max / median | 2.82 | 1.41 | +0.012 | -0.047 | +0.058 | 0.000 | pass | top-16 channel Jaccard across layers 0.241 vs control 0.004 |
| minicpm5-2b | outliers | v | column-norm max / median | 2.26 | 1.23 | -0.033 | -0.054 | +0.020 | 0.000 | pass | top-16 channel Jaccard across layers 0.057 vs control 0.004 |
| minicpm5-2b | outliers | o | column-norm max / median | 1.37 | 1.07 | -0.052 | -0.053 | +0.001 | 0.000 | fail | top-16 channel Jaccard across layers 0.150 vs control 0.004 (all hidden-input linears) |
| minicpm5-2b | outliers | gate | column-norm max / median | 2.34 | 1.20 | -0.033 | -0.051 | +0.018 | 0.000 | fail | top-16 channel Jaccard across layers 0.292 vs control 0.004 |
| minicpm5-2b | outliers | up | column-norm max / median | 2.30 | 1.18 | -0.029 | -0.051 | +0.022 | 0.000 | pass | top-16 channel Jaccard across layers 0.427 vs control 0.004 |
| minicpm5-2b | outliers | down | column-norm max / median | 1.85 | 1.62 | -0.039 | -0.049 | +0.010 | 0.000 | fail | top-16 channel Jaccard across layers 0.150 vs control 0.004 (all hidden-input linears) |
| minicpm5-2b | interlayer | q | min residual energy over alignments, adjacent pairs | 0.587 | 0.511 | -0.493 | -0.493 | -0.000 | - | fail (1 seed) | control 0.511; gain at the best alignment |
| minicpm5-2b | interlayer | k | min residual energy over alignments, adjacent pairs | 0.276 | 0.083 | -0.503 | -0.503 | +0.000 | - | fail (1 seed) | control 0.083; gain at the best alignment |
| minicpm5-2b | interlayer | v | min residual energy over alignments, adjacent pairs | 0.128 | 0.077 | -0.488 | -0.487 | -0.000 | - | fail (1 seed) | control 0.077; gain at the best alignment |
| minicpm5-2b | interlayer | o | min residual energy over alignments, adjacent pairs | 0.579 | 0.497 | -0.487 | -0.487 | -0.000 | - | fail (1 seed) | control 0.497; gain at the best alignment |
| minicpm5-2b | interlayer | gate | min residual energy over alignments, adjacent pairs | 0.186 | 0.167 | -0.405 | -0.443 | +0.038 | - | pass (1 seed) | control 0.167; gain at the best alignment |
| minicpm5-2b | interlayer | up | min residual energy over alignments, adjacent pairs | 0.160 | 0.167 | -0.429 | -0.446 | +0.017 | - | fail (1 seed) | control 0.167; gain at the best alignment |
| minicpm5-2b | interlayer | down | min residual energy over alignments, adjacent pairs | 0.161 | 0.168 | -0.412 | -0.442 | +0.030 | - | pass (1 seed) | control 0.168; gain at the best alignment |
| minicpm5-2b | acts | o | Gini of per-channel input energy | 0.367 | 0.969 | - | - | - | - | stable | control = split-half Spearman 0.969; channels for 50 % / 90 % of the energy: 0.253 / 0.718 |
| minicpm5-2b | acts | down | Gini of per-channel input energy | 0.460 | 0.902 | - | - | - | - | stable | control = split-half Spearman 0.902; channels for 50 % / 90 % of the energy: 0.184 / 0.687 |
| minicpm5-2b | sensitivity | q | s_t / n_t | 3.64e-06 | - | - | - | - | - | no prediction (Spearman 0.25) | @only KL 4 bit 0.0236 |
| minicpm5-2b | sensitivity | k | s_t / n_t | 8.03e-06 | - | - | - | - | - | no prediction (Spearman 0.25) | @only KL 4 bit 0.0117 |
| minicpm5-2b | sensitivity | v | s_t / n_t | 6.96e-06 | - | - | - | - | - | no prediction (Spearman 0.25) | @only KL 4 bit 0.0318 |
| minicpm5-2b | sensitivity | o | s_t / n_t | 3.64e-07 | - | - | - | - | - | no prediction (Spearman 0.25) | @only KL 4 bit 0.0355 |
| minicpm5-2b | sensitivity | gate | s_t / n_t | 2.02e-06 | - | - | - | - | - | no prediction (Spearman 0.25) | @only KL 4 bit 0.0306 |
| minicpm5-2b | sensitivity | up | s_t / n_t | 2.14e-06 | - | - | - | - | - | no prediction (Spearman 0.25) | @only KL 4 bit 0.0461 |
| minicpm5-2b | sensitivity | down | s_t / n_t | 1.25e-06 | - | - | - | - | - | no prediction (Spearman 0.25) | @only KL 4 bit 0.0819 |

### P7 super-weight candidates

| model | tensor | layer | type | [row, col] | value | dPPL % | KL | dPPL % magnitude control | dPPL % random | candidate | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| minicpm5-2b | model.layers.7.mlp.down_proj | 7 | down | [106, 1460] | 1.05 | 205.59 | 1.2415 | 0.08 | 0.01 | spike | super weight |
| minicpm5-2b | model.layers.40.mlp.down_proj | 40 | down | [106, 271] | 0.973 | -0.14 | 0.0014 | -0.13 | 0.01 | spike | not a super weight |
| minicpm5-2b | model.layers.41.mlp.down_proj | 41 | down | [106, 4621] | -0.824 | -0.01 | 0.0002 | -0.08 | 0.00 | spike_no_input | not a super weight |
| minicpm5-2b | model.layers.40.mlp.gate_proj | 40 | gate | [1331, 106] | 2.19 | 0.02 | 0.0009 | -0.20 | 0.01 | top16 | not a super weight |
| minicpm5-2b | model.layers.40.mlp.up_proj | 40 | up | [271, 750] | 1.98 | 0.25 | 0.0027 | 0.06 | 0.01 | top16 | not a super weight |
| minicpm5-2b | model.layers.7.mlp.gate_proj | 7 | gate | [4426, 1108] | -1.84 | 1.23 | 0.0118 | 0.11 | -0.00 | top16 | not a super weight |
| minicpm5-2b | model.layers.41.mlp.gate_proj | 41 | gate | [5497, 106] | -1.84 | -0.03 | 0.0002 | 1.26 | -0.00 | top16 | not a super weight |
| minicpm5-2b | model.layers.31.mlp.gate_proj | 31 | gate | [4411, 106] | -1.83 | 0.33 | 0.0008 | 0.01 | 0.02 | top16 | not a super weight |
| minicpm5-2b | model.layers.41.mlp.up_proj | 41 | up | [5497, 106] | 1.8 | 0.05 | 0.0002 | -0.08 | -0.00 | top16 | not a super weight |
| minicpm5-2b | model.layers.40.mlp.up_proj | 40 | up | [1331, 106] | 1.74 | 0.06 | 0.0008 | 0.25 | 0.01 | top16 | not a super weight |
| minicpm5-2b | model.layers.40.mlp.up_proj | 40 | up | [1290, 106] | 1.73 | 3.02 | 0.0100 | 0.25 | 0.01 | top16 | not a super weight |
| minicpm5-2b | model.layers.40.mlp.gate_proj | 40 | gate | [271, 106] | 1.71 | -0.20 | 0.0014 | 0.02 | 0.01 | top16 | not a super weight |
| minicpm5-2b | model.layers.7.mlp.gate_proj | 7 | gate | [3516, 1108] | -1.68 | 0.11 | 0.0019 | 1.23 | -0.00 | top16 | not a super weight |
| minicpm5-2b | model.layers.7.mlp.gate_proj | 7 | gate | [3919, 1108] | -1.62 | 0.11 | 0.0012 | 1.23 | -0.00 | top16 | not a super weight |
| minicpm5-2b | model.layers.28.mlp.down_proj | 28 | down | [106, 4293] | 1.62 | 0.01 | 0.0007 | 0.03 | 0.01 | top16 | not a super weight |
| minicpm5-2b | model.layers.22.mlp.up_proj | 22 | up | [3256, 106] | -1.6 | 0.06 | 0.0008 | 0.00 | 0.02 | top16 | not a super weight |
| minicpm5-2b | model.layers.14.mlp.down_proj | 14 | down | [106, 1564] | 1.56 | 0.15 | 0.0008 | 0.01 | 0.04 | top16 | not a super weight |
| minicpm5-2b | model.layers.7.mlp.up_proj | 7 | up | [3127, 1108] | 1.53 | 0.02 | 0.0011 | 0.06 | 0.04 | top16 | not a super weight |
| minicpm5-2b | model.layers.36.mlp.down_proj | 36 | down | [106, 3611] | -1.52 | 0.07 | 0.0006 | -0.00 | -0.00 | top16 | not a super weight |


## Skipped

- qwen3.5-2b: fisher.json missing (not in the Phase 2 run list, DESIGN.md)
- minicpm5-2b: lowrank_weighted.json missing (not in the Phase 2 run list, DESIGN.md)
- minicpm5-2b: fisher.json missing (not in the Phase 2 run list, DESIGN.md)
