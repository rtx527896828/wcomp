#!/bin/sh
# Round-2 fix smoke (sequential on the one GPU). Outputs stay under results/smoke/.
cd /d/repos/wcomp || exit 1
L=results/smoke
P=results/smoke/probe/qwen3.5-0.8b
step() { echo "=== $1 start $(date +%H:%M:%S) t=$(date +%s)"; }
done_() { echo "=== $1 exit $? end $(date +%H:%M:%S) t=$(date +%s)"; }

step regress_rtn
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods fp16,rtn-sym-g128-4 --limit-windows 8 --kl-windows 8 > $L/fix2_regress_rtn.log 2>&1
done_ regress_rtn

step probe_weights
uv run python scripts/probe_weights.py --model qwen3.5-0.8b --limit-tensors 12 --include-embedding > $L/fix2_probe_weights.log 2>&1
done_ probe_weights

step fisher_1024
uv run python scripts/probe_acts.py --model qwen3.5-0.8b --probes p6b --fisher-windows 2 --out-dir results/smoke/probe_fisher1024/qwen3.5-0.8b > $L/fix2_fisher_1024.log 2>&1
done_ fisher_1024

step probe_acts
uv run python scripts/probe_acts.py --model qwen3.5-0.8b --windows 4 --fisher-windows 4 --fisher-tokens 256 --eval-windows 4 > $L/fix2_probe_acts.log 2>&1
done_ probe_acts

step ablation_skip
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods "rtn-sym-g128-4+keepsuper=$P/superweight.json" --limit-windows 4 --kl-windows 4 --results-dir results/smoke/probe --out-name ablation.jsonl > $L/fix2_ablation_skip.log 2>&1
done_ ablation_skip

step ablation
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods "fp16,rtn-sym-g128-4,rtn-sym-g128-4@only=down,rtn-sym-g128-4+keepsuper=$P/superweight.json" --keepsuper-any-kind --limit-windows 4 --kl-windows 4 --results-dir results/smoke/probe --out-name ablation.jsonl > $L/fix2_ablation.log 2>&1
done_ ablation

step hessian_refused
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods gptq-sym-g128-3 --results-dir results/probe --out-name ablation.jsonl > $L/fix2_hessian_refused.log 2>&1
done_ hessian_refused

step hessian_paths
uv run python scripts/hessian_paths.py --model qwen3.5-0.8b --calib-n 4 --layer 0 --bits 3 > $L/fix2_hessian_paths.log 2>&1
done_ hessian_paths

step plots
uv run python scripts/plot_probe.py --model qwen3.5-0.8b --results-dir results/smoke --out-dir results/smoke > $L/fix2_plot_probe.log 2>&1
done_ plots
step table
uv run python scripts/structure_table.py --models qwen3.5-0.8b --results-dir results/smoke --out results/smoke/structure_table.md > $L/fix2_structure_table.log 2>&1
done_ table
step plot_rd
uv run python scripts/plot_rd.py --model qwen3.5-0.8b --results-dir results/smoke --include-smoke --extra $P/ablation.jsonl --out results/smoke/rd_qwen3.5-0.8b_extra.png > $L/fix2_plot_rd.log 2>&1
done_ plot_rd
echo SMOKE_DONE
