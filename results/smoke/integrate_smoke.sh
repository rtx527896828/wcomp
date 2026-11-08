#!/bin/sh
# Phase 2 integration smoke (sequential on the one GPU). Outputs stay under results/smoke/.
cd /d/repos/wcomp || exit 1
L=results/smoke
P=results/smoke/probe/qwen3.5-0.8b
step() { echo "=== $1 start $(date +%H:%M:%S) t=$(date +%s)"; }
done_() { echo "=== $1 exit $? end $(date +%H:%M:%S) t=$(date +%s)"; }

step regress_rtn
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods fp16,rtn-sym-g128-4 --limit-windows 8 --kl-windows 8 > $L/integrate_regress_rtn.log 2>&1
done_ regress_rtn

step regress_gptq_f32
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods gptq-sym-g128-4 --calib-n 16 --limit-windows 8 --kl-windows 8 > $L/integrate_regress_gptq_f32.log 2>&1
done_ regress_gptq_f32

step probe_weights
uv run python scripts/probe_weights.py --model qwen3.5-0.8b --limit-tensors 12 --include-embedding --out-dir $P > $L/integrate_probe_weights.log 2>&1
done_ probe_weights

step probe_acts
uv run python scripts/probe_acts.py --model qwen3.5-0.8b --windows 4 --fisher-windows 4 --fisher-tokens 256 --eval-windows 4 --out-dir $P > $L/integrate_probe_acts.log 2>&1
done_ probe_acts

step ablation
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods "fp16,rtn-sym-g128-4,rtn-sym-g32-4,rtn-sym-g128-4@only=down,rtn-sym-g128-4+keepcols=1,rtn-sym-g128-4+keepsuper=$P/superweight.json" --limit-windows 4 --kl-windows 4 --results-dir results/smoke/probe --out-name ablation.jsonl > $L/integrate_ablation.log 2>&1
done_ ablation

step p6a_recheck
uv run python scripts/probe_acts.py --model qwen3.5-0.8b --probes p6a --windows 4 --out-dir $P > $L/integrate_p6a_recheck.log 2>&1
done_ p6a_recheck

step plots
uv run python scripts/plot_probe.py --model qwen3.5-0.8b --results-dir results/smoke --out-dir results/smoke > $L/integrate_plot_probe.log 2>&1
done_ plots
step table
uv run python scripts/structure_table.py --models qwen3.5-0.8b --results-dir results/smoke --out results/smoke/structure_table.md > $L/integrate_structure_table.log 2>&1
done_ table
step plot_rd
uv run python scripts/plot_rd.py --model qwen3.5-0.8b --results-dir results/smoke --include-smoke --extra $P/ablation.jsonl --out results/smoke/rd_qwen3.5-0.8b_extra.png > $L/integrate_plot_rd.log 2>&1
done_ plot_rd
echo SMOKE_DONE
