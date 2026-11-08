#!/bin/sh
# Post-fix smoke (sequential on the one GPU). Outputs stay under results/smoke/.
cd /d/repos/wcomp || exit 1
L=results/smoke
P=results/smoke/probe/qwen3.5-0.8b
step() { echo "=== $1 start $(date +%H:%M:%S)"; }
done_() { echo "=== $1 exit $? end $(date +%H:%M:%S)"; }

step regress_rtn
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods fp16,rtn-sym-g128-4 --limit-windows 8 --kl-windows 8 > $L/fix_regress_rtn.log 2>&1
done_ regress_rtn

step probe_weights
uv run python scripts/probe_weights.py --model qwen3.5-0.8b --limit-tensors 12 --include-embedding --out-dir $P > $L/fix_probe_weights.log 2>&1
done_ probe_weights

step probe_acts
uv run python scripts/probe_acts.py --model qwen3.5-0.8b --windows 4 --fisher-windows 4 --fisher-tokens 256 --eval-windows 4 --out-dir $P --include-smoke > $L/fix_probe_acts.log 2>&1
done_ probe_acts

step ablation
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods "rtn-sym-g128-4+keepsuper=$P/superweight.json" --limit-windows 4 --kl-windows 4 --results-dir results/smoke/probe --out-name ablation.jsonl > $L/fix_ablation.log 2>&1
done_ ablation

step p6a_recheck
uv run python scripts/probe_acts.py --model qwen3.5-0.8b --probes p6a --windows 4 --out-dir $P --include-smoke > $L/fix_p6a_recheck.log 2>&1
done_ p6a_recheck

step table
uv run python scripts/structure_table.py --models qwen3.5-0.8b --results-dir results/smoke --include-smoke --out results/smoke/structure_table.md > $L/fix_structure_table.log 2>&1
done_ table
step plots
uv run python scripts/plot_probe.py --model qwen3.5-0.8b --results-dir results/smoke --out-dir results/smoke --include-smoke > $L/fix_plot_probe.log 2>&1
done_ plots
echo SMOKE_DONE
