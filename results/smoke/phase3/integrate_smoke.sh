#!/bin/sh
# Phase 3 integration: regression + end-to-end smoke (sequential on the one GPU). Outputs stay under results/smoke/phase3/.
cd /d/repos/wcomp || exit 1
L=results/smoke/phase3
R=results/smoke/phase3
M=qwen3.5-0.8b
step() { echo "=== $1 start $(date +%H:%M:%S) t=$(date +%s)"; }
done_() { echo "=== $1 exit $? end $(date +%H:%M:%S) t=$(date +%s)"; }

step regress_rtn
uv run python scripts/run_baselines.py --model $M --methods fp16,rtn-sym-g128-4 --limit-windows 8 --kl-windows 8 --results-dir $R --out-name regress.jsonl > $L/regress_rtn.log 2>&1
done_ regress_rtn

step regress_gptq_f64
uv run python scripts/run_baselines.py --model $M --methods gptq-sym-g128-4 --limit-windows 8 --kl-windows 4 --calib-n 16 --hessian-f64 --results-dir $R --out-name regress.jsonl > $L/regress_gptq_f64.log 2>&1
done_ regress_gptq_f64

step damage_c4
uv run python scripts/damage_c4.py --model $M --units type --windows 2 --results-dir $R > $L/damage_c4.log 2>&1
done_ damage_c4

step smoke_rows
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R \
  --methods fp16,gptq32-sym-g128-3,had-gptq32-sym-g128-3,lloyd-gptq32-g128-3,nf-gptq32-g128-3,alloc-type-gptq32-avg3,alloc-type-shuf1-gptq32-avg3,rowalloc-fisher-rtn-avg4,eig-gptq32-sym-g128-3 > $L/smoke_rows.log 2>&1
done_ smoke_rows

step combo
printf '{"had": true, "lloyd": true, "rowalloc": true}\n' > results/phase3/$M/combo.json
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R --methods combo-gptq32-avg3 > $L/smoke_combo.log 2>&1
done_ combo
rm -f results/phase3/$M/combo.json   # hand-written for the smoke only; the run agent writes the real one after the decision rules

step plots
uv run python scripts/plot_rd3.py --model $M --results-dir $R --include-smoke --out $L/rd3_${M}_smoke.png > $L/plot_rd3.log 2>&1
done_ plots
step summary
uv run python scripts/phase3_summary.py --models $M --results-dir $R --include-smoke --out $L/summary_smoke.md > $L/summary.log 2>&1
done_ summary
echo ALL DONE
