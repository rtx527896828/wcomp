#!/bin/sh
# After the second review-fix round (run on the final code, sequential on the one GPU): the Phase 1 regression
# (fp16 / rtn-sym-g128-4 must give 14.946913 / 19.007506; gptq-sym-g128-4 f64 16.248178), then every Phase 3 smoke row
# on the final code (the earlier smoke rows under qwen3.5-0.8b/stale_pre_fixes/ predate the row-width clamp, the
# coder-bound Shannon definition and the damage-set exclusion), the combo row with the smoke tree's own combo.json
# (read from --results-dir, the protocol tree is not touched), then the smoke plot / summary.
cd /d/repos/wcomp || exit 1
L=results/smoke/phase3
R=results/smoke/phase3
M=qwen3.5-0.8b
step() { echo "=== $1 start $(date +%H:%M:%S) t=$(date +%s)"; }
done_() { echo "=== $1 exit $? end $(date +%H:%M:%S) t=$(date +%s)"; }

step regress_rtn
uv run python scripts/run_baselines.py --model $M --methods fp16,rtn-sym-g128-4 --limit-windows 8 --kl-windows 8 --results-dir $R --out-name regress.jsonl > $L/regress_rtn_fixes2.log 2>&1
done_ regress_rtn

step regress_gptq_f64
uv run python scripts/run_baselines.py --model $M --methods gptq-sym-g128-4 --limit-windows 8 --kl-windows 4 --calib-n 16 --hessian-f64 --results-dir $R --out-name regress.jsonl > $L/regress_gptq_f64_fixes2.log 2>&1
done_ regress_gptq_f64

step smoke_rows
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R \
  --methods fp16,gptq32-sym-g128-3,alloc-type-gptq32-avg3,lloyd-gptq32-g128-3 > $L/smoke_rows_fixes2.log 2>&1
done_ smoke_rows

step smoke_rows_more
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R \
  --methods nf-gptq32-g128-3,nfq-gptq32-g128-3,alloc-type-shuf1-gptq32-avg3,alloc-type-shuf2-gptq32-avg3,rowalloc-fisher-rtn-avg4,had-gptq32-sym-g128-3,eig-gptq32-sym-g128-3 > $L/smoke_rows_fixes2b.log 2>&1
done_ smoke_rows_more

step combo
printf '{"had": true, "lloyd": true, "rowalloc": true}\n' > $R/$M/combo.json   # the smoke tree's own decision file (all components on, to exercise every part)
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R --methods combo-gptq32-avg3 > $L/smoke_combo_fixes2.log 2>&1
done_ combo

step plots
uv run python scripts/plot_rd3.py --model $M --results-dir $R --include-smoke --out $L/rd3_${M}_smoke.png > $L/plot_rd3.log 2>&1
done_ plots
step summary
uv run python scripts/phase3_summary.py --models $M --results-dir $R --include-smoke --out $L/summary_smoke.md > $L/summary.log 2>&1
done_ summary
echo ALL DONE
