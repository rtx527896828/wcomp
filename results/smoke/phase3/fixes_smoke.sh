#!/bin/sh
# After the review fixes: Phase 1 regression (fp16 / rtn-sym-g128-4 must give 14.946913 / 19.007506), the Phase 3
# end-to-end smoke (gptq32-3, alloc-type-avg3, lloyd-3), then the rows whose Shannon figure predated the coder-bound
# definition (nf-3, shuf1) plus the new nfq-3 row, and the smoke plot / summary. Sequential on the one GPU.
cd /d/repos/wcomp || exit 1
L=results/smoke/phase3
R=results/smoke/phase3
M=qwen3.5-0.8b
step() { echo "=== $1 start $(date +%H:%M:%S) t=$(date +%s)"; }
done_() { echo "=== $1 exit $? end $(date +%H:%M:%S) t=$(date +%s)"; }

step regress_rtn
uv run python scripts/run_baselines.py --model $M --methods fp16,rtn-sym-g128-4 --limit-windows 8 --kl-windows 8 --results-dir $R --out-name regress.jsonl > $L/regress_rtn_fixes.log 2>&1
done_ regress_rtn

step smoke_rows
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R \
  --methods gptq32-sym-g128-3,alloc-type-gptq32-avg3,lloyd-gptq32-g128-3 > $L/smoke_rows_fixes.log 2>&1
done_ smoke_rows

step smoke_rows_stale_shannon
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R \
  --methods nf-gptq32-g128-3,nfq-gptq32-g128-3,alloc-type-shuf1-gptq32-avg3 > $L/smoke_rows_fixes2.log 2>&1
done_ smoke_rows_stale_shannon

step plots
uv run python scripts/plot_rd3.py --model $M --results-dir $R --include-smoke --out $L/rd3_${M}_smoke.png > $L/plot_rd3.log 2>&1
done_ plots
step summary
uv run python scripts/phase3_summary.py --models $M --results-dir $R --include-smoke --out $L/summary_smoke.md > $L/summary.log 2>&1
done_ summary
echo ALL DONE
