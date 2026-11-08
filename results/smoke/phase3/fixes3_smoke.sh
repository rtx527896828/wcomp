#!/bin/sh
# Third review-fix round (shuffle-control validity, exact-DP pre-registration, nfq in the run list, dPPL-rule note, plot
# labels, tensor-signal choice, Huffman plain fallback, interior-width row tables, summary fixes). Sequential on the one GPU:
# the full pytest, the Phase 1 regression (fp16 / rtn-sym-g128-4 must give 14.946913 / 19.007506), the small Phase 3
# end-to-end smoke (3 rows) into the smoke tree, then the smoke plot and summary on the new code.
cd /d/repos/wcomp || exit 1
L=results/smoke/phase3
R=results/smoke/phase3
M=qwen3.5-0.8b
step() { echo "=== $1 start $(date +%H:%M:%S) t=$(date +%s)"; }
done_() { echo "=== $1 exit $? end $(date +%H:%M:%S) t=$(date +%s)"; }

step pytest
uv run python -m pytest -q > $L/pytest_fixes3.log 2>&1
done_ pytest

step regress_rtn
uv run python scripts/run_baselines.py --model $M --methods fp16,rtn-sym-g128-4 --limit-windows 8 --kl-windows 8 --results-dir $R --out-name regress.jsonl > $L/regress_rtn_fixes3.log 2>&1
done_ regress_rtn

step smoke_rows
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R \
  --methods gptq32-sym-g128-3,alloc-type-gptq32-avg3,lloyd-gptq32-g128-3 > $L/smoke_rows_fixes3.log 2>&1
done_ smoke_rows

step plots
uv run python scripts/plot_rd3.py --model $M --results-dir $R --include-smoke --out $L/rd3_${M}_smoke.png > $L/plot_rd3_fixes3.log 2>&1
done_ plots
step summary
uv run python scripts/phase3_summary.py --models $M --results-dir $R --include-smoke --out $L/summary_smoke.md > $L/summary_fixes3.log 2>&1
done_ summary
echo ALL DONE
