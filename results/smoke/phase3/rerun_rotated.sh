#!/bin/sh
# Re-run the three rotated smoke rows after the rd_bound basis fix (wcomp/rdbound.py, wcomp/quant/rotate.py), then replot.
cd /d/repos/wcomp || exit 1
L=results/smoke/phase3
R=results/smoke/phase3
M=qwen3.5-0.8b
D=$R/$M
step() { echo "=== $1 start $(date +%H:%M:%S) t=$(date +%s)"; }
done_() { echo "=== $1 exit $? end $(date +%H:%M:%S) t=$(date +%s)"; }

for f in runs.jsonl rd_bound.jsonl; do
  grep -v '"method": "had-gptq32-sym-g128-3"\|"method": "eig-gptq32-sym-g128-3"\|"method": "combo-gptq32-avg3"' $D/$f > $D/$f.tmp && mv $D/$f.tmp $D/$f
done
step rotated
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R --methods had-gptq32-sym-g128-3,eig-gptq32-sym-g128-3 > $L/smoke_rotated_rerun.log 2>&1
done_ rotated
step combo
printf '{"had": true, "lloyd": true, "rowalloc": true}\n' > results/phase3/$M/combo.json
uv run python scripts/run_baselines.py --model $M --limit-windows 4 --kl-windows 4 --calib-n 8 --results-dir $R --methods combo-gptq32-avg3 > $L/smoke_combo_rerun2.log 2>&1
done_ combo
rm -f results/phase3/$M/combo.json
step plots
uv run python scripts/plot_rd3.py --model $M --results-dir $R --include-smoke --out $L/rd3_${M}_smoke.png > $L/plot_rd3.log 2>&1
done_ plots
step summary
uv run python scripts/phase3_summary.py --models $M --results-dir $R --include-smoke --out $L/summary_smoke.md > $L/summary.log 2>&1
done_ summary
echo ALL DONE
