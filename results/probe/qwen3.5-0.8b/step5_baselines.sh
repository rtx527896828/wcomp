#!/bin/sh
# Phase 2 run list item (5) for qwen3.5-0.8b: full-model baselines -> results/qwen3.5-0.8b/runs.jsonl.
# GPTQ uses the Phase 1 f64 Hessian path (--hessian-f64): the f32 acceptance row (gptq-sym-g128-3 seed 0 ->
# results/probe/qwen3.5-0.8b/ablation.jsonl) gave PPL 23.32 / KL 0.2999 vs Phase 1 24.45 / 0.3131, outside 0.5% / 2%.
cd /d/repos/wcomp || exit 1
LOG=results/probe/qwen3.5-0.8b/run.log
step() { echo "=== $(date -u +%FT%TZ) STEP5-$1 start" >> $LOG; }
fin() { echo "=== $(date -u +%FT%TZ) EXIT_STEP5-$1 $2" >> $LOG; }

step gptq4_seed1
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods gptq-sym-g128-4 --seed 1 --hessian-f64 >> $LOG 2>&1
fin gptq4_seed1 $?

step had
uv run python scripts/run_baselines.py --model qwen3.5-0.8b --methods had-fp16,had-gptq-sym-g128-3 --hessian-f64 >> $LOG 2>&1
fin had $?

echo "STEP5_DONE" >> $LOG
