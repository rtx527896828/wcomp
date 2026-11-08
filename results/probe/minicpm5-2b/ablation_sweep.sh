#!/bin/sh
# Phase 2 ablation sweep for minicpm5-2b (DESIGN.md run list / NOTES.md pre-registration), sequential on the one GPU.
# Full protocol (all windows, kl_windows 32). Rows -> results/probe/minicpm5-2b/ablation.jsonl.
cd /d/repos/wcomp || exit 1
LOG=results/probe/minicpm5-2b/run.log
RB="uv run python scripts/run_baselines.py --model minicpm5-2b --results-dir results/probe --out-name ablation.jsonl --methods"
step() { echo "=== $(date -u +%FT%TZ) STEP3-$1 start" >> $LOG; }
fin() { echo "=== $(date -u +%FT%TZ) EXIT_STEP3-$1 $2" >> $LOG; }

step only4
$RB rtn-sym-g128-4@only=q,rtn-sym-g128-4@only=k,rtn-sym-g128-4@only=v,rtn-sym-g128-4@only=o,rtn-sym-g128-4@only=gate,rtn-sym-g128-4@only=up,rtn-sym-g128-4@only=down >> $LOG 2>&1
fin only4 $?

step keepcols
$RB rtn-sym-g128-4+keepcols=1 >> $LOG 2>&1
fin keepcols $?

step keepsuper
$RB rtn-sym-g128-4+keepsuper=results/probe/minicpm5-2b/superweight.json >> $LOG 2>&1
fin keepsuper $?

echo "SWEEP3_DONE" >> $LOG
