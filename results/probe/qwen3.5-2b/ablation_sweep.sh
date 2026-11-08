#!/bin/sh
# Phase 2 ablation sweep for qwen3.5-2b (DESIGN.md run list / NOTES.md pre-registration), sequential on the one GPU.
# Full protocol (all windows, kl_windows 32). Rows -> results/probe/qwen3.5-2b/ablation.jsonl.
cd /d/repos/wcomp || exit 1
LOG=results/probe/qwen3.5-2b/run.log
RB="uv run python scripts/run_baselines.py --model qwen3.5-2b --results-dir results/probe --out-name ablation.jsonl --methods"
step() { echo "=== $(date -u +%FT%TZ) STEP3-$1 start" >> $LOG; }
fin() { echo "=== $(date -u +%FT%TZ) EXIT_STEP3-$1 $2" >> $LOG; }

step only4
$RB rtn-sym-g128-4@only=q,rtn-sym-g128-4@only=k,rtn-sym-g128-4@only=v,rtn-sym-g128-4@only=o,rtn-sym-g128-4@only=gate,rtn-sym-g128-4@only=up,rtn-sym-g128-4@only=down,rtn-sym-g128-4@only=in_proj_qkv,rtn-sym-g128-4@only=in_proj_z,rtn-sym-g128-4@only=in_proj_ab,rtn-sym-g128-4@only=out_proj >> $LOG 2>&1
fin only4 $?

step layer3
$RB rtn-sym-g128-3@layer=0,rtn-sym-g128-3@layer=1,rtn-sym-g128-3@layer=2,rtn-sym-g128-3@layer=3,rtn-sym-g128-3@layer=4,rtn-sym-g128-3@layer=5,rtn-sym-g128-3@layer=6,rtn-sym-g128-3@layer=7,rtn-sym-g128-3@layer=8,rtn-sym-g128-3@layer=9,rtn-sym-g128-3@layer=10,rtn-sym-g128-3@layer=11,rtn-sym-g128-3@layer=12,rtn-sym-g128-3@layer=13,rtn-sym-g128-3@layer=14,rtn-sym-g128-3@layer=15,rtn-sym-g128-3@layer=16,rtn-sym-g128-3@layer=17,rtn-sym-g128-3@layer=18,rtn-sym-g128-3@layer=19,rtn-sym-g128-3@layer=20,rtn-sym-g128-3@layer=21,rtn-sym-g128-3@layer=22,rtn-sym-g128-3@layer=23 >> $LOG 2>&1
fin layer3 $?

step keepcols
$RB rtn-sym-g128-4+keepcols=1 >> $LOG 2>&1
fin keepcols $?

step keepsuper
$RB rtn-sym-g128-4+keepsuper=results/probe/qwen3.5-2b/superweight.json >> $LOG 2>&1
fin keepsuper $?

echo "SWEEP3_DONE" >> $LOG
