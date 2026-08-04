#!/usr/bin/env bash
# Kernel-level profile of ONE DFlash verify workload.
#
# Why this exists: every verify attribution we have so far was produced by
# subtracting known phases from the step total, and the residual disagreed with
# the kernel microbenchmarks by ~56 ms (the m=6 expert GEMV microtest says 213
# GB/s, which cannot coexist with "MoE is 81% of a 140 ms verify"). Subtraction
# is not evidence. This traces actual kernel time.
#
# Graphs are left ON so the profile matches production; nsys resolves kernels
# inside captured graphs. Model load is inside the capture window — filter the
# summary by kernel name, or note that load kernels are one-shot while decode
# kernels have counts in the thousands.
#
# Usage: scripts/nsys-verify.sh <out-name> [ENV=VAL ...]
set -euo pipefail

NAME="${1:?usage: nsys-verify.sh <out-name> [ENV=VAL ...]}"
shift

WORKTREE=/home/flocka/dsflash-combined
MODEL=/home/flocka/models/DeepSeek-V4-Flash-162B
DRAFTER=/home/flocka/models/DeepSeek-V4-Flash-0731-drafter
OUT="/tmp/$NAME"
LOG="$WORKTREE/serve-$NAME.log"

ENV_ARGS=(
  ATLAS_UNIFIED_MOE_LAYOUT=1
  ATLAS_V4_ATTN_NVFP4=1
  ATLAS_V4_ATTN_RELEASE_BF16=1
  ATLAS_MOE_MROW_PARTITION=1
  ATLAS_DSPARK_CAPTURE=1
)
for kv in "$@"; do ENV_ARGS+=("$kv"); done

cd "$WORKTREE"
: >"$LOG"
env "${ENV_ARGS[@]}" nsys profile \
  -t cuda \
  -s none \
  --cpuctxsw=none \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  -o "$OUT" \
  ./target/release/spark serve "$MODEL" \
    --port 8977 \
    --kv-cache-dtype fp8 \
    --lm-head-dtype fp8 \
    --gpu-memory-utilization 0.91 \
    --max-seq-len 1024 \
    --max-num-seqs 1 \
    --max-batch-size 1 \
    --max-prefill-tokens 1024 \
    --kv-cache-cap-tokens 1040 \
    --dflash --draft-model "$DRAFTER" --dflash-gamma 6 \
    >>"$LOG" 2>&1 &

echo "pid=$! log=$LOG out=$OUT.nsys-rep"
