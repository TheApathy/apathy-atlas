#!/usr/bin/env bash
# Kernel-level profile of ONE PLAIN decode workload (no drafter). Graphs ON.
set -euo pipefail
NAME="${1:?usage: nsys-plain.sh <out-name> [ENV=VAL ...]}"
shift || true
WORKTREE="${WORKTREE:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:?set MODEL to the DeepSeek-V4-Flash checkpoint directory}"
OUT="/tmp/$NAME"
LOG="$WORKTREE/serve-$NAME.log"
ENV_ARGS=( ATLAS_UNIFIED_MOE_LAYOUT=1 ATLAS_V4_ATTN_NVFP4=1 ATLAS_V4_ATTN_RELEASE_BF16=1 ATLAS_MOE_MROW_PARTITION=1 ATLAS_MOE_T_BLOCK=128 )
for kv in "$@"; do ENV_ARGS+=("$kv"); done
cd "$WORKTREE"
: >"$LOG"
env "${ENV_ARGS[@]}" nsys profile -t cuda -s none --cpuctxsw=none --cuda-memory-usage=false --force-overwrite=true -o "$OUT" \
  ./target/release/spark serve "$MODEL" --port 8977 --kv-cache-dtype fp8 --lm-head-dtype fp8 \
  --gpu-memory-utilization 0.91 --max-seq-len 1024 --max-num-seqs 1 --max-batch-size 1 \
  --max-prefill-tokens 1024 --kv-cache-cap-tokens 1040 >>"$LOG" 2>&1 &
echo "pid=$! log=$LOG out=$OUT.nsys-rep"
