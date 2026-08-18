#!/usr/bin/env bash
# Kernel-level profile of ONE PREFILL. Answers "which kernel eats the 97% of
# prefill wall that the MoE stage timers do not account for".
#
# Prefill never runs under CUDA graphs, so nsys sees real kernel launches with
# no graph opacity — unlike the decode profile, this needs no NO_GRAPH flag.
#
# Usage: scripts/nsys-prefill.sh <out-name> [ENV=VAL ...]
# Then drive one prefill (scratchpad/prefill_scan.py 512) and stop the server;
# `nsys stats --report cuda_gpu_kern_sum <out>.nsys-rep` ranks the kernels.
set -euo pipefail
NAME="${1:?usage: nsys-prefill.sh <out-name> [ENV=VAL ...]}"
shift || true
WORKTREE="${WORKTREE:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:?set MODEL to the DeepSeek-V4-Flash checkpoint directory}"
OUT="/tmp/$NAME"
LOG="$WORKTREE/serve-$NAME.log"

ENV_ARGS=( ATLAS_UNIFIED_MOE_LAYOUT=1 )
for kv in "$@"; do ENV_ARGS+=("$kv"); done

# Capture must NOT start at process launch: the ~6-minute model load issues
# enough CUDA activity to overrun nsys's buffers, and the prefill trace is then
# silently dropped (a run captured only 2 load-time quantize kernels). DELAY
# seconds skips the load so the trace holds prefill only. Drive prefills in a
# loop once the server answers so at least one lands inside the window.
DELAY="${DELAY:-400}"

cd "$WORKTREE"
: >"$LOG"
env "${ENV_ARGS[@]}" nsys profile -t cuda -s none --cpuctxsw=none \
  --delay "$DELAY" \
  --cuda-memory-usage=false --force-overwrite=true -o "$OUT" \
  ./target/release/spark serve "$MODEL" --port 8977 \
  --kv-cache-dtype fp8 --lm-head-dtype fp8 \
  --gpu-memory-utilization 0.96 --max-seq-len 4096 --max-num-seqs 1 \
  --max-batch-size 1 --max-prefill-tokens 4096 >>"$LOG" 2>&1 &
echo "pid=$! log=$LOG out=$OUT.nsys-rep"
