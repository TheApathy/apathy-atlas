#!/usr/bin/env bash
# DSpark speculative server on the dsflash-combined worktree binary.
#
# This is the config all 2026-08-04 measurements use: residency wins
# (RELEASE_BF16 frees 8.06 GiB so model+drafter fit), MROW MoE, the DSpark
# block drafter, adaptive spec with the low-gear fallback, and STEP_TIMING
# so verify/propose can be attributed per step.
#
# Usage:  scripts/cmb-serve.sh <log-name> [gamma] [ENV=VAL ...]
#   scripts/cmb-serve.sh chaindev 6
#   scripts/cmb-serve.sh chainhost 6 ATLAS_DSPARK_CHAIN_DEV=0
set -euo pipefail

NAME="${1:?usage: cmb-serve.sh <log-name> [gamma] [ENV=VAL ...]}"
GAMMA="${2:-6}"
shift 2 || shift 1 || true

WORKTREE="${WORKTREE:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:?set MODEL to the DeepSeek-V4-Flash checkpoint directory}"
DRAFTER="${DRAFTER:?set DRAFTER to the DSpark drafter directory}"
PORT="${PORT:-8977}"
LOG="$WORKTREE/serve-$NAME.log"

ENV_ARGS=(
  ATLAS_UNIFIED_MOE_LAYOUT=1
  ATLAS_V4_ATTN_NVFP4=1
  ATLAS_V4_ATTN_RELEASE_BF16=1
  ATLAS_MOE_MROW_PARTITION=1
  ATLAS_DSPARK_CAPTURE=1
  ATLAS_DFLASH_ADAPTIVE=1
  ATLAS_DFLASH_LOW_GEAR=1
  ATLAS_DFLASH_STEP_TIMING=1
  ATLAS_MOE_T_BLOCK=128
)
for kv in "$@"; do ENV_ARGS+=("$kv"); done

{
  echo "serve: $WORKTREE/target/release/spark"
  echo "model: $MODEL   drafter: $DRAFTER   gamma: $GAMMA"
  echo "env  : ${ENV_ARGS[*]}"
} >"$LOG"

cd "$WORKTREE"
env "${ENV_ARGS[@]}" ./target/release/spark serve "$MODEL" \
  --port "$PORT" \
  --kv-cache-dtype fp8 \
  --lm-head-dtype fp8 \
  --gpu-memory-utilization 0.91 \
  --max-seq-len 1024 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --max-prefill-tokens 1024 \
  --kv-cache-cap-tokens 1040 \
  --dflash --draft-model "$DRAFTER" --dflash-gamma "$GAMMA" >>"$LOG" 2>&1 &

echo "pid=$! log=$LOG"
