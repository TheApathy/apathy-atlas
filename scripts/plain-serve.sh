#!/usr/bin/env bash
# Plain (no-drafter) single-stream decode server for the plain-residency A/B.
#
# Runs the codex plain-residency worktree binary, which carries the residency
# fixes (ATLAS_V4_ATTN_RELEASE_BF16, --kv-cache-cap-tokens) and the exact-output
# MoE precompute (ATLAS_MOE_MROW_PARTITION). Single stream, batch 1, 1024 ctx.
#
# Usage:  scripts/plain-serve.sh <log-name> [ENV=VAL ...]
#   scripts/plain-serve.sh plain-control                          # RELEASE_BF16 off
#   scripts/plain-serve.sh plain-release ATLAS_V4_ATTN_RELEASE_BF16=1
#   scripts/plain-serve.sh plain-moe ATLAS_V4_ATTN_RELEASE_BF16=1 ATLAS_MOE_MROW_PARTITION=1
#
# Only ONE server at a time on the shared GB10. Stop with scripts/dspark-stop.sh.
set -euo pipefail

NAME="${1:?usage: plain-serve.sh <log-name> [ENV=VAL ...]}"
shift 1 || true

WORKTREE="${WORKTREE:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:?set MODEL to the DeepSeek-V4-Flash checkpoint directory}"
PORT="${PORT:-8977}"
LOG="${LOG:-$WORKTREE/serve-$NAME.log}"

# ATLAS_UNIFIED_MOE_LAYOUT=1 is REQUIRED (V4 experts only assembled in unified-T).
# ATLAS_V4_ATTN_NVFP4=1 selects the fast NVFP4 attention projections. No DSpark
# capture flag: this is plain decode.
ENV_ARGS=(
  ATLAS_UNIFIED_MOE_LAYOUT=1
  ATLAS_V4_ATTN_NVFP4=1
)
for kv in "$@"; do ENV_ARGS+=("$kv"); done

{
  echo "serve: $WORKTREE/target/release/spark  (plain-residency worktree)"
  echo "model: $MODEL"
  echo "flags: gpu_mem=0.91 max_seq=1024 num_seqs=1 batch=1 prefill=1024 kv_cap=1040 kv=fp8 lm_head=fp8"
  echo "env  : ${ENV_ARGS[*]}"
  echo "spec : <none, plain decode>"
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
  --kv-cache-cap-tokens 1040 >>"$LOG" 2>&1 &

echo "pid=$! log=$LOG"
