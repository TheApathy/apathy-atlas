#!/usr/bin/env bash
# Serve from the *combined-residency* worktree binary.
#
# This is the branch that merges both campaigns:
#   - codex plain-residency: RELEASE_BF16 (frees 8.06 GiB), --kv-cache-cap-tokens,
#     native-FP8 V4 prefill, lane-owned MLA reduction, MROW_PARTITION MoE
#   - deepseek-flash: T_BLOCK=64 MoE occupancy, MLA KV-alias, rope-tail fix
#
# The residency flags are not optional on a 120 GB unified box: without
# --kv-cache-cap-tokens the budget-driven arm turns *all* free memory into KV
# blocks, even though one 1024-token stream needs only ~51.5 MB. That is what
# OOMs the loader once the 10.86 GB drafter and graph pools land on top.
#
# Usage:  scripts/cb-serve.sh <log-name> <gamma|-> [ENV=VAL ...]
#   scripts/cb-serve.sh spec-g6 6
#   scripts/cb-serve.sh plain   -
#   scripts/cb-serve.sh spec-g6-tb32 6 ATLAS_MOE_T_BLOCK=32
set -euo pipefail

NAME="${1:?usage: cb-serve.sh <log-name> <gamma|-> [ENV=VAL ...]}"
GAMMA="${2:-6}"
shift 2 || shift 1 || true

WORKTREE=/home/flocka/dsflash-combined
MODEL=/home/flocka/models/DeepSeek-V4-Flash-162B
DRAFTER=/home/flocka/models/DeepSeek-V4-Flash-0731-drafter
PORT="${PORT:-8977}"
GPU_MEM="${GPU_MEM:-0.91}"
KV_CAP="${KV_CAP:-1040}"   # 65 blocks @16 = 64 for the 1024-tok seq + 1 safety
LOG="$WORKTREE/serve-$NAME.log"

ENV_ARGS=(
  ATLAS_UNIFIED_MOE_LAYOUT=1
  ATLAS_V4_ATTN_NVFP4=1
  ATLAS_V4_ATTN_RELEASE_BF16=1
  ATLAS_MOE_MROW_PARTITION=1
)
SPEC=()
if [ "$GAMMA" != "-" ]; then
  ENV_ARGS+=(ATLAS_DSPARK_CAPTURE=1)
  SPEC=(--dflash --draft-model "$DRAFTER" --dflash-gamma "$GAMMA")
fi
for kv in "$@"; do ENV_ARGS+=("$kv"); done

{
  echo "serve: $WORKTREE/target/release/spark  (combined-residency)"
  echo "commit: $(cd "$WORKTREE" && git rev-parse --short HEAD)"
  echo "model: $MODEL"
  echo "gamma: $GAMMA   spec: ${SPEC[*]:-<none, plain decode>}"
  echo "flags: gpu_mem=$GPU_MEM kv_cap=$KV_CAP max_seq=1024 num_seqs=1 batch=1 prefill=1024"
  echo "env  : ${ENV_ARGS[*]}"
} >"$LOG"

cd "$WORKTREE"
env "${ENV_ARGS[@]}" ./target/release/spark serve "$MODEL" \
  --port "$PORT" \
  --kv-cache-dtype fp8 \
  --lm-head-dtype fp8 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-seq-len 1024 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --max-prefill-tokens 1024 \
  --kv-cache-cap-tokens "$KV_CAP" \
  "${SPEC[@]}" >>"$LOG" 2>&1 &

echo "pid=$! log=$LOG"
