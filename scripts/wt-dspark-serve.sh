#!/usr/bin/env bash
# DSpark/DFlash speculative server on the plain-residency worktree binary.
#
# Combines the codex residency + MoE wins with the DSpark drafter:
#   - ATLAS_V4_ATTN_RELEASE_BF16=1  frees 8.06 GiB so model+drafter fit
#   - ATLAS_MOE_MROW_PARTITION=1    exact-output +3.77% on the wide verify
#   - --dflash + drafter            speculative decode toward the ~39 tok/s ceiling
#
# Usage:  scripts/wt-dspark-serve.sh <log-name> [gamma] [ENV=VAL ...]
#   scripts/wt-dspark-serve.sh dspark-g6 6
#   scripts/wt-dspark-serve.sh dspark-g6-nomoe 6 ATLAS_MOE_MROW_PARTITION=0
set -euo pipefail

NAME="${1:?usage: wt-dspark-serve.sh <log-name> [gamma] [ENV=VAL ...]}"
GAMMA="${2:-6}"
shift 2 || shift 1 || true

WORKTREE=/home/flocka/deepseek-flash/codex-work/atlas-plain-exp
MODEL=/home/flocka/models/DeepSeek-V4-Flash-162B
DRAFTER=/home/flocka/models/DeepSeek-V4-Flash-0731-drafter
PORT="${PORT:-8977}"
LOG="/home/flocka/deepseek-flash/serve-$NAME.log"

# Residency + MoE wins ON by default; ATLAS_DSPARK_CAPTURE arms the hc-mean
# capture the block drafter conditions on. Overridable via trailing ENV=VAL.
ENV_ARGS=(
  ATLAS_UNIFIED_MOE_LAYOUT=1
  ATLAS_V4_ATTN_NVFP4=1
  ATLAS_V4_ATTN_RELEASE_BF16=1
  ATLAS_MOE_MROW_PARTITION=1
  ATLAS_DSPARK_CAPTURE=1
)
for kv in "$@"; do ENV_ARGS+=("$kv"); done

{
  echo "serve: $WORKTREE/target/release/spark  (plain-residency worktree)"
  echo "model: $MODEL   drafter: $DRAFTER   gamma: $GAMMA"
  echo "flags: gpu_mem=0.91 max_seq=1024 num_seqs=1 batch=1 prefill=1024 kv_cap=1040 kv=fp8 lm_head=fp8"
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
