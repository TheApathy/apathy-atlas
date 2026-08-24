#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Launch a DeepSeek-V4 EXL3 K2/K3 checkpoint on one GB10.
#
# Usage:  scripts/exl3-serve.sh <log-name> [extra env assignments...]
#   MODEL=/models/ds4-k2 scripts/exl3-serve.sh k2 GAMMA=5 MAX_SEQ_LEN=1000000
#
# Only ONE server may run at a time (single shared GPU). Stop with
# scripts/dspark-stop.sh.
set -euo pipefail

NAME="${1:?usage: exl3-serve.sh <log-name> [ENV=VAL ...]}"
shift 1 || true

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:-/home/flocka/models/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1}"
PORT="${PORT:-8977}"
if [ -n "${GAMMA:-}" ]; then
  # DSpark's current capture is `[3, max_seq_len, hidden]` BF16. Keep the
  # speculative default resident until that absolute-position buffer becomes
  # a windowed ring; plain K2 can still expose the checkpoint-native 1M YaRN.
  DEFAULT_MAX_SEQ_LEN=131072
else
  DEFAULT_MAX_SEQ_LEN=1000000
fi
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$DEFAULT_MAX_SEQ_LEN}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.99}"
LOG="$REPO/serve-$NAME.log"

ENV_ARGS=(
  ATLAS_EXL3_PREFILL_CHUNK=1
  ATLAS_KV_OVERCOMMIT=1
)
if [ -n "${GAMMA:-}" ]; then
  # The embedded DSpark block conditions on the target's HC-mean capture.
  # Keep this before caller arguments so ATLAS_DSPARK_CAPTURE=0 remains an
  # explicit diagnostic override.
  ENV_ARGS+=(
    ATLAS_DSPARK_CAPTURE=1
    ATLAS_V4_ATTN_NVFP4=1
    ATLAS_V4_ATTN_RELEASE_BF16=1
    ATLAS_MOE_MROW_PARTITION=1
  )
fi
for kv in "$@"; do ENV_ARGS+=("$kv"); done

{
  echo "serve: $REPO/target/release/spark"
  echo "model: $MODEL (DeepSeek-V4 EXL3)"
  echo "port : 127.0.0.1:$PORT  kv=fp8 lm_head=fp8 gpu_mem=$GPU_MEMORY_UTILIZATION max_seq=$MAX_SEQ_LEN batch=1"
  echo "env  : ${ENV_ARGS[*]:-<none>}"
  echo "spec : ${GAMMA:+embedded DSpark gamma=$GAMMA}"
} >"$LOG"

# GAMMA=<n> arms the checkpoint's embedded five-token DSpark block.
SPEC=()
if [ -n "${GAMMA:-}" ]; then
  SPEC=(--dflash --draft-model "$MODEL" --dflash-gamma "$GAMMA")
fi

nohup env "${ENV_ARGS[@]}" "$REPO/target/release/spark" serve "$MODEL" \
  --port "$PORT" \
  --kv-cache-dtype fp8 \
  --lm-head-dtype fp8 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --oom-guard-mb "${OOM_GUARD:-512}" \
  "${SPEC[@]}" >>"$LOG" 2>&1 &

echo "pid=$! log=$LOG"
