#!/usr/bin/env bash
# Launch the DeepSeek-V4-Flash-162B DSpark/dflash server on a single GB10.
#
# Usage:  scripts/dspark-serve.sh <log-name> [gamma] [extra env assignments...]
#   scripts/dspark-serve.sh graphs-on 2
#   scripts/dspark-serve.sh eager 2 ATLAS_DEBUG_NO_GRAPH=1
#   scripts/dspark-serve.sh plain  -            # '-' gamma => no speculation
#
# Only ONE server may run at a time (single shared GPU). Stop with
# scripts/dspark-stop.sh, which waits for the process to actually exit before
# returning — relaunching early makes the new server OOM.
set -euo pipefail

NAME="${1:?usage: dspark-serve.sh <log-name> [gamma] [ENV=VAL ...]}"
GAMMA="${2:-2}"
shift 2 || shift 1 || true

REPO=/home/flocka/deepseek-flash
MODEL=/home/flocka/models/DeepSeek-V4-Flash-162B
DRAFTER=/home/flocka/models/DeepSeek-V4-Flash-0731-drafter
PORT="${PORT:-8977}"
LOG="$REPO/serve-$NAME.log"

# ATLAS_UNIFIED_MOE_LAYOUT=1 is REQUIRED (the V4 expert weights are only
# assembled in the unified-T layout). ATLAS_DSPARK_CAPTURE=1 arms the hc-mean
# capture the block drafter conditions on — without it the drafter is starved.
ENV_ARGS=(
  ATLAS_UNIFIED_MOE_LAYOUT=1
  ATLAS_DSPARK_CAPTURE=1
)
for kv in "$@"; do ENV_ARGS+=("$kv"); done

SPEC=()
if [ "$GAMMA" != "-" ]; then
  SPEC=(--dflash --draft-model "$DRAFTER" --dflash-gamma "$GAMMA")
fi

{
  echo "serve: $REPO/target/release/spark"
  echo "model: $MODEL"
  echo "port : 127.0.0.1:$PORT  kv=fp8 lm_head=fp8 gpu_mem=0.95 max_seq=1024 batch=1"
  echo "env  : ${ENV_ARGS[*]}"
  echo "spec : ${SPEC[*]:-<none, plain decode>}"
} >"$LOG"

env "${ENV_ARGS[@]}" "$REPO/target/release/spark" serve "$MODEL" \
  --port "$PORT" \
  --kv-cache-dtype fp8 \
  --lm-head-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-seq-len 1024 \
  --max-num-seqs 1 \
  --max-prefill-tokens 1024 \
  "${SPEC[@]}" >>"$LOG" 2>&1 &

echo "pid=$! log=$LOG"
