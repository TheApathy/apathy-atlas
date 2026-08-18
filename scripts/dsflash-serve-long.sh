#!/usr/bin/env bash
# Launch the combined-tree DeepSeek-V4-Flash-162B server in the tool-eval-bench
# configuration: DSpark gamma=5 + adaptive depth, 4096-token context.
#
# Runs with CWD = the combined repo so the `jinja-templates/` override dir
# resolves — the deepseek_v4.jinja tool-role branch lives there.
#
# Usage:  scripts/dsflash-serve-bench.sh <log-name> [gamma] [ENV=VAL ...]
# Only ONE server at a time (single shared GPU).
set -euo pipefail

NAME="${1:?usage: dsflash-serve-bench.sh <log-name> [gamma] [ENV=VAL ...]}"
GAMMA="${2:-5}"
shift 2 || shift 1 || true

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:?set MODEL to the DeepSeek-V4-Flash checkpoint directory}"
DRAFTER="${DRAFTER:?set DRAFTER to the DSpark drafter directory}"
PORT="${PORT:-8977}"
LOG="$REPO/serve-$NAME.log"

ENV_ARGS=(
  ATLAS_UNIFIED_MOE_LAYOUT=1
  ATLAS_DSPARK_CAPTURE=1
  ATLAS_DFLASH_ADAPTIVE=1
)
for kv in "$@"; do ENV_ARGS+=("$kv"); done

SPEC=()
if [ "$GAMMA" != "-" ]; then
  SPEC=(--dflash --draft-model "$DRAFTER" --dflash-gamma "$GAMMA")
fi

cd "$REPO"
{
  echo "serve: $REPO/target/release/spark"
  echo "cwd  : $REPO  (jinja-templates override dir)"
  echo "port : 127.0.0.1:$PORT  kv=fp8 lm_head=fp8 gpu_mem=0.96 max_seq=16384 batch=1"
  echo "env  : ${ENV_ARGS[*]}"
  echo "spec : ${SPEC[*]:-<none, plain decode>}"
} >"$LOG"

env "${ENV_ARGS[@]}" "$REPO/target/release/spark" serve "$MODEL" \
  --port "$PORT" \
  --kv-cache-dtype fp8 \
  --lm-head-dtype fp8 \
  --gpu-memory-utilization 0.96 \
  --max-seq-len 16384 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --max-prefill-tokens 16384 \
  "${SPEC[@]}" >>"$LOG" 2>&1 &

echo "pid=$! log=$LOG"
