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
# S1 of the 28-tok/s plan (docs/EXPERT-3BPW-PLAN.md): LMHEAD=nvfp4 halves the
# 529 MB/token lm_head stream. Default stays fp8 until the quality gate signs
# off on the argmax-flip risk.
LMHEAD="${LMHEAD:-fp8}"
# OOM guard (MB of free GPU memory to keep in reserve during load). The
# cuBLASLt prefill arm needs the BF16 mirrors resident (+8.06 GiB), which
# lands peak within ~1.4 GB of the default 4096 MB guard on this box —
# OOM_GUARD=2048 buys the headroom without changing any allocation.
OOM_GUARD="${OOM_GUARD:-4096}"
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
  echo "port : 127.0.0.1:$PORT  kv=fp8 lm_head=$LMHEAD gpu_mem=0.96 max_seq=4096 batch=1"
  echo "env  : ${ENV_ARGS[*]}"
  echo "spec : ${SPEC[*]:-<none, plain decode>}"
} >"$LOG"

env "${ENV_ARGS[@]}" "$REPO/target/release/spark" serve "$MODEL" \
  --port "$PORT" \
  --kv-cache-dtype fp8 \
  --lm-head-dtype "$LMHEAD" \
  --gpu-memory-utilization 0.96 \
  --max-seq-len 4096 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --max-prefill-tokens 4096 \
  --oom-guard-mb "$OOM_GUARD" \
  "${SPEC[@]}" >>"$LOG" 2>&1 &

echo "pid=$! log=$LOG"
