#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

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
MODEL="${MODEL:-/home/flocka/models/DeepSeek-V4-Flash-162B}"
DRAFTER="${DRAFTER:-/home/flocka/models/DeepSeek-V4-Flash-0731-drafter}"
BIN="${BIN:-$REPO/target/release/spark}"
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
LOG="${LOG:-$REPO/serve-$NAME.log}"

ENV_ARGS=(
  "_=$BIN"
  ATLAS_BENCH_ENGINE_USAGE=1
  ATLAS_UNIFIED_MOE_LAYOUT=1
  ATLAS_DSPARK_CAPTURE=1
  ATLAS_DFLASH_ADAPTIVE=1
)
for kv in "$@"; do ENV_ARGS+=("$kv"); done
for kv in "${ENV_ARGS[@]}"; do
  case "$kv" in
    _=*|[A-Za-z_][A-Za-z0-9_]*=*) ;;
    *) echo "invalid environment assignment: $kv" >&2; exit 2 ;;
  esac
done

SPEC=()
if [ "$GAMMA" != "-" ]; then
  SPEC=(--dflash --draft-model "$DRAFTER" --dflash-gamma "$GAMMA")
fi

CMD=(
  "$BIN" serve "$MODEL"
  --port "$PORT"
  --kv-cache-dtype fp8
  --lm-head-dtype "$LMHEAD"
  --gpu-memory-utilization 0.96
  --max-seq-len 4096
  --max-num-seqs 1
  --max-batch-size 1
  --max-prefill-tokens 4096
  --oom-guard-mb "$OOM_GUARD"
  "${SPEC[@]}"
)

if [ ! -x "$BIN" ]; then
  echo "benchmark binary is missing or not executable: $BIN" >&2
  exit 2
fi
if [ ! -f "$MODEL/config.json" ]; then
  echo "model config is missing: $MODEL/config.json" >&2
  exit 2
fi
if [ "$GAMMA" != "-" ] && [ ! -f "$DRAFTER/config.json" ]; then
  echo "drafter config is missing: $DRAFTER/config.json" >&2
  exit 2
fi

cd "$REPO"
ARGV_JSON=$(python3 - "${CMD[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)
ENV_JSON=$(python3 - "${ENV_ARGS[@]}" <<'PY'
import json
import sys
values = {}
for assignment in sys.argv[1:]:
    key, value = assignment.split("=", 1)
    values[key] = value
print(json.dumps(values, sort_keys=True))
PY
)
PLANNED_RECEIPT="$LOG.planned.receipt.json"
ACTIVE_RECEIPT="$LOG.receipt.json"
RECEIPT_ARGS=(
  --repo "$REPO"
  --binary "$BIN"
  --model "$MODEL"
  --argv-json "$ARGV_JSON"
  --environment-json "$ENV_JSON"
  --require-string deepseek_v4
  --require-string atlas_engine
  --require-string ATLAS_BENCH_ENGINE_USAGE
  --require-string ATLAS_UNIFIED_MOE_LAYOUT
  --require-string ATLAS_DSPARK_CAPTURE
  --require-string ATLAS_DFLASH_ADAPTIVE
  --output "$PLANNED_RECEIPT"
)
if [ "$GAMMA" != "-" ]; then
  RECEIPT_ARGS+=(--drafter "$DRAFTER")
fi
python3 "$REPO/scripts/benchmark_receipt.py" "${RECEIPT_ARGS[@]}" >/dev/null

{
  echo "serve: $BIN"
  echo "cwd  : $REPO  (jinja-templates override dir)"
  echo "port : 127.0.0.1:$PORT  kv=fp8 lm_head=$LMHEAD gpu_mem=0.96 max_seq=4096 batch=1"
  echo "env  : ${ENV_ARGS[*]}"
  echo "spec : ${SPEC[*]:-<none, plain decode>}"
  echo "planned receipt: $PLANNED_RECEIPT"
} >"$LOG"

if [ "${PRINT_CONFIG_ONLY:-0}" = 1 ]; then
  cat "$LOG"
  exit 0
fi

env "${ENV_ARGS[@]}" "${CMD[@]}" >>"$LOG" 2>&1 &
SERVER_PID=$!
if ! python3 "$REPO/scripts/benchmark_receipt.py" \
  --activate-receipt "$PLANNED_RECEIPT" \
  --pid "$SERVER_PID" \
  --base-url "http://127.0.0.1:$PORT" \
  --port "$PORT" \
  --ready-timeout "${ATLAS_BENCH_READY_TIMEOUT:-600}" \
  --output "$ACTIVE_RECEIPT" >/dev/null; then
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  echo "server failed active benchmark receipt verification; see $LOG" >&2
  exit 1
fi

echo "pid=$SERVER_PID log=$LOG receipt=$ACTIVE_RECEIPT"
