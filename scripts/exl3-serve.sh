#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Launch a DeepSeek-V4 EXL3 K2/K3 checkpoint on one GB10.
#
# Usage:  scripts/exl3-serve.sh <log-name> [extra env assignments...]
#   MODEL=/models/ds4-k2 DSPARK_TOKENS=5 scripts/exl3-serve.sh k2
#   MODEL=/models/ds4-k2 DRAFTER=/models/ds4-dflash2 GAMMA=16 \
#     scripts/exl3-serve.sh dflash2
#
# Only ONE server may run at a time (single shared GPU). Stop with
# scripts/dspark-stop.sh.
set -euo pipefail

NAME="${1:?usage: exl3-serve.sh <log-name> [ENV=VAL ...]}"
shift 1 || true

# Atlas names the verify width gamma; one row is the target bonus. Expose the
# checkpoint-native vocabulary here: five DSpark proposal tokens map to gamma
# six. GAMMA remains available as a raw expert/debug override.
if [ -n "${DSPARK_TOKENS:-}" ]; then
  if [ -n "${GAMMA:-}" ]; then
    echo "set either DSPARK_TOKENS or raw GAMMA, not both" >&2
    exit 2
  fi
  case "$DSPARK_TOKENS" in
    ''|*[!0-9]*|0) echo "DSPARK_TOKENS must be a positive integer" >&2; exit 2 ;;
  esac
  GAMMA=$((DSPARK_TOKENS + 1))
fi

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:-/home/flocka/models/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1}"
DRAFTER="${DRAFTER:-$MODEL}"
# Same-dir checkpoints are the embedded DSpark layout. A separate checkpoint
# defaults to the native DFlash2 ABI; DRAFTER_KIND remains overrideable for a
# separately packaged DSpark checkpoint.
DRAFTER_KIND="${DRAFTER_KIND:-auto}"
if [ "$DRAFTER_KIND" = auto ]; then
  if [ "$DRAFTER" = "$MODEL" ]; then
    DRAFTER_KIND=dspark
  else
    DRAFTER_KIND=dflash2
  fi
fi
case "$DRAFTER_KIND" in
  dspark|dflash2) ;;
  *) echo "DRAFTER_KIND must be auto, dspark, or dflash2" >&2; exit 2 ;;
esac
PORT="${PORT:-8977}"
if [ -n "${GAMMA:-}" ] && [ "$DRAFTER_KIND" = dflash2 ]; then
  # Native DFlash2 still owns a per-sequence five-layer context accumulator.
  # Keep its default bounded until that accumulator is windowed.
  DEFAULT_MAX_SEQ_LEN=131072
else
  # Plain K2 and embedded DSpark both use the checkpoint-native 1M YaRN. The
  # DSpark history is a fixed 256-row circular buffer, not max-seq-sized.
  DEFAULT_MAX_SEQ_LEN=1048576
fi
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$DEFAULT_MAX_SEQ_LEN}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.99}"
LOG="${LOG:-$REPO/serve-$NAME.log}"

ENV_ARGS=(
  ATLAS_EXL3_PREFILL_CHUNK=1
  ATLAS_KV_OVERCOMMIT=1
)
# Plain-target training capture. This deliberately reuses the mHC-aware
# DSpark dump primitive with five DFlash target layers: DeepSeek keeps its
# residual in four FP32 highway streams between layers, so copying the normal
# single-stream hidden scratch would record stale data. Keep this server
# private; the capture driver rejects any interleaved request records.
if [ -n "${DFLASH_TRAIN_DUMP:-}" ]; then
  if [ -n "${GAMMA:-}" ]; then
    echo "DFLASH_TRAIN_DUMP requires a plain target run (unset GAMMA/DSPARK_TOKENS)" >&2
    exit 2
  fi
  ENV_ARGS+=(
    "ATLAS_DSPARK_DUMP=$DFLASH_TRAIN_DUMP"
    "ATLAS_DSPARK_CAPTURE_LAYERS=0,10,21,31,42"
  )
fi
if [ -n "${GAMMA:-}" ]; then
  ENV_ARGS+=(
    ATLAS_V4_ATTN_NVFP4=1
    ATLAS_V4_ATTN_RELEASE_BF16=1
    ATLAS_MOE_MROW_PARTITION=1
    ATLAS_V4_DECODE_FUSED=1
    ATLAS_VERIFY_GEMV_V2=1
    ATLAS_DFLASH_ADAPTIVE=1
    ATLAS_DFLASH_LOW_GEAR=1
  )
  if [ "$DRAFTER_KIND" = dspark ]; then
    # DSpark conditions on the target's HC-mean capture. Native DFlash2 uses
    # its config-declared target-layer captures and must not pay for this
    # separate three-row DSpark buffer.
    ENV_ARGS+=(ATLAS_DSPARK_CAPTURE=1)
  fi
fi
for kv in "$@"; do ENV_ARGS+=("$kv"); done

{
  echo "serve: $REPO/target/release/spark"
  echo "model: $MODEL (DeepSeek-V4 EXL3)"
  echo "draft: $DRAFTER ($DRAFTER_KIND)"
  echo "port : 127.0.0.1:$PORT  kv=fp8 lm_head=fp8 gpu_mem=$GPU_MEMORY_UTILIZATION max_seq=$MAX_SEQ_LEN batch=1"
  echo "env  : ${ENV_ARGS[*]:-<none>}"
  if [ -n "${GAMMA:-}" ]; then
    echo "spec : $DRAFTER_KIND gamma=$GAMMA verify rows, $((GAMMA - 1)) drafts"
  else
    echo "spec : <none>"
  fi
} >"$LOG"

if [ "${PRINT_CONFIG_ONLY:-0}" = 1 ]; then
  cat "$LOG"
  exit 0
fi

# DSPARK_TOKENS=5 is the normal embedded-block interface. GAMMA=<n> is the
# raw verify-width override retained for controlled ablations.
SPEC=()
if [ -n "${GAMMA:-}" ]; then
  SPEC=(--dflash --draft-model "$DRAFTER" --dflash-gamma "$GAMMA")
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
