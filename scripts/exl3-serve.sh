#!/usr/bin/env bash
# Launch the reference EXL3 3.0bpw checkpoint (216-expert REAP tp1) on a
# single GB10 — bring-up smoke for the trellis expert path (plan §5 S6).
#
# Usage:  scripts/exl3-serve.sh <log-name> [extra env assignments...]
#   scripts/exl3-serve.sh exl3-smoke
#   scripts/exl3-serve.sh exl3-chunk16 ATLAS_EXL3_PREFILL_CHUNK=16
#
# Differences vs dspark-serve.sh:
#   - MODEL points at /home/flocka/sparkinfer-ref/data/tp1 (quant_method
#     "exl3": routed experts EXL3 trellis, everything else FP8; config.json
#     model_type deepseek_v4, 216 experts, top-6).
#   - PLAIN decode only: the DSpark drafter ships in-dir (mtp.*) but the
#     EXL3 wide-verify twin is not built yet (forward_km declines → per-row),
#     so speculation would serialize; smoke without it.
#   - NO ATLAS_UNIFIED_MOE_LAYOUT: that flag is for the MXFP4 unified-T
#     assembly; EXL3 tiles load as-is (no transpose pass, null NVFP4 tables).
#   - OOM pre-flight: quant_method "exl3" auto-selects the 1.05x multiplier
#     (experts get no copies); ATLAS_PEAK_MEM_MULT overrides if needed.
#
# Only ONE server may run at a time (single shared GPU). Stop with
# scripts/dspark-stop.sh.
set -euo pipefail

NAME="${1:?usage: exl3-serve.sh <log-name> [ENV=VAL ...]}"
shift 1 || true

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:-/home/flocka/sparkinfer-ref/data/tp1}"
PORT="${PORT:-8977}"
LOG="$REPO/serve-$NAME.log"

ENV_ARGS=()
for kv in "$@"; do ENV_ARGS+=("$kv"); done

{
  echo "serve: $REPO/target/release/spark"
  echo "model: $MODEL (EXL3 3.0bpw routed experts, 216-expert REAP)"
  echo "port : 127.0.0.1:$PORT  kv=fp8 lm_head=fp8 gpu_mem=0.95 max_seq=1024 batch=1"
  echo "env  : ${ENV_ARGS[*]:-<none>}"
  echo "spec : <none, plain decode — EXL3 wide-verify twin not built>"
} >"$LOG"

# GAMMA=<n> arms DSpark. The tp1 checkpoint carries its own DSpark drafter
# (mtp.* in the carried-* shards), so --draft-model is the model dir itself.
# --max-batch-size 1 is required once the drafter is resident: at
# --max-seq-len 1024 the KV pool holds 7 sequences and the default 8 aborts.
SPEC=()
if [ -n "${GAMMA:-}" ]; then
  SPEC=(--dflash --draft-model "$MODEL" --dflash-gamma "$GAMMA")
fi

env "${ENV_ARGS[@]}" "$REPO/target/release/spark" serve "$MODEL" \
  --port "$PORT" \
  --kv-cache-dtype fp8 \
  --lm-head-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-seq-len 1024 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --max-prefill-tokens 1024 \
  --oom-guard-mb "${OOM_GUARD:-2048}" \
  "${SPEC[@]}" >>"$LOG" 2>&1 &

echo "pid=$! log=$LOG"
