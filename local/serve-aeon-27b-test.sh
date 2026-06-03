#!/bin/bash
# Atlas perf-gate bisect runner.
# Boots Atlas with baseline production gates + whatever ATLAS_* extras
# are exported BEFORE invoking this script.
set -euo pipefail
PORT=${PORT:-8889}
if pgrep -x spark >/dev/null 2>&1; then
  pkill -9 -x spark || true
  sleep 4
fi
if ss -tnlp 2>/dev/null | grep -q ":${PORT} "; then
  echo "[serve-test] ERROR: port ${PORT} still bound." >&2
  exit 1
fi
FREE_GB=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${FREE_GB:-0}" -lt 40 ]; then
  echo "[serve-test] ERROR: only ${FREE_GB} GB free." >&2
  exit 1
fi
# Production baseline gates
export ATLAS_GDN_PREFILL_TUNED=1
export ATLAS_LM_HEAD_BATCH3=1
export ATLAS_SSM_OUT_BATCH3=1
export ATLAS_PREFILL_FFN_FAST=1
export ATLAS_FFN_M16_TRANSPOSED=1
# Inherited from caller's env: ATLAS_TC_NVFP4_K3, ATLAS_FFN_DUAL_TUNED,
# ATLAS_SSM_BA_BATCHED, ATLAS_E2M1_GEMM_DOWN_ONLY, etc.
echo "[serve-test] gates active:"
env | grep -E "^ATLAS_" | sort
exec /home/flocka/atlas-src/target/release/spark serve \
  --model-from-path /home/flocka/models/AEON-Q36-27B-XS \
  --model-name aeon-27b \
  --port "${PORT}" \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-seq-len 16384 \
  --max-batch-size 8 \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --speculative \
  --num-drafts 2 \
  --mtp-quantization nvfp4 \
  --mtp-vocab 32000 \
  --ssm-cache-slots 16 \
  --ssm-checkpoint-interval 16 \
  --max-prefill-tokens 1024 \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas-src/local/warmup.txt
