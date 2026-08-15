#!/bin/bash
set -euo pipefail
PORT=${PORT:-8890}
pkill -9 -x spark 2>/dev/null || true; sleep 4
export ATLAS_TARGET_LMHEAD_VOCAB=120000
exec /home/flocka/atlas/src/target/release/spark serve \
  --model-from-path /home/flocka/models/AEON-27B-MTP \
  --model-name aeon-mtp --port "${PORT}" \
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --max-seq-len 8192 \
  --max-batch-size 1 \
  --speculative --num-drafts ${NDRAFTS:-1} --mtp-quantization nvfp4 --mtp-vocab 100000
