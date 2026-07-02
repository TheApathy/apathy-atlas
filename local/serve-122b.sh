#!/bin/bash
set -euo pipefail
PORT=${PORT:-8897}
pkill -9 -x spark 2>/dev/null || true; sleep 4
exec /home/flocka/atlas-src/target/release/spark serve \
  --model-from-path /home/flocka/models/Qwen3.5-122B-heretic-MTP-NVFP4 \
  --model-name q122b-atlas \
  --port "${PORT}" \
  --kernel-target qwen3.5-122b-a10b \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --max-seq-len 8192
