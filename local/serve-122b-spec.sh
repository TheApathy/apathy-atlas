#!/bin/bash
# Qwen3.5-122B-A10B-heretic on single GB10 via Atlas + MTP-1 speculation.
# Tuned 2026-06-13: nd=1 + mtp-vocab 100k = 45/37/38 tok/s (vs 35 dense,
# vs vLLM-NVFP4 16.6). num_drafts>1 loses (pos-2 acceptance drops); 100k
# vocab covers >99% BPE, cuts draft-head cost. fp8 KV negligible loss.
set -euo pipefail
PORT=${PORT:-8897}
pkill -9 -x spark 2>/dev/null || true; sleep 4
export ATLAS_TARGET_LMHEAD_VOCAB=120000
exec /home/flocka/atlas-src/target/release/spark serve \
  --model-from-path /home/flocka/models/Qwen3.5-122B-heretic-MTP-NVFP4 \
  --model-name q122b-atlas --port "${PORT}" \
  --kernel-target qwen3.5-122b-a10b \
  --gpu-memory-utilization 0.93 --kv-cache-dtype fp8 --max-seq-len 8192 \
  --max-batch-size 1 \
  --speculative --num-drafts 1 --mtp-quantization nvfp4 --mtp-vocab 100000
