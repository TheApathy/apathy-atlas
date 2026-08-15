#!/bin/bash
# AEON-27B tuned for AGENT use (Hermes/kimi/aider). Reliable + as snappy as possible.
# Pair with Hermes config: provider=atlas base_url=http://127.0.0.1:8890/v1 (IPv4!),
# terminal.persistent_shell=false (avoids cached /home/user cwd bug), mesh+sectools MCP off.
set -euo pipefail
PORT=${PORT:-8890}
pkill -9 -x spark 2>/dev/null || true; sleep 8
exec env ATLAS_DFLASH_ATTN_KGAMMA=1 ATLAS_DFLASH_QUANT=nvfp4 ATLAS_DFLASH_CTX_WINDOW=2048 \
  ATLAS_DFLASH_NOISE_ONLY=1 ATLAS_DFLASH_FFN_KGAMMA=1 ATLAS_DFLASH_GRAMMAR_MODE=verify \
  /home/flocka/atlas/src/target/release/spark serve \
  --model-from-path /home/flocka/models/AEON-Q36-27B-Full --model-name aeon-27b-dflash \
  --port "${PORT}" --kernel-target qwen3.6-27b --gpu-memory-utilization 0.80 --kv-cache-dtype fp8 \
  --max-seq-len 32768 --max-batch-size 1 --max-num-seqs 1 \
  --dflash --draft-model /home/flocka/atlas/dflash/retrain/drafter-v1 --dflash-gamma 16 --mtp-vocab 100000 \
  --dflash-quantization nvfp4 --max-thinking-budget 256 \
  --enable-prefix-caching --ssm-cache-slots 32 --ssm-checkpoint-interval 16
