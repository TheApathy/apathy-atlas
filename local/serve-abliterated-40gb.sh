#!/bin/bash
# qwen36-abl on 40 GB-class GPU (A6000 48GB, RTX 6000 Ada, etc.)
#
# Tested 2026-05-10: 101 tok/s decode, fits in ~40 GB GPU memory.
# Same per-stream speed as the 91 GB production config (decode is
# bandwidth-bound on the 22 GB weight read, not on KV size).
#
# Trade-offs vs production:
#   - max context 4096 (vs 16384)
#   - batch size 1, single concurrent sequence (vs 8/128)
#   - no Marconi prefix-cache snapshots (no multi-turn TTFT speedup)
#   - block-table-only prefix cache still works
#
# On a literal 40 GB GPU, change --gpu-memory-utilization 0.34 to 1.0.
# The 0.34 here simulates a 40 GB cap on our 119.7 GB GB10.
exec /home/flocka/atlas-src/target/release/spark serve \
  --model-from-path /home/flocka/models/Huihui-NVFP4-Sehyo-MTP \
  --model-name qwen36-abl \
  --port 8889 \
  --kernel-target qwen3.6-35b-a3b-abl \
  --gpu-memory-utilization 0.34 \
  --kv-cache-dtype turbo4 \
  --kv-high-precision-layers auto \
  --max-seq-len 4096 \
  --max-batch-size 1 \
  --max-num-seqs 1 \
  --enable-prefix-caching \
  --speculative \
  --num-drafts 1 \
  --mtp-quantization nvfp4 \
  --mtp-vocab 100000 \
  --ssm-cache-slots 0 \
  --swap-space-gb 0 \
  --oom-guard-mb 1024 \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas-src/local/warmup.txt
