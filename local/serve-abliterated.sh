#!/usr/bin/env bash
# Serve abliterated NVFP4 checkpoints with tuned defaults.
#
# Until per-checkpoint kernel-target override lands upstream, the
# easiest way to apply abliterated-tuned defaults is to (a) serve
# from the canonical kernel target, (b) override the few CLI knobs
# that matter, and (c) leave sampling tuning to the client (since
# spark serve has no --temperature flag — clients pass per-request).
#
# Usage:
#   ./serve-abliterated.sh huihui-q36       # Huihui-NVFP4-Sehyo-MTP (35B-A3B)
#   ./serve-abliterated.sh heretic-122b     # Qwen3.5-122B-heretic-MTP-NVFP4
#   ./serve-abliterated.sh aeon7-27b        # AEON-7-DFlash-Qwen3.5-27B (BLOCKED — see task #11)
#
# Common defaults:
#   --gpu-memory-utilization 0.85
#   --kv-cache-dtype fp8
#   --kv-high-precision-layers auto
#   --enable-prefix-caching
#   --speculative + --ngram-speculative + --num-drafts 1 (35B) or 2 (122B)
#   --mtp-quantization nvfp4
#   --max-thinking-budget tuned per checkpoint

set -euo pipefail

SPARK="${SPARK_BIN:-/path/to/atlas-src/target/release/spark}"
PORT="${SPARK_PORT:-8889}"
MODELS="${MODELS_DIR:-/path/to/models}"

case "${1:-}" in
  huihui-q36)
    exec "$SPARK" serve \
      --model-from-path "$MODELS/Huihui-NVFP4-Sehyo-MTP" \
      --model-name qwen36-abl \
      --port "$PORT" \
      --gpu-memory-utilization 0.85 \
      --kv-cache-dtype fp8 \
      --kv-high-precision-layers auto \
      --max-seq-len 16384 \
      --enable-prefix-caching \
      --speculative \
      --ngram-speculative \
      --num-drafts 1 \
      --mtp-quantization nvfp4 \
      --max-thinking-budget 768 \
      --fp8-kv-calibration-tokens 256
    ;;

  heretic-122b)
    # 76 GB on disk × 1.05 mult = ~80 GB peak load + ~4 GB live KV/scratch.
    # Total budget: 0.95 of 119.7 GB ≈ 113.7 GB minus inference reserve
    # (~13 GB) ≈ 100 GB. Tight. Drop oom-guard from default 4 GB to 2 GB
    # and trim max-seq-len from 16K to 8K (KV cache scales linearly).
    exec "$SPARK" serve \
      --model-from-path "$MODELS/Qwen3.5-122B-heretic-MTP-NVFP4" \
      --model-name qwen35-122b-heretic \
      --port "$PORT" \
      --gpu-memory-utilization 0.97 \
      --oom-guard-mb 512 \
      --max-batch-size 1 \
      --kv-cache-dtype fp8 \
      --kv-high-precision-layers auto \
      --max-seq-len 2048 \
      --max-prefill-tokens 2048 \
      --speculative \
      --ngram-speculative \
      --num-drafts 1 \
      --mtp-quantization nvfp4 \
      --max-thinking-budget 512 \
      --fp8-kv-calibration-tokens 128
    ;;

  aeon7-27b)
    echo "BLOCKED: AEON-7-DFlash-Qwen3.5-27B crashes at layer-2 transpose post loader patch." >&2
    echo "Tracking in task #11 (Diagnose AEON-7 layer-2 transpose crash)." >&2
    echo "Try after the downstream kernel/transpose fix lands." >&2
    exit 2
    ;;

  '' | -h | --help)
    grep -E '^# ' "$0" | sed 's/^# //'
    exit 0
    ;;

  *)
    echo "unknown checkpoint: $1" >&2
    echo "see: $0 --help" >&2
    exit 1
    ;;
esac
