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
    # WINNING CONFIG (validated 2026-05-07 spec sweep, 4-prompt mix):
    #
    #   A — MTP + ngram K=2:                    79-93 tok/s,  mean 87.9
    #   B — MTP K=2 only (NO ngram):           104-122 tok/s, mean 112  🏆
    #   C — MTP K=3 (num_drafts=2):             71-109 tok/s, mean 88.9
    #   D — MTP K=2 + self-spec:                62-64  tok/s, mean 62.8
    #
    # The ngram K=2 reject path was net-negative on diverse text
    # (each rejected verify costs 15.8ms = full forward pass).
    # Self-spec adds overhead with no benefit on this checkpoint.
    # K=3 is 21% slower than K=2 (Atlas docs predicted, confirmed).
    #
    # Result: +37.4% over closed-source Atlas Alpha 2.99 production
    # docker (81.5 tok/s baseline) on the same Huihui-Q36-abl checkpoint.
    exec "$SPARK" serve \
      --model-from-path "$MODELS/Huihui-NVFP4-Sehyo-MTP" \
      --model-name qwen36-abl \
      --port "$PORT" \
      --kernel-target qwen3.6-35b-a3b-abl \
      --gpu-memory-utilization 0.85 \
      --kv-cache-dtype nvfp4 \
      --kv-high-precision-layers auto \
      --max-seq-len 16384 \
      --enable-prefix-caching \
      --speculative \
      --num-drafts 1 \
      --mtp-quantization nvfp4 \
      --max-thinking-budget 768 \
      --warmup-prompt "$(dirname "$0")/warmup.txt"
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
      --kernel-target qwen3.5-122b-a10b-heretic \
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
    # Loads + serves with our patches but output quality is degraded —
    # Chinese/Japanese tokens from training distribution rather than
    # coherent English. Tested both auto-selected qwen3.6-27b kernel
    # target and forced qwen3.5-27b (this model has no MRoPE so 3.5
    # is correct); both produce garbled output at ~12 tok/s.
    #
    # This is NOT the loader-crash bug we fixed — the model loads
    # fully. It's a deeper kernel/weight-layout issue specific to
    # AEON-7's modelopt-format SSM hybrid 27B. Likely candidates:
    #   1. AEON-7's abliteration process produced corrupted SSM weights
    #   2. modelopt input_scale not threaded through quantized_auto
    #   3. shard_quantized_nvfp4 layout incompatibility for modelopt
    #
    # If you want to repro: spark serve with --kernel-target qwen3.5-27b
    exec "$SPARK" serve \
      --model-from-path "$MODELS/AEON-7-DFlash-Qwen3.5-27B-Uncensored-NVFP4" \
      --model-name aeon7-27b \
      --port "$PORT" \
      --kernel-target qwen3.5-27b \
      --gpu-memory-utilization 0.7 \
      --kv-cache-dtype nvfp4 \
      --kv-high-precision-layers auto \
      --max-seq-len 4096 \
      --enable-prefix-caching \
      --speculative \
      --num-drafts 1 \
      --mtp-quantization nvfp4
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
