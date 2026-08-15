#!/usr/bin/env bash
# Per-metric sweep — bench candidates, log per-prompt to /tmp/per_metric_sweep.log.
# Strict no-regression rule: any config that loses on essay OR creative vs the
# K=3 MTP reference (22.75 / 21.70) gets a ❌ marker.
set -uo pipefail
ATLAS_BIN=/home/flocka/atlas/src/target/release/spark
MODELS=/home/flocka/models
PORT=8890
LOG=/tmp/per_metric_sweep.log

run() {
  local label="$1"; shift
  echo "=== $label ===" | tee -a "$LOG"
  pkill -9 -x spark 2>/dev/null || true; sleep 4
  nohup "$@" > /tmp/spark_$label.log 2>&1 &
  SPARK_PID=$!
  for i in $(seq 1 80); do
    if curl -sf "http://localhost:$PORT/v1/models" -o /dev/null 2>&1; then break; fi
    sleep 5
  done
  cd /home/flocka/atlas/src/bench
  python3 bench_aeon27b.py "$PORT" "$label" 2 512 2>&1 | tee -a "$LOG" | tail -7
  kill -9 $SPARK_PID 2>/dev/null || true
  wait $SPARK_PID 2>/dev/null || true
  sleep 2
}

: > "$LOG"
echo "=== Reference: Atlas K=3 MTP (current production default) ===" | tee -a "$LOG"

run k3_mtp_baseline env "$ATLAS_BIN" serve \
  --model-from-path "$MODELS/AEON-Q36-27B-XS" \
  --model-name aeon-27b --port $PORT \
  --kernel-target qwen3.6-27b --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 --max-seq-len 16384 \
  --max-batch-size 8 --max-num-seqs 8 \
  --enable-prefix-caching --speculative --num-drafts 2 \
  --mtp-quantization nvfp4 --mtp-vocab 32000 \
  --ssm-cache-slots 16 --ssm-checkpoint-interval 16 --max-prefill-tokens 1024 \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt

run k3_mtp_fp8quant env "$ATLAS_BIN" serve \
  --model-from-path "$MODELS/AEON-Q36-27B-XS" \
  --model-name aeon-27b --port $PORT \
  --kernel-target qwen3.6-27b --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 --max-seq-len 16384 \
  --enable-prefix-caching --speculative --num-drafts 2 \
  --mtp-quantization fp8 --mtp-vocab 32000 \
  --ssm-cache-slots 16 --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt

run k3_mtp_vocab100k env "$ATLAS_BIN" serve \
  --model-from-path "$MODELS/AEON-Q36-27B-XS" \
  --model-name aeon-27b --port $PORT \
  --kernel-target qwen3.6-27b --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 --max-seq-len 16384 \
  --enable-prefix-caching --speculative --num-drafts 2 \
  --mtp-quantization nvfp4 --mtp-vocab 100000 \
  --ssm-cache-slots 16 --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt

run k3_mtp_prefillfast env ATLAS_PREFILL_FFN_FAST=1 ATLAS_FFN_M16_TRANSPOSED=1 \
  "$ATLAS_BIN" serve \
  --model-from-path "$MODELS/AEON-Q36-27B-XS" \
  --model-name aeon-27b --port $PORT \
  --kernel-target qwen3.6-27b --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 --max-seq-len 16384 \
  --enable-prefix-caching --speculative --num-drafts 2 \
  --mtp-quantization nvfp4 --mtp-vocab 32000 \
  --ssm-cache-slots 16 --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt

# DFlash with smaller γ + v2 drafter
run dflash_g4_v2 env ATLAS_DFLASH_DRAFT_CAP=4 ATLAS_DFLASH_QUANT=nvfp4 \
  ATLAS_FLASH_ATTN_KGAMMA=1 ATLAS_KGAMMA_VECDEQUANT=1 \
  ATLAS_FFN_KGAMMA_M16=1 ATLAS_FFN_M16_TRANSPOSED=1 \
  ATLAS_DFLASH_FFN_KGAMMA=1 ATLAS_DFLASH_ATTN_KGAMMA=1 \
  "$ATLAS_BIN" serve \
  --model-from-path "$MODELS/AEON-Q36-27B-Full" \
  --model-name aeon-27b-dflash --port $PORT \
  --kernel-target qwen3.6-27b --gpu-memory-utilization 0.65 \
  --kv-cache-dtype fp8 --max-seq-len 4096 \
  --max-batch-size 1 --max-num-seqs 1 \
  --dflash --draft-model "$MODELS/z-lab-Qwen3.6-27B-DFlash-aeon-v2" \
  --dflash-gamma 4 --dflash-quantization nvfp4 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt

run dflash_g4_orig env ATLAS_DFLASH_DRAFT_CAP=4 ATLAS_DFLASH_QUANT=nvfp4 \
  ATLAS_FLASH_ATTN_KGAMMA=1 ATLAS_KGAMMA_VECDEQUANT=1 \
  ATLAS_FFN_KGAMMA_M16=1 ATLAS_FFN_M16_TRANSPOSED=1 \
  ATLAS_DFLASH_FFN_KGAMMA=1 ATLAS_DFLASH_ATTN_KGAMMA=1 \
  "$ATLAS_BIN" serve \
  --model-from-path "$MODELS/AEON-Q36-27B-Full" \
  --model-name aeon-27b-dflash --port $PORT \
  --kernel-target qwen3.6-27b --gpu-memory-utilization 0.65 \
  --kv-cache-dtype fp8 --max-seq-len 4096 \
  --max-batch-size 1 --max-num-seqs 1 \
  --dflash --draft-model "$MODELS/z-lab-Qwen3.6-27B-DFlash" \
  --dflash-gamma 4 --dflash-quantization nvfp4 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt

run dflash_g8_orig env ATLAS_DFLASH_DRAFT_CAP=8 ATLAS_DFLASH_QUANT=nvfp4 \
  ATLAS_FLASH_ATTN_KGAMMA=1 ATLAS_KGAMMA_VECDEQUANT=1 \
  ATLAS_FFN_KGAMMA_M16=1 ATLAS_FFN_M16_TRANSPOSED=1 \
  ATLAS_DFLASH_FFN_KGAMMA=1 ATLAS_DFLASH_ATTN_KGAMMA=1 \
  "$ATLAS_BIN" serve \
  --model-from-path "$MODELS/AEON-Q36-27B-Full" \
  --model-name aeon-27b-dflash --port $PORT \
  --kernel-target qwen3.6-27b --gpu-memory-utilization 0.65 \
  --kv-cache-dtype fp8 --max-seq-len 4096 \
  --max-batch-size 1 --max-num-seqs 1 \
  --dflash --draft-model "$MODELS/z-lab-Qwen3.6-27B-DFlash" \
  --dflash-gamma 8 --dflash-quantization nvfp4 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt

echo ""
echo "=== SWEEP COMPLETE ===" | tee -a "$LOG"
echo "results: $LOG"
