#!/usr/bin/env bash
# Sweep γ on Atlas DFlash with all kgamma gates ON, all 4 prompt classes.
# γ=16 (current default) is the "kgamma maximum"; smaller γ trades per-step
# work for higher accept-rate-per-attempt on natural language.
set -uo pipefail
# Repo root is derived from this script's location so the harness runs from
# any checkout. Override MODELS to point at your checkpoint directory.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODELS="${MODELS:-$HOME/models}"
ATLAS_BIN=$REPO_ROOT/target/release/spark
MODELS=$MODELS
PORT=8890
LOG=/tmp/gamma_sweep.log

run_gamma() {
  local gamma="$1"
  local label="g${gamma}"
  echo "=== γ=$gamma ===" | tee -a "$LOG"
  pkill -9 -x spark || true; sleep 4

  # Note: ATLAS_DFLASH_DRAFT_CAP must equal γ (else NaN bug per memory).
  # γ ∈ {2, 4} use distinct SSM kernels (wy2, wy4); γ=16 uses wy17.
  env ATLAS_DFLASH_DRAFT_CAP=$gamma ATLAS_DFLASH_CTX_WINDOW=512 ATLAS_DFLASH_QUANT=nvfp4 \
      ATLAS_FLASH_ATTN_KGAMMA=1 ATLAS_KGAMMA_VECDEQUANT=1 \
      ATLAS_FFN_KGAMMA_M16=1 ATLAS_FFN_M16_TRANSPOSED=1 \
      ATLAS_DFLASH_FFN_KGAMMA=1 ATLAS_DFLASH_ATTN_KGAMMA=1 \
      "$ATLAS_BIN" serve \
        --model-from-path "$MODELS/AEON-Q36-27B-Full" \
        --model-name aeon-27b-dflash --port $PORT \
        --kernel-target qwen3.6-27b \
        --gpu-memory-utilization 0.65 --kv-cache-dtype fp8 \
        --max-seq-len 4096 --max-batch-size 1 --max-num-seqs 1 \
        --dflash --draft-model "$MODELS/z-lab-Qwen3.6-27B-DFlash" \
        --dflash-gamma $gamma --dflash-quantization nvfp4 \
        --warmup-prompt $REPO_ROOT/local/warmup.txt \
        > /tmp/spark_g${gamma}.log 2>&1 &
  SPARK_PID=$!
  for i in $(seq 1 80); do
    if curl -sf "http://localhost:$PORT/v1/models" -o /dev/null 2>&1; then break; fi
    sleep 5
  done
  cd $REPO_ROOT/bench
  python3 bench_aeon27b.py "$PORT" "$label" 2 512 | tee -a "$LOG"
  kill -9 $SPARK_PID 2>/dev/null || true
  wait $SPARK_PID 2>/dev/null || true
  sleep 2
}

: > "$LOG"
for g in 2 4 8 16; do
  run_gamma $g
done
echo ""
echo "=== summary ==="
grep -E "^  (count100|code_long|essay|creative|MEAN)|=== γ" "$LOG"
