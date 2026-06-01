#!/usr/bin/env bash
# Sweep ATLAS_*_KGAMMA gates on Atlas DFlash γ=16 path.
# Each gate stands alone; combined-winners test runs last.
# Output: appends one line per config to /tmp/kgamma_sweep.log
set -uo pipefail

ATLAS_BIN=/home/flocka/atlas-src/target/release/spark
MODELS=/home/flocka/models
PORT=8890
LOG=/tmp/kgamma_sweep.log

run_one() {
  local label="$1"; shift
  local env_args=("$@")
  echo "=== $label  env=${env_args[*]:-<none>} ===" | tee -a "$LOG"
  pkill -9 -x spark || true; sleep 4

  env ATLAS_DFLASH_DRAFT_CAP=16 ATLAS_DFLASH_QUANT=nvfp4 "${env_args[@]}" \
    "$ATLAS_BIN" serve \
      --model-from-path "$MODELS/AEON-Q36-27B-Full" \
      --model-name aeon-27b-dflash \
      --port "$PORT" \
      --kernel-target qwen3.6-27b \
      --gpu-memory-utilization 0.65 \
      --kv-cache-dtype fp8 \
      --max-seq-len 4096 \
      --max-batch-size 1 \
      --max-num-seqs 1 \
      --dflash \
      --draft-model "$MODELS/z-lab-Qwen3.6-27B-DFlash" \
      --dflash-gamma 16 \
      --dflash-quantization nvfp4 \
      --warmup-prompt /home/flocka/atlas-src/local/warmup.txt \
      > /tmp/atlas_kgamma_${label}.log 2>&1 &
  SPARK_PID=$!

  for i in $(seq 1 80); do
    if curl -sf "http://localhost:$PORT/v1/models" -o /dev/null 2>&1; then
      break
    fi
    sleep 5
  done

  python3 /home/flocka/atlas-src/bench/bench_dflash_quick.py "$PORT" "$label" 2 | tee -a "$LOG"

  kill -9 $SPARK_PID 2>/dev/null || true
  wait $SPARK_PID 2>/dev/null || true
  sleep 2
}

: > "$LOG"
# baseline (no kgamma)
run_one baseline
# individual gates
run_one g_FLASH_ATTN_KGAMMA          ATLAS_FLASH_ATTN_KGAMMA=1
run_one g_VECDEQUANT                 ATLAS_FLASH_ATTN_KGAMMA=1 ATLAS_KGAMMA_VECDEQUANT=1
run_one g_FFN_KGAMMA_M16             ATLAS_FFN_KGAMMA_M16=1 ATLAS_FFN_M16_TRANSPOSED=1
run_one g_DFLASH_FFN_KGAMMA          ATLAS_DFLASH_FFN_KGAMMA=1
run_one g_DFLASH_ATTN_KGAMMA         ATLAS_DFLASH_ATTN_KGAMMA=1
# combined winners (all on)
run_one g_ALL_KGAMMA \
  ATLAS_FLASH_ATTN_KGAMMA=1 ATLAS_KGAMMA_VECDEQUANT=1 \
  ATLAS_FFN_KGAMMA_M16=1 ATLAS_FFN_M16_TRANSPOSED=1 \
  ATLAS_DFLASH_FFN_KGAMMA=1 ATLAS_DFLASH_ATTN_KGAMMA=1

echo ""
echo "=== summary ==="
grep -E "^\[" "$LOG"
