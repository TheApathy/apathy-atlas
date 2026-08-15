#!/usr/bin/env bash
# Run Atlas serve with arbitrary CLI overrides + env, bench, stop.
# Usage:
#   run_config.sh <label> <kv_dtype> <num_drafts> [extra_env]
set -euo pipefail

LABEL="${1:?label}"
KV_DTYPE="${2:-fp8}"
NUM_DRAFTS="${3:-2}"
EXTRA_ENV="${4:-}"

ATLAS_BIN=/home/flocka/atlas/src/target/release/spark
PORT="${PORT:-8889}"

pkill -9 -x spark || true
sleep 4

CMD=(env)
if [[ -n "$EXTRA_ENV" ]]; then
  IFS=',' read -ra envs <<< "$EXTRA_ENV"
  for e in "${envs[@]}"; do
    CMD+=("$e")
  done
fi

echo "[run_config $LABEL] kv=$KV_DTYPE drafts=$NUM_DRAFTS env=$EXTRA_ENV"

"${CMD[@]}" "$ATLAS_BIN" serve \
  --model-from-path /home/flocka/models/AEON-Q36-27B-XS \
  --model-name aeon-27b \
  --port "$PORT" \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype "$KV_DTYPE" \
  --max-seq-len 16384 \
  --enable-prefix-caching \
  --speculative \
  --num-drafts "$NUM_DRAFTS" \
  --mtp-quantization nvfp4 \
  --mtp-vocab 32000 \
  --ssm-cache-slots 16 \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt &
SPARK_PID=$!

for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/v1/models" -o /dev/null 2>&1; then
    echo "[run_config $LABEL] ready in $((i*5))s"
    break
  fi
  sleep 5
done

cd /home/flocka/atlas/src/bench
python3 bench_aeon27b.py "$PORT" "$LABEL" 3 512 || true

kill -9 $SPARK_PID 2>/dev/null || true
wait $SPARK_PID 2>/dev/null || true
sleep 2
