#!/usr/bin/env bash
# Ablation harness for serve-aeon-27b.sh ATLAS_* gates.
# Usage: ablate_gates.sh <label> [unset_var]
#
# Sets all 5 gates ON, then unsets the named one (if any), restarts spark,
# runs bench, writes /tmp/bench_aeon_<label>.json
set -euo pipefail

LABEL="${1:?label required}"
UNSET_VAR="${2:-}"

ATLAS_BIN=/home/flocka/atlas-src/target/release/spark
MODELS=/home/flocka/models
PORT="${PORT:-8889}"

# Kill anything on the port
pkill -9 -x spark || true
sleep 4
if ss -tnlp 2>/dev/null | grep -q ":${PORT} "; then
  echo "[ablate] ERROR: port ${PORT} still bound after pkill" >&2
  exit 1
fi

# Five gates ON by default
declare -A GATES=(
  [ATLAS_GDN_PREFILL_TUNED]=1
  [ATLAS_LM_HEAD_BATCH3]=1
  [ATLAS_SSM_OUT_BATCH3]=1
  [ATLAS_PREFILL_FFN_FAST]=1
  [ATLAS_FFN_M16_TRANSPOSED]=1
)
# If the user asks to unset one, drop it
if [[ -n "$UNSET_VAR" && -v "GATES[$UNSET_VAR]" ]]; then
  unset "GATES[$UNSET_VAR]"
fi

ENV_ARGS=()
for k in "${!GATES[@]}"; do
  ENV_ARGS+=("$k=${GATES[$k]}")
done

echo "[ablate $LABEL] env: ${ENV_ARGS[*]:-<none>}"

env "${ENV_ARGS[@]}" "$ATLAS_BIN" serve \
  --model-from-path "$MODELS/AEON-Q36-27B-XS" \
  --model-name aeon-27b \
  --port "$PORT" \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-seq-len 16384 \
  --max-batch-size 8 \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --speculative \
  --num-drafts 2 \
  --mtp-quantization nvfp4 \
  --mtp-vocab 32000 \
  --ssm-cache-slots 16 \
  --ssm-checkpoint-interval 16 \
  --max-prefill-tokens 1024 \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas-src/local/warmup.txt &
SPARK_PID=$!

# Wait for ready
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/v1/models" -o /dev/null 2>&1; then
    echo "[ablate $LABEL] ready in $((i*5))s"
    break
  fi
  sleep 5
done

# Run bench
cd /home/flocka/atlas-src/bench
python3 bench_aeon27b.py "$PORT" "$LABEL" 3 512 || true

# Stop spark
kill -9 $SPARK_PID 2>/dev/null || true
wait $SPARK_PID 2>/dev/null || true
sleep 2
