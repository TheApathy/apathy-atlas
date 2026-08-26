#!/usr/bin/env bash
# Full Qwen3.8-Flash-Next on one DGX Spark / GB10. No expert, layer, PLE-row,
# vocabulary, or vision pruning. The converted checkpoint keeps the 320M-row
# PLE table in sparse NVFP4 sidecars and stages only selected rows per token.
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the Atlas Flash-Next NVFP4-Offload checkpoint}"
BIN="${BIN:-target/release/spark}"
PORT="${PORT:-8898}"
MODEL_NAME="${MODEL_NAME:-qwen3.8-flash-next}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
PLE_CACHE_MB="${PLE_CACHE_MB:-512}"

[[ "$MAX_SEQ_LEN" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_SEQ_LEN must be a positive integer" >&2
  exit 2
}
[[ "$MAX_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_BATCH_SIZE must be a positive integer" >&2
  exit 2
}
[[ "$PLE_CACHE_MB" =~ ^[1-9][0-9]*$ ]] || {
  echo "PLE_CACHE_MB must be a positive integer" >&2
  exit 2
}

exec env ATLAS_PLE_CACHE_MB="$PLE_CACHE_MB" \
  "$BIN" serve \
    --model-from-path "$MODEL_DIR" \
    --model-name "$MODEL_NAME" \
    --kernel-target qwen3.8-flash-next \
    --port "$PORT" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --max-num-seqs "$MAX_BATCH_SIZE" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --ssm-cache-slots 8 \
    --kv-cache-dtype bf16 \
    --no-tui \
    "$@"
