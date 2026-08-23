#!/bin/bash
# CROSS-SEQ BATCHED DFLASH VERIFY (#39) — serve /tmp/spark_v39 with the
# batched verify gated ON, concurrency-capable (BATCH>=2). Mirrors
# serve-aeon-27b-dflash.sh env but uses the v39 binary + distinct port.
set -euo pipefail

PORT=${PORT:-8899}
BINARY=${BINARY:-/tmp/spark_v39}

# GPU etiquette: never boot over a foreign server holding the 128GB pool.
if pgrep -x spark >/dev/null 2>&1 || pgrep -af 'spark_.*serve' | grep -qv 'grep'; then
  echo "[v39] a spark server is already running; refusing to boot." >&2
  pgrep -af 'spark' | grep -v grep | grep -v claude >&2 || true
  exit 1
fi
if ss -tnlp 2>/dev/null | grep -q ":${PORT} "; then
  echo "[v39] ERROR: port ${PORT} still bound" >&2
  exit 1
fi

export ATLAS_DFLASH_DRAFT_CAP=${ATLAS_DFLASH_DRAFT_CAP:-16}
export ATLAS_LM_HEAD_T="${ATLAS_LM_HEAD_T:-1}"
export ATLAS_DFLASH_CTX_WINDOW=${ATLAS_DFLASH_CTX_WINDOW:-4096}
export ATLAS_DFLASH_QUANT=${ATLAS_DFLASH_QUANT:-nvfp4}
export ATLAS_FFN_KGAMMA_M16=${ATLAS_FFN_KGAMMA_M16:-1}
export ATLAS_FFN_M16_TRANSPOSED=${ATLAS_FFN_M16_TRANSPOSED:-1}
export ATLAS_FFN_KGAMMA_M128=${ATLAS_FFN_KGAMMA_M128:-1}
# NEW #39: the WIDE m128 window (32<M<=256) that batches c*17 FFN rows.
export ATLAS_FFN_KGAMMA_WIDE=${ATLAS_FFN_KGAMMA_WIDE:-1}
# NEW #39: enable the cross-seq batched verify step.
export ATLAS_DFLASH_BATCHED_VERIFY=${ATLAS_DFLASH_BATCHED_VERIFY:-1}
export ATLAS_DFLASH_BATCHED_VERIFY_LOG=${ATLAS_DFLASH_BATCHED_VERIFY_LOG:-1}
export ATLAS_DFLASH_FFN_KGAMMA=${ATLAS_DFLASH_FFN_KGAMMA:-1}
export ATLAS_DFLASH_ATTN_KGAMMA=${ATLAS_DFLASH_ATTN_KGAMMA:-1}
export ATLAS_ATTN_QKV_SPLITK=${ATLAS_ATTN_QKV_SPLITK:-4}
export ATLAS_FFN_DOWN_SPLITK=${ATLAS_FFN_DOWN_SPLITK:-4}
export ATLAS_ATTN_QKV_BATCHED=${ATLAS_ATTN_QKV_BATCHED:-1}
# Lossless default.  Values greater than one plus lazy commit are an
# experimental replay path and have produced nondeterministic GDN state.
export ATLAS_WY17_LAZY=${ATLAS_WY17_LAZY:-1}
export ATLAS_WY17_LAZY_COMMIT=${ATLAS_WY17_LAZY_COMMIT:-0}
export ATLAS_DISABLE_TREE_WY=${ATLAS_DISABLE_TREE_WY:-1}
# Keep SAM/tree OFF for the batched path (flat-chain only in v1).
export ATLAS_DFLASH_SAM=${ATLAS_DFLASH_SAM:-0}
export ATLAS_THINK_SPEC=${ATLAS_THINK_SPEC:-0}
export ATLAS_TC_NVFP4_M16=0
export ATLAS_TC_NVFP4_M16_MS_ATTN=0

DRAFT_MODEL=${DRAFT_MODEL:-/home/flocka/atlas/dflash/retrain/v5-ckpt-goheavy/epoch_2_step_16732}
BATCH=${BATCH:-8}

echo "[v39] booting ${BINARY} port ${PORT} BATCH=${BATCH} batched_verify=${ATLAS_DFLASH_BATCHED_VERIFY}"
exec "${BINARY}" serve \
  --model-from-path "${TARGET_MODEL:-/home/flocka/models/AEON-Q36-27B-Full}" \
  --model-name aeon-27b-v39 \
  --port ${PORT} \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization ${UTIL:-0.75} \
  --kv-cache-dtype "${KV_DTYPE:-fp8}" \
  --max-seq-len ${MAX_SEQ_LEN:-8192} \
  --max-batch-size ${BATCH} \
  --max-num-seqs ${BATCH} \
  --dflash \
  --draft-model "${DRAFT_MODEL}" \
  --dflash-gamma ${DFLASH_GAMMA:-16} \
  --mtp-vocab "${MTP_VOCAB:-96000}" \
  --dflash-quantization "$ATLAS_DFLASH_QUANT" \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt
