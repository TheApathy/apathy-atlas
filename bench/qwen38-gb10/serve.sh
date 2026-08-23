#!/usr/bin/env bash
# Single-stream decode configuration for Qwen3.8-27B on GB10 (DGX Spark).
#
# Single-stream speed profile. See README.md in this directory for build
# requirements, drafter selection, and measurement caveats.
#
#   MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 \
#   DRAFT=/path/to/dflash-drafter \
#   ./bench/qwen38-gb10/serve.sh
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-27B NVFP4 checkpoint}"
: "${DRAFT:?set DRAFT to the DFlash drafter directory}"
BIN="${BIN:-target/release/spark}"
PORT="${PORT:-8896}"

# Derive gamma from the drafter's block_size: trained_drafts = block_size - 1.
# 15 corresponds to a block_size=16 drafter. Asking for more than the drafter
# was trained for is refused by the loader.
GAMMA="${GAMMA:-15}"

exec env \
  ATLAS_FFN_TC=1 \
  ATLAS_SSM_PROJ_TC=1 \
  ATLAS_LM_HEAD_TC=1 \
  ATLAS_ACCEPT_FAST_ARGMAX=1 \
  ATLAS_PREFILL_PROJ_FAST=0 \
  ATLAS_PREFILL_FFN_FAST=0 \
  ATLAS_DFLASH_FREE_SLOTS=0 \
  ATLAS_SSM_GDN_SEQ_PERSISTENT=1 \
  ATLAS_SSM_GDN_LAZY=1 \
  ATLAS_ATTN_QKV_FUSED=1 \
  ATLAS_DFLASH_DRAFT_SPLITK=8 \
  ATLAS_WEIGHT_CACHE=1 \
  ATLAS_DDTREE_MAX_NODES=$((GAMMA + 1)) \
  "$BIN" serve \
    --model-from-path "$MODEL_DIR" \
    --model-name qwen38 --port "$PORT" \
    --kernel-target gb10 \
    --gpu-memory-utilization 0.55 \
    --kv-cache-dtype bf16 \
    --kv-high-precision-layers 0 \
    --max-seq-len 8192 \
    --max-batch-size 1 --max-num-seqs 1 \
    --dflash --draft-model "$DRAFT" \
    --dflash-gamma "$GAMMA" --mtp-vocab 96000 \
    --disable-confidence-early-stop \
    --disable-simhash-watchdog \
    --disable-loop-watchdog
