#!/usr/bin/env bash
# Single-stream decode configuration for Qwen3.8-27B on GB10 (DGX Spark).
#
# This is the exact flag set behind the 63.9 tok/s median reported in the PR.
# Every non-obvious knob is justified in docs/QWEN38_PERFORMANCE_RECIPE.md §2;
# the short version is inline below so this file stands alone.
#
#   MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 \
#   DRAFT=/path/to/dflash-drafter \
#   ./bench/qwen38-gb10/serve.sh
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-27B NVFP4 checkpoint}"
: "${DRAFT:?set DRAFT to the DFlash drafter directory}"
BIN="${BIN:-target/release/spark}"
PORT="${PORT:-8896}"

# gamma 15 is optimal AND maximal: the drafter's block_size=16 yields
# trained_drafts = block_size - 1. gamma 20 is refused by the loader.
# Measured 10/12/15 -> 56.71/59.30/62.92 tok/s, monotonic into the ceiling.
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
