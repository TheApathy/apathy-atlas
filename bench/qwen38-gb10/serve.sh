#!/usr/bin/env bash
# Qwen3.8-27B single-stream decode profile for GB10 (DGX Spark).
#
# This reproduces the environment the reported median was measured under. It is
# the full flag set, not a reduced one: an earlier version of this file set 13
# ATLAS_* variables against the measured run's 38 and omitted
# --dflash-quantization, whose clap default is bf16 (~134 ms/step) rather than
# nvfp4. That version could not reach the reported number.
#
# See README.md in this directory for build requirements, drafter selection,
# and measurement caveats. Comments live above the exec: a comment inside a
# line-continuation block silently truncates the argument list.
#
#   MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 \
#   DRAFT=/path/to/dflash-drafter \
#   ./bench/qwen38-gb10/serve.sh
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-27B NVFP4 checkpoint}"
: "${DRAFT:?set DRAFT to the DFlash drafter directory}"
BIN="${BIN:-target/release/spark}"
PORT="${PORT:-8896}"

# --kernel-target names a MODEL directory under kernels/<hw>/, NOT the hardware
# directory. Passing `gb10` here is rejected: the loader errors with the list of
# available targets. The named target must also have been compiled in, which is
# what ATLAS_TARGET_MODEL=qwen3.8-27b at build time does.
KTARGET="${KTARGET:-qwen3.8-27b}"

# Derive gamma from the drafter's block_size: trained_drafts = block_size - 1.
# 15 corresponds to a block_size=16 drafter. Asking for more than the drafter
# was trained for is refused by the loader.
GAMMA="${GAMMA:-15}"

exec env \
  ATLAS_FFN_TC=1 \
  ATLAS_SSM_PROJ_TC=1 \
  ATLAS_LM_HEAD_TC=1 \
  ATLAS_LM_HEAD_T=1 \
  ATLAS_ACCEPT_FAST_ARGMAX=1 \
  ATLAS_PREFILL_PROJ_FAST=0 \
  ATLAS_PREFILL_FFN_FAST=0 \
  ATLAS_SSM_GDN_SEQ_PERSISTENT=1 \
  ATLAS_SSM_GDN_LAZY=1 \
  ATLAS_ATTN_QKV_FUSED=1 \
  ATLAS_ATTN_QKV_EXACT_STRIDED=1 \
  ATLAS_ATTN_QKV_BATCHED=0 \
  ATLAS_ATTN_QKV_SPLITK=4 \
  ATLAS_DFLASH_DRAFT_SPLITK=8 \
  ATLAS_DFLASH_QUANT=nvfp4 \
  ATLAS_DFLASH_LM_HEAD_NVFP4=1 \
  ATLAS_DFLASH_LM_HEAD_FP8=1 \
  ATLAS_DFLASH_FFN_KGAMMA=1 \
  ATLAS_DFLASH_ATTN_KGAMMA=1 \
  ATLAS_DFLASH_NOISE_ONLY=1 \
  ATLAS_DFLASH_CTX_WINDOW=4096 \
  ATLAS_DFLASH_DRAFT_CAP="$GAMMA" \
  ATLAS_DFLASH_FREE_SLOTS=0 \
  ATLAS_DFLASH_FREE_SLOTS_TAIL=4 \
  ATLAS_DFLASH_SAM=0 \
  ATLAS_DFLASH_ASYNC=0 \
  ATLAS_DFLASH_RETR_WIDE=31 \
  ATLAS_DFLASH_KERNEL_PROFILE=0 \
  ATLAS_DFLASH_TREE_COMMIT=0 \
  ATLAS_THINK_SPEC=1 \
  ATLAS_FFN_FUSED_GATEUP=1 \
  ATLAS_FFN_KGAMMA_M16=1 \
  ATLAS_FFN_KGAMMA_M128=1 \
  ATLAS_FFN_M16_TRANSPOSED=1 \
  ATLAS_FFN_DOWN_SPLITK=4 \
  ATLAS_NVFP4_GATE_UP_M128=1 \
  ATLAS_NO_GEMV_SW=1 \
  ATLAS_TC_NVFP4_M16=0 \
  ATLAS_TC_NVFP4_M16_MS_ATTN=0 \
  ATLAS_SSM_OUT_SPLITK=4 \
  ATLAS_SSM_QKVZ_SPLITK=4 \
  ATLAS_SSM_BA_BATCH=1 \
  ATLAS_WY17_LAZY=1 \
  ATLAS_WY17_LAZY_COMMIT=0 \
  ATLAS_WY17_SPLIT=2 \
  ATLAS_DISABLE_TREE_WY=1 \
  ATLAS_DSPARK_ASYMMETRIC_ATTN=1 \
  ATLAS_FLASH_ATTN_KGAMMA_SPLITK=1 \
  ATLAS_FA2_KGAMMA=1 \
  ATLAS_PAGED_DECODE_SPLITK=1 \
  ATLAS_LM_HEAD_BATCH3=1 \
  ATLAS_DDTREE_MAX_NODES=$((GAMMA + 1)) \
  ATLAS_DDTREE_UNCAP=0 \
  ATLAS_DDTREE_TREE_AWARE_VERIFY=0 \
  ATLAS_DDTREE_TREE_TOKENS_VERIFY=0 \
  ATLAS_DDTREE_TREE_CONV_EXACT=0 \
  ATLAS_TREE_AWARE_ATTN=0 \
  ATLAS_MULTISEQ_GRAPHS=0 \
  ATLAS_DFLASH_SPEC_CYCLE_V2=0 \
  ATLAS_WEIGHT_CACHE=1 \
  "$BIN" serve \
    --model-from-path "$MODEL_DIR" \
    --model-name qwen38 --port "$PORT" \
    --kernel-target "$KTARGET" \
    --gpu-memory-utilization 0.55 \
    --kv-cache-dtype bf16 \
    --kv-high-precision-layers 0 \
    --max-seq-len 8192 \
    --max-batch-size 1 --max-num-seqs 1 \
    --dflash --draft-model "$DRAFT" \
    --dflash-gamma "$GAMMA" \
    --dflash-quantization nvfp4 \
    --mtp-vocab 96000 \
    --max-thinking-budget 2048 \
    --request-timeout 300 \
    --disable-confidence-early-stop \
    --disable-simhash-watchdog \
    --disable-loop-watchdog
