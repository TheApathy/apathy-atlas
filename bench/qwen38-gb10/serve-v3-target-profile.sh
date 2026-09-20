#!/usr/bin/env bash
# Shared target-side half of the measured V3 profile. Source only.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "serve-v3-target-profile.sh must be sourced by a runtime wrapper" >&2
  exit 2
fi

export MODEL_NAME="${MODEL_NAME:-qwen38-atlas-fork}"
export GAMMA="${GAMMA:-15}"
export MTP_VOCAB="${MTP_VOCAB:-248320}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4}"
export KV_HIGH_PRECISION_LAYERS="${KV_HIGH_PRECISION_LAYERS:-0}"

# This versioned profile must not depend on the generic launcher's defaults.
export DFLASH_PREFILL_PIPE="${DFLASH_PREFILL_PIPE:-1}"
export E2M1_GEMM="${E2M1_GEMM:-0}"
export E2M1_GEMM_DOWN_ONLY="${E2M1_GEMM_DOWN_ONLY:-0}"
export E2M1_STATIC_SCALE="${E2M1_STATIC_SCALE:-0}"
export E2M1_KMAJOR="${E2M1_KMAJOR:-0}"
export E2M1_KMAJOR_M256="${E2M1_KMAJOR_M256:-0}"
export E2M1_SILU_QUANT="${E2M1_SILU_QUANT:-0}"
export PREFILL_FFN_FLASHINFER="${PREFILL_FFN_FLASHINFER-${ATLAS_PREFILL_FFN_FLASHINFER-0}}"
export PREFILL_PROJ_FLASHINFER="${PREFILL_PROJ_FLASHINFER-${ATLAS_PREFILL_PROJ_FLASHINFER-0}}"
export PREFILL_SSM_FLASHINFER="${PREFILL_SSM_FLASHINFER-${ATLAS_PREFILL_SSM_FLASHINFER-0}}"
export PREFILL_FFN_FUSED_EPILOGUE="${PREFILL_FFN_FUSED_EPILOGUE-0}"
export PREFILL_FFN_DUAL_FUSED="${PREFILL_FFN_DUAL_FUSED-${ATLAS_PREFILL_FFN_DUAL_FUSED-0}}"
export SSM_PREFILL_CONV_L2="${SSM_PREFILL_CONV_L2:-0}"
export GDN_PREFILL_GATECACHE="${GDN_PREFILL_GATECACHE:-0}"
export GDN_C143_PREFILL="${GDN_C143_PREFILL-${ATLAS_GDN_C143_PREFILL-0}}"
export PREFILL_QKNORM_ROPE="${PREFILL_QKNORM_ROPE:-0}"
export PREFILL_KV_DUAL="${PREFILL_KV_DUAL:-0}"
# Preserve an explicit empty value so serve.sh rejects it instead of silently
# coercing an invalid qualification request back to the control route.
export PREFILL_ATTN_GATE_FUSED="${PREFILL_ATTN_GATE_FUSED-${ATLAS_PREFILL_ATTN_GATE_FUSED-0}}"
