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
#
# Extra arguments are forwarded to the binary, which is how you add flags this
# profile does not set. In a container you MUST add --bind 0.0.0.0: the server
# defaults to loopback, and a loopback bind inside a container is unreachable
# through -p port mapping.
#
#   ... ./bench/qwen38-gb10/serve.sh --bind 0.0.0.0
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the Qwen3.8-27B NVFP4 checkpoint}"
BIN="${BIN:-target/release/spark}"
PORT="${PORT:-8896}"
RUNTIME_MODE="${RUNTIME_MODE-dflash-v3}"

# --kernel-target names a MODEL directory under kernels/<hw>/, NOT the hardware
# directory. Passing `gb10` here is rejected: the loader errors with the list of
# available targets. The named target must also have been compiled in, which is
# what ATLAS_TARGET_MODEL=qwen3.8-27b at build time does.
KTARGET="${KTARGET:-qwen3.8-27b}"

# Derive gamma from the drafter's block_size: trained_drafts = block_size - 1.
# 15 corresponds to a block_size=16 drafter. Asking for more than the drafter
# was trained for is refused by the loader.
GAMMA="${GAMMA:-15}"
MODEL_NAME="${MODEL_NAME:-qwen38}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}"
KV_HIGH_PRECISION_LAYERS="${KV_HIGH_PRECISION_LAYERS:-0}"
MTP_VOCAB="${MTP_VOCAB:-96000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"
E2M1_GEMM="${E2M1_GEMM:-0}"
E2M1_GEMM_DOWN_ONLY="${E2M1_GEMM_DOWN_ONLY:-0}"
E2M1_STATIC_SCALE="${E2M1_STATIC_SCALE:-0}"
E2M1_KMAJOR="${E2M1_KMAJOR:-0}"
E2M1_KMAJOR_M256="${E2M1_KMAJOR_M256:-0}"
E2M1_SILU_QUANT="${E2M1_SILU_QUANT:-0}"
if [[ -v PREFILL_FFN_FLASHINFER && -v ATLAS_PREFILL_FFN_FLASHINFER \
      && "${PREFILL_FFN_FLASHINFER-}" != "${ATLAS_PREFILL_FFN_FLASHINFER-}" ]]; then
  echo "PREFILL_FFN_FLASHINFER and ATLAS_PREFILL_FFN_FLASHINFER disagree" >&2
  exit 2
fi
PREFILL_FFN_FLASHINFER="${PREFILL_FFN_FLASHINFER-${ATLAS_PREFILL_FFN_FLASHINFER-0}}"
if [[ -v PREFILL_PROJ_FLASHINFER && -v ATLAS_PREFILL_PROJ_FLASHINFER \
      && "${PREFILL_PROJ_FLASHINFER-}" != "${ATLAS_PREFILL_PROJ_FLASHINFER-}" ]]; then
  echo "PREFILL_PROJ_FLASHINFER and ATLAS_PREFILL_PROJ_FLASHINFER disagree" >&2
  exit 2
fi
PREFILL_PROJ_FLASHINFER="${PREFILL_PROJ_FLASHINFER-${ATLAS_PREFILL_PROJ_FLASHINFER-0}}"
if [[ -v PREFILL_SSM_FLASHINFER && -v ATLAS_PREFILL_SSM_FLASHINFER \
      && "${PREFILL_SSM_FLASHINFER-}" != "${ATLAS_PREFILL_SSM_FLASHINFER-}" ]]; then
  echo "PREFILL_SSM_FLASHINFER and ATLAS_PREFILL_SSM_FLASHINFER disagree" >&2
  exit 2
fi
PREFILL_SSM_FLASHINFER="${PREFILL_SSM_FLASHINFER-${ATLAS_PREFILL_SSM_FLASHINFER-0}}"
FLASHINFER_SM121_LIB="${FLASHINFER_SM121_LIB-${ATLAS_FLASHINFER_SM121_LIB-}}"
PREFILL_PROJ_FAST="${PREFILL_PROJ_FAST:-0}"
PREFILL_FFN_FAST="${PREFILL_FFN_FAST:-0}"
PREFILL_PROJ_PIPE="${PREFILL_PROJ_PIPE:-1}"
PREFILL_FFN_PIPE="${PREFILL_FFN_PIPE:-1}"
PREFILL_FFN_FUSED_EPILOGUE="${PREFILL_FFN_FUSED_EPILOGUE:-0}"
if [[ -v PREFILL_FFN_DUAL_FUSED && -v ATLAS_PREFILL_FFN_DUAL_FUSED \
      && "${PREFILL_FFN_DUAL_FUSED-}" != "${ATLAS_PREFILL_FFN_DUAL_FUSED-}" ]]; then
  echo "PREFILL_FFN_DUAL_FUSED and ATLAS_PREFILL_FFN_DUAL_FUSED disagree" >&2
  exit 2
fi
PREFILL_FFN_DUAL_FUSED="${PREFILL_FFN_DUAL_FUSED-${ATLAS_PREFILL_FFN_DUAL_FUSED-0}}"
PREFILL_KV_DUAL="${PREFILL_KV_DUAL:-0}"
if [[ -v PREFILL_ATTN_GATE_FUSED && -v ATLAS_PREFILL_ATTN_GATE_FUSED \
      && "${PREFILL_ATTN_GATE_FUSED-}" != "${ATLAS_PREFILL_ATTN_GATE_FUSED-}" ]]; then
  echo "PREFILL_ATTN_GATE_FUSED and ATLAS_PREFILL_ATTN_GATE_FUSED disagree" >&2
  exit 2
fi
PREFILL_ATTN_GATE_FUSED="${PREFILL_ATTN_GATE_FUSED-${ATLAS_PREFILL_ATTN_GATE_FUSED-0}}"
if [[ -v PREFILL_ATTN_BR128 && -v ATLAS_PREFILL_ATTN_BR128 \
      && "${PREFILL_ATTN_BR128-}" != "${ATLAS_PREFILL_ATTN_BR128-}" ]]; then
  echo "PREFILL_ATTN_BR128 and ATLAS_PREFILL_ATTN_BR128 disagree" >&2
  exit 2
fi
# Accept the documented ATLAS_* spelling as well as the launcher's short alias.
# `${x-y}` deliberately preserves an explicit empty value so validation rejects
# it instead of silently coercing the request to the default-off route.
PREFILL_ATTN_BR128="${PREFILL_ATTN_BR128-${ATLAS_PREFILL_ATTN_BR128-0}}"
if [[ -v PREFILL_QKNORM_ROPE && -v ATLAS_PREFILL_QKNORM_ROPE \
      && "${PREFILL_QKNORM_ROPE-}" != "${ATLAS_PREFILL_QKNORM_ROPE-}" ]]; then
  echo "PREFILL_QKNORM_ROPE and ATLAS_PREFILL_QKNORM_ROPE disagree" >&2
  exit 2
fi
# Preserve explicit empty values so they fail validation rather than silently
# selecting the default-off route.
PREFILL_QKNORM_ROPE="${PREFILL_QKNORM_ROPE-${ATLAS_PREFILL_QKNORM_ROPE-0}}"
DFLASH_PREFILL_PIPE="${DFLASH_PREFILL_PIPE:-1}"
SSM_PREFILL_PACK="${SSM_PREFILL_PACK:-1}"
SSM_PREFILL_CONV_L2="${SSM_PREFILL_CONV_L2:-0}"
GDN_PREFILL_GATECACHE="${GDN_PREFILL_GATECACHE:-0}"
if [[ -v GDN_C143_PREFILL && -v ATLAS_GDN_C143_PREFILL \
      && "${GDN_C143_PREFILL-}" != "${ATLAS_GDN_C143_PREFILL-}" ]]; then
  echo "GDN_C143_PREFILL and ATLAS_GDN_C143_PREFILL disagree" >&2
  exit 2
fi
GDN_C143_PREFILL="${GDN_C143_PREFILL-${ATLAS_GDN_C143_PREFILL-0}}"
if [[ -v GDN_C143_SM121_LIB && -v ATLAS_GDN_C143_SM121_LIB \
      && "${GDN_C143_SM121_LIB-}" != "${ATLAS_GDN_C143_SM121_LIB-}" ]]; then
  echo "GDN_C143_SM121_LIB and ATLAS_GDN_C143_SM121_LIB disagree" >&2
  exit 2
fi
GDN_C143_SM121_LIB="${GDN_C143_SM121_LIB-${ATLAS_GDN_C143_SM121_LIB-}}"
if [[ -v ATTN_QKV_M17_ASTAGE && -v ATLAS_ATTN_QKV_EXACT_M17_ASTAGE \
      && "${ATTN_QKV_M17_ASTAGE-}" != "${ATLAS_ATTN_QKV_EXACT_M17_ASTAGE-}" ]]; then
  echo "ATTN_QKV_M17_ASTAGE and ATLAS_ATTN_QKV_EXACT_M17_ASTAGE disagree" >&2
  exit 2
fi
# Accept the public ATLAS_* spelling and preserve explicit empty values so the
# boolean validation below rejects them instead of silently selecting default off.
ATTN_QKV_M17_ASTAGE="${ATTN_QKV_M17_ASTAGE-${ATLAS_ATTN_QKV_EXACT_M17_ASTAGE-0}}"
ATTN_GATE_BATCHED="${ATTN_GATE_BATCHED:-0}"

case "$RUNTIME_MODE" in
  dflash-v3)
    : "${DRAFT:?set DRAFT to the DFlash drafter directory}"
    RUNTIME_ARGS=(
      --dflash --draft-model "$DRAFT"
      --dflash-gamma "$GAMMA"
      --dflash-quantization nvfp4
    )
    ;;
  no-spec)
    if [[ -n "${DRAFT-}" ]]; then
      echo "DRAFT must be unset or empty when RUNTIME_MODE=no-spec" >&2
      exit 2
    fi
    for argument in "$@"; do
      case "$argument" in
        --dflash|--dflash=*|--dflash-*|--draft-model|--draft-model=*|\
        --speculative|--speculative=*|\
        --self-speculative|--self-speculative=*|\
        --ngram-speculative|--ngram-speculative=*|\
        --mtp-gate|--mtp-gate=*)
          echo "speculative argument $argument is forbidden when RUNTIME_MODE=no-spec" >&2
          exit 2
          ;;
      esac
    done
    RUNTIME_ARGS=()
    ;;
  *)
    echo "RUNTIME_MODE must be dflash-v3 or no-spec" >&2
    exit 2
    ;;
esac

case "$KV_CACHE_DTYPE" in
  bf16|fp8|nvfp4) ;;
  *) echo "unsupported KV_CACHE_DTYPE: $KV_CACHE_DTYPE" >&2; exit 2 ;;
esac
[[ "$KV_HIGH_PRECISION_LAYERS" =~ ^[0-9]+$ ]] || {
  echo "KV_HIGH_PRECISION_LAYERS must be a non-negative integer" >&2
  exit 2
}
[[ "$MTP_VOCAB" =~ ^[1-9][0-9]*$ ]] || {
  echo "MTP_VOCAB must be a positive integer" >&2
  exit 2
}
[[ "$MAX_SEQ_LEN" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_SEQ_LEN must be a positive integer" >&2
  exit 2
}
if (( ${#MAX_SEQ_LEN} > 7 )); then
  echo "MAX_SEQ_LEN exceeds Qwen3.8's supported 1000000-token ceiling" >&2
  exit 2
fi
# The length gate above makes this decimal conversion safe from signed Bash
# arithmetic overflow. The regex forbids leading zeroes, so 10# is unambiguous.
MAX_SEQ_LEN=$((10#$MAX_SEQ_LEN))
[[ "$MAX_PREFILL_TOKENS" =~ ^[0-9]+$ ]] || {
  echo "MAX_PREFILL_TOKENS must be a non-negative integer" >&2
  exit 2
}
for bool_name in \
  E2M1_GEMM E2M1_GEMM_DOWN_ONLY E2M1_STATIC_SCALE E2M1_KMAJOR E2M1_KMAJOR_M256 E2M1_SILU_QUANT \
  PREFILL_FFN_FLASHINFER PREFILL_PROJ_FLASHINFER PREFILL_SSM_FLASHINFER \
  PREFILL_PROJ_FAST PREFILL_FFN_FAST PREFILL_PROJ_PIPE PREFILL_FFN_PIPE \
  PREFILL_FFN_FUSED_EPILOGUE PREFILL_FFN_DUAL_FUSED PREFILL_KV_DUAL PREFILL_ATTN_GATE_FUSED \
  PREFILL_ATTN_BR128 \
  PREFILL_QKNORM_ROPE \
  DFLASH_PREFILL_PIPE SSM_PREFILL_PACK SSM_PREFILL_CONV_L2 \
  GDN_PREFILL_GATECACHE GDN_C143_PREFILL ATTN_QKV_M17_ASTAGE ATTN_GATE_BATCHED
do
  bool_value="${!bool_name}"
  case "$bool_value" in
    0|1) ;;
    *) echo "$bool_name must be 0 or 1" >&2; exit 2 ;;
  esac
done
if [[ "$GDN_C143_PREFILL" == 1 && "$GDN_PREFILL_GATECACHE" == 1 ]]; then
  echo "GDN_C143_PREFILL=1 is mutually exclusive with GDN_PREFILL_GATECACHE=1" >&2
  exit 2
fi
if [[ "$E2M1_STATIC_SCALE" == 1 && "$E2M1_GEMM" == 0 && "$E2M1_GEMM_DOWN_ONLY" == 0 \
      && "$PREFILL_FFN_FLASHINFER" == 0 ]]; then
  echo "E2M1_STATIC_SCALE=1 requires an E2M1 or FlashInfer FFN route" >&2
  exit 2
fi
if [[ "$E2M1_KMAJOR" == 1 && "$E2M1_GEMM" == 0 && "$E2M1_GEMM_DOWN_ONLY" == 0 ]]; then
  echo "E2M1_KMAJOR=1 requires E2M1_GEMM=1 or E2M1_GEMM_DOWN_ONLY=1" >&2
  exit 2
fi
if [[ "$E2M1_KMAJOR_M256" == 1 && "$E2M1_KMAJOR" == 0 ]]; then
  echo "E2M1_KMAJOR_M256=1 requires E2M1_KMAJOR=1" >&2
  exit 2
fi
if [[ "$E2M1_SILU_QUANT" == 1 && "$E2M1_STATIC_SCALE" == 0 ]]; then
  echo "E2M1_SILU_QUANT=1 requires E2M1_STATIC_SCALE=1" >&2
  exit 2
fi
FLASHINFER_ENV=()
if [[ -n "$FLASHINFER_SM121_LIB" ]]; then
  if [[ "$FLASHINFER_SM121_LIB" != /* || ! -f "$FLASHINFER_SM121_LIB" ]]; then
    echo "FLASHINFER_SM121_LIB must name an absolute regular file" >&2
    exit 2
  fi
  FLASHINFER_ENV=("ATLAS_FLASHINFER_SM121_LIB=$FLASHINFER_SM121_LIB")
fi
if [[ ( "$PREFILL_FFN_FLASHINFER" == 1 || "$PREFILL_PROJ_FLASHINFER" == 1 \
        || "$PREFILL_SSM_FLASHINFER" == 1 ) \
      && -z "$FLASHINFER_SM121_LIB" ]]; then
  echo "a FlashInfer FFN/attention/SSM projection route requires FLASHINFER_SM121_LIB" >&2
  exit 2
fi
GDN_C143_ENV=()
if [[ -n "$GDN_C143_SM121_LIB" ]]; then
  if [[ "$GDN_C143_SM121_LIB" != /* || ! -f "$GDN_C143_SM121_LIB" ]]; then
    echo "GDN_C143_SM121_LIB must name an absolute regular file" >&2
    exit 2
  fi
  GDN_C143_ENV=("ATLAS_GDN_C143_SM121_LIB=$GDN_C143_SM121_LIB")
fi
if [[ "$GDN_C143_PREFILL" == 1 && -z "$GDN_C143_SM121_LIB" ]]; then
  echo "GDN_C143_PREFILL=1 requires GDN_C143_SM121_LIB" >&2
  exit 2
fi

CONTEXT_ARGS=(--max-seq-len "$MAX_SEQ_LEN")
if (( MAX_SEQ_LEN > 262144 )); then
  if (( MAX_SEQ_LEN > 1000000 )); then
    echo "Qwen3.8 static YaRN is supported through exactly 1000000 total tokens" >&2
    exit 2
  fi
  if [[ -v ROPE_THETA_OVERRIDE ]]; then
    echo "ROPE_THETA_OVERRIDE is not Qwen3.8's 1M recipe; use ROPE_YARN_FACTOR=4" >&2
    exit 2
  fi
  ROPE_YARN_FACTOR="${ROPE_YARN_FACTOR:-4}"
  if [[ "$ROPE_YARN_FACTOR" != 4 && "$ROPE_YARN_FACTOR" != 4.0 ]]; then
    echo "Qwen3.8 long context requires ROPE_YARN_FACTOR=4" >&2
    exit 2
  fi
  if [[ "$RUNTIME_MODE" != no-spec ]]; then
    echo "1M target YaRN is ready only in RUNTIME_MODE=no-spec; the V3 drafter still needs its own long-position qualification" >&2
    exit 2
  fi
  CONTEXT_ARGS+=(--rope-yarn-factor "$ROPE_YARN_FACTOR")
elif [[ -v ROPE_YARN_FACTOR ]]; then
  echo "ROPE_YARN_FACTOR is only valid when MAX_SEQ_LEN exceeds 262144" >&2
  exit 2
fi

QUALIFICATION_ENV=()
if [[ -v ATLAS_QUALIFICATION_RUN_NONCE ]]; then
  if [[ ! "$ATLAS_QUALIFICATION_RUN_NONCE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ATLAS_QUALIFICATION_RUN_NONCE must be 64 lowercase hexadecimal characters" >&2
    exit 2
  fi
  QUALIFICATION_ENV=(
    "ATLAS_QUALIFICATION_RUN_NONCE=$ATLAS_QUALIFICATION_RUN_NONCE"
  )
  printf 'ATLAS_QUALIFICATION_RUN pid=%s nonce=%s\n' \
    "$$" "$ATLAS_QUALIFICATION_RUN_NONCE" >&2
fi

exec env \
  -u ATLAS_PROFILE \
  -u ATLAS_PROFILE_FIRST \
  -u ATLAS_FULL_PROFILE \
  -u ATLAS_PREFILL_PHASE_PROFILE \
  -u ATLAS_GDN_PROFILE \
  -u ATLAS_DFLASH_KERNEL_PROFILE \
  -u ATLAS_DFLASH_ASYNC_PROBE \
  -u ATLAS_PROPOSE_PROBE \
  -u ATLAS_SSM_KERNEL_PROFILE \
  -u ATLAS_LAYER_RESOLUTION_PROBE \
  -u ATLAS_DFLASH_EARLY_EXIT_PROFILE \
  -u ATLAS_PREFILL_FFN_FLASHINFER \
  -u ATLAS_PREFILL_PROJ_FLASHINFER \
  -u ATLAS_PREFILL_SSM_FLASHINFER \
  -u ATLAS_FLASHINFER_SM121_LIB \
  -u ATLAS_GDN_C143_PREFILL \
  -u ATLAS_GDN_C143_SM121_LIB \
  -u PREFILL_FFN_FLASHINFER \
  -u PREFILL_PROJ_FLASHINFER \
  -u PREFILL_SSM_FLASHINFER \
  -u FLASHINFER_SM121_LIB \
  -u GDN_C143_PREFILL \
  -u GDN_C143_SM121_LIB \
  -u E2M1_GEMM \
  -u E2M1_GEMM_DOWN_ONLY \
  -u E2M1_STATIC_SCALE \
  -u E2M1_KMAJOR \
  -u E2M1_KMAJOR_M256 \
  -u E2M1_SILU_QUANT \
  -u PREFILL_PROJ_FAST \
  -u PREFILL_FFN_FAST \
  -u PREFILL_PROJ_PIPE \
  -u PREFILL_FFN_PIPE \
  -u PREFILL_FFN_FUSED_EPILOGUE \
  -u PREFILL_FFN_DUAL_FUSED \
  -u PREFILL_KV_DUAL \
  -u PREFILL_ATTN_GATE_FUSED \
  -u PREFILL_ATTN_BR128 \
  -u PREFILL_QKNORM_ROPE \
  -u DFLASH_PREFILL_PIPE \
  -u SSM_PREFILL_PACK \
  -u SSM_PREFILL_CONV_L2 \
  -u GDN_PREFILL_GATECACHE \
  -u ATTN_QKV_M17_ASTAGE \
  -u ATTN_GATE_BATCHED \
  "${QUALIFICATION_ENV[@]}" \
  "${FLASHINFER_ENV[@]}" \
  "${GDN_C143_ENV[@]}" \
  ATLAS_FFN_TC=1 \
  ATLAS_SSM_PROJ_TC=1 \
  ATLAS_LM_HEAD_TC=1 \
  ATLAS_DECODE_TC_PARITY=1 \
  ATLAS_LM_HEAD_T=1 \
  ATLAS_ACCEPT_FAST_ARGMAX=1 \
  ATLAS_E2M1_GEMM="$E2M1_GEMM" \
  ATLAS_E2M1_GEMM_DOWN_ONLY="$E2M1_GEMM_DOWN_ONLY" \
  ATLAS_E2M1_STATIC_SCALE="$E2M1_STATIC_SCALE" \
  ATLAS_E2M1_KMAJOR="$E2M1_KMAJOR" \
  ATLAS_E2M1_KMAJOR_M256="$E2M1_KMAJOR_M256" \
  ATLAS_E2M1_SILU_QUANT="$E2M1_SILU_QUANT" \
  ATLAS_PREFILL_FFN_FLASHINFER="$PREFILL_FFN_FLASHINFER" \
  ATLAS_PREFILL_PROJ_FLASHINFER="$PREFILL_PROJ_FLASHINFER" \
  ATLAS_PREFILL_SSM_FLASHINFER="$PREFILL_SSM_FLASHINFER" \
  ATLAS_PREFILL_PROJ_FAST="$PREFILL_PROJ_FAST" \
  ATLAS_PREFILL_FFN_FAST="$PREFILL_FFN_FAST" \
  ATLAS_PREFILL_PROJ_PIPE="$PREFILL_PROJ_PIPE" \
  ATLAS_PREFILL_FFN_PIPE="$PREFILL_FFN_PIPE" \
  ATLAS_PREFILL_FFN_FUSED_EPILOGUE="$PREFILL_FFN_FUSED_EPILOGUE" \
  ATLAS_PREFILL_FFN_DUAL_FUSED="$PREFILL_FFN_DUAL_FUSED" \
  ATLAS_PREFILL_KV_DUAL="$PREFILL_KV_DUAL" \
  ATLAS_PREFILL_ATTN_GATE_FUSED="$PREFILL_ATTN_GATE_FUSED" \
  ATLAS_PREFILL_ATTN_BR128="$PREFILL_ATTN_BR128" \
  ATLAS_PREFILL_QKNORM_ROPE="$PREFILL_QKNORM_ROPE" \
  ATLAS_SSM_GDN_SEQ_PERSISTENT=1 \
  ATLAS_SSM_GDN_LAZY=1 \
  ATLAS_ATTN_QKV_FUSED=1 \
  ATLAS_ATTN_QKV_EXACT_STRIDED=1 \
  ATLAS_ATTN_QKV_EXACT_M17_ASTAGE="$ATTN_QKV_M17_ASTAGE" \
  ATLAS_ATTN_GATE_BATCHED="$ATTN_GATE_BATCHED" \
  ATLAS_ATTN_QKV_BATCHED=0 \
  ATLAS_ATTN_QKV_SPLITK=4 \
  ATLAS_DFLASH_DRAFT_SPLITK=8 \
  ATLAS_DFLASH_PREFILL_PIPE="$DFLASH_PREFILL_PIPE" \
  ATLAS_SSM_PREFILL_PACK="$SSM_PREFILL_PACK" \
  ATLAS_SSM_PREFILL_CONV_L2="$SSM_PREFILL_CONV_L2" \
  ATLAS_GDN_PREFILL_GATECACHE="$GDN_PREFILL_GATECACHE" \
  ATLAS_GDN_C143_PREFILL="$GDN_C143_PREFILL" \
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
  ATLAS_DFLASH_ECHO=0 \
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
  ATLAS_FFN_W3_LAYERS= \
  ATLAS_FFN_W3_SIDECAR= \
  ATLAS_DFLASH_SPEC_CYCLE_V2=1 \
  ATLAS_WEIGHT_CACHE=1 \
  "$BIN" serve \
    --model-from-path "$MODEL_DIR" \
    --model-name "$MODEL_NAME" --port "$PORT" \
    --kernel-target "$KTARGET" \
    --gpu-memory-utilization 0.55 \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --kv-high-precision-layers "$KV_HIGH_PRECISION_LAYERS" \
    "${CONTEXT_ARGS[@]}" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --max-batch-size 1 --max-num-seqs 1 \
    "${RUNTIME_ARGS[@]}" \
    --mtp-vocab "$MTP_VOCAB" \
    --max-thinking-budget 2048 \
    --request-timeout 300 \
    --disable-confidence-early-stop \
    --disable-simhash-watchdog \
    --disable-loop-watchdog \
    "$@"
