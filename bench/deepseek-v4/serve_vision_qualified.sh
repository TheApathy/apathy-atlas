#!/usr/bin/env bash
set -euo pipefail
campaign=/var/tmp/atlas-deepseek-vision-spec.EKOuMy1l
if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {plain|dspark|dspark-max2410}\n' "$0" >&2
  exit 64
fi
arm=$1
case "$arm" in
  plain|dspark|dspark-max2410) ;;
  *)
    printf 'Unsupported mode: %s (expected plain, dspark, or dspark-max2410)\n' "$arm" >&2
    exit 64
    ;;
esac
run=$(mktemp -d "$campaign/$arm-live.XXXXXXXX")
printf 'Run directory: %s\n' "$run"
ATLAS_PORT=8977
QWEN38_GPU_LOCK_ID=all
source /home/flocka/atlas/qwen38/benchmark/arms/common.sh
acquire_benchmark_lock 8977 1
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
[[ -z "$(ss -H -ltn 'sport = :8977')" ]]
cd /home/flocka/atlas/apathy-deepseek
sha256sum -c "$campaign/source-r11.sha256" --quiet
server="$campaign/native-r11/release/spark"
target=/home/flocka/models/DeepSeek-V4-Flash-Vision-EXL3-K2-c171bea5
settings=(
  PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 RUST_LOG=info
  ATLAS_DEBUG_NO_GRAPH=1 ATLAS_DFLASH_DEBUG_NO_GRAPH=1 ATLAS_DSPARK_CAPTURE=1
  ATLAS_DFLASH_MASKED_VERIFY=1 ATLAS_MOE_GATE_EXACT=1
  ATLAS_V4_SHARED_NATIVE_FP8=1 ATLAS_V4_WOA_INPLACE=0 ATLAS_CUBLAS_TUNED=0
  ATLAS_EXL3_PREFILL_DIRECT=1 ATLAS_EXL3_PREFILL_PERSISTENT=1
  ATLAS_EXL3_PREFILL_FIXED_K2=1 ATLAS_EXL3_PREFILL_FIXED_SHAPE=1
  ATLAS_EXL3_PREFILL_K64=1 ATLAS_EXL3_PREFILL_N128=1
  ATLAS_EXL3_PREFILL_N256=0 ATLAS_EXL3_PREFILL_M128=0
  ATLAS_EXL3_PREFILL_FUSED_POST=1 ATLAS_EXL3_PREFILL_DUAL_PRE=0
  ATLAS_EXL3_PREFILL_FUSED_UNPERMUTE=0 ATLAS_EXL3_PREFILL_FUSED_BLEND=0
  ATLAS_EXL3_PREFILL_W2A8=0 ATLAS_PREFILL_MAX_REQUIRE_ARMS=0
)
# The plain and dspark modes retain the broadly exercised BF16/N128 recipe.
# Only dspark-max2410 enables the strict, shape-locked W2A8 performance arm.
args=(serve --model-from-path "$target" --model-name deepseek-v4-flash-vision
  --bind 127.0.0.1 --port 8977 --max-seq-len 12288 --max-prefill-tokens 4096
  --max-num-seqs 1 --max-batch-size 1 --kv-cache-dtype fp8
  --kv-high-precision-layers 0 --lm-head-dtype bf16 --kv-cache-cap-tokens 12304
  --gpu-memory-utilization 0.87 --enable-prefix-caching false --oom-guard-mb 2048
  --disable-thinking)
if [[ "$arm" == dspark-max2410 ]]; then
  # Qualification-only exact-shape profile. Strict mode intentionally rejects
  # prompts whose expanded routed rows do not equal 2410 * topk(6).
  settings+=(
    ATLAS_V4_PREFILL_CUBLASLT=1 ATLAS_V4_ATTN_RELEASE_BF16=0
    ATLAS_V4_PREFILL_HC_RMS_FUSED=0 ATLAS_V4_ATTN_NVFP4=0
    ATLAS_V4_PREFILL_TC=1 ATLAS_V4_PREFILL_TC2=1 ATLAS_V4_PREFILL_TC2_WARP0=0
    ATLAS_V4_PREFILL_QB_ROPE_FUSED=0 ATLAS_V4_PREFILL_QB_ROPE_CACHE_FUSED=0
    ATLAS_V4_COMP_GEMM_TC=1 ATLAS_V4_KV_PIPELINED=1 ATLAS_V4_WOA_INPLACE=1
    ATLAS_HC_TILED=1 ATLAS_V4_PREFILL_KV_ALIAS=1
    ATLAS_V4_PREFILL_INVERSE_ROPE_FUSED=1
    ATLAS_EXL3_PREFILL_K64=0 ATLAS_EXL3_PREFILL_N128=0
    ATLAS_EXL3_PREFILL_N256=0 ATLAS_EXL3_PREFILL_M128=0
    ATLAS_EXL3_PREFILL_DUAL_PRE=1 ATLAS_EXL3_PREFILL_W2A8=1
    ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN=1
    ATLAS_EXL3_PREFILL_W2A8_FUSED_GU_DOWN_N256=0
    ATLAS_EXL3_PREFILL_W2A8_N256_DOWN=1
    ATLAS_EXL3_PREFILL_FUSED_UNPERMUTE=1 ATLAS_EXL3_HROW_FIXED_SHAPE=1
    ATLAS_EXL3_PREFILL_FUSED_BLEND=0 ATLAS_MOE_SHARED_K64=4
    ATLAS_PREFILL_MAX_REQUIRE_ARMS=1
  )
fi
if [[ "$arm" == dspark || "$arm" == dspark-max2410 ]]; then
  settings+=(ATLAS_DEEPSEEK_VISION_DSPARK=1 ATLAS_DSPARK_CONF=0 ATLAS_DFLASH_DIAG=1)
  args+=(--dflash --draft-model "$target" --dflash-gamma 6)
fi
env -i --default-signal=INT "${settings[@]}" "$server" "${args[@]}" >"$run/server.log" 2>&1 &
owned_server=$!
handoff_benchmark_lock_to_server "$owned_server"
if ! node "$campaign/provenance-r11.mjs" "$owned_server" "$run/provenance.json"; then
  if [[ "$(readlink "/proc/$owned_server/exe" 2>/dev/null || true)" == "$server" ]]; then
    kill -INT "$owned_server"
  fi
  wait "$owned_server" || true
  exit 1
fi
set +e
wait "$owned_server"
result=$?
printf 'owned_server=%s exit=%s\n' "$owned_server" "$result" >"$run/exit.txt"
exit "$result"
