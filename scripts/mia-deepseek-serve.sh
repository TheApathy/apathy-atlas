#!/usr/bin/env bash
# Launch the pinned local Mia runtime. This script never stops another GPU job.
set -euo pipefail

ALLOW_GPU=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --allow-gpu) ALLOW_GPU=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [[ "$ALLOW_GPU" != 1 && "$DRY_RUN" != 1 ]]; then
  echo "usage: $0 --allow-gpu | --dry-run" >&2
  echo "Refusing to allocate the GPU without the explicit flag." >&2
  exit 2
fi

IMAGE="${MIA_IMAGE:-ds4-mia-exl3-k2-1spark:local}"
MODEL="${MIA_MODEL:-/home/flocka/models/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1}"
NAME="${MIA_CONTAINER:-atlas-mia-deepseek-k2}"
HF_CACHE="${HF_CACHE:-/home/flocka/.cache/huggingface}"

[[ -d "$MODEL" ]] || { echo "model directory not found: $MODEL" >&2; exit 1; }
shards=$(find "$MODEL" -maxdepth 1 -type f -name '*.safetensors' | wc -l)
[[ "$shards" -eq 10 ]] || {
  echo "expected 10 K2 checkpoint shards in $MODEL, found $shards" >&2
  exit 1
}
docker image inspect "$IMAGE" >/dev/null

if [[ "$DRY_RUN" != 1 ]] && docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "container $NAME already exists; remove it explicitly before relaunch" >&2
  exit 1
fi

if [[ "$DRY_RUN" != 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
  if [[ -n "$busy" && "${MIA_ALLOW_BUSY_GPU:-0}" != 1 ]]; then
    echo "GPU is already in use; Mia was not started:" >&2
    printf '%s\n' "$busy" >&2
    echo "Set MIA_ALLOW_BUSY_GPU=1 only after intentionally stopping the other workload." >&2
    exit 1
  fi
fi

args=(
  --gpus all --network host --ipc host --shm-size 64g
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576
  --ulimit stack=67108864:67108864
  -v "$MODEL:/model:ro"
  -v "$HF_CACHE:/cache/huggingface"
  -e MODEL_KIND=k2 -e MODEL_PATH=/model
  -e SERVED_MODEL_NAME=deepseek-v4-flash-k2
  -e TP_SIZE=1 -e NNODES=1
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-1000000}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
  -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
  -e DSPARK_TOKENS="${DSPARK_TOKENS:-5}"
  -e DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
  -e DEFAULT_THINKING="${DEFAULT_THINKING:-off}"
  -e PREFIX_CACHE="${PREFIX_CACHE:-1}"
  -e CUTE_DSL_ARCH=sm_121a
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  -e VLLM_USE_B12X_MOE=1
)
[[ -e /dev/infiniband ]] && args+=(--device /dev/infiniband:/dev/infiniband)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'docker run -d --name %q ' "$NAME"
  printf '%q ' "${args[@]}" "$IMAGE"
  printf '\n'
  exit 0
fi

docker run -d --name "$NAME" "${args[@]}" "$IMAGE"
echo "Mia started as $NAME on port 8888. First compile can take 15-30 minutes."
echo "Follow: docker logs -f $NAME"
