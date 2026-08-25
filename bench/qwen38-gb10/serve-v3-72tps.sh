#!/usr/bin/env bash
# Exact full-vocabulary v3/NVFP4-KV Weschera speed profile.
# NVFP4 target KV changes output versus BF16; qualify quality separately.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_NAME="${MODEL_NAME:-qwen38-atlas-fork}"
export GAMMA="${GAMMA:-15}"
export MTP_VOCAB="${MTP_VOCAB:-248320}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4}"
export KV_HIGH_PRECISION_LAYERS="${KV_HIGH_PRECISION_LAYERS:-0}"

exec "$HERE/serve.sh" "$@"
