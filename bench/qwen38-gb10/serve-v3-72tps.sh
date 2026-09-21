#!/usr/bin/env bash
# Exact full-vocabulary v3/NVFP4-KV Weschera speed profile.
# NVFP4 target KV changes output versus BF16; qualify quality separately.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -v RUNTIME_MODE && "$RUNTIME_MODE" != dflash-v3 ]]; then
  echo "serve-v3-72tps.sh requires RUNTIME_MODE=dflash-v3" >&2
  exit 2
fi
export RUNTIME_MODE=dflash-v3

# Dynamic path; the sourced profile is checked separately.
# shellcheck disable=SC1091
source "$HERE/serve-v3-target-profile.sh"

exec "$HERE/serve.sh" "$@"
