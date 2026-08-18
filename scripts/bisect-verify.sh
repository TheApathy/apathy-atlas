#!/usr/bin/env bash
# One bisection leg for the batched-verify numerics gap (task #45).
# Starts a forced-spec DSpark server with ONE feature toggled, waits for
# ready, and leaves it running. Pair with scripts/bisect-probe.py.
#
# Usage: scripts/bisect-verify.sh <name> [ENV=VAL ...]
set -euo pipefail
NAME="${1:?usage: bisect-verify.sh <name> [ENV=VAL ...]}"
shift || true

WORKTREE="${WORKTREE:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MODEL:?set MODEL to the DeepSeek-V4-Flash checkpoint directory}"
DRAFTER="${DRAFTER:?set DRAFTER to the DSpark drafter directory}"
LOG="$WORKTREE/serve-bisect-$NAME.log"

ENV_ARGS=(
  ATLAS_UNIFIED_MOE_LAYOUT=1
  ATLAS_V4_ATTN_NVFP4=1
  ATLAS_V4_ATTN_RELEASE_BF16=1
  ATLAS_MOE_MROW_PARTITION=1
  ATLAS_MOE_T_BLOCK=128
  ATLAS_DSPARK_CAPTURE=1
  ATLAS_DFLASH_STEP_TIMING=1
  ATLAS_MTP_GATE_FORCE=1
)
for kv in "$@"; do ENV_ARGS+=("$kv"); done

echo "env: ${ENV_ARGS[*]}" >"$LOG"
cd "$WORKTREE"
env "${ENV_ARGS[@]}" ./target/release/spark serve "$MODEL" \
  --port 8977 --kv-cache-dtype fp8 --lm-head-dtype fp8 \
  --gpu-memory-utilization 0.91 --max-seq-len 1024 --max-num-seqs 1 \
  --max-batch-size 1 --max-prefill-tokens 1024 --kv-cache-cap-tokens 1040 \
  --dflash --draft-model "$DRAFTER" \
  --dflash-gamma 6 >>"$LOG" 2>&1 &
echo "pid=$! log=$LOG"
