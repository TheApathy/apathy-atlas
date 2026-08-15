#!/bin/bash
# Atlas Spark — AEON-7 Qwen3.6-27B + DDTree experimental path (M11E defaults).
#
# Ports AEON-7's "deployable-safe" DDTree recipe to Atlas:
#   - flat accepted prefix only (no non-flat branch commit) — enforced by
#     adapt_to_flat_safe_contract() in ddtree.rs M5B sampler
#   - root-leaf bonus branches handled by greedy tree walk
#   - DFlash drafter-context compaction inherited from shrink-noise fix
#   - NO recurrent branch-state compaction (Triton GDN replay path disabled)
#   - NO inline parent fallback in attention metadata
#
# Defaults (mirror AEON-7 README §M11E):
#   MAX_MODEL_LEN=2048, MAX_NUM_SEQS=1, NUM_SPECULATIVE_TOKENS=15,
#   DDTREE_BUDGET=15, DDTREE_TOP_K=8, DDTREE_ROOT_LEAF_ALT_COUNT=4,
#   DDTREE_MIN_ROOT_BRANCHES=5, DDTREE_TRITON_TREE_GDN=0.
#
# DO NOT enable these for serving (research-only):
#   DDTREE_TRITON_TREE_GDN=1, DDTREE_COMPACT_RECURRENT_STATE=1,
#   DDTREE_ALLOW_BRANCH_STATE_COMPACTION=1, DDTREE_FULL_BRANCH_COMMIT=1.
#
# Status: M8A (tree-aware GDN kernel) NOT yet implemented in Atlas. With
# the safe defaults this script falls back to flat DFlash behavior — the
# `ddtree` method just exercises the M3/M4B/M6 payload bridge end-to-end
# without changing throughput vs `serve-aeon-27b-dflash.sh`. The actual
# throughput unlock waits for M8A landing.
set -euo pipefail

PORT=${PORT:-8890}

if pgrep -x spark >/dev/null 2>&1; then
  echo "[serve-aeon-27b-ddtree] killing prior spark serve..."
  pkill -9 -x spark || true
  sleep 4
fi
if ss -tnlp 2>/dev/null | grep -q ":${PORT} "; then
  echo "[serve-aeon-27b-ddtree] ERROR: port ${PORT} still bound" >&2
  exit 1
fi
FREE_GB=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${FREE_GB:-0}" -lt 50 ]; then
  echo "[serve-aeon-27b-ddtree] ERROR: only ${FREE_GB} GB free, need ≥50 GB" >&2
  free -h >&2
  exit 1
fi
echo "[serve-aeon-27b-ddtree] preflight ok: ${FREE_GB} GB free (M11E defaults)"

# M11E safe defaults (env vars consumed inside the dflash_head module).
export ATLAS_DFLASH_METHOD=${ATLAS_DFLASH_METHOD:-ddtree}
export ATLAS_DFLASH_DRAFT_CAP=${ATLAS_DFLASH_DRAFT_CAP:-32}
export ATLAS_DFLASH_CTX_WINDOW=${ATLAS_DFLASH_CTX_WINDOW:-512}
export ATLAS_DFLASH_QUANT=${ATLAS_DFLASH_QUANT:-nvfp4}
# DDTree-specific (M11E):
export DDTREE_BUDGET=${DDTREE_BUDGET:-15}
export DDTREE_TOP_K=${DDTREE_TOP_K:-8}
export DDTREE_ROOT_LEAF_ALT_COUNT=${DDTREE_ROOT_LEAF_ALT_COUNT:-4}
export DDTREE_MIN_ROOT_BRANCHES=${DDTREE_MIN_ROOT_BRANCHES:-5}
# Research-only switches stay OFF for deployable serving.
export DDTREE_TRITON_TREE_GDN=${DDTREE_TRITON_TREE_GDN:-0}
export DDTREE_COMPACT_RECURRENT_STATE=${DDTREE_COMPACT_RECURRENT_STATE:-0}
export DDTREE_ALLOW_BRANCH_STATE_COMPACTION=${DDTREE_ALLOW_BRANCH_STATE_COMPACTION:-0}
export DDTREE_FULL_BRANCH_COMMIT=${DDTREE_FULL_BRANCH_COMMIT:-0}

exec /home/flocka/atlas/src/target/release/spark serve \
  --model-from-path /home/flocka/models/AEON-Q36-27B-Full \
  --model-name aeon-27b-ddtree \
  --port "${PORT}" \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype fp8 \
  --max-seq-len 2048 \
  --max-batch-size 1 \
  --max-num-seqs 1 \
  --dflash \
  --dflash-method ddtree \
  --ddtree-budget "${DDTREE_BUDGET}" \
  --ddtree-top-k "${DDTREE_TOP_K}" \
  --draft-model /home/flocka/models/z-lab-Qwen3.6-27B-DFlash \
  --dflash-gamma 16 \
  --dflash-quantization "$ATLAS_DFLASH_QUANT" \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas/src/local/warmup.txt
