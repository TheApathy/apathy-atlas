#!/bin/bash
# Atlas Spark — AEON-7 Qwen3.6-27B target + DSpark-AEON-draft drafter
# (Hikari07jp checkpoint: 5-layer DFlash + rank-256 VanillaMarkov head,
#  block_size 11 → γ=11 → K=12 verify).
#
# The Markov head adds a per-position bigram logit bias
#   B(prev) = markov_w2 @ markov_w1[prev]
# applied LEFT-TO-RIGHT over the γ block BEFORE argmax (position k's chosen
# token biases position k+1). This is auto-on when the checkpoint ships the
# head; set ATLAS_DFLASH_MARKOV=0 to A/B it OFF (plain DFlash argmax).
# LOSSLESS w.r.t. committed output — the target verify still commits its own
# greedy token; the Markov head only changes which drafts are proposed.
#
# ── K=12 VERIFY PATH (block_size 11) ─────────────────────────────────────
# γ=11 → DRAFT_CAP=11 → K = γ+1 = 12. K=12 is NOT one of the dedicated fused
# kernels {K=2,3,4,17}, so it routes through the CHUNKED WY path in
# trait_decode_batched_conv_gdn.rs (the `else` K∈{5..16} branch), which splits
# 12 = 4+4+4 across the fused wy4 kernel and saves ALL per-token SSM
# intermediates for partial-accept rollback. This is the modern, NaN-SAFE
# replacement for the old per-token sequential loop (whose NaN-at-K-3..K-1 bug
# the serve-aeon-27b-dflash.sh comments warn about — that warning applies only
# to the OLD path, which this chunked path supersedes).
#
# ATLAS_DISABLE_TREE_WY=1 is REQUIRED here (task #34). A prior revision of
# this script left it unset, reasoning the injection was "also safe" — it is
# NOT: with graphs on and no tree payload, verify_d.rs substitutes the
# persistent linear-chain parent_ids whenever k == ddtree_parent_ids_capacity
# (= dflash_kgamma = 12 at γ=11), rerouting the SSM verify through
# gated_delta_rule_tree_wy. That kernel (a) leaves the live h_state UNTOUCHED
# — pre-fix, every FULL accept then committed the STALE pre-verify state (SSM
# state froze → non-lossless exactly at high acceptance) — and (b) is not
# bit-equivalent to the wy17-class kernels on linear chains (numeric drift,
# see c588b34). The commit-side staleness is fixed in-engine
# (dflash_flat_tree_route, trait_impl/commit_plan.rs), but the chunked path
# remains the proven-lossless route for K≠17 (md5-exact at K=21/25 in the
# 2026-06 γ sweep), so keep the injection disabled.
#
# All γ/K plumbing is fully dynamic: --dflash-gamma 11 → num_drafts=10 →
# dflash_kgamma = num_drafts+2 = 12 → num_intermediates sized from γ (=13).
# Nothing is hardcoded to 16/17.
set -euo pipefail

PORT=${PORT:-8890}
if pgrep -x spark >/dev/null 2>&1; then
  echo "[serve-aeon-27b-dspark] killing prior spark serve..."
  pkill -9 -x spark || true
  sleep 4
fi
if ss -tnlp 2>/dev/null | grep -q ":${PORT} "; then
  echo "[serve-aeon-27b-dspark] ERROR: port ${PORT} still bound" >&2
  exit 1
fi
FREE_GB=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${FREE_GB:-0}" -lt 50 ]; then
  echo "[serve-aeon-27b-dspark] ERROR: only ${FREE_GB} GB free, need ≥50 GB" >&2
  free -h >&2
  exit 1
fi
echo "[serve-aeon-27b-dspark] preflight ok: ${FREE_GB} GB free"

# γ=11 → DRAFT_CAP=11 → K=12 (chunked wy path, NaN-safe). DRAFT_CAP is clamped
# to min(cap, γ) inside forward_block, so 11 is the effective and correct cap.
export ATLAS_DFLASH_DRAFT_CAP=${ATLAS_DFLASH_DRAFT_CAP:-11}
export ATLAS_LM_HEAD_T="${ATLAS_LM_HEAD_T:-1}"
export ATLAS_DFLASH_CTX_WINDOW=${ATLAS_DFLASH_CTX_WINDOW:-4096}

# nvfp4 drafter weights (the Markov W1/W2 stay BF16 regardless — the head runs
# in shared-vocab logit space with a tiny K=256 GEMV; quantizing it buys
# nothing). Token-exact: verify is the source of truth.
export ATLAS_DFLASH_QUANT=${ATLAS_DFLASH_QUANT:-nvfp4}

# DSpark Markov head control. Auto-on when the checkpoint has it. Set to 0 to
# A/B the head OFF (falls back to plain per-position DFlash argmax).
export ATLAS_DFLASH_MARKOV=${ATLAS_DFLASH_MARKOV:-1}

# Verify-side speedups (target-side, drafter-agnostic — safe to keep on).
export ATLAS_FFN_KGAMMA_M16=${ATLAS_FFN_KGAMMA_M16:-1}
export ATLAS_FFN_FUSED_GATEUP=${ATLAS_FFN_FUSED_GATEUP:-1}
export ATLAS_FFN_M16_TRANSPOSED=${ATLAS_FFN_M16_TRANSPOSED:-1}
export ATLAS_FFN_DOWN_SPLITK=${ATLAS_FFN_DOWN_SPLITK:-4}
export ATLAS_ATTN_QKV_BATCHED=${ATLAS_ATTN_QKV_BATCHED:-1}
export ATLAS_FFN_KGAMMA_M128=${ATLAS_FFN_KGAMMA_M128:-1}

# Drafter-side kgamma transposed weights (propose speedup).
export ATLAS_DFLASH_FFN_KGAMMA=${ATLAS_DFLASH_FFN_KGAMMA:-1}
export ATLAS_DFLASH_ATTN_KGAMMA=${ATLAS_DFLASH_ATTN_KGAMMA:-1}
export ATLAS_DFLASH_NOISE_ONLY=${ATLAS_DFLASH_NOISE_ONLY:-1}

# Task #34 fix: force the chunked WY path at K=12 (see header comment). The
# flat-chain tree_wy injection fires at k == capacity (=12) with graphs on and
# froze the SSM state on every full accept pre-fix; tree_wy also drifts
# numerically on linear chains. Chunked wy4/wy3/wy2 is the proven-lossless
# K≠17 route.
export ATLAS_DISABLE_TREE_WY=${ATLAS_DISABLE_TREE_WY:-1}

# SAM retrieval augmentation (lossless, adaptive-gated) — orthogonal to Markov.
export ATLAS_DFLASH_SAM=${ATLAS_DFLASH_SAM:-1}

# DSpark-AEON draft checkpoint (5-layer DFlash + rank-256 VanillaMarkov head).
DRAFT_MODEL=${DRAFT_MODEL:-/home/flocka/models/DSpark-AEON-draft}

exec /home/flocka/atlas-src/target/release/spark serve \
  --model-from-path "${TARGET_MODEL:-/home/flocka/models/AEON-Q36-27B-Full}" \
  --model-name aeon-27b-dspark \
  --port "${PORT}" \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype "${KV_DTYPE:-fp8}" \
  --max-seq-len ${MAX_SEQ_LEN:-8192} \
  --max-batch-size ${BATCH:-1} \
  --max-num-seqs ${BATCH:-1} \
  --dflash \
  --draft-model "${DRAFT_MODEL}" \
  --dflash-gamma ${DFLASH_GAMMA:-11} \
  --mtp-vocab "${MTP_VOCAB:-32000}" \
  --dflash-quantization "$ATLAS_DFLASH_QUANT" \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas-src/local/warmup.txt
