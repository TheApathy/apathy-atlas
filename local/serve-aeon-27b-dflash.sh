#!/bin/bash
# Atlas Spark — AEON-7 Qwen3.6-27B target + Qwen3.6-27B-DFlash drafter
# Tight mem because γ=16 inflates SSM-MTP-pool + KV by ~24 GB.
#
# CRASH-SAFE WRAPPER (2026-05-16): preflight kill + port + memory checks
# (host crashed during repeated launches without these). 0.65 utilization
# is already conservative; this script just refuses to start in unsafe state.
set -euo pipefail

PORT=${PORT:-8890}
if pgrep -x spark >/dev/null 2>&1; then
  echo "[serve-aeon-27b-dflash] killing prior spark serve..."
  pkill -9 -x spark || true
  sleep 4
fi
if ss -tnlp 2>/dev/null | grep -q ":${PORT} "; then
  echo "[serve-aeon-27b-dflash] ERROR: port ${PORT} still bound" >&2
  exit 1
fi
FREE_GB=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${FREE_GB:-0}" -lt 50 ]; then
  echo "[serve-aeon-27b-dflash] ERROR: only ${FREE_GB} GB free, need ≥50 GB (DFlash adds 24 GB pool)" >&2
  free -h >&2
  exit 1
fi
echo "[serve-aeon-27b-dflash] preflight ok: ${FREE_GB} GB free"
#
# ATLAS_DFLASH_DRAFT_CAP=16 — full γ=16 drafts + 1 prefix = K=17 verify.
# K=17 triggers gdn_decode_wy17 which saves all 17 intermediates.
# caps 4..15 fall through to the sequential path (no intermediates)
# and corrupt SSM rollback — DO NOT USE.
#
# ATLAS_DFLASH_CTX_WINDOW=512 — drafter trained on full prefix; capping
# at γ cripples accept rate. 512 ≈ 280 MB scratch, affordable.
#
# ATLAS_DFLASH_QUANT={bf16|nvfp4} — drafter weight precision. Defaults to
# bf16 to preserve the pre-existing production path. `nvfp4` runtime-
# quantizes every dense projection in the drafter (7/layer + `fc`) at
# model-load time so the per-step forward runs through the same fast
# `w4a16_gemm` kernels the target model uses, cutting propose latency
# from ~134 ms → ~25-40 ms on GB10 at γ=16, ctx_window=512. RMSNorm and
# bias tensors stay BF16. Frees ~3.3 GB of BF16 source weights post-
# quantize; verify-side parity is preserved because the target's logits
# are always the source of truth.
# IMPORTANT: ATLAS_DFLASH_DRAFT_CAP MUST equal γ (=16) so total verify tokens
# K = γ + 1 = 17 hits the fused `gdn_wy17_k` SSM kernel. Any DRAFT_CAP < γ
# (e.g. 15 → K=16) routes through the sequential per-token SSM path which
# has a NaN bug at positions K-3..K-1 for K>4. Symptom: target output
# becomes `correct_first_token + !!!!!`. Confirmed via 64-layer HF reference
# (modelforge inspect-batched) — atlas_kgamma_layer0_pos13..15 = NaN at
# DRAFT_CAP=15 but pos0..16 all valid at DRAFT_CAP=16.
export ATLAS_DFLASH_DRAFT_CAP=${ATLAS_DFLASH_DRAFT_CAP:-16}
export ATLAS_DFLASH_CTX_WINDOW=${ATLAS_DFLASH_CTX_WINDOW:-512}
export ATLAS_DFLASH_QUANT=${ATLAS_DFLASH_QUANT:-bf16}

exec /path/to/atlas-src/target/release/spark serve \
  --model-from-path /path/to/models/AEON-Q36-27B-Full \
  --model-name aeon-27b-dflash \
  --port 8890 \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype turbo4 \
  --kv-high-precision-layers auto \
  --max-seq-len 8192 \
  --max-batch-size 1 \
  --max-num-seqs 1 \
  --enable-prefix-caching \
  --dflash \
  --draft-model /path/to/models/z-lab-Qwen3.6-27B-DFlash \
  --dflash-gamma 16 \
  --dflash-quantization "$ATLAS_DFLASH_QUANT" \
  --max-thinking-budget 768 \
  --warmup-prompt /path/to/atlas-src/local/warmup.txt
