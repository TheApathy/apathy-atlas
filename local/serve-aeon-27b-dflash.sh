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
# EMPIRICAL 2026-05-17 (post shrink-noise fix): γ=2 (K=3 verify) is the
# robust sweet spot for mixed workloads — 5.4 tok/s mean across
# counting/fibonacci/capital/haiku vs γ=4's 4.5 mean. K=3 hits the proven
# 88ms graphed verify path. After moving DRAFT_CAP enforcement INSIDE
# forward_block.rs (shrink the noise block to γ_eff+1 instead of full γ
# noise + post-filter), drafter latency scales with γ_eff so smaller γ
# now actually translates to lower per-step wall.
#
# Tradeoff: γ=4 still wins on counting alone (7.76 vs 5.64) due to longer
# accepted runs on structured prompts. Switch to DRAFT_CAP=4 for
# counting-heavy / code-gen workloads. γ=2 wins on creative/QA/fibonacci.
# Container's 97 tok/s counting remains out of reach without K=γ kernel
# rewrite (DUET-style state-stationary SSM dataflow).
#
# γ=16 (original published default) is preserved by setting
# DRAFT_CAP=16 explicitly — only useful when running the gdn_wy17_k
# fused safe path matters more than throughput.
export ATLAS_DFLASH_DRAFT_CAP=${ATLAS_DFLASH_DRAFT_CAP:-32}
export ATLAS_LM_HEAD_T="${ATLAS_LM_HEAD_T:-1}"
export ATLAS_DFLASH_CTX_WINDOW=${ATLAS_DFLASH_CTX_WINDOW:-2048}
export ATLAS_DFLASH_QUANT=${ATLAS_DFLASH_QUANT:-bf16}

# 2026-06-10: batched K=γ FFN. KPROF showed the K=17 verify spending
# 760ms/step (of 900ms total) in ssm_ffn_per_token_loop_n17 +
# attn_ffn_per_token_loop_n17 — 64 layers × 17 M=1 GEMVs re-reading FFN
# weights 17× per step. ATLAS_FFN_KGAMMA_M16=1 routes verify through
# forward_kgamma; ATLAS_FFN_M16_TRANSPOSED=1 upgrades the FFN GEMMs to
# w4a16_gemm_t_m16, which was flag-isolated CLEAN at M=17 (token-exact
# greedy counting, 15.6/16 mean acceptance). Measured on counting:
# verify 900→244ms, 42.6 tok/s at accepted 15.62/16.
# ATLAS_DISABLE_TREE_WY=1 is the γ=16 correctness fix from c588b34.
export ATLAS_FFN_KGAMMA_M16=${ATLAS_FFN_KGAMMA_M16:-1}
export ATLAS_FFN_M16_TRANSPOSED=${ATLAS_FFN_M16_TRANSPOSED:-1}
export ATLAS_DISABLE_TREE_WY=${ATLAS_DISABLE_TREE_WY:-1}

# 2026-06-10: noise-rows-only drafter layers (upstream dflash.py
# alignment — ctx enters attention as cached K/V only; input_norm / q /
# o / FFN / residuals run on the γ+1 noise rows instead of all
# ctx+noise rows). propose 145→88ms. Validated token-exact +
# deterministic + acceptance 15.90/16. Counting: 51.0 tok/s
# (step 331ms = verify 243 + propose 88).
export ATLAS_DFLASH_NOISE_ONLY=${ATLAS_DFLASH_NOISE_ONLY:-1}

# 2026-06-10: batched K=17 attention QKV via plain w4a16_gemm (M=17, one
# weight read instead of 17 per layer). Validated token-exact +
# deterministic + acceptance 15.90/16. verify 243→231ms → 52.8 tok/s
# counting. Also proves the gated-layer corruption lives in the
# w4a16_gemm_t_m16/NVFP4-T path, NOT in deinterleave_qg.
export ATLAS_ATTN_QKV_BATCHED=${ATLAS_ATTN_QKV_BATCHED:-1}

# 2026-06-11: nsys-guided kernel routing. Ground-truth trace showed the
# K=17 verify dominated by small-M GEMM inefficiency (GPU ~80% busy, no
# graph gaps): plain w4a16_gemm ran ~5x off the bandwidth floor at M=17
# (strided B reads), and w4a16_gemm_t_m16 re-reads weights per 16-row
# tile. Routing FFN + batched qkv through the transposed M_TILE=128
# kernel (single weight read, coalesced): verify 238->199ms.
# ATLAS_DFLASH_FFN_KGAMMA=1 quantizes the drafter FFN to m16-transposed
# at load: propose 88->61ms. Combined: counting 38->43.4, coding
# 23.9->26.2, prose ~14->16.2 tok/s (usage-method, vs container
# 74.3/51.6/21.0).
export ATLAS_FFN_KGAMMA_M128=${ATLAS_FFN_KGAMMA_M128:-1}
export ATLAS_DFLASH_FFN_KGAMMA=${ATLAS_DFLASH_FFN_KGAMMA:-1}

# ── DO NOT ENABLE: TC_NVFP4_M16 attention path corrupts at K=17 ───────
# ATLAS_TC_NVFP4_M16=1 + ATLAS_TC_NVFP4_M16_MS_ATTN=1 (ms_phase_qkv
# M_TILE=16 q/k/v path) was flag-isolated as the corruptor by greedy
# controlled A/B 2026-06-10: with it, verify deep-slot argmax repeats
# earlier digits ("39, 40, 40, 4443..."), greedy determinism breaks
# across requests, and acceptance collapses 15.6/16 → ~1.5/16 (which
# masquerades as a drafter-quality problem). All other combinations
# (flags off / KGAMMA only / KGAMMA+TRANSPOSED) are token-exact. It
# bought only ~20ms of verify — not worth debugging until propose-side
# work is done. Suspect: qkv N-dims (q=8192? kv=1024) vs the kernel's
# (gns + 15 < N) predicates, or q/k/v transposed-weight build.
export ATLAS_TC_NVFP4_M16=${ATLAS_TC_NVFP4_M16:-0}
export ATLAS_TC_NVFP4_M16_MS_ATTN=${ATLAS_TC_NVFP4_M16_MS_ATTN:-0}

# Drafter checkpoint. Default = generic z-lab drafter (trained for base
# Qwen3.6-27B). The AEON-tuned variants (-aeon-tuned/-aeon-v2/
# -aeon-v3-balanced) match the abliterated target's distribution and
# should accept more drafts per step on prose.
DRAFT_MODEL=${DRAFT_MODEL:-/path/to/models/z-lab-Qwen3.6-27B-DFlash}

exec /path/to/atlas-src/target/release/spark serve \
  --model-from-path /path/to/models/AEON-Q36-27B-Full \
  --model-name aeon-27b-dflash \
  --port 8890 \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype "${KV_DTYPE:-fp8}" \
  --max-seq-len 8192 \
  --max-batch-size 1 \
  --max-num-seqs 1 \
  --dflash \
  --draft-model "${DRAFT_MODEL}" \
  --dflash-gamma 16 \
  --mtp-vocab "${MTP_VOCAB:-131072}" \
  --dflash-quantization "$ATLAS_DFLASH_QUANT" \
  --max-thinking-budget 768 \
  --warmup-prompt /path/to/atlas-src/local/warmup.txt
