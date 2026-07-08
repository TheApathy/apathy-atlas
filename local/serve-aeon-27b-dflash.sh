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
export ATLAS_DFLASH_DRAFT_CAP=${ATLAS_DFLASH_DRAFT_CAP:-16}
export ATLAS_LM_HEAD_T="${ATLAS_LM_HEAD_T:-1}"
# 4096 matches the jun26 drafter's trained SWA window (was 2048 for the 3.6 drafter).
export ATLAS_DFLASH_CTX_WINDOW=${ATLAS_DFLASH_CTX_WINDOW:-4096}
# 2026-06-17: default MUST be nvfp4. With bf16 every drafter projection runs
# ops::dense_gemm over all n_attn = eff_ctx(≤CTX_WINDOW) + γ rows — propose
# scaled to ~1.75s/step at seq 800 (counting 13 tok/s, coding TIMEOUT).
# nvfp4 routes the same projections through w4a16_gemm (the target's kernel):
# propose 1751→48ms, step 1910→206ms, counting 13.1→64.9, coding TIMEOUT→16.8,
# prose ~2→10.9. Token-exact (verify is the source of truth; drafter quant only
# affects acceptance, never committed greedy tokens).
export ATLAS_DFLASH_QUANT=${ATLAS_DFLASH_QUANT:-nvfp4}

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
# 2026-07-02: fused gate+up+SiLU kernel (one launch, shared A-tile, dual cp.async
# weight streams, BF16-round-trip reproduced in-kernel → counting md5 byte-identical).
# Measured: gate+up+silu 42.8→34.8ms/step, counting AND coding +5.5%.
export ATLAS_FFN_FUSED_GATEUP=${ATLAS_FFN_FUSED_GATEUP:-1}
export ATLAS_FFN_M16_TRANSPOSED=${ATLAS_FFN_M16_TRANSPOSED:-1}
export ATLAS_DISABLE_TREE_WY=${ATLAS_DISABLE_TREE_WY:-1}

# 2026-06-10: noise-rows-only drafter layers (upstream dflash.py
# alignment — ctx enters attention as cached K/V only; input_norm / q /
# o / FFN / residuals run on the γ+1 noise rows instead of all
# ctx+noise rows). propose 145→88ms. Validated token-exact +
# deterministic + acceptance 15.90/16. Counting: 51.0 tok/s
# (step 331ms = verify 243 + propose 88).
export ATLAS_DFLASH_NOISE_ONLY=${ATLAS_DFLASH_NOISE_ONLY:-1}

# 2026-06-19: SAM longest-suffix RETRIEVAL augmentation (retrieval.rs),
# "retrieval-when-confident, diffusion-otherwise". LOSSLESS by construction —
# retrieval drafts pass the SAME greedy DFlash verify, so output is always the
# target's argmax (changes SPEED not output; counting MD5 bit-exact either way).
#
# DEFAULT ON, made safe by the ADAPTIVE GATE (2026-06-21). SAM alone is a big
# win on reuse-heavy code EDITING (add_method, ~100-line context: 45→71 tok/s)
# but REGRESSED counting 77→65 (digit runs produce strong-but-wrong suffix
# matches → wasted drafts). The adaptive gate (propose.rs, default ON via
# ATLAS_DFLASH_SAM_ADAPTIVE) tracks retrieval accept per-seq and auto-disables
# it after 3 consecutive misfires for a 24-step cooldown — so it stays active
# on editing and backs off on counting/novel content. LOSSLESS either way
# (verify commits the target's greedy token). Disable retrieval: ATLAS_DFLASH_SAM=0.
export ATLAS_DFLASH_SAM=${ATLAS_DFLASH_SAM:-1}

# 2026-07-02: PORTFOLIO verify (ATLAS_DFLASH_PORTFOLIO=1, default OFF).
# Verifies TWO independent flat chains in ONE K=32 pass — the DFlash drafter's
# 16-token chain AND the SAM retrieval chain — as a 2-root forest (both attach
# to the shared root). At each step the target rides whichever chain it actually
# continues onto, so a divergence the drafter mispredicts can still be covered by
# the retrieval sibling in the SAME bandwidth-bound verify (K=32 costs ≈ K=17).
# LOSSLESS by the greedy-oracle argument (verify commits only the target's argmax
# — proven byte-identical counting md5). Retrieval no longer PRE-EMPTS the
# drafter here, so its hybrid gate is loosened (fires on any real match, not only
# the strongest). Requires the wide verify buffers + depth-aware verify path:
#   ATLAS_DDTREE_MAX_NODES=32          → allocate 32-node parent_ids buffers
#   ATLAS_DDTREE_TREE_TOKENS_VERIFY=1  → embed each slot's tree-topology token
#   ATLAS_DDTREE_DFS_REORDER=1         → DFS pre-order = contiguous ancestor
#                                        reads per chain (validated lossless)
# Chain A (drafter) is never truncated (byte-exact drafter baseline when no
# retrieval sibling fires); chain B (retrieval) takes the remaining budget.
# Enable all four together:
#   ATLAS_DFLASH_PORTFOLIO=1 ATLAS_DDTREE_MAX_NODES=32 \
#   ATLAS_DDTREE_TREE_TOKENS_VERIFY=1 ATLAS_DDTREE_DFS_REORDER=1
if [ "${ATLAS_DFLASH_PORTFOLIO:-0}" = "1" ]; then
  export ATLAS_DDTREE_MAX_NODES=${ATLAS_DDTREE_MAX_NODES:-32}
  export ATLAS_DDTREE_TREE_TOKENS_VERIFY=${ATLAS_DDTREE_TREE_TOKENS_VERIFY:-1}
  export ATLAS_DDTREE_DFS_REORDER=${ATLAS_DDTREE_DFS_REORDER:-1}
  echo "[serve-aeon-27b-dflash] PORTFOLIO verify ON (2-root forest, K=32, lossless)"
fi

# 2026-07-05: FREE-SLOTS branch verify (ATLAS_DFLASH_FREE_SLOTS=<N>, default OFF).
# Keeps the FULL γ=16 spine AND spends the free K=32 verify slots on N sibling
# branches placed at the LOW-CONFIDENCE draft positions where the linear chain
# statistically dies (the "cliffs"), each carrying the drafter's top-2 at the
# cliff + a short re-rooted tail. Same bandwidth-bound verify (K=32 ≈ K=17) →
# more of the target's greedy continuation accepted (DDTree: up to +46% accept on
# coding). LOSSLESS by the greedy-oracle argument (verify commits only the
# target's argmax). Requires the same wide-verify + depth-aware commit path as
# PORTFOLIO, plus ATLAS_DFLASH_TREE_COMMIT=1 so the deep branch tail commits
# (not just the flat prefix). Companion knobs:
#   ATLAS_DFLASH_FREE_SLOTS=<N>        → number of sibling branches (each ≈1+tail)
#   ATLAS_DFLASH_FREE_SLOTS_TAIL=<L>   → per-branch tail length (default 4)
#   ATLAS_DFLASH_BRANCH_MARGIN=<m>     → cliff = top1-top2 margin < m (default 2.0)
# Enable:
#   ATLAS_DFLASH_FREE_SLOTS=3 ATLAS_DDTREE_MAX_NODES=32 \
#   ATLAS_DDTREE_TREE_TOKENS_VERIFY=1 ATLAS_DDTREE_DFS_REORDER=1 \
#   ATLAS_DFLASH_TREE_COMMIT=1
if [ "$(printf '%s' "${ATLAS_DFLASH_FREE_SLOTS:-0}" | tr -cd '0-9')" != "0" ] \
   && [ "${ATLAS_DFLASH_FREE_SLOTS:-0}" != "0" ]; then
  export ATLAS_DDTREE_MAX_NODES=${ATLAS_DDTREE_MAX_NODES:-32}
  export ATLAS_DDTREE_TREE_TOKENS_VERIFY=${ATLAS_DDTREE_TREE_TOKENS_VERIFY:-1}
  export ATLAS_DDTREE_DFS_REORDER=${ATLAS_DDTREE_DFS_REORDER:-1}
  export ATLAS_DFLASH_TREE_COMMIT=${ATLAS_DFLASH_TREE_COMMIT:-1}
  echo "[serve-aeon-27b-dflash] FREE-SLOTS branch verify ON (N=${ATLAS_DFLASH_FREE_SLOTS}, K=32, lossless)"
fi

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
# 2026-06-11 records-grade: also route the drafter's attention projections
# (q/k/v/o, 7 GEMM sites) through the transposed m32_n64 kernel by building
# drafter T-weights at load. Cuts propose; with the nvfp4 drafter fix this
# restores counting to records parity (65->74.4). Token-exact (drafter-only).
export ATLAS_DFLASH_ATTN_KGAMMA=${ATLAS_DFLASH_ATTN_KGAMMA:-1}

# 2026-06-18: split-K down_proj. full_profile exposed the K=17 verify's #1
# kernel sink as ffn_down_kgamma — the down projection ([M=17,N=5120,K=16384])
# fields only 80 CTAs on the single-slice w4a16_gemm_t_m32_n64 kernel (vs
# gate/up's 256 at N=16384) and runs at ~91 GB/s vs gate/up's ~163 on the same
# 47MB weight, i.e. SM-starved on the long K-loop. ATLAS_FFN_DOWN_SPLITK=4
# slices K across gridDim.z (80->320 CTAs) into an FP32 workspace, then
# reduce_splitk_f32_to_bf16 sums+downcasts. Lossless (FP32 partials) and
# token-exact (counting md5 unchanged). Clean A/B: counting 75.2->82.0 (+9%),
# coding 16.5->17.9, prose 12.3->12.9 — lifts every workload since verify runs
# every step. Requires the w4a16_gemm_t_m32_n64_splitk + reduce kernels (built
# into the qwen3.6-27b cache); handle-0 fallback keeps the single-slice path.
export ATLAS_FFN_DOWN_SPLITK=${ATLAS_FFN_DOWN_SPLITK:-4}

# 2026-07-04 (kernel wave 2): split-K for the attention K/V projections on the
# K=γ verify QKV path. K/V are N=nkv*hd=1024 → only 16 CTAs on the single-slice
# w4a16_gemm_t_m32_n64 (severely SM-starved on the 48-SM GB10); Q at
# N=q_proj_dim=12288 fields 192 CTAs and stays single-slice. ATLAS_ATTN_QKV_
# SPLITK=4 slices K across gridDim.z (16→64 CTAs each) into an FP32 workspace +
# reduce — exactly the proven ffn_down pattern. Clean A/B (idle box, n=5):
# counting 84.9→86.7 (+2.1%), coding 41.2→42.5 (+3.2%), step 157.9→155.0ms;
# counting md5 byte-identical. Wave-2 siblings measured same-day, left OFF:
#   ATLAS_WY17_SPLIT=2 (v-dim split of gated_delta_rule_wy17): honest NO-OP
#     (84.8/41.5, step 158.2ms) — 48 CTAs at 1/SM isn't starved enough to pay
#     for the per-split kd_flat recompute. md5 clean; kernel kept in tree.
#   ATLAS_SSM_BA_BATCH=1 (17 BA GEMVs → 1 dense_gemv_bf16_batchn launch):
#     marginal (85.7/41.6, step 155.9ms, ~+1%). md5 clean via the bit-exact
#     batchn kernel (a dense_gemm variant was NOT bit-exact — md5 mismatch;
#     do not swap back).
#   All three combined: 85.7 counting / 42.7 coding, step 152.3ms.
export ATLAS_ATTN_QKV_SPLITK=${ATLAS_ATTN_QKV_SPLITK:-4}

# 2026-07-05 (WY17 lazy Hi-writes): the gated_delta_rule_wy17 verify kernel
# writes 16 per-token intermediate states as a partial-accept safety net —
# 86% of its DRAM traffic. Lazy mode writes only checkpoint slots {0, K-2,
# every J-th} and reconstructs the rare skipped-slot partial-accept via a
# bit-exact root-replay kernel (gdn_wy17_replay). Microbench 306→111µs/call
# (-64%). Validated e2e A/B (idle box): counting 86.6→88.1 (+1.7%), coding
# 39.2→41.8 (+6.6%), prose 13.5→14.0 (+3.7%); counting md5 == 91a6ff90
# constitution (replay path exercised, LOSSLESS). Prose (low-accept → more
# replays) did NOT regress — each replay is one cheap partial kernel. SHIP.
export ATLAS_WY17_LAZY=${ATLAS_WY17_LAZY:-8}
export ATLAS_WY17_LAZY_COMMIT=${ATLAS_WY17_LAZY_COMMIT:-1}

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
# 2026-07-02: default drafter switched to the mid-June z-lab Qwen3.5-27B-DFlash
# refresh (6 layers, SWA window 4096) — A/B vs the stale Apr-27 3.6 drafter:
# counting 71.7 vs 68.3, code_novel 15.9 vs 15.2, prose par, counting-md5
# LOSSLESS. Cross-target (3.5-trained) but wins anyway; larger SWA window may
# help more at long context. Old drafter: z-lab-Qwen3.6-27B-DFlash.
# 2026-07-08: default → v5-goheavy (2-epoch warm-start from v4-scale on the
# combined 8712-sample NVFP4-captured corpus, go share 1.6%→11.5%). Same-
# harness A/B vs v4-scale: go +4.6%, counting +1.2%, py neutral; counting md5
# byte-identical (lossless swap) and ABBA HumanEval x60 chat pass@1 gate:
# 95.0% vs 95.0%, delta CI [0,0] — identical outputs on every problem. SHIP.
DRAFT_MODEL=${DRAFT_MODEL:-/path/to/dflash-retrain/v5-ckpt-goheavy/epoch_2_step_16732}

exec /path/to/atlas-src/target/release/spark serve \
  --model-from-path "${TARGET_MODEL:-/path/to/models/AEON-Q36-27B-Full}" \
  --model-name aeon-27b-dflash \
  --port 8890 \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype "${KV_DTYPE:-fp8}" \
  --max-seq-len ${MAX_SEQ_LEN:-8192} \
  --max-batch-size ${BATCH:-1} \
  --max-num-seqs ${BATCH:-1} \
  --dflash \
  --draft-model "${DRAFT_MODEL}" \
  --dflash-gamma ${DFLASH_GAMMA:-16} \
  --mtp-vocab "${MTP_VOCAB:-32000}" \
  --dflash-quantization "$ATLAS_DFLASH_QUANT" \
  --max-thinking-budget 768 \
  --warmup-prompt /path/to/atlas-src/local/warmup.txt
