# Serve-profile flag reference

Every `ATLAS_*` variable `serve.sh` sets, with the value used for the published
decode result and the description taken from the source that reads it. Where the
source carries no comment this table says so rather than guessing — those are
flags whose behaviour you should read at the use site before changing.

Values here are the measured configuration. Changing any of them invalidates the
comparison to the published number; several are not bit-exact (see the numerics
section of the performance recipe).

| Flag | Value | What it does |
|---|---|---|
| `ATLAS_ACCEPT_FAST_ARGMAX` | `1` | Environment kill-switch for the raw-BF16 accept fast path.  Default ON. Set `ATLAS_ACCEPT_FAST_ARGMAX=0` to force every row through the full `process_seq_logits` walk (use only when debugging a drift the fast path might have introduced —… |
| `ATLAS_ATTN_QKV_BATCHED` | `0` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_ATTN_QKV_EXACT_STRIDED` | `1` | scratch and scattering with per-row d2d copies. Bit-exact (same bytes to the same addresses; the exact kernels already deinterleave QG at the store), it only removes the 3 copy launches per row plus their bandwidth. |
| `ATLAS_ATTN_QKV_FUSED` | `1` | Cached `ATLAS_ATTN_QKV_FUSED` env-var lookup. When `1`/`true` the batch3 QKV projection writes directly into the interleaved `qkv_buf` layout and uses a single batched RMS norm launch instead of 6 separate ones. Default off for A/B safety. |
| `ATLAS_ATTN_QKV_SPLITK` | `4` | parsed factor clamped to [2, 8]. A/B against the single-slice baseline — the win is only real if the extra CTAs raise effective bandwidth on the tiny K/V weights. |
| `ATLAS_DDTREE_MAX_NODES` | `$((GAMMA` | for partial-accept rollback. Force `has_mtp` on whenever DFlash is active so the checkpoint pools exist. |
| `ATLAS_DDTREE_TREE_AWARE_VERIFY` | `0` | indirection to land for a real win on non-flat topologies.  depth[t] is derived from the kernel-frame parent_ids stashed by set_ddtree_parent_ids — index 0 is the bonus (depth 0), index i+1 is draft i. parent[i] = -1 means "child of pre-… |
| `ATLAS_DDTREE_TREE_CONV_EXACT` | `0` | ── M8A: DDTree tree-aware GDN verify ── parent_ids_dev is a [num_tokens × i32] device tensor uploaded by verify_d.rs from a.pending_tree_payload before the layer loop. Each token's state load follows parent_ids[i] instead of i-1, letting… |
| `ATLAS_DDTREE_TREE_TOKENS_VERIFY` | `0` | Backward compatibility: when chain_only=1 / chain_seed=true with no branching, `tree_token_ids == drafts` and the two paths produce identical bytes. When the tree is non-flat (M4B v2 with branches), this fix puts the RIGHT tokens at the … |
| `ATLAS_DDTREE_UNCAP` | `0` | ddtree gamma=15 and flat gamma=15 both reported k=16 and accepted 7.03 to three significant figures, with no error logged), and why no env flag could switch it on — including ATLAS_DDTREE_MAX_NODES=32, which raises `physical_drafts` but … |
| `ATLAS_DFLASH_ASYNC` | `0` | Master gate: `ATLAS_DFLASH_ASYNC=1` (default OFF). Cached. ATLAS_DFLASH_FUSED=1: record the propose-ordering event pre-commit so the drafter runs in parallel with SSM commit + KV reshape (~10ms overlap). Requires ATLAS_DFLASH_ASYNC=1. Ca… |
| `ATLAS_DFLASH_ATTN_KGAMMA` | `1` | Like the FFN gate, requires the transposed `nvfp4_t` weight layout (~one-time `transpose_for_gemm` per projection at model build). Default off so the M_TILE=64 path stays baseline until A/B-validated. Cached via `OnceLock`. |
| `ATLAS_DFLASH_CTX_WINDOW` | `4096` | Phase 2.5n: ctx_window controls how many captured target positions the drafter attends to per step. The drafter was trained over the FULL captured prefix (paper §A.1), but capping at γ=16 cripples it on prompts past a tiny window — Atlas… |
| `ATLAS_DFLASH_DRAFT_CAP` | `$GAMMA` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_DFLASH_DRAFT_SPLITK` | `8` | and every dispatch below is byte-for-byte the pre-existing kernel choice. Values `1` and below are treated as off; the kernel's workspace is sized for at most 8 slices, matching `ffn_down_splitk`'s clamp. |
| `ATLAS_DFLASH_FFN_KGAMMA` | `1` | the loader runs an additional `transpose_for_gemm` per FFN projection (one-time at model build, ~1.3 GB H↔D round-trip across 5 layers × 3 projections — measured separately). Default off so the M_TILE=64 path stays the baseline until A/B… |
| `ATLAS_DFLASH_FREE_SLOTS` | `0` | under DFS corrupt — the scheduler now degrades them to flat-safe (see `build_free_slots_payload` doc). Default off ⇒ byte-identical to the flat γ=16 path.  N = the number of sibling branches to place (each ≈ 1 + tail_len extra nodes). Th… |
| `ATLAS_DFLASH_FREE_SLOTS_TAIL` | `4` | Per-branch tail length (post-cliff continuation carried on the sibling). 0 = bare 1-node fork leaves. Default 4 — enough of the predictable post-cliff structure (indentation/closers) to re-accept. |
| `ATLAS_DFLASH_KERNEL_PROFILE` | `0` | Historically only the verify path answered to `ATLAS_FULL_PROFILE`; the drafter's 6 transformer layers used this separate accumulator and so were invisible in a full profile — the drafter's lm_head was the only kernel that appeared, beca… |
| `ATLAS_DFLASH_LM_HEAD_FP8` | `1` | lm_vocab × K (~0.5 GB result, ~1 GB transient). The compensating 1/s goes into the final `norm` weight — its only consumer is the propose lm_head GEMM (noise_pass Step 4), so all downstream logit consumers (argmax, top-2 cliff margins) s… |
| `ATLAS_DFLASH_LM_HEAD_NVFP4` | `1` | ATLAS_DFLASH_LM_HEAD_FP8 (2026-07-31, second attempt): halve the read instead of quartering it — pre-scaled E4M3 slice (built at load; see lm_head_shared_fp8 field doc) keeps ~3 mantissa bits of logit fidelity where the NVFP4 slice's E2M… |
| `ATLAS_DFLASH_NOISE_ONLY` | `1` | With this gate ON, the per-row ops shrink to the noise slice [eff_ctx .. n_attn). Ops that genuinely cover ctx rows are unchanged: k_norm + RoPE (ctx K is cached pre-rope; positions shift per step) and the attention kernel itself (ctx ro… |
| `ATLAS_DFLASH_QUANT` | `nvfp4` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_DFLASH_RETR_WIDE` | `31` | N is clamped to [γ_eff, K_MAX-1] where K_MAX=32 is the tree-WY / ddtree buffer capacity (parent_ids, SSM intermediates, hidden_save). The head carries the exact model allocation and the immutable outer budget clamps N to that capacity mi… |
| `ATLAS_DFLASH_SAM` | `0` | (async defers the syncs the per-kernel timers depend on), so the async-eligibility check below must see the same value. ATLAS_DFLASH_SAM_ASYNC=1 overrides this: when the mtp_step collect always precedes propose, the retrieval early-retur… |
| `ATLAS_DFLASH_SPEC_CYCLE_V2` | `1` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_DFLASH_TREE_COMMIT` | `0` | accepted path (incl. a sibling-fork tail), not just the contiguous flat prefix. `tree_accepted_path` carries the (possibly non-contiguous) compact-index path + the bonus row so the emit + KV-compaction below can lay tokens/KV correctly. … |
| `ATLAS_DISABLE_TREE_WY` | `1` | comment around line 343 — "first-pass tree kernel isn't bit- equivalent to wy17 — flat-chain tokens drift numerically and drafter accept collapses"). When γ matches `ddtree_parent_ids_capacity`, the auto-injection at this site would othe… |
| `ATLAS_DSPARK_ASYMMETRIC_ATTN` | `1` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_FA2_KGAMMA` | `1` | - fa2 kernel resolved at init (`paged_decode_kgamma_fa2_k`)  Takes precedence over the VEC variant when both are enabled. Default off until proven. Cached via `OnceLock`. |
| `ATLAS_FFN_DOWN_SPLITK` | `4` | the single-slice `w4a16_gemm_t_m32_n64` kernel: N=5120 → only 80 CTAs vs gate/up's 256 at N=16384, and it grinds a 512-iteration K-loop. Per full_profile (2026-06-18) it runs at ~91 GB/s vs gate/up ~163 GB/s on the same-size weight. Spli… |
| `ATLAS_FFN_FUSED_GATEUP` | `1` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_FFN_KGAMMA_M128` | `1` | kernel covers M=17 in ONE tile (single weight read); its compute waste on 111 phantom rows is irrelevant because the kernel is bandwidth-bound (see forward_prefill's fp8_fast_path note). Expected ~40ms verify reduction at K=17 on Qwen3.6… |
| `ATLAS_FFN_KGAMMA_M16` | `1` | step (~145 GB of redundant LPDDR5X traffic); the batched path loads each weight once per layer (~8.6 GB) — an 18× reduction on the dominant cost in the K=γ profile. Default off so the per-token loop stays the baseline until A/B-validated… |
| `ATLAS_FFN_M16_TRANSPOSED` | `1` | projections × ~780 KB each). One-time host-side transpose cost at load (~89 MB H↔D per projection). When this gate is off OR the `w4a16_gemm_t_m16` kernel symbol is missing, the existing M_TILE=64 path stays the baseline. Cached via `Onc… |
| `ATLAS_FFN_TC` | `1` | (m32_n64 / m128). The MMA reduction order, BF16 weight rounding and dequant re-association differ from the serial FMA oracle, so enabling this changes token output: the reference completion hash MUST be re-established ("refreeze") after … |
| `ATLAS_FLASH_ATTN_KGAMMA_SPLITK` | `1` | reclaim SM occupancy on long-context decode (4 CTAs → 48 CTAs on a 48-SM GB10). Only consulted when `flash_attn_kgamma_enabled()` is also true; default off until proven. Cached via `OnceLock`. |
| `ATLAS_LM_HEAD_BATCH3` | `1` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_LM_HEAD_T` | `1` | bytes and inject garbage into attention scores. ATLAS_LM_HEAD_T=1: transposed NVFP4 lm_head copy retained for post-construction sharing with the DFlash drafter's propose head (`dflash_lm_head_t`). Target verification no longer reads this… |
| `ATLAS_LM_HEAD_TC` | `1` | so the weight layout and MMA rounding are proven coherent; the target committed token may differ from the scalar oracle by MMA-vs-FMA rounding (a re-reference in the same class as `ATLAS_FFN_TC=1`). Default OFF: the exact scalar path rem… |
| `ATLAS_MULTISEQ_GRAPHS` | `0` | table) is uploaded to those fixed addresses *before* each segment's replay, OUTSIDE the captured region, mirroring vLLM's persistent-batch gather-before-replay. Default off; opt-in via env var. Cached via `OnceLock`. |
| `ATLAS_NO_GEMV_SW` | `1` | Resolved once per process, like the other kernel-path levers in [`crate::layers`]. `std::env::var` is too expensive to call per decode hop. |
| `ATLAS_NVFP4_GATE_UP_M128` | `1` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_PAGED_DECODE_SPLITK` | `1` | — the splitk kernel doesn't do BC=4 batching). Allocation is gated so the larger arena only exists when the env var opts in. max_decode_seqs ≈ 32 (covers K=γ DFlash verify with γ=16 → 17 + headroom) MAX_SPLITS      = max_seq_len / 512, c… |
| `ATLAS_PREFILL_FFN_FAST` | `0` | `qwen3_attention/prefill_weights.rs:14`). Requires the transposed weights AND the `w4a16_gemm_t_m128` kernel symbol; falls back to the M_TILE=64 path silently when either is missing. Default off until A/B-validated. Cached via `OnceLock`… |
| `ATLAS_PREFILL_PROJ_FAST` | `0` | routes them through the non-transposed M_TILE=64 `w4a16_gemm` instead, mirroring the FFN fix. This only affects the PREFILL projections; the transposed weights remain installed for the decode/verify path. Cached via `OnceLock`. |
| `ATLAS_SSM_BA_BATCH` | `1` | γ=16 that is 17 launches/layer × 48 SSM layers = 816 tiny GEMV launches per K=γ verify, each computing only N=64 outputs × K=5120 reductions. The weight (`in_proj_ba`) is IDENTICAL across tokens, so the 17 GEMVs collapse into ONE `dense_… |
| `ATLAS_SSM_GDN_LAZY` | `1` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_SSM_GDN_SEQ_PERSISTENT` | `1` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_SSM_OUT_SPLITK` | `4` | Same lossless FP32-partials + `reduce_splitk_f32_to_bf16` pattern as the shipped `ffn_down` split-K. Returns 0 when unset/0/1; else the factor clamped to [2, 8]. Default OFF pending the counting-md5 gate. |
| `ATLAS_SSM_PROJ_TC` | `1` | partials + `reduce_splitk_f32_to_bf16`, so the reduction order differs from the serial FMA oracle and token output can change: the reference completion hash must be re-established after enabling. Default off (the exact branches keep shad… |
| `ATLAS_SSM_QKVZ_SPLITK` | `4` | 132µs floor); split-K×2 = 167.3µs (85%, 232 GB/s) — ~2.2ms/step across 48 layers. Lossless FP32 partials, mirrors `ffn_down` split-K. Returns 0 when unset/0/1; else clamped to [2, 8]. Default OFF. |
| `ATLAS_TC_NVFP4_M16` | `0` | M ≤ 32) so the parent `w4a16_gemm_n128` (M_TILE=64) stays the default until the new kernel is benched and validated. Cached via `OnceLock` — `std::env::var` is non-trivial to call on every decode hop. |
| `ATLAS_TC_NVFP4_M16_MS_ATTN` | `0` | identified — the dispatch wiring is correct (set_prefill_weights now populates q_nvfp4_t / k_nvfp4_t / v_nvfp4_t for the qwen35_dense loader; m16 + deinterleave_qg matches the FP8/dense gated decode shape; SSM uses the same kernel at the… |
| `ATLAS_THINK_SPEC` | `1` | ! bonus. `</think>` and EOS are phase-boundary tokens — they always end ! the walk as the bonus (never as an accepted draft) so `a.last_token` ! keeps the plain-path contract of "committed but not yet fed to the ! model" across the trans… |
| `ATLAS_TREE_AWARE_ATTN` | `0` | Default OFF until the kernel single-position path is batched (e.g. by re-introducing BC=4 over indirected ancestors, which requires same-block packing — non-trivial because ancestor slots are scattered across blocks for deep trees). |
| `ATLAS_WEIGHT_CACHE` | `1` | *(no comment at the use site — read the source before changing)* |
| `ATLAS_WY17_LAZY` | `1` | Returns 1 (disabled — write all, bit-identical to the historical kernel) when unset/0/1; else the parsed J clamped to [2, 16]. Outputs and final h_state are byte-identical for every J. md5-gated. Cached via `OnceLock`. |
| `ATLAS_WY17_LAZY_COMMIT` | `0` | `gated_delta_rule_wy17_replay` kernel instead of a plain intermediate → h_state D2D copy. Requires the per-layer k/v/gate/beta retention buffers (see `SsmLayerState::wy17_kv_retain` / `wy17_gate_retain`). Default OFF. Cached via `OnceLock`. |
| `ATLAS_WY17_SPLIT` | `2` | (per-column FP32 math + reduction order unchanged). Returns 0 (disabled) when unset/0/1; else the parsed factor clamped to [2, 4] (v_dim=128 → 64 or 32 columns/CTA; beyond 4 the kd_flat recompute dominates). |

49 of 60 carry a description at their use site in `crates/`.

Undocumented at source: `ATLAS_ATTN_QKV_BATCHED`, `ATLAS_DFLASH_DRAFT_CAP`, `ATLAS_DFLASH_QUANT`, `ATLAS_DFLASH_SPEC_CYCLE_V2`, `ATLAS_DSPARK_ASYMMETRIC_ATTN`, `ATLAS_FFN_FUSED_GATEUP`, `ATLAS_LM_HEAD_BATCH3`, `ATLAS_NVFP4_GATE_UP_M128`, `ATLAS_SSM_GDN_LAZY`, `ATLAS_SSM_GDN_SEQ_PERSISTENT`, `ATLAS_WEIGHT_CACHE`.
