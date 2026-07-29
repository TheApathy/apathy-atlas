// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash block-diffusion draft head implementing [`DraftProposer`].
//!
//! Block-diffusion drafter (Z Lab, arXiv 2602.06036): a small Qwen3-architecture
//! transformer (8 layers, hidden=2048, GQA 32:4, head_dim=128) that emits γ=16
//! tokens **in a single forward pass** via bidirectional in-block attention.
//! Conditioned on five intermediate hidden states captured from the target
//! model at `target_layer_ids` (e.g., `[1, 10, 19, 28, 37]` for
//! Qwen3.6-35B-A3B-DFlash), projected through a single `fc` layer at model
//! entry — NOT per-layer KV injection (early plan was wrong; cf. vLLM
//! `qwen3_dflash.py`).
//!
//! Phase 1 deliverable: type + trait wiring. The actual γ-block forward kernel
//! (`inferspark_dflash_block_attn_fp8`) lands in Phase 2; until then `propose()`
//! returns the bonus token repeated `num_drafts` times so the verify path
//! degenerates to single-token decode (acceptance ~100% but no speedup).

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Kernel handles for the DFlash γ-block forward chain. All resolved once
/// at `BlockDiffusionDraftHead::from_weights` against the active GPU backend
/// (which compiles target-specific PTX at startup); subsequent
/// `propose()` calls just `KernelLaunch::new(...).launch(stream)`.
pub struct DflashKernels {
    pub rms_norm: KernelHandle,
    pub residual_rms_norm: KernelHandle,
    pub dense_gemv: KernelHandle,
    pub dense_gemm: KernelHandle,
    /// NVFP4 GEMM for the final logits when the shared lm_head is NVFP4
    /// (e.g. Holo): a BF16 `dense_gemm` on NVFP4-packed bytes reads garbage
    /// (and ~4× OOB → CUDA-700). `.0 == 0` when the target lm_head is BF16.
    pub w4a16_gemm: KernelHandle,
    pub dense_gemm_pipelined: KernelHandle,
    /// Small-M (M ≤ 16) BF16 weight-streaming GEMM
    /// (`dense_gemm_bf16_mtile16`, kernels/gb10/common/). Drop-in for
    /// `dense_gemm_pipelined` on the drafter's M=γ propose GEMMs when
    /// `ATLAS_DFLASH_DRAFTER_FASTGEMM=1` (default OFF). Output is
    /// bit-identical (same m16n8k16 ascending-K FP32-accumulate chain);
    /// the win is pure weight-read bandwidth: N_TILE=64 grid (2-6× the
    /// CTA count of the 128×128 tile at drafter N's) + 4-stage cp.async
    /// ring (~24 KB of B in flight per CTA vs ~8 KB). `.0 == 0` when the
    /// target's kernel set lacks it (non-gb10 targets) — dispatch falls
    /// back to `dense_gemm_pipelined`. Used for N ≤ 2048 (kv/g_proj).
    pub dense_gemm_mtile16: KernelHandle,
    /// N_TILE=128 wide-stream sibling of `dense_gemm_mtile16` (kernel
    /// `dense_gemm_bf16_mtile16_n128`, same .cu). Used for N > 2048 —
    /// fewer, longer B streams win on LPDDR5x at large N. `.0 == 0` →
    /// pipelined fallback.
    pub dense_gemm_mtile16_n128: KernelHandle,
    pub rope_qwen3: KernelHandle,
    pub reshape_cache_fp8: KernelHandle,
    /// BF16 KV cache writeback. Used by Phase 2 `precompute_ctx_kv` and
    /// the per-layer γ-block `reshape_and_cache` call to populate the
    /// drafter's BF16 paged cache before each `prefill_attention_paged_dflash`.
    pub reshape_cache_bf16: KernelHandle,
    pub prefill_attn_dflash_fp8: KernelHandle,
    /// BF16 paged-attention dispatcher for the DFlash γ-block.
    /// Calls `inferspark_prefill_paged` with `causal_mask_enabled=0`,
    /// reading BF16 K/V from the per-layer paged cache pool. Phase 2
    /// (Option B) drafter attention runs through this kernel; the FP8
    /// variant above is retained for a future quality-validated FP8 KV
    /// path. See `ops::prefill_attention_paged_dflash`.
    pub prefill_attn_dflash_bf16: KernelHandle,
    /// Phase 5 (CUDA graph) variant of `prefill_attn_dflash_bf16` that reads
    /// `kv_len` and `q_offset` from device pointers instead of taking them as
    /// kernel scalar args. Used by the graph-captured forward_block path so a
    /// single graph instance can be replayed across steps with different
    /// dynamic values written to the indirect-args buffer pre-launch.
    /// Resolves to kernel `inferspark_prefill_paged_indirect`.
    pub prefill_attn_dflash_bf16_indirect: KernelHandle,
    pub silu_mul: KernelHandle,
    pub residual_add: KernelHandle,
    pub argmax: KernelHandle,
    /// Per-row top-2 over the drafter logits (block-fork tree cliff
    /// detection, doc 16). `0` when the target's kernel set lacks it.
    pub top2: KernelHandle,
    pub batched_embed: KernelHandle,
    /// Phase 2 Option B: builds `[count]` i32 slot indices on-device
    /// from a host-provided block_table. Used by propose.rs to populate
    /// the slot_mapping passed to reshape_and_cache and precompute_ctx_kv.
    pub fill_slots: KernelHandle,
    /// Non-paged prefill attention (used for the γ-block self-attention
    /// when there's no persistent K/V cache to walk).
    pub prefill_attn: KernelHandle,
    /// Phase G — BF16 → FP8 E4M3 per-row weight quantization. Used at
    /// model load time to convert the seven dense-GEMM drafter weights
    /// (q/k/v/o/gate/up/down) when `ATLAS_DFLASH_DRAFTER_FP8=1`. Never
    /// on the hot path.
    pub quantize_bf16_to_fp8: KernelHandle,
    /// Phase G — Row-scaled BF16 × FP8 → BF16 GEMM. Consumes the
    /// `Fp8DenseWeight` (FP8 weight + per-row f32 scale) produced at
    /// load time by `quantize_bf16_to_fp8`. Wraps
    /// `kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu fp8_gemm_t_row_scaled`.
    /// Replaces `dense_gemm_bf16` on the seven dense-GEMM call sites in
    /// `forward_block_layer_pre_attn` / `_post_attn` when
    /// `self.quant == DflashQuantization::Fp8Weights`.
    pub fp8_gemm_n128_row_scaled: KernelHandle,
    /// Phase G — Row-scaled BF16 × FP8 → BF16 GEMV (M=1) for the
    /// lm_head fall-back. At γ=16 vs vocab=248320 the row-scaled GEMM
    /// wastes 75% of its M_TILE; the GEMV in a γ-loop is faster.
    pub dense_gemv_fp8w: KernelHandle,
    /// Phase G — Small-M (M≤16) row-scaled FP8 GEMM. Drop-in replacement
    /// for `fp8_gemm_n128_row_scaled` when M=γ=16. Single warp per CTA,
    /// no wasted M_TILE rows. Used by the lm_head GEMM.
    pub fp8_gemm_n128_row_scaled_m16: KernelHandle,
    /// Weight-read-bound M ≤ 8 row-scaled FP8 GEMM
    /// (`w4a16::fp8_gemm_t_row_scaled_mtile8`, N_TILE=64, 4-stage cp.async
    /// ring). Preferred over `_m16` for the FP8 lm_head tail when γ ≤ 8 and
    /// K % 32 == 0 — it streams the vocab×hidden mirror once at near-wall
    /// GB/s (the single-warp _m16 tile measured ~40% lower cold weight-read
    /// bandwidth at the mirror shapes). 0-handle on miss — dispatch falls
    /// back to `_m16`.
    pub fp8_gemm_row_scaled_mtile8: KernelHandle,
    /// N_TILE=32 sibling of `_mtile8` for small-N/large-K M ≤ 8 shapes
    /// (`w4a16::fp8_gemm_t_row_scaled_mtile8_n32`). At N=3072 the N_TILE=64
    /// tile launches ceil(3072/64) = 48 CTAs = exactly 1/SM on GB10, and one
    /// 4-stage cp.async ring per SM cannot hide LPDDR5x latency; the N_TILE=32
    /// grid doubles that to 96 CTAs = 2/SM. Bit-identical accumulation chain.
    /// Serves the drafter's o_proj (N=3072, K=9216) and down_proj (N=3072,
    /// K=12288). 0-handle on miss — dispatch falls back to `_mtile8`.
    pub fp8_gemm_row_scaled_mtile8_n32: KernelHandle,
    /// Laguna per-head attention-output gate: applies
    /// `out[t,h,d] = in[t,h,d] * softplus(gate[t,h])`, broadcasting one
    /// softplus scalar per head across `head_dim`. Consumes the `[γ, num_q_heads]`
    /// gate produced by GEMV'ing the layer input hidden through `g_proj`.
    /// Resolves to `softplus_gate_mul_head_broadcast` (kernels/gb10/common/
    /// residual_add.cu). Only used when the drafter ships a per-head `g_proj`.
    pub softplus_gate: KernelHandle,
}

/// Per-step scratch buffers for the γ-block forward.
///
/// Sized for `n_attn_slots = ctx_window + γ` rows, where ctx_window is the
/// max number of past target positions the drafter attends to per step. The
/// first `ctx_window` slots hold post-`fc` projected target context (K/V
/// only — Q is zero-padded); the next γ slots hold the noise tokens.
///
/// At γ=16 and ctx_window=γ=16: 32 rows × 2048 BF16 × ~10 buffers = ~1.3 MB
/// per head. lm_head logits buffer is the largest single alloc:
/// 32 × 248320 × 2 = 15 MB.
pub struct DflashScratch {
    pub stream_buf: DevicePtr,
    pub norm_buf: DevicePtr,
    pub q_buf: DevicePtr,
    pub k_buf: DevicePtr,
    pub v_buf: DevicePtr,
    pub attn_out: DevicePtr,
    pub mlp_intermediate: DevicePtr,
    pub mlp_up: DevicePtr,
    pub stream_acc: DevicePtr,
    /// `[ctx_window, draft_hidden]` BF16 — fc-projected + hidden_norm'd
    /// ctx for the most recent `ctx_window` target positions.
    pub fc_proj: DevicePtr,
    /// Phase 2 (Option B) scratch for `precompute_ctx_kv`: fused KV
    /// GEMM output, shape `[max_new_ctx, L * 2 * kv_dim]` BF16.
    /// `max_new_ctx` = `ctx_window` (worst case: first propose runs
    /// precompute over the entire prefix).
    pub fused_kv_out: DevicePtr,
    /// Phase 2 scratch: i32 slot mapping for the per-layer
    /// `reshape_and_cache` calls. Sized `[ctx_window]`.
    pub slot_mapping_dev: DevicePtr,
    /// Phase 5 (CUDA graph) scratch: 8 bytes (`[u32 kv_len, u32 q_offset]`)
    /// holding the per-call dynamic values that the indirect paged-attention
    /// kernel reads at entry. Host writes via `copy_h2d` BEFORE entering the
    /// captured region so the graph itself sees a stable device pointer.
    pub option_b_indirect_args_dev: DevicePtr,
    /// Phase E.2: pinned host buffer (`γ × 4` bytes) for the per-propose
    /// draft-token D2H copy. Allocated once at construction via
    /// `gpu.alloc_host_pinned`; the async D2H lands here without touching
    /// the system pageable allocator each call.
    ///
    /// Wrapped in `AtomicPtr` to keep `DflashScratch: Send + Sync` (the
    /// proposer is stored as `Arc<dyn DraftProposer>` which requires both
    /// auto-traits). Reads via `Ordering::Relaxed` are safe: the pointer
    /// itself never changes after construction; we only need atomic
    /// access for the Send/Sync bound, not for any actual concurrency.
    pub draft_tokens_host_pinned: std::sync::atomic::AtomicPtr<u8>,
    /// Phase E.2: CUDA event recorded against the draft-tokens D2H so the
    /// host can block on completion just before reading the pinned buffer,
    /// without a full `cuStreamSynchronize`. Created once at construction.
    pub draft_tokens_event: u64,
    pub logits: DevicePtr,
    pub draft_tokens_dev: DevicePtr,
    /// `[ctx_window + γ]` i32 positions. First ctx_window are
    /// historical target positions (decoded indices); last γ are
    /// the to-be-predicted noise positions.
    pub position_ids: DevicePtr,

    /// Laguna per-capture aux-norm pre-pass scratch. `[ctx_window, L_t*h_t]`
    /// BF16 — holds a per-capture RMS-normalised copy of the captured target
    /// hiddens (each `h_t`-wide capture slice normalised by its own
    /// `aux_hidden_norms[k]`) that `fc` then reads instead of the raw
    /// accumulator. `DevicePtr::NULL` when the drafter has no aux norms
    /// (Qwen3.6-DFlash), in which case the fc path reads the accumulator
    /// directly (unchanged).
    pub aux_normed: DevicePtr,
    /// Laguna aux-norm gather temp. `[ctx_window, h_t]` BF16 — one capture
    /// index's slices for all rows, gathered contiguously so the contiguous
    /// `rms_norm` kernel can normalise them, then scattered back into
    /// `aux_normed`. `DevicePtr::NULL` when the drafter has no aux norms.
    pub aux_slice: DevicePtr,
    /// Laguna per-head gate scratch: `[γ, num_q_heads]` BF16 — the raw
    /// `hidden_in @ g_proj.T` logits (pre-softplus) for the γ noise rows.
    /// `softplus_gate_mul_head_broadcast` reads this and applies the softplus
    /// in fp32 internally. `DevicePtr::NULL` when the drafter has no `g_proj`
    /// (Qwen3.6-DFlash) — the gate is then skipped entirely.
    pub gate_buf: DevicePtr,
}

/// Drafter-side weight precision. Defaults to BF16. **Phase G (2026-05-28)**
/// adds `Fp8Weights`, gated by env var `ATLAS_DFLASH_DRAFTER_FP8`. The
/// historical SM12.x acceptance collapse note applied to drafter FP8 KV
/// cache (different concern — bidirectional attention math); Phase G
/// targets weight FP8 only, so the risk surface is dynamic-range loss
/// in MLP intermediate activations, which per-row scales mitigate.
/// `--mtp-quantization fp8` is still not honored for the DFlash drafter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashQuantization {
    Bf16,
    /// Weight-only FP8: q/k/v/o/gate/up/down BF16 → FP8 E4M3 with per-row
    /// f32 scales at model load. Activations stay BF16; KV cache stays
    /// BF16. GEMMs use `fp8_gemm_n128` (BF16 × FP8 → BF16).
    Fp8Weights,
    /// Attention-only weight FP8 (`ATLAS_DFLASH_DRAFTER_FP8_ATTN=1`):
    /// only q/k/v/o are mirrored to FP8 per layer; gate/up/down and the
    /// shared lm_head stay BF16. Motivation: the full-FP8 accept collapse
    /// (3.09 → 2.59) is attributed to the MLP/lm_head — the target model's
    /// q/k/v/o at FP8 measured accept-neutral. Dispatch is per-GEMM on FP8
    /// mirror presence, so this variant is observability-only.
    Fp8AttnWeights,
}

/// Per-drafter-layer Qwen3-style weights. Phase 1 is BF16-only; **Phase G**
/// (2026-05-28) adds optional FP8 weight fields populated at model load
/// when `ATLAS_DFLASH_DRAFTER_FP8=1`. The BF16 fields are always present
/// (Fp8 path falls back to them for any GEMM whose Fp8 weight is None).
#[allow(dead_code)]
pub struct DflashLayer {
    // Norms
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    // Attention (Qwen3: per-head Q/K RMSNorm)
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    /// Per-head attention output gate (`self_attn.g_proj.weight`, `[num_q_heads, hidden]`).
    /// Present ONLY on the Laguna drafter (`gating: per-head`); `None` for the
    /// Qwen3.6-DFlash drafters. When set, the forward path computes
    /// `g = softplus(hidden_in @ g_proj.T)` [tokens, num_q_heads] and multiplies
    /// it head-wise into the attention output (broadcast over head_dim) BEFORE
    /// o_proj — see `forward_block_layer_post_attn`.
    pub g_proj: Option<DenseWeight>,
    // MLP
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,

    // Phase G — optional FP8 mirrors of the seven dense-GEMM weights.
    // All seven populated at load time when `ATLAS_DFLASH_DRAFTER_FP8=1`;
    // only q/k/v/o when `ATLAS_DFLASH_DRAFTER_FP8_ATTN=1` (attn-only mode,
    // gate/up/down stay None → BF16). Consumed by
    // forward_block_layer_pre_attn / _post_attn, dispatched per GEMM on
    // mirror presence. All None when the BF16 path is active.
    pub q_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub k_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub v_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub o_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub gate_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub up_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub down_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
}

/// Per-sequence DFlash drafter state. One paged KV cache per drafter layer
/// (8 typical), shared block table across layers since attention shape is
/// identical layer-to-layer for a vanilla Qwen3 architecture. Mirrors
/// `MtpProposerState` in spirit; the multi-layer cache keeps it distinct.
pub struct DflashProposerState {
    /// Block table for the drafter's KV cache (shared across all drafter layers).
    pub block_table: Vec<u32>,
    /// Current logical sequence length in the drafter's KV cache. Tracks how
    /// many target-aligned positions have been written via
    /// `precompute_and_store_context_kv`.
    pub seq_len: usize,
    /// Drafts produced in the last `propose()` call. `after_verify` consults
    /// this to know how many KV positions to roll back when the accept
    /// prefix is shorter than γ.
    pub last_num_drafted: usize,
    /// Whether the prompt-time `precompute_and_store_context_kv` has been
    /// called. The first `propose()` after model build needs to run prefill
    /// over the full prompt's captured hiddens; subsequent steps incrementally
    /// append the latest accepted tokens' projections.
    pub prefill_done: bool,
    /// Multi-token accumulator for captured target hidden states. Layout:
    /// `[max_ctx_len, 5 * target_hidden]` BF16 packed. The scheduler appends
    /// the model's `dflash_hidden_save` (latest decoded position's 5 hiddens)
    /// into slot `ctx_len` after each successful verify. `propose()` reads
    /// the full populated prefix and projects all positions through `fc`
    /// at forward time. Sized for `max_seq_len` total positions; not
    /// circular — fail-fast if exceeded (drafter can't handle longer
    /// context than allocated).
    pub ctx_hidden_acc: DevicePtr,
    /// Number of populated slots in `ctx_hidden_acc`. Capped at `max_ctx_len`.
    pub ctx_len: usize,
    /// Drafts accepted in the verify that immediately preceded this propose.
    /// Set by `after_verify` so propose can label row-0 with its TRUE position.
    pub last_num_accepted: usize,
    /// EAGLE-fix one-shot: when set, the next `propose()` skips its internal
    /// decode-append because the verify step (K=2 accept) already appended
    /// row 0 + row 1 in EAGLE order before calling propose. Consumed (reset to
    /// false) by propose. Only set under ATLAS_DFLASH_EAGLE_FIX=1.
    pub skip_next_decode_append: bool,
    /// Allocation cap for `ctx_hidden_acc` (in slot count). Mirrors the
    /// `max_seq_len` build arg so we can clamp without re-fetching it.
    pub max_ctx_len: usize,
    /// Width (bytes) of one `ctx_hidden_acc` slot — `5 * target_hidden * bf16`.
    /// Stored to avoid re-deriving on every append.
    pub ctx_slot_bytes: usize,

    // ─── Phase 2 Option B fields (paged KV cache for ctx) ───────────────
    /// Device-side block table for the drafter's paged KV cache. Allocated
    /// once at first propose with enough u32 slots to cover `max_seq_len`
    /// at block_size=16. Read by `prefill_attention_paged_dflash` to map
    /// logical block indices to physical pool block indices. Mirrors the
    /// host-side `block_table` Vec, copied to GPU after each `alloc_block`.
    pub block_table_dev: Option<DevicePtr>,
    /// Number of paged-cache slots populated with ctx K/V for this sequence.
    /// Distinct from `ctx_len` (which counts target_hidden_acc slots). The
    /// drafter writes one ctx K/V slot per accepted target token; the
    /// γ-block then attends over `[0..ctx_count_drafter+γ)`. Bumped by γ
    /// per propose (γ slots written for the noise rows) and trimmed in
    /// `after_verify` by `(γ - num_accepted)`.
    pub ctx_count_drafter: usize,
    /// Cap for `ctx_count_drafter`. Mirrors `block_table.len() * block_size`.
    pub max_ctx_count_drafter: usize,
    /// Phase I — incremental ctx precompute watermark. Number of ctx slots
    /// `[0..ctx_committed)` whose K/V is already valid in the paged cache
    /// from a prior propose. Each step we only precompute the new tail
    /// `[ctx_committed..ctx_len)` instead of rebuilding the whole prefix
    /// (the old O(ctx_len²) waste — see design doc §18). Reset to the
    /// current `ctx_len` on any rewind so stale slots can't be read.
    /// `0` forces a full rebuild (first propose, or the debug escape hatch).
    pub ctx_committed: usize,
    /// Phase I (v2) — per-slot TRUE absolute decoded position, stamped once
    /// when a ctx slot is appended and never recomputed. Indexed by ctx
    /// slot (parallel to `ctx_hidden_acc` slots, len == `ctx_len`). This is
    /// the vLLM convention: a cached token's rope position is fixed at
    /// insert time, so committed slots never go stale when later accepts
    /// shift the live `position`. Replaces the sliding `absolute_start_pos
    /// + i` formula in `precompute_ctx_kv`. Prefill positions are seeded
    /// `0..prompt_len` in `update_dflash_ctx_len_after_prefill`.
    pub ctx_positions: Vec<i32>,
    /// ATLAS_DFLASH_ASYNC: this sequence's `pending_drafts` currently holds a
    /// PLACEHOLDER chain from an async (second-stream) propose launch; the
    /// real drafts are collected via `collect_async_drafts` at the top of
    /// the next scheduler step. Cleared on collect / resolve.
    pub async_placeholder: bool,

    // ── Accept-lift draft sources (atlas-src port, Phase A) ──
    /// Whether `propose_drafts` has returned at least once for this sequence.
    /// The source precision gates require a real drafter/accept history
    /// before pre-empting.
    pub first_propose_done: bool,
    /// Host copy of the sequence's committed tokens (prompt + generated),
    /// refreshed by the caller each propose when retrieval/SAM is on — the
    /// retrieval haystack.
    pub pld_tokens: Vec<u32>,
    // Adaptive retrieval gate (ATLAS_DFLASH_SAM auto-disable): attribute the
    // last step's accept to retrieval, count consecutive misfires, cool down.
    pub retr_used_last: bool,
    pub retr_misfire_streak: u32,
    pub retr_cooldown: u32,
    // ATLAS_DFLASH_RECYCLE: the previous step's discarded draft tail
    // `drafts[num_accepted+1..]`, keyed by the corrected (bonus) token; the
    // drafter's structural continuation often survives a one-token
    // substitution. Offer at most every other step (recycle_last_offered).
    pub recycle_tail: Vec<u32>,
    pub recycle_key: u32,
    pub recycle_valid: bool,
    pub recycle_last_offered: bool,
    // ATLAS_DFLASH_ECHO: the TARGET'S own verify argmaxes downstream of the
    // bonus (`verified[num_accepted+1..]`) — target-authored salvage drafts
    // at zero propose cost, keyed by the bonus token, streak-capped.
    pub echo_tail: Vec<u32>,
    pub echo_key: u32,
    pub echo_valid: bool,
    pub echo_streak: u32,
    pub echo_offered_last: bool,
    /// Block-fork tree payload (doc 16): `(cliff_draft_index, fork_token)`.
    /// Set by the drafter path when ATLAS_DFLASH_BLOCKFORK=1 (the lowest-
    /// margin draft position + the drafter's top-2 token there); drained by
    /// the scheduler via `dflash_take_block_fork` alongside the drafts.
    /// Cleared at the top of every propose so source paths never carry a
    /// stale fork.
    pub pending_block_fork: Option<(usize, u32)>,
    /// DDTree M0 gate (ATLAS_DFLASH_TREE_M0=1): per-draft top-2 from the
    /// drafter logits `(top1_tok, top1_val, top2_tok, top2_val)`, index =
    /// draft position. Measurement only — the verify step compares the
    /// target's correction at the death position against `top2_tok` to
    /// estimate the tree-verify accept ceiling. Cleared at propose entry.
    pub pending_m0_top2: Option<Vec<(u32, f32, u32, f32)>>,
    /// DDTree M1 (ATLAS_DFLASH_TREE=1): free-slots tree payload built by the
    /// drafter path from the fresh top-2 logits (spine + one low-margin
    /// cliff fork + 1-token re-rooted tail). Drained by the scheduler via
    /// `dflash_take_tree_payload`; UNUSED by verify until M2. Cleared at the
    /// top of every propose so source paths never carry a stale tree.
    pub pending_tree_payload: Option<ddtree::TreePayload>,
    /// ATLAS_DFLASH_SPEC_PROPOSE: host watermark snapshot taken when a
    /// speculative (full-accept-bet) propose was launched during the verify;
    /// restored by `spec_propose::spec_rollback` on discard. `None` when no
    /// speculative launch is outstanding (or it was adopted).
    pub spec_watermark: Option<spec_propose::SpecWatermark>,
}

impl ProposerState for DflashProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Block-diffusion draft head. Public API is the [`DraftProposer`] trait.
///
/// The drafter shares `embed_tokens` and `lm_head` with the target — these
/// are NOT in the drafter's safetensors checkpoint (verified against
/// `z-lab/Qwen3.6-35B-A3B-DFlash` commit 42d3b34). The constructor takes
/// the target's `embed_tokens_shared` and `lm_head_shared` device pointers
/// at build time and slots them in alongside the drafter's own `fc`,
/// `hidden_norm`, `norm`, and per-layer weights.
#[allow(dead_code)]
pub struct BlockDiffusionDraftHead {
    // Drafter-architecture config (mirrors the drafter's HF config.json).
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub gamma: usize,
    pub mask_token_id: u32,
    pub window_size: Option<usize>,
    /// `target_layer_ids`. Same data as `TransformerModel::dflash_capture_layers`,
    /// repeated here so the loader is the single source of truth; the model
    /// reads these to size its capture buffer.
    pub target_layer_ids: Vec<usize>,
    /// Target-side hidden_size (used for the `fc` projection input width:
    /// `target_layer_ids.len() * target_hidden_size`).
    pub target_hidden_size: usize,

    // === Weights shared with the target ===
    /// Target's embed_tokens GPU pointer. The drafter's checkpoint has no
    /// own embeddings — both vocab and embedding dim must match the target
    /// (Qwen3.6-35B-A3B-DFlash: vocab=248320, hidden=2048 — same as target).
    pub embed_tokens_shared: DevicePtr,
    /// Target's lm_head GPU pointer. Used for the drafter's per-position
    /// argmax over `[γ, vocab]` logits. Valid only when the target lm_head is
    /// BF16; when `lm_head_nvfp4` is `Some`, the NVFP4 path is used instead.
    pub lm_head_shared: DevicePtr,
    /// Target's NVFP4 lm_head (packed + scales), shared with the drafter for
    /// the final logits GEMM. `Some` when the target ships an NVFP4 lm_head
    /// (e.g. Holo) — required because a BF16 `dense_gemm` on the NVFP4 buffer
    /// reads garbage and OOB. `None` → use the BF16 `lm_head_shared`.
    pub lm_head_nvfp4: Option<QuantizedWeight>,
    /// Phase G — optional FP8 mirror of the shared lm_head weight,
    /// `[vocab_size, hidden_size]` FP8 E4M3 + per-row f32 scales.
    /// Built at model load when `ATLAS_DFLASH_DRAFTER_FP8=1`. Owned by
    /// the drafter (separate allocation from the shared BF16 ptr) since
    /// it must not mutate the target model's lm_head. `None` on the
    /// BF16 path.
    pub lm_head_shared_fp8: Option<crate::weight_map::Fp8DenseWeight>,

    // === Weights from the drafter checkpoint ===
    /// Hidden-norm applied to the projected target context before mixing
    /// with the embedded tokens (Qwen3-DFlash convention; see vLLM
    /// `DFlashQwen3Model.hidden_norm`).
    pub hidden_norm: DenseWeight,
    /// Final RMSNorm before LM head.
    pub norm: DenseWeight,
    /// `fc` projection — `[draft_hidden, target_layer_ids.len() * target_hidden_size]`
    /// BF16. Maps the stack of captured target hiddens to drafter's input space
    /// once at model entry. Replaces the earlier (incorrect) "per-layer KV
    /// injection" design.
    pub fc: DenseWeight,
    /// Optional draft-vocab-id → target-vocab-id remap. `None` when the
    /// drafter shares vocab with the target (Qwen3.6-35B-A3B-DFlash case:
    /// vocab_size == draft_vocab_size == 248320).
    pub draft_id_to_target_id: Option<DevicePtr>,

    /// Per-captured-target-hidden RMSNorms (`aux_hidden_norms.{k}.weight`,
    /// each `[target_hidden_size]`). Present ONLY on the poolside **Laguna**
    /// drafter: each captured target hidden is RMS-normalised with its own
    /// norm BEFORE the `fc` projection (per-capture conditioning). Empty for
    /// the Qwen3.6-DFlash drafters, which apply only the single post-`fc`
    /// `hidden_norm`. When non-empty, the fc-projection sites (`forward_block`
    /// step 0 and `precompute_ctx_kv`) pre-normalise the captured hiddens
    /// slice-by-slice through these before running `fc`.
    pub aux_hidden_norms: Vec<DenseWeight>,

    /// `true` for the poolside Laguna drafter (`dflash_config.causal`). Gates
    /// the causal γ-block attention mask in the CONTIG attention path
    /// (`forward_block_layer*`). Default `false` so the Qwen3.6-DFlash drafters
    /// keep the bidirectional block-diffusion γ-block attention unchanged.
    pub causal: bool,
    /// Drafter transformer layers (8 for Qwen3.6-35B-A3B-DFlash).
    pub layers: Vec<DflashLayer>,

    /// Phase 2 (Option B) fused K/V projection across all L drafter layers.
    /// Shape: `[L × 2 × kv_dim, h]` BF16 — concatenated `[K0; V0; K1; V1; …]`
    /// (per-layer K then V interleaved). Built once at construction by
    /// `copy_d2d`-stitching the per-layer `k_proj.weight` and `v_proj.weight`
    /// pointers from `layers[i]`. Lets `precompute_ctx_kv` derive every
    /// drafter layer's ctx K/V via a single `dense_gemm` of shape
    /// `[new_ctx_count, h] × [h, L·2·kv_dim]` instead of 2·L per-layer GEMMs.
    ///
    /// `None` until Phase 2 lands the build (stage 1: kernel/dispatcher
    /// scaffolding; stage 2: this allocation + the precompute_ctx_kv module;
    /// stage 3: pyref bit-exact diff). Layout (K then V per layer) chosen
    /// to match vLLM's `_fused_kv_weight` in `qwen3_dflash.py:381-389`.
    pub fused_kv_weight: Option<DevicePtr>,

    /// Paged FP8 KV cache. One cache holding all `num_layers` drafter layers,
    /// laid out the same way the target's KV cache is — block-table-keyed,
    /// `num_layers × num_kv_heads × head_dim` per slot. Allocating a single
    /// multi-layer cache (vs. one per drafter layer) matches Atlas's existing
    /// `PagedKvCache` ABI and lets us reuse the existing `reshape_and_cache`
    /// kernel without per-layer dispatch overhead.
    pub kv_cache: Mutex<PagedKvCache>,

    /// Per-step scratch buffers (allocated once at construction, reused).
    pub scratch: DflashScratch,

    /// All kernel handles needed by `propose()` and the eventual prefill
    /// projection (`precompute_and_store_context_kv`).
    pub kernels: DflashKernels,

    /// Per-sequence ctx accumulator capacity (mirrors model's `max_seq_len`).
    /// Used by `alloc_state` to size each new sequence's `ctx_hidden_acc`.
    pub max_seq_len: usize,

    /// Pre-computed yarn inv_freq table (`[head_dim/2]` f32 on GPU).
    /// Drafter rope_scaling: factor=64, beta_fast=32, beta_slow=1,
    /// original_max_position_embeddings=4096 (per drafter config.json).
    pub yarn_inv_freq: DevicePtr,

    /// rope_theta (10000000 for Qwen3.6-DFlash). Stored to pass into the
    /// rope_yarn kernel each step.
    pub rope_theta: f32,

    /// rotary_dim. Drafter uses full-rotation (rotary_dim = head_dim = 128).
    pub rotary_dim: usize,

    /// RMSNorm epsilon (drafter inherits Qwen3 default 1e-6).
    pub rms_norm_eps: f32,

    /// Max number of past target positions injected into the drafter's K/V
    /// per step. Default γ — drafter sees at most γ ctx + γ noise = 2γ
    /// attention positions per step. ctx_window=0 disables ctx conditioning
    /// (degraded quality, ablation only).
    pub ctx_window: usize,

    // === Phase D (CUDA graph capture) → Phase F (piecewise) ===
    /// Per-subgraph captured handles. `None` until warm-up completes and
    /// the first capture pass lands; on the capture pass we fill this
    /// `Vec` with `2 × num_layers + 1` handles laid out as
    /// `[pre_0, post_0, pre_1, post_1, ..., pre_{N-1}, post_{N-1}, tail]`.
    /// Slot index = `layer_idx * 2 + half` for the layer halves
    /// (half = 0 for pre_attn, 1 for post_attn) and `num_layers * 2` for
    /// the tail (final norm + lm_head + argmax). `GraphHandle(0)` is the
    /// "empty capture" sentinel and means that slot replays eager.
    ///
    /// Phase F.2 (2026-05-28): replaces the single full-region capture
    /// with one capture per subgraph. Attention is NEVER captured —
    /// it's the natural sync barrier between captured subgraphs
    /// (vLLM piecewise convention). See design doc §15.
    pub propose_graphs: Mutex<Option<Vec<spark_runtime::gpu::GraphHandle>>>,
    /// ATLAS_DFLASH_PROPOSE_ONEGRAPH=1 (default off): ONE captured graph for
    /// the whole propose compute — all layers (pre-attn GEMMs + INDIRECT
    /// paged attention + post-attn GEMMs) + tail (norm + lm_head + argmax).
    /// Collapses the ~19 host launch boundaries of the piecewise path
    /// (6 × [pre graph, eager attn, post graph] + tail graph) into a single
    /// `launch_graph`. The attention launch is capture-safe because the
    /// default path already runs `inferspark_prefill_paged_indirect`, which
    /// reads its per-step `[kv_len, q_offset, q_rope_pos]` from
    /// `option_b_indirect_args_dev` at kernel entry; all other per-step
    /// dynamics (position_ids, token ids incl. the spec-propose indirect
    /// last_token, γ slot mapping via the pre-graph eager `fill_slots`
    /// launch) also ride device buffers written before the replay.
    ///
    /// Keyed by the `block_table_dev` pointer (`u64`) — the ONLY per-sequence
    /// device pointer baked into the captured region (the attention launches
    /// bind it as a kernel arg). Under ONEGRAPH the block-table transport
    /// buffer is SLOT-STABLE (borrowed from `bt_dev_pool`, head-lifetime —
    /// see the pool docs below), so a key is captured against AT MOST ONCE
    /// for the head's lifetime: sequences reuse the pool buffer, the pointer
    /// stays valid, and only the buffer CONTENTS change (uploaded H2D at
    /// each sequence's lazy block-table init, read by the kernels at replay
    /// time). The map holds one entry per pool buffer ever created (bounded
    /// by peak concurrent sequences; 1-2 in production) — entries are never
    /// destroyed or re-captured, eliminating the per-request ~200-400 ms
    /// re-capture that made the first landing of this feature net-negative.
    /// `GraphHandle(0)` = empty-capture sentinel → replay eager forever.
    pub propose_onegraph: Mutex<Vec<(u64, spark_runtime::gpu::GraphHandle)>>,
    /// ONEGRAPH slot-stable block-table transport pool. When
    /// `dflash_propose_onegraph_enabled()`, `propose.rs` borrows a buffer
    /// from here (or allocates one sized `bt_pool_buf_bytes()` — the
    /// head-max block count, identical for every sequence since
    /// `max_ctx_len == max_seq_len` for all states) instead of a fresh
    /// per-sequence `gpu.alloc`, and `free_state` RETURNS the buffer here
    /// instead of freeing it. The device pointer therefore survives across
    /// sequences and the captured onegraph's baked pointer stays valid; a
    /// new sequence only overwrites the buffer contents (H2D upload at lazy
    /// block-table init, before any replay for that sequence — safe because
    /// every sync propose ends with an event sync on the drafts D2H and
    /// `free_state` resolves async in-flight launches first). Buffers are
    /// head-lifetime: never freed, bounded by peak concurrent sequences.
    /// Empty and untouched when the ONEGRAPH env gate is off (byte-identical
    /// default path: plain alloc/free per sequence).
    pub bt_dev_pool: Mutex<Vec<DevicePtr>>,
    /// When set, all `forward_block` calls run eagerly. Mirrors target-model
    /// `TransformerModel::suppress_graphs` so external code can disable
    /// graphs at runtime (e.g. while calibrating FP8 KV).
    pub suppress_graphs: std::sync::atomic::AtomicBool,
    /// How many eager warm-up calls we've executed against the graph path.
    /// Default warmup target is 2 (override via `ATLAS_DFLASH_PROPOSE_WARMUP_N`).
    /// Two eager passes warm the PTX→SASS cache, ramp GB10 clocks to steady
    /// state, and bring hot weight tiles into L2 before the capture freezes
    /// SASS variants the driver picks. Shared across all subgraphs — every
    /// subgraph captures on the same propose call after the warmup target
    /// is hit.
    pub propose_warmup_count: std::sync::atomic::AtomicUsize,

    /// Block-fork tree (doc 16): `[γ, 4]` u32 top-2 results
    /// (idx1, bits(val1), idx2, bits(val2)) per drafter-logits row.
    pub top2_out: DevicePtr,

    // ── ATLAS_DFLASH_ASYNC (ported from atlas-src task #20) ──
    /// At most one in-flight async (second-stream) propose — the head owns a
    /// single scratch buffer set. See `async_propose.rs`.
    pub async_inflight: Mutex<Option<async_propose::AsyncInflight>>,
    /// Dedicated non-blocking CUDA stream for async propose launches. Lazily
    /// created on first eligible launch; `0` = creation failed → disabled.
    pub async_propose_stream: std::sync::OnceLock<u64>,
    /// Event ordering the propose stream after the default stream's prior
    /// writes (ctx-append D2Ds, verify captures).
    pub async_order_event: std::sync::atomic::AtomicU64,

    // Quantization mode (BF16 only for Phase 1).
    pub quant: DflashQuantization,
}

mod async_propose;
pub mod ddtree;
pub(crate) mod echo;
mod forward_block;
mod forward_block_layer;
mod forward_block_layer_paged;
mod from_weights;
mod precompute_ctx_kv;
mod propose;
mod retrieval;
pub mod spec_propose;

// Re-export the DDTree payload so the scheduler / traits layer can carry it
// as `Option<DDTreePayload>` without reaching into the module path.
pub use ddtree::TreePayload as DDTreePayload;

/// ATLAS_DFLASH_DRAFTER_FASTGEMM=1 (default OFF): route the drafter's
/// M=γ BF16 propose GEMMs through the small-M weight-streaming kernel
/// `dense_gemm_bf16_mtile16` instead of `dense_gemm_bf16_pipelined`.
/// Read once (OnceLock) so the decision is stable across CUDA graph
/// capture/replay.
pub(crate) fn drafter_fastgemm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_DRAFTER_FASTGEMM")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// ATLAS_DFLASH_DRAFTER_FP8_SMALLM=1 (default OFF): route the drafter's
/// seven per-layer FP8 GEMMs off the stock `fp8_gemm_t_row_scaled`
/// (M_TILE=64) onto the small-M tile ladder.
///
/// At the propose shape M=γ=5 the M_TILE=64 kernel wastes ~92% of its
/// MMA cycles and smem_A traffic: grid Y is ceil(M/64)=1, but each CTA
/// still runs a full 64-row A tile. The `_mtile8` header (w4a16_gemm.cu)
/// puts the stock tile at ~43% of the LPDDR5x bound at M=7 vs ≥85%
/// expected. Phase G checks these kernels are loaded as an arming
/// precondition and then never calls them on the layer path — only the
/// lm_head tail reaches them.
///
/// The ladder mirrors the target's `fp8_mirror_gemm` (qwen3_attention/
/// trait_impl/multi_seq/qkv.rs), measured at 82-88% of the wall at M=5:
///
/// ```text
/// m<=8 && n<=3072 && k>=4096  -> _mtile8_n32  (N_TILE=32)
/// m<=8                        -> _mtile8      (N_TILE=64)
/// m<=16                       -> _m16
/// else                        -> stock M_TILE=64
/// ```
///
/// The n32 rung matters for o_proj (N=3072, K=9216) and down_proj
/// (N=3072, K=12288): at N=3072 the N_TILE=64 grid is ceil(3072/64) = 48
/// CTAs = exactly 1/SM on GB10, and one 4-stage cp.async ring per SM
/// cannot hide LPDDR5x latency. N_TILE=32 doubles the grid to 2/SM.
///
/// Guarded on `m` rather than γ so a future γ > 8 falls through to `_m16`
/// instead of tripping the `debug_assert!(m <= 8)` in the smallm
/// launchers. All rungs share the m16n8k32 MMA shape, ascending-K
/// accumulate order and scale-then-cast epilogue, so this is intended as
/// a pure occupancy change with identical output — but note `g_proj`'s
/// GEMV was bit-exact against its GEMM while the MoE router-gate GEMV
/// diverged on 5/6 hashes, so parity is asserted by the bench, not
/// assumed. Read once (OnceLock) so graph capture and replay agree.
pub(crate) fn dflash_drafter_fp8_smallm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_DRAFTER_FP8_SMALLM")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// ATLAS_DFLASH_DRAFTER_FP8_SMALLM_NO_MTILE8=1 (default OFF, BENCH ONLY):
/// with `ATLAS_DFLASH_DRAFTER_FP8_SMALLM=1` also set, skips both mtile8
/// rungs so the ladder falls through to `_m16`. Exists so the ablation
/// can price the two rungs against each other in one binary — the
/// drafter's lm_head tail already records `_m16` as ~40% slower cold, and
/// if the two rungs also disagree *numerically* we want that isolated
/// from the M_TILE=64 arm rather than confounded with it. No production
/// arm sets this. Read once (OnceLock) so graph capture and replay agree.
pub(crate) fn dflash_drafter_fp8_smallm_no_mtile8() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_DRAFTER_FP8_SMALLM_NO_MTILE8")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// ATLAS_DFLASH_BF16_NTILE64_3072=1 (default OFF): raise `try_fastgemm`'s
/// N_TILE=64 cutoff from 2048 to 3072, so o_proj (N=3072, K=9216) and
/// down_proj (N=3072, K=12288) take the 64-wide kernel instead of the
/// 128-wide one.
///
/// The 2048 threshold appears never to have been measured at N=3072. Its
/// doc scores both arms against `dense_gemm_bf16_pipelined` — kv_proj at
/// N=1024, o/down/fc/lm_head at N>2048 — but not against each other, and
/// 2048 sits as a round number between two measured endpoints. Three
/// facts collide at N=3072: (1) the N_TILE=64 kernel's header names
/// "o/down = 48 CTAs" as a design goal; (2) this dispatcher sends those
/// two shapes to N_TILE=128, i.e. ceil(3072/128) = 24 CTAs on 48 SMs,
/// half the machine idle; (3) the 128-wide kernel's stated rationale is
/// relieving DRAM page thrash from "MANY concurrent 64-row B streams
/// (96+ CTAs)" — but the 64-wide tile makes 48 streams here, not 96+, so
/// that rationale does not obviously reach this shape. q_proj (144 CTAs)
/// and gate/up (192 CTAs) DO clear 96+, which is why the cutoff moves to
/// exactly 3072 and not higher.
///
/// Both kernels document the same ascending-K m16n8k16 accumulate chain
/// and both claim bit-identity with the pipelined kernel, so this is
/// expected to be a pure occupancy change with 6/6 hash parity — asserted
/// by the bench, not assumed. A divergence would falsify one of those two
/// bit-identity claims and is worth reporting on its own.
///
/// Read once (OnceLock) so graph capture and replay agree.
pub(crate) fn dflash_bf16_ntile64_3072_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_BF16_NTILE64_3072")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// ATLAS_DFLASH_PROPOSE_ONEGRAPH=1 (default OFF): single full-propose
/// captured graph (see `propose_onegraph`). Shared gate for the three
/// coupled sites — the `forward_block` capture/replay branch, the
/// `propose.rs` block-table borrow (pool vs fresh alloc), and the
/// `free_state` return (pool vs `gpu.free`) — so the slot-stable
/// transport and the graph keying can never disagree. The CONTIG
/// attention ablation injects D2H + sync inside the layer loop — not
/// capture-safe — so it forces the gate off. Read once (OnceLock) so
/// capture and replay always agree.
pub(crate) fn dflash_propose_onegraph_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_PROPOSE_ONEGRAPH")
            .ok()
            .as_deref()
            == Some("1")
            && std::env::var("ATLAS_DFLASH_CONTIG_ATTN").is_err()
    })
}

impl BlockDiffusionDraftHead {
    /// Size (bytes) of one `bt_dev_pool` block-table transport buffer: the
    /// head-max block count × 4 (u32 block ids). Mirrors the `propose.rs`
    /// lazy-init formula `(max_ctx_len + γ + 1).div_ceil(BLOCK_SIZE)` with
    /// `max_ctx_len == max_seq_len` (every state's cap — set in
    /// `alloc_state`), so every sequence's `bt_bytes` fits in every pool
    /// buffer and pooled buffers are interchangeable across slots.
    pub(crate) fn bt_pool_buf_bytes(&self) -> usize {
        const BLOCK_SIZE: usize = 16; // matches propose.rs / from_weights.rs:68
        (self.max_seq_len + self.gamma + 1).div_ceil(BLOCK_SIZE) * std::mem::size_of::<u32>()
    }

    /// Dispatch one row-scaled FP8 drafter GEMM through the small-M tile
    /// ladder when `ATLAS_DFLASH_DRAFTER_FP8_SMALLM=1`, else the stock
    /// M_TILE=64 kernel. See `dflash_drafter_fp8_smallm_enabled` for the
    /// ladder and its rationale.
    ///
    /// Both smallm launchers `debug_assert!(m <= 8)`, so the `m <= 8`
    /// guards are load-bearing, not cosmetic. `k % 32 == 0` is the
    /// kernels' documented contract (all drafter K ∈ {3072, 9216, 12288}
    /// qualify); a miss falls through rather than launching a kernel
    /// whose K-step ring would read past the tail.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fp8_gemm_dispatch(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        weight: &crate::weight_map::Fp8DenseWeight,
        dst: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if dflash_drafter_fp8_smallm_enabled() {
            // Bench-only: skip both mtile8 rungs so the ladder lands on `_m16`.
            // Lets the sweep price mtile8 against _m16 without rebuilding, and
            // isolates any mtile8-vs-_m16 numeric disagreement from the
            // M_TILE=64 comparison. Never set in production.
            let no_mtile8 = dflash_drafter_fp8_smallm_no_mtile8();
            if !no_mtile8
                && m <= 8
                && k.is_multiple_of(32)
                && n <= 3072
                && k >= 4096
                && self.kernels.fp8_gemm_row_scaled_mtile8_n32.0 != 0
            {
                // Small-N/large-K: o_proj and down_proj. N_TILE=64 would put
                // only 1 CTA/SM here — see the field docs.
                return crate::layers::ops::fp8_gemm_row_scaled_smallm_n32(
                    gpu,
                    self.kernels.fp8_gemm_row_scaled_mtile8_n32,
                    src,
                    weight,
                    dst,
                    m,
                    n,
                    k,
                    stream,
                );
            }
            if !no_mtile8
                && m <= 8
                && k.is_multiple_of(32)
                && self.kernels.fp8_gemm_row_scaled_mtile8.0 != 0
            {
                return crate::layers::ops::fp8_gemm_row_scaled_smallm(
                    gpu,
                    self.kernels.fp8_gemm_row_scaled_mtile8,
                    src,
                    weight,
                    dst,
                    m,
                    n,
                    k,
                    stream,
                );
            }
            if m <= 16 && self.kernels.fp8_gemm_n128_row_scaled_m16.0 != 0 {
                return crate::layers::ops::fp8_gemm_n128_row_scaled_m16(
                    gpu,
                    self.kernels.fp8_gemm_n128_row_scaled_m16,
                    src,
                    weight,
                    dst,
                    m,
                    n,
                    k,
                    stream,
                );
            }
        }
        crate::layers::ops::fp8_gemm_n128_row_scaled(
            gpu,
            self.kernels.fp8_gemm_n128_row_scaled,
            src,
            weight,
            dst,
            m,
            n,
            k,
            stream,
        )
    }

    /// Try to dispatch an `[m, n] = [m, k] · [n, k]ᵀ` BF16 GEMM through the
    /// small-M weight-streaming kernels. Returns `Ok(true)` when the fast
    /// path launched (caller skips the pipelined fallback), `Ok(false)`
    /// when ineligible: env gate off, kernels missing from the target's
    /// set, `m > 16` (one m16n8k16 row block), or `k % 8 != 0` (16-B
    /// cp.async row alignment). All drafter propose shapes
    /// (K ∈ {3072, 9216, 12288, 18432}) qualify.
    ///
    /// Tile pick (GB10 microbench, dflash_bf16gemm_smallm_microtest):
    /// N ≤ 2048 → N_TILE=64 (kv_proj N=1024: 22 µs vs pipelined's 113 µs —
    /// the 128-wide tile puts only ceil(N/128) ≤ 16 CTAs on 48 SMs);
    /// N > 2048 → N_TILE=128 wide-stream (o/down/fc/lm_head 1.06-1.28×
    /// over pipelined; fewer, longer DRAM streams win on LPDDR5x).
    /// Pure device args — CUDA graph-capture safe; the env gate is a
    /// OnceLock so capture and replay always agree.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_fastgemm(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        weight: &DenseWeight,
        dst: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        if !drafter_fastgemm_enabled() || m > 16 || k % 8 != 0 {
            return Ok(false);
        }
        // ATLAS_DFLASH_BF16_NTILE64_3072=1 raises the N_TILE=64 cutoff from
        // 2048 to 3072, moving o_proj and down_proj (both N=hidden=3072) off
        // the wide kernel. See `dflash_bf16_ntile64_3072_enabled`.
        let cutoff = if dflash_bf16_ntile64_3072_enabled() {
            3072
        } else {
            2048
        };
        let (kernel, wide) = if n <= cutoff {
            (self.kernels.dense_gemm_mtile16, false)
        } else {
            (self.kernels.dense_gemm_mtile16_n128, true)
        };
        if kernel.0 == 0 {
            return Ok(false);
        }
        if wide {
            crate::layers::ops::dense_gemm_bf16_mtile16_n128(
                gpu, kernel, src, weight, dst, m, n, k, stream,
            )?;
        } else {
            crate::layers::ops::dense_gemm_bf16_mtile16(
                gpu, kernel, src, weight, dst, m, n, k, stream,
            )?;
        }
        Ok(true)
    }
}

impl DraftProposer for BlockDiffusionDraftHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        // Per-seq ctx accumulator: `[max_seq_len, 5 * target_hidden] BF16`.
        // Sized once, re-used across the seq's lifetime; reset on
        // `free_state`. At max_seq_len=16384 and 5×2048 BF16: 320 MB per
        // seq — tolerable on a single Spark with max_batch_size=1; for
        // higher batch we may want to reduce to a smaller working window.
        let bf16 = 2usize;
        let ctx_slot_bytes = self.target_layer_ids.len() * self.target_hidden_size * bf16;
        let total = self.max_seq_len * ctx_slot_bytes;
        let ctx_hidden_acc = gpu.alloc(total)?;
        // Initialize to zero so stale data doesn't leak between sequences.
        gpu.memset(ctx_hidden_acc, 0, total)?;
        Ok(Box::new(DflashProposerState {
            block_table: Vec::with_capacity(64),
            seq_len: 0,
            last_num_drafted: 0,
            prefill_done: false,
            ctx_hidden_acc,
            ctx_len: 0,
            last_num_accepted: 0,
            skip_next_decode_append: false,
            max_ctx_len: self.max_seq_len,
            ctx_slot_bytes,
            // Phase 2 Option B: lazily allocated on first propose when
            // ATLAS_DFLASH_OPTION_B=1. None until then to keep alloc_state
            // cheap for sequences that never use Option B.
            block_table_dev: None,
            ctx_count_drafter: 0,
            max_ctx_count_drafter: 0,
            ctx_committed: 0,
            ctx_positions: Vec::new(),
            async_placeholder: false,
            first_propose_done: false,
            pld_tokens: Vec::new(),
            retr_used_last: false,
            retr_misfire_streak: 0,
            retr_cooldown: 0,
            recycle_tail: Vec::new(),
            recycle_key: 0,
            recycle_valid: false,
            recycle_last_offered: false,
            echo_tail: Vec::new(),
            echo_key: 0,
            echo_valid: false,
            echo_streak: 0,
            echo_offered_last: false,
            pending_block_fork: None,
            pending_m0_top2: None,
            pending_tree_payload: None,
            spec_watermark: None,
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: spark_runtime::gpu::DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        draft_embed_target: Option<spark_runtime::gpu::DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<spark_runtime::gpu::DevicePtr>,
    ) -> Result<Vec<u32>> {
        self.propose_drafts(
            last_token,
            target_hidden,
            position,
            num_drafts,
            state,
            ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            target_hidden_stack,
        )
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        // Phase 1: no real KV trim because `propose()` is a stub. Phase 2
        // adds the rollback that drops `(last_num_drafted - num_accepted)`
        // tokens from each layer's paged cache.
        //
        // Phase I invariant: `ctx_committed` is the watermark of ctx slots
        // already precomputed into the paged cache. It is monotonic only as
        // long as `ctx_len` is monotonic (today it is — ctx is append-only
        // and never rewound here). IF a future rollback ever shrinks the
        // committed ctx (rewinds `ctx_len`), it MUST also reset
        // `dstate.ctx_committed = dstate.ctx_len` so the next propose
        // recomputes the rolled-back tail instead of reading stale K/V.
        // The `.min(ctx_len)` clamp in propose() is the defensive backstop.
        //
        // Accept-lift port: record the REAL accept count — the echo/recycle
        // precision gates and the SAM adaptive cooldown key off it. (Before
        // this port nothing ever set it; it was only zeroed on free_state.)
        dstate.last_num_accepted = num_accepted;
        dstate.last_num_drafted = 0;
        Ok(())
    }

    fn collect_async_drafts(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut dyn ProposerState,
    ) -> Result<Option<Vec<u32>>> {
        let dstate = match state.as_any_mut().downcast_mut::<DflashProposerState>() {
            Some(d) => d,
            None => return Ok(None),
        };
        self.collect_async_drafts_impl(gpu, dstate)
    }

    fn spec_propose_launch(
        &self,
        gpu: &dyn GpuBackend,
        default_stream: u64,
        device_last_token: DevicePtr,
        hidden_save: DevicePtr,
        ctx_rows: usize,
        base_pos: usize,
        state: &mut dyn ProposerState,
        ctx: &crate::layer::ForwardContext,
    ) -> Result<bool> {
        let dstate = match state.as_any_mut().downcast_mut::<DflashProposerState>() {
            Some(d) => d,
            None => return Ok(false),
        };
        self.spec_propose_launch_impl(
            gpu,
            default_stream,
            device_last_token,
            hidden_save,
            ctx_rows,
            base_pos,
            dstate,
            ctx,
        )
    }

    fn spec_pending(&self, state: &mut dyn ProposerState) -> bool {
        match state.as_any_mut().downcast_mut::<DflashProposerState>() {
            Some(d) => self.spec_pending_impl(d),
            None => false,
        }
    }

    fn spec_discard(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let dstate = match state.as_any_mut().downcast_mut::<DflashProposerState>() {
            Some(d) => d,
            None => return Ok(()),
        };
        self.spec_discard_impl(gpu, dstate)
    }

    fn spec_adopt_placeholder(&self, state: &mut dyn ProposerState) -> Result<Option<Vec<u32>>> {
        let dstate = match state.as_any_mut().downcast_mut::<DflashProposerState>() {
            Some(d) => d,
            None => return Ok(None),
        };
        self.spec_adopt_impl(dstate)
    }

    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        // Phase 2 (Option B) reclaim: return the drafter's lazily-allocated
        // paged KV blocks to the pool on request completion. Without this the
        // ~257-block Option-B drafter cache (allocated in propose.rs when
        // block_table_dev.is_none()) is never freed, so the SECOND request to
        // a long-lived server starts with zero free drafter blocks and floods
        // "DFlash Option B: paged KV cache exhausted". Mirrors MtpHead::free_state.
        let dstate = match state.as_any_mut().downcast_mut::<DflashProposerState>() {
            Some(s) => s,
            // Phase 1 / non-DFlash proposer state: nothing allocated, nothing to free.
            None => return Ok(()),
        };
        // ATLAS_DFLASH_ASYNC / ATLAS_DFLASH_SPEC_PROPOSE: an in-flight
        // second-stream propose reads this sequence's ctx buffers — sync +
        // discard (and, for spec, roll the watermark back) before freeing.
        if async_propose::dflash_async_enabled() || spec_propose::dflash_spec_enabled() {
            self.resolve_async_inflight_impl(gpu, Some(dstate))?;
        }
        if !dstate.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&dstate.block_table);
            dstate.block_table.clear();
        }
        // Free the per-seq ctx accumulator — the dominant per-request
        // allocation (`max_seq_len × 5 × target_hidden` BF16; ~320 MB at
        // max_seq_len=16384). `DevicePtr` has no Drop, so without this every
        // finished sequence leaks it for the server's lifetime. Guarded on a
        // non-null pointer so a double free_state is a no-op.
        if dstate.ctx_hidden_acc.0 != 0 {
            gpu.free(dstate.ctx_hidden_acc)?;
            dstate.ctx_hidden_acc = DevicePtr(0);
        }
        // Release the device-side block table (lazily allocated in
        // propose.rs). Under ONEGRAPH the buffer is slot-stable transport
        // borrowed from `bt_dev_pool` — RETURN it (never free) so the
        // captured graph's baked pointer stays valid and the next sequence
        // reuses it (contents re-uploaded at its lazy init). Default path:
        // plain free, byte-identical to pre-ONEGRAPH behavior.
        if let Some(bt) = dstate.block_table_dev.take() {
            if dflash_propose_onegraph_enabled() {
                self.bt_dev_pool.lock().push(bt);
            } else {
                gpu.free(bt)?;
            }
        }
        // Reset the lazy-alloc guard + watermarks so the NEXT request's first
        // propose re-allocates fresh blocks and re-precomputes ctx from a clean
        // slate (propose.rs gates alloc on block_table_dev.is_none()).
        dstate.max_ctx_count_drafter = 0;
        dstate.ctx_count_drafter = 0;
        dstate.ctx_committed = 0;
        dstate.ctx_positions.clear();
        dstate.seq_len = 0;
        dstate.ctx_len = 0;
        dstate.prefill_done = false;
        dstate.last_num_drafted = 0;
        dstate.last_num_accepted = 0;
        dstate.skip_next_decode_append = false;
        Ok(())
    }
}
