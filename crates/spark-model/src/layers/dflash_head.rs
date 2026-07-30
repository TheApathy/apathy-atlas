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

/// Cached gate for `ATLAS_DFLASH_DEEPLOOP=1` (multi-pass denoise residual scaling).
///
/// Defined in the head module so `forward_block_layer` and `noise_pass` can
/// both import it without a circular dependency.
pub(crate) fn dflash_deeploop_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let on = std::env::var("ATLAS_DFLASH_DEEPLOOP").ok().as_deref() == Some("1");
        if on {
            tracing::info!("DFlash DeepLoop residual scaling ENABLED (ATLAS_DFLASH_DEEPLOOP=1)");
        }
        on
    })
}

/// Compile-time cap on the per-position top-K used by the DDTree (M4B v2)
/// builder. Must match `MAX_TOP_K` in `kernels/gb10/nvfp4/argmax_bf16.cu`.
/// Runtime `top_k` comes from `ATLAS_DDTREE_TOP_K` (default 8) and is
/// clamped to this maximum.
pub const DDTREE_TOP_K_MAX: usize = 16;

/// Kernel handles for the DFlash γ-block forward chain. All resolved once
/// at `BlockDiffusionDraftHead::from_weights` against the active GPU backend
/// (which compiles target-specific PTX at startup); subsequent
/// `propose()` calls just `KernelLaunch::new(...).launch(stream)`.
pub struct DflashKernels {
    pub rms_norm: KernelHandle,
    pub residual_rms_norm: KernelHandle,
    pub dense_gemv: KernelHandle,
    pub dense_gemm: KernelHandle,
    pub rope_qwen3: KernelHandle,
    pub reshape_cache_fp8: KernelHandle,
    pub prefill_attn_dflash_fp8: KernelHandle,
    pub silu_mul: KernelHandle,
    pub residual_add: KernelHandle,
    /// BF16 scaled accumulate: `output[i] += scale * src[i]`. Same module
    /// as `residual_add` (`residual_add.cu`). Used by DeepLoop multi-pass
    /// residual scaling (`ATLAS_DFLASH_DEEPLOOP=1`).
    pub scaled_add: KernelHandle,
    /// GPU-side token recommit for DeepLoop multi-pass: reconstructs
    /// `draft_tokens_dev` for pass N+1 without host D2H (async-safe).
    /// Same module as `scaled_add` (`residual_add.cu`).
    pub token_recommit: KernelHandle,
    pub argmax: KernelHandle,
    /// Top-K over BF16 logits — used by the DDTree (M4B v2) propose path
    /// to seed per-position branch candidates. Same `argmax` module as
    /// `argmax_bf16` (shared .cu file). Resolved unconditionally; never
    /// invoked under flat DFlash, only when the propose path opts into
    /// the real (non-chain-only) DDTree topology builder.
    pub topk: KernelHandle,
    pub batched_embed: KernelHandle,
    /// Non-paged prefill attention (used for the γ-block self-attention
    /// when there's no persistent K/V cache to walk).
    pub prefill_attn: KernelHandle,
    /// NVFP4 W4A16 GEMV (M=1) — used for the per-row `fc` projection when
    /// the drafter is built with `DflashQuantization::Nvfp4`. BF16 build
    /// leaves this set to the same handle as `dense_gemv` and never
    /// dispatches to it.
    pub w4a16_gemv: KernelHandle,
    /// NVFP4 W4A16 GEMM (M>1) — used for q/k/v/o + gate/up/down per-layer
    /// projections under NVFP4. BF16 build leaves this set to the same
    /// handle as `dense_gemm` and never dispatches to it.
    pub w4a16_gemm: KernelHandle,
    /// NVFP4 W4A16 GEMM M_TILE=16 specialization (`w4a16_gemm_t_m16`) —
    /// requires transposed `nvfp4_t` weight layout. Only used by the
    /// drafter FFN when `ATLAS_DFLASH_FFN_KGAMMA=1`. Resolved via
    /// `try_kernel`; handle is `KernelHandle(0)` (sentinel) on miss so
    /// the dispatch can degrade gracefully to the M_TILE=64 path.
    pub w4a16_gemm_t_m16: KernelHandle,
    /// `w4a16_gemm_t_m32_n64` — single B read × full occupancy at M ≤ 32.
    /// Preferred over m16 (2× B reads at M=17) for drafter kgamma GEMMs.
    pub w4a16_gemm_t_m32_n64: KernelHandle,
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
    pub logits: DevicePtr,
    pub draft_tokens_dev: DevicePtr,
    /// `[ctx_window + γ]` i32 positions. First ctx_window are
    /// historical target positions (decoded indices); last γ are
    /// the to-be-predicted noise positions.
    pub position_ids: DevicePtr,
    /// DDTree M4B v2: per-MASK-position top-K token IDs. Layout
    /// `[γ, DDTREE_TOP_K_MAX]` u32, written by the `topk_bf16` kernel
    /// after lm_head and read host-side to seed the DDTree builder.
    /// Allocated for the maximum compile-time K (16); per-call `k` may
    /// be smaller (env-configurable via `ATLAS_DDTREE_TOP_K`).
    pub topk_tokens_dev: DevicePtr,
    /// Matching f32 logit scores for `topk_tokens_dev`, shape
    /// `[γ, DDTREE_TOP_K_MAX]`. Caller converts to log-probs in Rust by
    /// row-wise softmax over the K selected logits (or simpler heuristic
    /// — see `propose.rs` for the score-translation policy).
    pub topk_logits_dev: DevicePtr,
    /// EAGLE-3.1 per-layer FC-normalization scratch (ATLAS_DFLASH_FC_LAYERNORM=1).
    /// Holds one fc-input slot's `[n_target_layers * target_hidden]` BF16 after
    /// each captured target-layer slice has been unit-variance RMS-normalized
    /// independently (so no single high-magnitude late-layer capture dominates
    /// the fused fc input). The fc GEMV reads from here instead of the raw
    /// `ctx_hidden_acc` slot when the flag is on. Sized
    /// `n_target_layers * target_hidden` BF16. Unused (and may be a null/stub
    /// pointer) when the flag is off — Step 0 only touches it on the ON path.
    pub fc_norm_in: DevicePtr,
    /// All-zeros BF16 weight of length `target_hidden`, used as the per-slice
    /// RMS-norm scale for the unit-variance FC-layernorm variant. The
    /// `rms_norm` kernel computes `x * rms * (1 + w)`, so a zero weight yields
    /// the plain `x * rms` unit-variance normalization (variant (a): no learned
    /// gamma, the safest / least-OOD choice given `self.fc` was trained on
    /// un-normalized concat). Allocated + zeroed once at construction.
    pub fc_norm_zero_w: DevicePtr,
    /// DSpark Markov head scratch — the gathered `markov_w1[prev]` row
    /// (`[rank]` BF16). Only allocated (non-null) when the head is present;
    /// `DevicePtr::NULL` otherwise. Input to the `w2` bias GEMV.
    pub markov_w1_row: DevicePtr,
    /// DSpark Markov head scratch — the per-position bias `B(prev)`
    /// (`[vocab]` BF16), the output of `dense_gemv(w1_row, w2)`. Added to the
    /// base logit row before argmax. `DevicePtr::NULL` when no Markov head.
    pub markov_bias: DevicePtr,
    /// DSpark Markov head scratch — the previous token id (`[1]` u32) fed to
    /// the `batched_embed` gather that selects `markov_w1[prev]`. Rewritten
    /// each block position with the token sampled at the prior position.
    /// `DevicePtr::NULL` when no Markov head.
    pub markov_prev_dev: DevicePtr,
}

/// Drafter-side weight precision.
///
/// * `Bf16` — default. Loads weights verbatim from the drafter checkpoint
///   (BF16 in both `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash`). Stable but slow:
///   the BF16 `dense_gemm` is bandwidth-bound on GB10 and dominates the
///   ~134 ms / propose-step at γ=16, ctx_window=16.
/// * `Nvfp4` — runtime-quantize the 7 dense projections per layer plus
///   the `fc` target-context projection to NVFP4 at model-load time, then
///   dispatch the per-step forward through the same fast `w4a16_gemv` /
///   `w4a16_gemm` kernels the target model already uses. RMSNorm + bias
///   weights stay BF16 (kernels expect BF16 inputs for those small ops).
///   Cuts propose latency by ~3-5× on the GEMM-dominated path. Acceptance
///   parity vs BF16 has been observed for the target NVFP4 path; matching
///   parity for the drafter is acceptable because the verify step always
///   uses the target's logits, so a small drafter-side numerical drift just
///   reduces accept rate, never produces wrong tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashQuantization {
    Bf16,
    Nvfp4,
}

/// DSpark VanillaMarkov head (runtime form) — the low-rank bigram bias
/// `B(prev) = markov_w2 @ markov_w1[prev]` applied per block position.
///
/// Both weights are BF16 on device. `w1` is `[vocab, rank]` (an embedding
/// gather picks row `prev`); `w2` is `[vocab, rank]` (`nn.Linear(rank, vocab)`
/// weight), so `B = dense_gemv(input=w1[prev], weight=w2, n=vocab, k=rank)`.
/// The per-block bias scratch (`[vocab]` BF16) lives in [`DflashScratch`].
pub struct MarkovHead {
    pub w1: DenseWeight,
    pub w2: DenseWeight,
    pub rank: usize,
}

/// Per-drafter-layer Qwen3-style BF16 weights (default path).
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
    // MLP
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// Per-drafter-layer NVFP4 weights (one of these is built per layer when
/// `DflashQuantization::Nvfp4` is selected). The 7 dense projections
/// (q/k/v/o + gate/up/down) are quantized via `quantize_to_nvfp4`; the
/// RMSNorm weights (`input_layernorm`, `post_attention_layernorm`,
/// `q_norm`, `k_norm`) stay BF16 because the rms_norm kernel reads them
/// as BF16 scaling factors.
#[allow(dead_code)]
pub struct DflashLayerNvfp4 {
    // Norms — BF16 (small, kernel-required dtype).
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    // Attention projections — NVFP4 packed.
    pub q_proj: QuantizedWeight,
    pub k_proj: QuantizedWeight,
    pub v_proj: QuantizedWeight,
    pub o_proj: QuantizedWeight,
    // MLP projections — NVFP4 packed.
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
    /// Transposed (`nvfp4_t` layout) MLP projections — populated only when
    /// `ATLAS_DFLASH_FFN_KGAMMA=1` was set at model build time and the
    /// quantization path took the NVFP4 branch. When `Some`, the per-layer
    /// forward routes the gate/up/down FFN GEMMs through
    /// `w4a16_gemm_n128_m16` (M_TILE=16) instead of the standard M_TILE=64
    /// `w4a16_gemm`. Each is a freshly allocated GPU buffer pair (packed
    /// weight + scales transposed via `QuantizedWeight::transpose_for_gemm`).
    /// `None` falls back to the original M_TILE=64 path.
    pub gate_proj_t: Option<QuantizedWeight>,
    pub up_proj_t: Option<QuantizedWeight>,
    pub down_proj_t: Option<QuantizedWeight>,
    /// Transposed (`nvfp4_t` layout) attention projections — populated only
    /// when `ATLAS_DFLASH_ATTN_KGAMMA=1` was set at model build time and the
    /// quantization path took the NVFP4 branch. When `Some`, the per-layer
    /// forward routes q/k/v/o GEMMs through `w4a16_gemm_n128_m16` (M_TILE=16)
    /// instead of the standard M_TILE=64 `w4a16_gemm`. Each is a freshly
    /// allocated GPU buffer pair (packed weight + scales transposed via
    /// `QuantizedWeight::transpose_for_gemm`). `None` falls back to the
    /// original M_TILE=64 path. Mirrors the FFN-kgamma pattern (cf.
    /// `gate_proj_t` / `up_proj_t` / `down_proj_t` above).
    pub q_proj_t: Option<QuantizedWeight>,
    pub k_proj_t: Option<QuantizedWeight>,
    pub v_proj_t: Option<QuantizedWeight>,
    pub o_proj_t: Option<QuantizedWeight>,
}

/// Per-layer weight bundle dispatched on at forward time. Holds either the
/// BF16 [`DflashLayer`] or its NVFP4 sibling [`DflashLayerNvfp4`]; the
/// per-layer forward pass (`forward_block_layer`) reads the active variant
/// once and dispatches every projection through the right kernel.
#[allow(clippy::large_enum_variant)]
pub enum DflashLayerQuantWeights {
    Bf16(DflashLayer),
    Nvfp4(DflashLayerNvfp4),
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
    /// Allocation cap for `ctx_hidden_acc` (in slot count). Mirrors the
    /// `max_seq_len` build arg so we can clamp without re-fetching it.
    pub max_ctx_len: usize,
    /// Width (bytes) of one `ctx_hidden_acc` slot — `5 * target_hidden * bf16`.
    /// Stored to avoid re-deriving on every append.
    pub ctx_slot_bytes: usize,
    /// Index into the multi-token `dflash_hidden_save` buffer of the
    /// last accepted token's hidden state. Set by `after_verify` to
    /// `num_accepted` so `propose_drafts` appends the correct capture.
    pub last_capture_idx: usize,
    /// Number of drafts accepted in the previous verify step.
    /// Used by `propose_drafts` to know how many verify positions to
    /// append to `ctx_hidden_acc` (positions 1..=last_num_accepted for
    /// accepted drafts; position 0 when zero drafts were accepted).
    pub last_num_accepted: usize,
    /// FIX 1 (ATLAS_DFLASH_TREE_COMMIT): the accepted-path COMPACT indices
    /// from the last tree-fork verify (e.g. `[1, 2, 3, 7]`). Empty on every
    /// flat-chain / non-tree step. When non-empty, `propose_drafts` reads the
    /// captured ctx hiddens from these sparse `dflash_hidden_save` rows
    /// (verify slot 0 = last_token, slot `c` = compact index `c`) instead of
    /// the contiguous `0..last_num_accepted+1` — because a fork accept's
    /// hiddens are scattered, and reading `0..N+1` would pick up rejected
    /// sibling rows. Stays a strict superset-safe override: when the path is
    /// the contiguous `[1..N]` it produces identical reads.
    pub last_accepted_compact: Vec<usize>,
    /// Host copy of the sequence's committed tokens, refreshed by the
    /// caller each propose when ATLAS_DFLASH_PLD=1 (prompt-lookup drafts).
    pub pld_tokens: Vec<u32>,
    /// Whether `propose_drafts` has been called at least once. Used to
    /// skip the post-prefill append on the first call because
    /// `dflash_hidden_save` hasn't been populated yet.
    pub first_propose_done: bool,

    // ── Adaptive retrieval gate (ATLAS_DFLASH_SAM auto-disable) ──
    /// Whether the PREVIOUS propose pre-empted the neural drafter with a
    /// retrieval (SAM) draft. Set when the retrieval path fires; read on the
    /// next propose to attribute `last_num_accepted` to retrieval vs drafter.
    pub retr_used_last: bool,
    /// Consecutive retrieval steps whose accept came back poor. When it
    /// crosses the limit, retrieval enters a cooldown — this auto-disables SAM
    /// on content where its strong suffix matches mis-predict (e.g. counting:
    /// digit runs match but the next number is always new), while leaving it
    /// fully active on reuse-heavy code editing (where it keeps accepting).
    pub retr_misfire_streak: u32,
    /// Remaining propose steps to SKIP retrieval (cooldown). Decremented each
    /// step; retrieval is suppressed while > 0, then retried.
    pub retr_cooldown: u32,

    // ── Persistent context cache (eliminates O(seq_len) recompute) ──
    /// Cached fc_proj outputs for context tokens. Circular buffer of
    /// `[ctx_window, hidden_size]` BF16. Slot `p` holds absolute position
    /// `p % ctx_window`.
    pub ctx_fc_cache: DevicePtr,
    /// Cached K projections per layer. One buffer per layer,
    /// `[ctx_window, num_kv_heads * head_dim]` BF16.
    pub ctx_k_cache: Vec<DevicePtr>,
    /// Cached V projections per layer.
    pub ctx_v_cache: Vec<DevicePtr>,
    /// Per-layer cache range: first valid absolute position in the circular
    /// K cache for each layer. All layers cache the same positions in
    /// practice, but they must be tracked independently so layer N knows
    /// which positions layer N has already cached.
    pub cache_k_start: Vec<usize>,
    /// Per-layer K cache range end (one past last valid).
    pub cache_k_end: Vec<usize>,
    /// Per-layer V cache range start.
    pub cache_v_start: Vec<usize>,
    /// Per-layer V cache range end.
    pub cache_v_end: Vec<usize>,
    /// fc_proj cache range start (shared across layers).
    pub cache_fc_start: usize,
    /// fc_proj cache range end.
    pub cache_fc_end: usize,
    /// DDTree M4B: optional tree payload built by the drafter's most recent
    /// `propose_drafts()` when `--dflash-method=ddtree` is active. The
    /// scheduler drains this via `take_pending_tree_payload()` after each
    /// propose call and stashes it on `ActiveSeq.pending_tree_payload`
    /// (M3 plumbing). `None` for flat DFlash → preserves legacy behavior.
    pub pending_tree_payload: Option<ddtree::TreePayload>,
    /// Ring buffer of accept counts from the last `ACCEPT_HISTORY` verifies.
    /// `accept_history[accept_history_pos]` is the next slot to overwrite.
    /// Used by `ATLAS_DFLASH_ADAPTIVE_GAMMA=1` in `forward_block` to shrink
    /// the drafter noise block when recent accept is low.
    pub accept_history: [u8; 8],
    /// Next write index into `accept_history` (mod 8).
    pub accept_history_pos: usize,
    /// Number of verifies recorded so far (capped at `accept_history.len()`).
    /// Below 4 we don't adapt — let the drafter run full γ until we have a
    /// stable signal.
    pub accept_history_count: usize,
    /// Monotonic count of adaptive-engaged propose() calls. Unlike
    /// `accept_history_count` (saturates at 8), this counter never wraps
    /// during a sequence so it can drive periodic γ_max reprobes via
    /// `ATLAS_DFLASH_ADAPTIVE_PROBE_INTERVAL`. Without periodic reprobe,
    /// adaptive truncate is self-limiting: once γ_eff truncates to K=2/3/4,
    /// the history fills with the small accept counts from those truncated
    /// steps and the cutoff stays low forever even when content becomes
    /// more predictable (counting, lists, structured output). Reprobing
    /// re-measures the true accept ceiling.
    pub propose_steps: usize,
    /// ATLAS_DFLASH_ACCEPT_FALLBACK: number of remaining steps this sequence
    /// must spend in plain single-token decode (speculation suppressed)
    /// before the next re-probe. When > 0, `propose_drafts` returns an empty
    /// draft vector so the scheduler routes the sequence through the bootstrap
    /// plain-decode path. Decremented once per suppressed propose call. When
    /// it reaches 0, the next propose runs a full γ probe to re-measure
    /// acceptance. 0 = not suppressed (normal full-γ speculation).
    pub fallback_suppressed_remaining: usize,

    // ── ATLAS_DFLASH_RECYCLE=1: discarded-draft-tail recycling (default off) ──
    /// The tail of the PREVIOUS step's drafts that verify discarded after the
    /// first content miss — `drafts[num_accepted+1 .. γ_eff]`. The drafter
    /// often gets the STRUCTURAL continuation right even when the corrected
    /// token (committed by the target at the mismatch) differs, so re-offering
    /// this tail re-accepts the free structural part. Populated by
    /// `dflash_stash_recycle` from `verify_dflash_step` after `num_accepted`
    /// is known; consumed by `propose_drafts` on the next call. Empty ⇒ no
    /// tail available. LOSSLESS: these tokens are only PROPOSED — verify still
    /// commits target-greedy, so a wrong recycle costs one rejected
    /// speculation, never output.
    pub recycle_tail: Vec<u32>,
    /// The corrected token committed by the target at the mismatch position
    /// (= `verified[num_accepted]` = the bonus = next step's `last_token`).
    /// `propose_drafts` offers `recycle_tail` only when the new `last_token`
    /// equals this key — the tail was the drafter's continuation conditioned
    /// on this exact corrected token, so re-offering it is principled.
    pub recycle_key: u32,
    /// Whether `recycle_tail`/`recycle_key` hold a valid stash (distinguishes
    /// "empty tail because full-accept" from "no stash yet"; the offer path
    /// needs both a matching key AND a non-empty tail anyway, so this is
    /// belt-and-suspenders for the key==0 corner case).
    pub recycle_valid: bool,
    /// Whether the PREVIOUS propose returned a recycled tail. Prevents the
    /// self-sustaining low-accept trap: a recycled tail that re-accepts poorly
    /// keeps `last_num_accepted` low, which would re-open the recycle gate
    /// forever and STARVE the neural drafter (measured: counting collapsed
    /// 82→~9 tok/s because one early recycle offer trapped the sequence). With
    /// this flag, recycle never fires two steps in a row — after any offer the
    /// next step runs the real drafter, which re-establishes the true accept
    /// signal and produces a fresh tail. Recycle thus fires at most every other
    /// step, capping its worst-case throughput cost at one wasted speculation
    /// per two steps while still recovering the discarded tail on genuinely
    /// weak content.
    pub recycle_last_offered: bool,

    // ── ATLAS_DFLASH_ECHO=1: echo-drafting / Jacobi salvage (default off) ──
    /// The TARGET'S OWN verify argmaxes downstream of the bonus —
    /// `verified[num_accepted+1 ..]` from the previous flat-chain verify.
    /// Unlike `recycle_tail` (drafter-authored), these are target-authored:
    /// conditioned on a near-miss prefix, they are usually still right after
    /// the one-token bonus substitution. Populated by `dflash_stash_echo`
    /// (verify_dflash_step.rs, flat path only, gated on
    /// `num_accepted >= min_accept` + `tail >= min_tail`); consumed by
    /// `propose_drafts` on the next call, skipping the drafter forward
    /// entirely (the 25-50ms propose slice). LOSSLESS: proposal-only.
    pub echo_tail: Vec<u32>,
    /// The bonus token committed at the rejection (= next step's
    /// `last_token`). The echo tail is the target's continuation after this
    /// exact token's context, so it is offered only when the key matches.
    pub echo_key: u32,
    /// Whether `echo_tail`/`echo_key` hold a valid stash (belt-and-suspenders
    /// for the key==0 corner, mirroring `recycle_valid`).
    pub echo_valid: bool,
    /// Consecutive echo offers. Capped at `ATLAS_DFLASH_ECHO_MAX_STREAK`
    /// (default 2) so an echo-drafted step that itself rejects can not keep
    /// salvaging its own wreckage forever — after the cap the real drafter
    /// runs and re-establishes the true accept signal. Reset to 0 on any
    /// propose where echo does not fire.
    pub echo_streak: u32,
    /// Whether the PREVIOUS propose returned an echo tail. Read (and
    /// cleared) by the next propose to attribute `last_num_accepted` to the
    /// echo draft — the salvage-accept telemetry line.
    pub echo_offered_last: bool,
    /// ATLAS_DFLASH_ASYNC: this sequence's `pending_drafts` currently holds
    /// a PLACEHOLDER chain from an async (second-stream) propose launch; the
    /// real drafts are collected via `collect_async_drafts` at the top of
    /// the next scheduler step. Cleared on collect / resolve.
    pub async_placeholder: bool,
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
    /// Per-layer attention window in tokens. `Some(w)` for `sliding_attention`
    /// layers (vLLM PR #40898 — Qwen3.6-27B-DFlash has 4× sliding + 1× full),
    /// `None` for `full_attention`. Length == `num_layers`. Empty Vec when
    /// the drafter omits `layer_types` (older 3.5 drafter — treated as all
    /// full-attention via the propose codepath).
    pub layer_window_sizes: Vec<u32>,
    /// Per-layer causal mask flag (vLLM PR #40898 — DFlash uses `causal=True`
    /// for `sliding_attention` layers and `causal=False` for `full_attention`).
    /// Same length as `layer_window_sizes`. Empty Vec falls back to `false`
    /// for all layers (the original DFlash 3.5 behavior).
    pub layer_causal: Vec<bool>,
    /// `target_layer_ids`. Same data as `TransformerModel::dflash_capture_layers`,
    /// repeated here so the loader is the single source of truth; the model
    /// reads these to size its capture buffer.
    pub target_layer_ids: Vec<usize>,
    /// Target-side hidden_size (used for the `fc` projection input width:
    /// `target_layer_ids.len() * target_hidden_size`).
    pub target_hidden_size: usize,
    /// Target-side vocab_size. The drafter's `vocab_size` may be larger
    /// (e.g. 248320 vs target 248077) but the shared lm_head only has
    /// `target_vocab_size` rows. Capped at construction time.
    pub target_vocab_size: usize,

    // === Weights shared with the target ===
    /// Target's embed_tokens GPU pointer. The drafter's checkpoint has no
    /// own embeddings — both vocab and embedding dim must match the target
    /// (Qwen3.6-35B-A3B-DFlash: vocab=248320, hidden=2048 — same as target).
    pub embed_tokens_shared: DevicePtr,
    /// Target's lm_head GPU pointer. Used for the drafter's per-position
    /// argmax over `[γ, vocab]` logits.
    pub lm_head_shared: DevicePtr,

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
    ///
    /// Populated under `DflashQuantization::Bf16`. Under `Nvfp4` this field
    /// holds a stale (zeroed) pointer after the BF16 source is freed, and
    /// the forward path reads `fc_nvfp4` instead. Kept as a typed field
    /// (rather than `Option`) so the rest of `BlockDiffusionDraftHead` stays
    /// shape-stable.
    pub fc: DenseWeight,
    /// NVFP4-quantized version of `fc`. `Some` only when the drafter was
    /// built with `DflashQuantization::Nvfp4`. `forward_block` Step 0 reads
    /// this through `w4a16_gemv` when present, else falls back to `fc` via
    /// `dense_gemv`.
    pub fc_nvfp4: Option<QuantizedWeight>,
    /// Optional draft-vocab-id → target-vocab-id remap. `None` when the
    /// drafter shares vocab with the target (Qwen3.6-35B-A3B-DFlash case:
    /// vocab_size == draft_vocab_size == 248320).
    pub draft_id_to_target_id: Option<DevicePtr>,

    /// DSpark VanillaMarkov head. `Some` when the checkpoint ships the
    /// `markov_head.markov_w{1,2}.weight` tensors and `markov_rank > 0` and
    /// `ATLAS_DFLASH_MARKOV != 0`. When present, `propose_drafts` applies a
    /// per-position bigram logit bias `B(prev) = W2(W1[prev])` and samples the
    /// γ-block LEFT-TO-RIGHT (each position's chosen token biases the next),
    /// semi-autoregressively repairing suffix decay. LOSSLESS w.r.t. committed
    /// output — only changes which tokens are *proposed*; the target verify
    /// still commits its own greedy token.
    pub markov: Option<MarkovHead>,
    /// Drafter transformer layers (8 for Qwen3.6-35B-A3B-DFlash). Each
    /// layer carries either BF16 or NVFP4 weights — the `forward_block_layer`
    /// helper match-dispatches on the variant.
    pub layers: Vec<DflashLayerQuantWeights>,

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

    // Quantization mode (BF16 only for Phase 1).
    pub quant: DflashQuantization,

    // ── ATLAS_DFLASH_ASYNC (task #20) ──
    /// At most one in-flight async (second-stream) propose — the head owns a
    /// single scratch buffer set. `None` when idle / flag off. See
    /// `async_propose.rs` for the shared-scratch discipline.
    pub async_inflight: Mutex<Option<async_propose::AsyncInflight>>,
    /// Dedicated non-blocking CUDA stream for async propose launches.
    /// Lazily created on first eligible launch; `0` = creation failed →
    /// async permanently disabled.
    pub async_propose_stream: std::sync::OnceLock<u64>,
    /// Event used to order the propose stream after the default stream's
    /// prior writes (ctx-append D2Ds, verify captures). Set together with
    /// `async_propose_stream`.
    pub async_order_event: std::sync::atomic::AtomicU64,

    // ── ATLAS_DFLASH_FUSED ──
    /// Set by `arm_propose_overlap` immediately after verify returns (before
    /// commit is enqueued on the default stream). When true,
    /// `try_launch_async_propose` skips re-recording `async_order_event`
    /// (the pre-commit snapshot is already in place) so the propose stream
    /// only waits for verify, not for the ~10ms SSM commit + KV reshape.
    /// Cleared on each consume in `try_launch_async_propose`.
    pub fused_event_armed: std::sync::atomic::AtomicBool,
}

impl BlockDiffusionDraftHead {
    /// Copy a range of cached context slots from a circular buffer into a
    /// linear destination. Handles wrap-around with at most two D2D copies.
    ///
    /// `cache` is a circular buffer of `window` slots, each `slot_bytes`.
    /// Valid absolute positions are `[cache_start..cache_end)`.
    /// We need positions `[needed_start..needed_end)`.
    /// Copies overlap into `dst` at offsets `[pos - needed_start)`.
    fn cache_copy_range(
        &self,
        gpu: &dyn GpuBackend,
        cache: DevicePtr,
        cache_start: usize,
        cache_end: usize,
        window: usize,
        needed_start: usize,
        needed_end: usize,
        dst: DevicePtr,
        slot_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let copy_start = needed_start.max(cache_start);
        let copy_end = needed_end.min(cache_end);
        if copy_start >= copy_end {
            return Ok(());
        }
        // Find first wrap point in [copy_start..copy_end).
        let first_wrap = ((copy_start / window) + 1) * window;
        if first_wrap >= copy_end {
            // Entirely within one ring segment — single copy.
            let src_offset = (copy_start % window) * slot_bytes;
            let dst_offset = (copy_start - needed_start) * slot_bytes;
            let count = copy_end - copy_start;
            gpu.copy_d2d_async(
                cache.offset(src_offset),
                dst.offset(dst_offset),
                count * slot_bytes,
                stream,
            )?;
        } else {
            // Wraps around — two copies.
            let count1 = first_wrap - copy_start;
            let src1 = (copy_start % window) * slot_bytes;
            let dst1 = (copy_start - needed_start) * slot_bytes;
            gpu.copy_d2d_async(
                cache.offset(src1),
                dst.offset(dst1),
                count1 * slot_bytes,
                stream,
            )?;
            let count2 = copy_end - first_wrap;
            let src2 = 0usize;
            let dst2 = (first_wrap - needed_start) * slot_bytes;
            gpu.copy_d2d_async(
                cache.offset(src2),
                dst.offset(dst2),
                count2 * slot_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Write newly-computed context slots from a linear source into a
    /// circular cache. `src` holds `write_count` slots starting at absolute
    /// position `write_start`. Returns the updated `(cache_start, cache_end)`.
    fn cache_write_range(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        write_start: usize,
        write_count: usize,
        cache: DevicePtr,
        cache_start: usize,
        cache_end: usize,
        window: usize,
        slot_bytes: usize,
        stream: u64,
    ) -> Result<(usize, usize)> {
        if write_count == 0 {
            return Ok((cache_start, cache_end));
        }
        let write_end = write_start + write_count;
        let first_wrap = ((write_start / window) + 1) * window;
        if first_wrap >= write_end {
            // No wrap.
            let src_offset = 0usize;
            let dst_offset = (write_start % window) * slot_bytes;
            gpu.copy_d2d_async(
                src.offset(src_offset),
                cache.offset(dst_offset),
                write_count * slot_bytes,
                stream,
            )?;
        } else {
            // Wrap.
            let count1 = first_wrap - write_start;
            let src1 = 0usize;
            let dst1 = (write_start % window) * slot_bytes;
            gpu.copy_d2d_async(
                src.offset(src1),
                cache.offset(dst1),
                count1 * slot_bytes,
                stream,
            )?;
            let count2 = write_end - first_wrap;
            let src2 = count1 * slot_bytes;
            let dst2 = 0usize;
            gpu.copy_d2d_async(
                src.offset(src2),
                cache.offset(dst2),
                count2 * slot_bytes,
                stream,
            )?;
        }
        // Expand cache range to include the new positions.
        let (mut new_start, new_end) = if cache_start == 0 && cache_end == 0 {
            (write_start, write_end)
        } else {
            (cache_start.min(write_start), cache_end.max(write_end))
        };
        // Cap at window size (oldest positions are implicitly evicted).
        if new_end - new_start > window {
            new_start = new_end - window;
        }
        Ok((new_start, new_end))
    }
}

pub mod async_propose;
pub mod ddtree;
pub mod ddtree_gdn_contract;
pub mod ddtree_gdn_dispatch;
pub mod echo;
mod forward_block;
mod forward_block_layer;
mod from_weights;
mod markov;
mod noise_pass;
mod propose;
pub mod retrieval;

// Re-export DDTree payload so the scheduler can carry it as Option<DDTreePayload>
// in ActiveSeq (M3 milestone — pure plumbing, no behavior change).
pub use ddtree::TreePayload as DDTreePayload;

// ── Kernel profiler accumulator (ATLAS_DFLASH_KERNEL_PROFILE=1) ──
// Per-kernel μs accumulated across all drafter layers in one propose()
// call. Reset by `kprof_reset_layers()` at the top of forward_block;
// `forward_block_layer` adds into each field via `kp!`; snapshot is
// returned by `kprof_snapshot_layers()` at end of forward_block.
//
// Thread-local because Atlas serves one request per stream and we only
// want per-step aggregation, not global. Using `Cell<KprofAcc>` keeps
// the helpers cheap and Copy.
#[derive(Clone, Copy, Default)]
pub(super) struct KprofAcc {
    pub input_norm_us: u128,
    pub q_proj_us: u128,
    pub kv_ctx_copy_us: u128,
    pub kv_ctx_new_us: u128,
    pub kv_noise_us: u128,
    pub qk_norm_us: u128,
    pub rope_us: u128,
    pub cache_write_us: u128,
    pub prefill_attn_us: u128,
    pub o_proj_us: u128,
    pub resid1_us: u128,
    pub post_norm_us: u128,
    pub gate_up_us: u128,
    pub silu_mul_us: u128,
    pub down_proj_us: u128,
    pub resid2_us: u128,
}

thread_local! {
    static KPROF_ACC: std::cell::Cell<KprofAcc> = const { std::cell::Cell::new(KprofAcc {
        input_norm_us: 0, q_proj_us: 0, kv_ctx_copy_us: 0, kv_ctx_new_us: 0,
        kv_noise_us: 0, qk_norm_us: 0, rope_us: 0, cache_write_us: 0,
        prefill_attn_us: 0, o_proj_us: 0, resid1_us: 0, post_norm_us: 0,
        gate_up_us: 0, silu_mul_us: 0, down_proj_us: 0, resid2_us: 0,
    }) };
}

pub(super) fn kprof_reset_layers() {
    KPROF_ACC.with(|c| c.set(KprofAcc::default()));
}

pub(super) fn kprof_snapshot_layers() -> KprofAcc {
    KPROF_ACC.with(|c| c.get())
}

pub(super) fn kprof_add(f: impl FnOnce(&mut KprofAcc)) {
    KPROF_ACC.with(|c| {
        let mut a = c.get();
        f(&mut a);
        c.set(a);
    });
}

impl DraftProposer for BlockDiffusionDraftHead {
    fn is_dflash(&self) -> bool {
        true
    }

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
        // Allocate per-sequence persistent context caches.
        // These eliminate O(seq_len) recomputation of fc_proj and k_proj/v_proj
        // for previously-seen context positions.
        let ctx_fc_cache = gpu.alloc(self.ctx_window * self.hidden_size * bf16)?;
        let mut ctx_k_cache = Vec::with_capacity(self.num_layers);
        let mut ctx_v_cache = Vec::with_capacity(self.num_layers);
        let kv_dim = self.num_kv_heads * self.head_dim;
        for _ in 0..self.num_layers {
            ctx_k_cache.push(gpu.alloc(self.ctx_window * kv_dim * bf16)?);
            ctx_v_cache.push(gpu.alloc(self.ctx_window * kv_dim * bf16)?);
        }
        let zeros = vec![0usize; self.num_layers];

        Ok(Box::new(DflashProposerState {
            block_table: Vec::with_capacity(64),
            seq_len: 0,
            last_num_drafted: 0,
            prefill_done: false,
            ctx_hidden_acc,
            ctx_len: 0,
            max_ctx_len: self.max_seq_len,
            ctx_slot_bytes,
            last_capture_idx: 0,
            last_num_accepted: 0,
            last_accepted_compact: Vec::new(),
            pld_tokens: Vec::new(),
            first_propose_done: false,
            retr_used_last: false,
            retr_misfire_streak: 0,
            retr_cooldown: 0,
            ctx_fc_cache,
            ctx_k_cache,
            ctx_v_cache,
            cache_k_start: zeros.clone(),
            cache_k_end: zeros.clone(),
            cache_v_start: zeros.clone(),
            cache_v_end: zeros.clone(),
            cache_fc_start: 0,
            cache_fc_end: 0,
            pending_tree_payload: None,
            accept_history: [0u8; 8],
            accept_history_pos: 0,
            accept_history_count: 0,
            propose_steps: 0,
            fallback_suppressed_remaining: 0,
            recycle_tail: Vec::new(),
            recycle_key: 0,
            recycle_valid: false,
            recycle_last_offered: false,
            echo_tail: Vec::new(),
            echo_key: 0,
            echo_valid: false,
            echo_streak: 0,
            echo_offered_last: false,
            async_placeholder: false,
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
        // `num_accepted` equals the index of the last accepted token in the
        // verify batch (0 = prefix token, 1 = first draft, etc.). The
        // multi-token `dflash_hidden_save` buffer holds hidden states for
        // every verify position; `last_capture_idx` selects the correct one.
        dstate.last_capture_idx = num_accepted;
        dstate.last_num_accepted = num_accepted;
        // Clear any stale tree-fork path; the scheduler re-stamps it via
        // `set_dflash_accepted_compact` AFTER this when the step forked.
        dstate.last_accepted_compact.clear();
        dstate.last_num_drafted = 0;
        // Push accept count into the ring buffer for ATLAS_DFLASH_ADAPTIVE_GAMMA.
        // Saturating cast: num_accepted >= 256 cannot happen because dflash_kgamma
        // <= 64 in practice and the buffer width is u8.
        let slot = dstate.accept_history_pos % dstate.accept_history.len();
        dstate.accept_history[slot] = num_accepted.min(u8::MAX as usize) as u8;
        dstate.accept_history_pos = (dstate.accept_history_pos + 1) % dstate.accept_history.len();
        dstate.accept_history_count =
            (dstate.accept_history_count + 1).min(dstate.accept_history.len());
        Ok(())
    }

    /// DDTree M6: drain the tree payload stashed by `propose()` (if any).
    /// Returns + clears `dstate.pending_tree_payload`.
    fn take_pending_tree_payload(
        &self,
        state: &mut dyn ProposerState,
    ) -> Option<crate::layers::DDTreePayload> {
        state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .and_then(|s| s.pending_tree_payload.take())
    }

    fn free_state(&self, _state: &mut dyn ProposerState) -> Result<()> {
        // Per-sequence device allocations (ctx_hidden_acc, ctx_fc_cache,
        // ctx_k_cache, ctx_v_cache) are not freed here because free_state
        // lacks a GpuBackend reference. Total leak is ~15 MB per sequence,
        // acceptable for typical session lifetimes. The allocator reclaims
        // on process exit.
        Ok(())
    }

    fn collect_async_drafts(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut dyn ProposerState,
    ) -> Result<Option<Vec<u32>>> {
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        self.collect_async_drafts_impl(gpu, dstate)
    }

    fn resolve_async_inflight(
        &self,
        gpu: &dyn GpuBackend,
        state: Option<&mut dyn ProposerState>,
    ) -> Result<()> {
        let dstate = state.and_then(|s| s.as_any_mut().downcast_mut::<DflashProposerState>());
        self.resolve_async_inflight_impl(gpu, dstate)
    }

    fn arm_propose_overlap(
        &self,
        gpu: &dyn GpuBackend,
        default_stream: u64,
    ) -> Result<()> {
        BlockDiffusionDraftHead::arm_propose_overlap(self, gpu, default_stream)
    }
}
