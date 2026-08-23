// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash block-diffusion draft head implementing [`DraftProposer`].
//!
//! Block-diffusion drafter (Z Lab, arXiv 2602.06036): a small Qwen3-architecture
//! transformer (8 layers, hidden=2048, GQA 32:4, head_dim=128) that emits γ
//! draft tokens **in a single forward pass** via bidirectional in-block
//! attention. The training `block_size` includes one known anchor row, so a
//! block-16 checkpoint has γ=15 trained drafts.
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
use std::time::Instant;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle, PinnedHostBuffer};

use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Cross-request pool for the (multi-GB) per-sequence `ctx_hidden_acc`
/// accumulator. Allocating it fresh per request (up to 5.37 GB at
/// max_seq_len=65536) costs 200-950 ms of UMA first-touch page faults on
/// every request — measured 623/952/188 ms in the 2026-08-19 alloc split
/// vs 28 ms for the memset. Pooling the allocation across requests removes
/// that from TTFT while the per-request memset keeps stale-slot semantics
/// bit-identical (every read slot is written before read, but the zero-init
/// is retained as the defensive baseline). Keyed by byte size so different
/// arm/config sizes cannot cross-pollinate. max_batch_size=1 means at most
/// one buffer is in flight at a time; the Vec tolerates a future larger
/// batch. Buffers are deliberately never returned to CUDA (process-lifetime
/// reuse, same discipline as the SSM pool's fixed addresses).
pub(crate) fn ctx_acc_pool() -> &'static Mutex<std::collections::HashMap<usize, Vec<DevicePtr>>> {
    static POOL: std::sync::OnceLock<Mutex<std::collections::HashMap<usize, Vec<DevicePtr>>>> =
        std::sync::OnceLock::new();
    POOL.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Take a pooled `ctx_hidden_acc` buffer of exactly `bytes`, if one is idle.
pub(crate) fn ctx_acc_pool_take(bytes: usize) -> Option<DevicePtr> {
    ctx_acc_pool().lock().get_mut(&bytes)?.pop()
}

/// Return a `ctx_hidden_acc` buffer to the pool for the next request.
pub(crate) fn ctx_acc_pool_return(bytes: usize, ptr: DevicePtr) {
    if !ptr.is_null() {
        ctx_acc_pool().lock().entry(bytes).or_default().push(ptr);
    }
}

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

/// Resolve the effective neural/host-draft width against the trained maximum.
/// The scheduler's runtime arm override has precedence over the environment,
/// matching the forward path's established contract.
fn resolve_effective_draft_width(
    trained_drafts: usize,
    runtime_override: Option<usize>,
    env_cap: Option<usize>,
) -> usize {
    runtime_override
        .or(env_cap)
        .unwrap_or(trained_drafts)
        .min(trained_drafts)
        .max(1)
}

pub(crate) fn effective_draft_width(trained_drafts: usize) -> usize {
    let env_cap = std::env::var("ATLAS_DFLASH_DRAFT_CAP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    resolve_effective_draft_width(
        trained_drafts,
        crate::speculative::dflash_gamma_override(),
        env_cap,
    )
}

#[cfg(test)]
mod effective_draft_width_tests {
    use super::resolve_effective_draft_width;

    #[test]
    fn width_never_exceeds_the_trained_rows() {
        assert_eq!(resolve_effective_draft_width(15, None, None), 15);
        assert_eq!(resolve_effective_draft_width(15, None, Some(16)), 15);
        assert_eq!(resolve_effective_draft_width(6, None, Some(4)), 4);
    }

    #[test]
    fn runtime_arm_cap_precedes_environment_cap() {
        assert_eq!(resolve_effective_draft_width(15, Some(3), Some(8)), 3);
        assert_eq!(resolve_effective_draft_width(15, Some(0), Some(8)), 1);
    }

    #[test]
    fn pld_effective_width_cannot_recreate_the_untrained_tail() {
        // `propose.rs` uses this same resolver before slicing a PLD hit.
        assert_eq!(resolve_effective_draft_width(15, None, Some(16)), 15);
        assert_eq!(resolve_effective_draft_width(15, Some(4), Some(12)), 4);
        assert_eq!(resolve_effective_draft_width(6, None, None), 6);
    }
}

/// Compile-time cap on the per-position top-K used by the DDTree (M4B v2)
/// builder. Must match `MAX_TOP_K` in `kernels/gb10/common/argmax_bf16.cu`.
/// Runtime `top_k` comes from `ATLAS_DDTREE_TOP_K` (default 8) and is
/// validated against this maximum; invalid values fail closed without launch.
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
    /// DFlash 2 grouped-dynamic-conv stage 0 (`prepare`): convolves the
    /// normed noise rows in place and exports the stage-1 dynamic rows.
    /// Resolved only for DFlash2 checkpoints; sentinel zero otherwise.
    pub dflash2_conv_prepare: KernelHandle,
    /// DFlash 2 grouped-dynamic-conv stage 1 (`finish`): convolves the
    /// sublayer output with the stage-1 dynamic rows. Resolved only for
    /// DFlash2 checkpoints; sentinel zero otherwise.
    pub dflash2_conv_finish: KernelHandle,
    /// DFlash 2 candidate-selector greedy walk (top-k codebook scores).
    /// Resolved only for DFlash2 checkpoints; sentinel zero otherwise.
    pub dflash2_selector_walk: KernelHandle,
    /// Exact `argmax(base_logits + markov_bias)` with lowest-token tie-break.
    /// Keeps the DSpark left-to-right Markov chain on the producer stream.
    /// Resolved fail-closed for Markov checkpoints; sentinel zero for generic
    /// DFlash so this algorithm-specific kernel does not broaden its contract.
    pub argmax_add: KernelHandle,
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
    /// `w4a16_gemm_t_m32_n64_gateup_silu` — FUSED gate_proj + up_proj +
    /// SiLU·mul in one launch (A loaded once, both B streams, one [M,N]
    /// write). The drafter FFN dispatches to it when the transposed gate/up
    /// weights are present, replacing two m32_n64 GEMMs + the standalone
    /// silu_mul launch. `try_kernel`; sentinel 0 on miss → fall back to the
    /// separate-GEMM path.
    pub w4a16_gemm_t_m32_n64_gateup_silu: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_splitk` + `reduce_splitk_f32_to_bf16` — the
    /// K-sliced variant of `w4a16_gemm_t_m32_n64` and its FP32 band reducer.
    /// Used only when `ATLAS_DFLASH_DRAFT_SPLITK` >= 2; see `draft_splitk.rs`
    /// for why the drafter's narrow-N projections are occupancy-starved.
    pub w4a16_gemm_t_m32_n64_splitk: KernelHandle,
    pub reduce_splitk_k: KernelHandle,
    /// `fp8_gemm_t` (BF16 A × FP8-E4M3 B) — the propose lm_head FP8 fast
    /// path (`ATLAS_DFLASH_LM_HEAD_FP8=1`). `try_kernel`; sentinel 0 on miss.
    pub fp8_gemm_t: KernelHandle,
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
    /// DSpark Markov head scratch — the seed token id (`[1]` u32) fed to the
    /// first `batched_embed` gather. Later positions read the preceding u32
    /// directly from `draft_tokens_dev`, so no host round-trip or D2D copy is
    /// needed. `DevicePtr::NULL` when no Markov head.
    pub markov_prev_dev: DevicePtr,
    /// DFlash 2 conv scratch — the `kernel_projection` output
    /// `[n_attn, 2*kernel_size*groups]` BF16 (both stages). Allocated only
    /// for DFlash2 checkpoints; `DevicePtr::NULL` otherwise.
    pub conv_dyn: DevicePtr,
    /// DFlash 2 conv scratch — the stage-1 (finish) dynamic rows exported by
    /// `prepare`: `[n_attn, kernel_size*groups]` BF16. `DevicePtr::NULL` for
    /// non-DFlash2 checkpoints.
    pub conv_dyn1: DevicePtr,
    /// DFlash 2 conv scratch — non-aliased conv output `[n_attn, hidden]`
    /// BF16. The conv kernel's causal taps read a *previous row* of the same
    /// buffer it writes, so in-place would race; the kernel writes this
    /// buffer and the caller D2D-copies the noise slice back to
    /// norm_buf/stream_acc. `DevicePtr::NULL` for non-DFlash2 checkpoints.
    pub conv_out: DevicePtr,
    /// DFlash 2 selector scratch — the `hidden_projection` output
    /// `[gamma, selector_rank]` BF16. `DevicePtr::NULL` for non-DFlash2
    /// checkpoints.
    pub selector_hidden: DevicePtr,
    /// FP32 split-K partial-product workspace `[k_splits, 32, n]` for the
    /// drafter's own projections. `DevicePtr::NULL` unless
    /// `ATLAS_DFLASH_DRAFT_SPLITK` >= 2 was set at load time (allocation must
    /// happen before CUDA graph capture, so it is not lazily created).
    pub splitk_ws: DevicePtr,
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
    /// DFlash 2 two-tap dynamic grouped conv wrapping the attention sublayer
    /// (`attention_conv.base_kernel` + `attention_conv.kernel_projection`),
    /// present iff the checkpoint is DFlash2 (`incoai/Qwen3.8-27B-DFlash2`).
    /// `None` for plain DFlash / DSpark drafters — `forward_block_layer`
    /// skips the conv entirely in that case.
    pub attention_conv: Option<crate::weight_loader::DflashConvWeights>,
    /// DFlash 2 two-tap dynamic grouped conv wrapping the MLP sublayer.
    /// `None` for plain DFlash / DSpark drafters.
    pub mlp_conv: Option<crate::weight_loader::DflashConvWeights>,
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
    /// DFlash 2 two-tap dynamic grouped conv weights (BF16 — the checkpoint
    /// ships them BF16 and they are small; they stay unquantized even in the
    /// NVFP4 drafter variant). `None` for plain DFlash / DSpark drafters.
    pub attention_conv: Option<crate::weight_loader::DflashConvWeights>,
    /// DFlash 2 MLP conv (see `attention_conv`).
    pub mlp_conv: Option<crate::weight_loader::DflashConvWeights>,
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

/// Per-sequence DFlash drafter state for the active BF16 circular-context path.
///
/// There is deliberately no paged draft-KV block table here. The old state
/// carried a table, sequence length, and prefill flag for an FP8 paged design,
/// but no production forward ever read them. Context projection plus every
/// layer's K/V are instead retained in the circular buffers below.
pub struct DflashProposerState {
    /// Drafts produced in the last `propose()` call. `after_verify` consults
    /// this to know how many KV positions to roll back when the accept
    /// prefix is shorter than γ.
    pub last_num_drafted: usize,
    /// Immutable limits from the most recent outer `propose` call.
    pub draft_budget: Option<draft_budget::DflashDraftBudget>,
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
    /// Page-locked host mirror of the per-propose position-id buffer
    /// (`[ctx_pos_0..ctx_pos_{eff_ctx-1}, seq_pos..seq_pos+γ-1]`). The
    /// forward writes positions here host-side and enqueues a stream-ordered
    /// async H2D on the propose stream, avoiding the per-propose
    /// `cuStreamSynchronize(default_stream)` drain that `copy_h2d` performs
    /// (measured ~8.7 ms of hidden serialization per cycle). Sized for
    /// `(ctx_window + γ + 1) * 4` bytes.
    pub pos_pinned: PinnedHostBuffer,

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
    /// Throughput router state. Timing begins immediately before neural
    /// proposal and ends in `after_verify`, covering the cost that matters to
    /// server decode throughput rather than drafter acceptance in isolation.
    throughput_router: throughput_router::ThroughputRouter,
    /// Climb/drop adaptive depth controller (ATLAS_DFLASH_TPS_ROUTER_MODE=
    /// climbdrop, ported from llama.cpp PR #27210). Fed by `after_verify`
    /// with (last_num_drafted, num_accepted) and queried by `forward_block`
    /// for the draft cutoff. Mutually exclusive with the EWMA router.
    climbdrop_router: throughput_router::ClimbDropRouter,
    throughput_cycle_started: Option<Instant>,
    throughput_last_width: usize,
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

/// Family-specific draft-query/output geometry.
///
/// Generic DFlash trains an anchor followed by `gamma` MASK rows and emits
/// logits only from the MASK rows. DSpark instead runs exactly `gamma` query
/// rows (anchor plus `gamma - 1` MASK rows) and emits logits from every row,
/// including the anchor. The latter matches SGLang's DSpark proposal path:
/// row 0 is the target bonus token and all `gamma` raw hidden rows feed the
/// shared LM head and Markov sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DraftRowLayout {
    pub query_rows: usize,
    pub output_start: usize,
}

impl DraftRowLayout {
    pub fn for_family(family: crate::weight_loader::DrafterCheckpointFamily, gamma: usize) -> Self {
        match family {
            crate::weight_loader::DrafterCheckpointFamily::Dflash => Self {
                query_rows: gamma + 1,
                output_start: 1,
            },
            crate::weight_loader::DrafterCheckpointFamily::Dspark => Self {
                query_rows: gamma,
                output_start: 0,
            },
        }
    }

    pub fn feedback_rows(self) -> usize {
        self.query_rows.saturating_sub(1)
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
    pub checkpoint_family: crate::weight_loader::DrafterCheckpointFamily,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub gamma: usize,
    /// Verify-side physical token capacity, including the bonus/root row.
    pub physical_verify_k: usize,
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
    /// Target's transposed NVFP4 lm_head (shared pointer, no extra memory)
    /// for the propose lm_head fast path (`ATLAS_DFLASH_LM_HEAD_NVFP4=1`):
    /// reads the `--mtp-vocab` column prefix of the T-weight via
    /// `w4a16_gemm_t_m32_n64` (~0.25 GB at 96k vocab) instead of streaming
    /// the ~1 GB BF16 slice through scalar `dense_gemm`. Drafter-only —
    /// verify commits the target's own argmax, so this affects acceptance,
    /// never committed tokens. Wired post-construction in `factory/build.rs`.
    pub lm_head_shared_t: Option<crate::weight_map::QuantizedWeight>,
    /// Row stride (padded full-vocab N) of `lm_head_shared_t`.
    pub lm_head_shared_t_ldb: u32,
    /// Pre-scaled FP8-E4M3 copy of the `--mtp-vocab` lm_head slice
    /// (`ATLAS_DFLASH_LM_HEAD_FP8=1`, built at load in `from_weights`):
    /// `[lm_vocab, K]` row-major, weights multiplied by a power-of-2 scale
    /// `s` chosen so absmax·s ≈ 256 (keeps small weights out of the E4M3
    /// subnormal floor). The compensating `1/s` is folded into the drafter's
    /// final `norm` weight — whose ONLY consumer is this lm_head — so the
    /// logits come out in true scale and the DDTree cliff margins are
    /// undistorted. Halves the propose lm_head read vs BF16 (0.98→0.49 GB)
    /// at far higher logit fidelity than the NVFP4 slice (which measured
    /// accepted 5.88→5.36 and was refuted, 2026-07-31).
    pub lm_head_shared_fp8: Option<DevicePtr>,

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
    /// DFlash 2 candidate selector (3 BF16 tensors) replacing the per-row
    /// argmax in the propose tail. `Some` only for DFlash2 checkpoints
    /// (`candidate_selector.*` present + `selector_rank > 0`). When present,
    /// the propose tail runs top-16 per position and walks a coherent path
    /// via low-rank codebook scores instead of plain per-row argmax
    /// (reference: `CandidateSelector.select`, z-lab/dflash dflash/model.py).
    /// Drafter-only — the target verify still commits its own greedy token.
    pub selector: Option<crate::weight_loader::DflashSelectorWeights>,
    /// Drafter transformer layers (8 for Qwen3.6-35B-A3B-DFlash). Each
    /// layer carries either BF16 or NVFP4 weights — the `forward_block_layer`
    /// helper match-dispatches on the variant.
    pub layers: Vec<DflashLayerQuantWeights>,

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

    /// YaRN's post-RoPE cos/sin multiplier. Transformers derives this as
    /// `1 + 0.1 * ln(factor)` unless the checkpoint supplies an explicit
    /// `attention_factor`; vanilla RoPE uses 1.0.
    pub rope_attention_factor: f32,

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
pub mod dflash3;
mod draft_budget;
mod draft_splitk;
pub mod echo;
mod forward_block;
mod forward_block_layer;
mod from_weights;
mod logits_layout;
mod markov;
mod noise_pass;
mod pctree;
mod propose;
pub mod retrieval;
mod throughput_router;

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
    /// DFlash2 attention-conv `prepare` / `finish` (kernel_projection GEMM +
    /// conv + the D2D copy-back). Previously unlabelled on the NVFP4 path, so
    /// a profile could not tell "ran unwrapped" from "never ran".
    pub conv_prepare_us: u128,
    pub conv_finish_us: u128,
}

thread_local! {
    static KPROF_ACC: std::cell::Cell<KprofAcc> = const { std::cell::Cell::new(KprofAcc {
        input_norm_us: 0, q_proj_us: 0, kv_ctx_copy_us: 0, kv_ctx_new_us: 0,
        kv_noise_us: 0, qk_norm_us: 0, rope_us: 0, cache_write_us: 0,
        prefill_attn_us: 0, o_proj_us: 0, resid1_us: 0, post_norm_us: 0,
        gate_up_us: 0, silu_mul_us: 0, down_proj_us: 0, resid2_us: 0,
        conv_prepare_us: 0, conv_finish_us: 0,
    }) };
}

pub(super) fn kprof_reset_layers() {
    KPROF_ACC.with(|c| c.set(KprofAcc::default()));
}

pub(super) fn kprof_snapshot_layers() -> KprofAcc {
    KPROF_ACC.with(|c| c.get())
}

impl KprofAcc {
    /// Every field paired with the label used in the `DFLASH_KP` log line and
    /// in the `KPROF` table. Keeping one list means a new field cannot be
    /// added to the report without also being published to `ATLAS_FULL_PROFILE`.
    pub(super) fn labelled(&self) -> [(&'static str, u128); 18] {
        [
            ("draft_input_norm", self.input_norm_us),
            ("draft_q_proj", self.q_proj_us),
            ("draft_kv_ctx_copy", self.kv_ctx_copy_us),
            ("draft_kv_ctx_new", self.kv_ctx_new_us),
            ("draft_kv_noise", self.kv_noise_us),
            ("draft_qk_norm", self.qk_norm_us),
            ("draft_rope", self.rope_us),
            ("draft_cache_write", self.cache_write_us),
            ("draft_prefill_attn", self.prefill_attn_us),
            ("draft_o_proj", self.o_proj_us),
            ("draft_resid1", self.resid1_us),
            ("draft_post_norm", self.post_norm_us),
            ("draft_gate_up", self.gate_up_us),
            ("draft_silu_mul", self.silu_mul_us),
            ("draft_down_proj", self.down_proj_us),
            ("draft_resid2", self.resid2_us),
            ("draft_conv_prepare", self.conv_prepare_us),
            ("draft_conv_finish", self.conv_finish_us),
        ]
    }

    /// Sum of every attributed field, in microseconds.
    pub(super) fn attributed_us(&self) -> u128 {
        self.labelled().iter().map(|(_, v)| *v).sum()
    }
}

/// True when the per-kernel propose profiler should run: either its own
/// `ATLAS_DFLASH_KERNEL_PROFILE=1`, or `ATLAS_FULL_PROFILE=1` (including the
/// SIGUSR1 runtime override), so ONE flag attributes both the verify path and
/// the drafter's internal kernels.
///
/// Historically only the verify path answered to `ATLAS_FULL_PROFILE`; the
/// drafter's 6 transformer layers used this separate accumulator and so were
/// invisible in a full profile — the drafter's lm_head was the only kernel
/// that appeared, because it is the one propose-side launch wrapped in
/// `kprof!` rather than the layer-local `kp!`.
pub(super) fn kernel_profile_enabled() -> bool {
    std::env::var("ATLAS_DFLASH_KERNEL_PROFILE").ok().as_deref() == Some("1")
        || crate::full_profile::is_enabled()
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

    fn physical_verify_k(&self) -> Option<usize> {
        Some(self.physical_verify_k)
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
        let alloc_t0 = std::time::Instant::now();
        // Reuse a pooled buffer when one is idle (see `ctx_acc_pool`): a
        // fresh cuMemAlloc of this size page-faults ~200-950 ms per request
        // on UMA. The memset below still zeroes it every request, so pooled
        // reuse is bit-identical to the old per-request alloc+memset.
        let pool_hit = ctx_acc_pool_take(total);
        let ctx_hidden_acc = match pool_hit {
            Some(p) => p,
            None => gpu.alloc(total)?,
        };
        let alloc_us = alloc_t0.elapsed().as_secs_f64() * 1e6;
        let memset_t0 = std::time::Instant::now();
        // Initialize to zero so stale data doesn't leak between sequences.
        gpu.memset(ctx_hidden_acc, 0, total)?;
        let memset_us = memset_t0.elapsed().as_secs_f64() * 1e6;
        if std::env::var("ATLAS_PREFILL_PHASE_PROFILE").ok().as_deref() == Some("1") {
            tracing::info!(
                "DFLASH ALLOC_STATE | ctx_total_bytes={:.2}GB pool_hit={} alloc={:.1}ms memset={:.1}ms",
                total as f64 / 1e9,
                pool_hit.is_some(),
                alloc_us / 1000.0,
                memset_us / 1000.0,
            );
        }
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
        // Page-locked host buffer sized for the max position-id layout
        // (`ctx_window` context positions + up to γ+1 noise rows).
        let pos_pinned = gpu.alloc_host_pinned((self.ctx_window + self.gamma + 1) * 4)?;

        Ok(Box::new(DflashProposerState {
            last_num_drafted: 0,
            draft_budget: None,
            ctx_hidden_acc,
            ctx_len: 0,
            max_ctx_len: self.max_seq_len,
            ctx_slot_bytes,
            last_capture_idx: 0,
            last_num_accepted: 0,
            last_accepted_compact: Vec::new(),
            pld_tokens: Vec::new(),
            first_propose_done: false,
            pos_pinned,
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
            throughput_router: throughput_router::ThroughputRouter::default(),
            climbdrop_router: throughput_router::ClimbDropRouter::default(),
            throughput_cycle_started: None,
            throughput_last_width: 0,
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
        let result = self.propose_drafts(
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
        );
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        match result {
            Ok(drafts) => {
                let Some(budget) = dstate.draft_budget else {
                    draft_budget::clear_proposal_outputs(dstate);
                    anyhow::bail!("DFlash proposal returned without an outer draft budget")
                };
                let had_drafts = !drafts.is_empty();
                let finalized = draft_budget::finalize_proposal(dstate, budget, drafts);
                if had_drafts && finalized.is_empty() {
                    self.resolve_async_inflight_impl(ctx.gpu, Some(dstate))?;
                }
                Ok(finalized)
            }
            Err(error) => {
                draft_budget::clear_proposal_outputs(dstate);
                Err(error)
            }
        }
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
        if let Some(started) = dstate.throughput_cycle_started.take() {
            let elapsed = started.elapsed().as_secs_f64();
            let alpha = std::env::var("ATLAS_DFLASH_TPS_ROUTER_ALPHA")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.30);
            let width = dstate.throughput_last_width;
            let climbdrop = std::env::var("ATLAS_DFLASH_TPS_ROUTER_MODE")
                .ok()
                .as_deref()
                == Some("climbdrop");
            if climbdrop {
                // Climb/drop mode: feed (drafts offered, drafts accepted).
                // `num_accepted` is the count of accepted drafts (0 = only
                // the prefix/bonus token matched). The controller needs the
                // number of drafts actually verified this cycle, which is
                // `last_num_drafted` when it was non-zero.
                let offered = if dstate.last_num_drafted > 0 {
                    dstate.last_num_drafted
                } else {
                    width
                };
                dstate
                    .climbdrop_router
                    .update(offered, num_accepted.min(offered));
            } else {
                dstate
                    .throughput_router
                    .observe(width, num_accepted + 1, elapsed, alpha);
            }
            if let Some(score) = dstate.throughput_router.score(width) {
                tracing::debug!(
                    "DFlash TPS router observe: width={} delivered={} elapsed_ms={:.3} ewma_tps={:.3}",
                    width,
                    num_accepted + 1,
                    elapsed * 1000.0,
                    score
                );
            }
        }
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
        let dstate = state.as_any_mut().downcast_mut::<DflashProposerState>()?;
        let payload = dstate.pending_tree_payload.take()?;
        let Some(budget) = dstate.draft_budget else {
            tracing::warn!("DFlash tree rejected without an outer draft budget");
            return None;
        };
        if let Err(error) = budget.validate_tree(&payload) {
            tracing::warn!("DFlash tree rejected before scheduler exposure: {error:#}");
            return None;
        }
        Some(payload)
    }

    fn free_state(&self, _state: &mut dyn ProposerState) -> Result<()> {
        // Per-sequence device allocations (ctx_hidden_acc, ctx_fc_cache,
        // ctx_k_cache, ctx_v_cache) are not freed here because free_state
        // lacks a GpuBackend reference. No unused paged-FP8 pool is attached
        // to the head anymore. The remaining state-lifetime leak is separate
        // and the allocator reclaims it on process exit.
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

    fn arm_propose_overlap(&self, gpu: &dyn GpuBackend, default_stream: u64) -> Result<()> {
        BlockDiffusionDraftHead::arm_propose_overlap(self, gpu, default_stream)
    }
}

#[cfg(test)]
mod draft_kv_storage_contract_tests {
    #[test]
    fn dead_paged_fp8_path_cannot_silently_return() {
        let head = include_str!("dflash_head.rs");
        let constructor = include_str!("dflash_head/from_weights.rs");
        for dead in [
            concat!("Paged", "KvCache"),
            concat!("KvCache", "Config"),
            concat!("reshape_cache", "_fp8"),
            concat!("prefill_attn_dflash", "_fp8"),
        ] {
            assert!(!head.contains(dead), "head reintroduced dead `{dead}`");
            assert!(
                !constructor.contains(dead),
                "constructor reintroduced dead `{dead}`"
            );
        }
    }

    #[test]
    fn active_context_path_remains_bf16_circular() {
        let head = include_str!("dflash_head.rs");
        let layer = include_str!("dflash_head/forward_block_layer.rs");
        for required in ["ctx_fc_cache", "ctx_k_cache", "ctx_v_cache"] {
            assert!(head.contains(required), "missing active `{required}` state");
        }
        assert!(layer.contains("cache_write_range("));
        assert!(layer.contains("ops::prefill_attention("));
        assert!(!layer.contains(concat!("prefill_attention", "_paged")));
    }
}
