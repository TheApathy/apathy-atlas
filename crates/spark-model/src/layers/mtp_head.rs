// SPDX-License-Identifier: AGPL-3.0-only

//! MTP (Multi-Token Prediction) head implementing [`DraftProposer`].
//!
//! Single transformer decoder layer trained jointly with the target model.
//! Forward pass: embed+hidden concat → fc → attention → MoE → norm → lm_head → argmax.
//!
//! Weight precision is parameterized via [`MtpQuantization`]: NVFP4 (4-bit),
//! FP8 (8-bit), or BF16 (16-bit). Higher precision improves draft acceptance
//! at the cost of increased MTP forward latency.

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::ForwardContext;
use crate::layers::MoeLayer;
use crate::layers::fp8_calibration::Fp8KvCalibration;
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{
    DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight, quantize_to_fp8, quantize_to_nvfp4,
};

/// Returns true when `ATLAS_MTP_FP8_CALIB=1` is set in the process env.
///
/// Gates the MTP head's per-attention-layer FP8 KV-cache scale calibration.
/// Default OFF after empirical A/B (2026-05-22): enabling calibration on
/// Qwen3.6-27B/aeon-27b moved long-ctx accept 0.92→0.89 and short-ctx
/// 1.83→1.81 (i.e., slightly worse), and tok/s 10.05→9.60 at 4K ctx and
/// 29.77→29.44 at short ctx — the per-token `gpu.synchronize` inside
/// `Fp8KvCalibration::observe` adds latency, and Qwen3.6-27B's MTP K/V
/// magnitudes are well within the FP8 E4M3 [-448, 448] range at scale=1.0,
/// so calibration changes precision without removing clipping. Kept as an
/// opt-in (set `ATLAS_MTP_FP8_CALIB=1`) for models whose MTP K/V outputs
/// genuinely exceed ±448 (e.g., Gemma-4 26B, Mistral families). Cached via
/// `OnceLock`.
pub(crate) fn mtp_fp8_calib_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_MTP_FP8_CALIB")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Number of warmup tokens before MTP FP8 KV scales are frozen.
/// Mirrors the target attention layer's `MODEL.toml fp8_kv_calibration_tokens`
/// default (256) since the MTP head sees the same hidden-state distribution.
pub(crate) const MTP_FP8_WARMUP_TOKENS: usize = 256;

/// MTP head weight precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpQuantization {
    /// NVFP4 E2M1 (0.5 bytes/weight) — fastest MTP forward, lowest accuracy.
    Nvfp4,
    /// FP8 E4M3 (1 byte/weight) — balanced.
    Fp8,
    /// BF16 (2 bytes/weight) — highest accuracy, slowest MTP forward.
    Bf16,
}

impl std::str::FromStr for MtpQuantization {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "nvfp4" | "fp4" => Ok(Self::Nvfp4),
            "fp8" => Ok(Self::Fp8),
            "bf16" => Ok(Self::Bf16),
            _ => anyhow::bail!("Unknown MTP quantization: {s}. Expected: nvfp4, fp8, bf16"),
        }
    }
}

/// Weight storage that can hold any supported precision.
#[allow(dead_code)]
enum ProjectionWeight {
    Nvfp4(QuantizedWeight),
    Fp8(Fp8DenseWeight),
    /// FP8 E4M3 block-scaled from checkpoint (w8a16_gemv LUT kernel).
    /// Used when the checkpoint is FP8 native (native FP8 serving).
    Fp8BlockScaled(Fp8Weight),
    Bf16(DenseWeight),
}

/// Per-sequence MTP proposer state.
pub struct MtpProposerState {
    /// Block table for MTP's own KV cache.
    pub block_table: Vec<u32>,
    /// Current sequence length in MTP's KV cache.
    pub seq_len: usize,
    /// Number of drafts produced in the last propose() call.
    /// Used by after_verify to know how many entries to trim.
    pub last_num_drafted: usize,
}

impl ProposerState for MtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// MTP prediction head.
#[allow(dead_code)]
pub struct MtpHead {
    // Norms (always BF16)
    pre_fc_norm_embedding: DenseWeight,
    pre_fc_norm_hidden: DenseWeight,
    input_layernorm: DenseWeight,
    post_attn_layernorm: DenseWeight,
    norm: DenseWeight,

    // Projections (precision depends on MtpQuantization)
    fc: ProjectionWeight,
    q_proj: ProjectionWeight,
    k_proj: ProjectionWeight,
    v_proj: ProjectionWeight,
    o_proj: ProjectionWeight,

    // BF16 fallbacks for Q/K norms
    q_norm: DenseWeight,
    k_norm: DenseWeight,

    // MoE: NVFP4 uses fused MoeLayer; FP8/BF16 uses per-expert storage
    moe_nvfp4: Option<MoeLayer>,
    moe_experts_generic: Option<Vec<(ProjectionWeight, ProjectionWeight, ProjectionWeight)>>,
    moe_shared_generic: Option<(ProjectionWeight, ProjectionWeight, ProjectionWeight)>,
    moe_gate: DenseWeight,
    shared_expert_gate: DenseWeight,

    // Dense MLP path (Qwen3.5/3.6 27B-class). When `is_dense_mlp == true`,
    // the forward pass swaps the MoE block for a plain gate/up/down MLP
    // built from `MtpDenseWeights`. Quantized to NVFP4 at construction so
    // it shares the same w4a16_gemv kernel as the projections above.
    is_dense_mlp: bool,
    dense_mlp_gate: Option<QuantizedWeight>,
    dense_mlp_up: Option<QuantizedWeight>,
    dense_mlp_down: Option<QuantizedWeight>,
    dense_mlp_intermediate: usize,

    // Precision mode
    quant: MtpQuantization,

    /// Reduced vocab size for MTP LM head GEMV (0 = full vocab).
    mtp_vocab_size: u32,

    // Shared weights from target model
    embed_tokens: DenseWeight,
    lm_head_nvfp4: QuantizedWeight,

    // KV cache for MTP attention (1 layer, separate from target)
    kv_cache: Mutex<PagedKvCache>,
    attn_layer_idx: usize,

    /// Per-MTP-layer FP8 KV cache scale calibration.
    ///
    /// `Some` when `ATLAS_MTP_FP8_CALIB=1` (default ON) and the cache is FP8.
    /// Without calibration, MTP decode hardcodes `k_scale=v_scale=1.0`, which
    /// causes FP8 dequantization error to compound across positions at long
    /// context (4K+), dropping draft accept rate by ~50% (1.83 → 0.92
    /// tokens/verify). Mirrors `Qwen3AttentionLayer::fp8_calibration`.
    pub(super) fp8_calibration: Option<Fp8KvCalibration>,

    // Kernel handles (always needed)
    rms_norm_k: KernelHandle,
    rms_norm_residual_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemv_qg_k: KernelHandle,
    w4a16_gemv_dual_k: KernelHandle,
    rope_k: KernelHandle,
    reshape_cache_k: KernelHandle,
    paged_decode_k: KernelHandle,
    residual_add_k: KernelHandle,
    residual_add_rms_norm_k: KernelHandle,
    sigmoid_gate_mul_k: KernelHandle,
    bf16_concat_k: KernelHandle,
    argmax_k: KernelHandle,
    embed_from_argmax_k: KernelHandle,
    /// Fixed device buffer (4 bytes) for deferred draft token ID readback.
    draft_token_id_dev: DevicePtr,
    // BF16/FP8 kernel handles (None if NVFP4 mode)
    dense_gemv_k: Option<KernelHandle>,
    dense_gemv_fp8w_k: Option<KernelHandle>,
    w8a16_gemv_k: Option<KernelHandle>,
    deinterleave_qg_k: Option<KernelHandle>,
    moe_topk_k: Option<KernelHandle>,
    moe_silu_mul_k: Option<KernelHandle>,
    moe_weighted_sum_blend_k: Option<KernelHandle>,
}

impl MtpHead {
    /// Acquire the MTP KV cache mutex. Used by the multi-module
    /// dispatcher (`mtp_multi`) to reclaim blocks during free_state.
    /// `parking_lot::Mutex` does not poison, so this can never fail.
    pub(crate) fn kv_cache_lock(&self) -> parking_lot::MutexGuard<'_, PagedKvCache> {
        self.kv_cache.lock()
    }

    /// Returns the effective FP8 K/V scales for the MTP attention layer.
    ///
    /// During warmup or when calibration is disabled, returns the bootstrap
    /// scales from `Fp8KvCalibration::scales()` (currently 2.0/2.0 during
    /// warmup). After warmup, returns the calibrated per-layer scales.
    /// Without a calibration tracker, falls back to 1.0/1.0 (legacy behavior).
    pub(super) fn effective_mtp_fp8_scales(&self) -> (f32, f32) {
        match &self.fp8_calibration {
            Some(cal) => cal.scales(),
            None => (1.0, 1.0),
        }
    }

    /// Dispatch GEMV to the appropriate kernel based on weight precision.
    fn gemv(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        proj: &ProjectionWeight,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        match proj {
            ProjectionWeight::Nvfp4(w) => {
                ops::w4a16_gemv(gpu, self.w4a16_gemv_k, input, w, output, n, k, stream)
            }
            ProjectionWeight::Fp8(w) => ops::dense_gemv_fp8w(
                gpu,
                self.dense_gemv_fp8w_k.unwrap(),
                input,
                w,
                output,
                n,
                k,
                stream,
            ),
            ProjectionWeight::Fp8BlockScaled(w) => ops::w8a16_gemv(
                gpu,
                self.w8a16_gemv_k.unwrap(),
                input,
                w.weight,
                w.row_scale,
                output,
                n,
                k,
                stream,
            ),
            ProjectionWeight::Bf16(w) => ops::dense_gemv(
                gpu,
                self.dense_gemv_k.unwrap(),
                input,
                w,
                output,
                n,
                k,
                stream,
            ),
        }
    }

    /// Quantize a BF16 weight to the target precision.
    fn quantize_proj(
        bf16: &DenseWeight,
        n: usize,
        k: usize,
        quant: MtpQuantization,
        gpu: &dyn GpuBackend,
        absmax_k: KernelHandle,
        nvfp4_k: KernelHandle,
        fp8_k: KernelHandle,
        stream: u64,
    ) -> Result<ProjectionWeight> {
        match quant {
            MtpQuantization::Nvfp4 => Ok(ProjectionWeight::Nvfp4(quantize_to_nvfp4(
                bf16, n, k, gpu, absmax_k, nvfp4_k, stream,
            )?)),
            MtpQuantization::Fp8 => Ok(ProjectionWeight::Fp8(quantize_to_fp8(
                bf16, n, k, gpu, fp8_k, stream,
            )?)),
            MtpQuantization::Bf16 => Ok(ProjectionWeight::Bf16(*bf16)),
        }
    }
}

mod forward;
mod moe_forward;
mod new;

impl DraftProposer for MtpHead {
    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(MtpProposerState {
            block_table: Vec::new(),
            seq_len: 0,
            last_num_drafted: 0,
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;

        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_token = last_token;
        let mut current_hidden = target_hidden;

        for i in 0..num_drafts {
            // Only the LAST draft gets GPU-side embedding (it's the one
            // used in the next verify step).
            let embed_target = if i == num_drafts - 1 {
                draft_embed_target
            } else {
                None
            };
            // Grammar-masked drafting (num_drafts==1 path only for now).
            // For num_drafts > 1 we would need to speculatively advance the
            // matcher between drafts and roll back before returning; the
            // current scheduler only uses num_drafts==1, so we pass the same
            // mask for every i and warn loudly if K>1 + grammar combine.
            if grammar_bitmask.is_some() && i > 0 {
                tracing::warn!(
                    "MTP grammar-masked drafting called with num_drafts>1 (i={i}); \
                     mask held fixed across draft positions — acceptance may drop."
                );
            }
            let mask_for_draft = grammar_bitmask;
            let draft = self.forward_one(
                current_token,
                current_hidden,
                position + i,
                mtp_state,
                ctx,
                stream,
                embed_target,
                mask_for_draft,
            )?;
            tracing::debug!(
                "MTP propose[{i}]: token={current_token} pos={} mtp_seq_len={} → draft={draft}",
                position + i,
                mtp_state.seq_len,
            );
            drafts.push(draft);
            current_token = draft;
            // For subsequent drafts, use the MTP head's own hidden state
            current_hidden = ctx.buffers.hidden_states();
        }

        mtp_state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        self.read_deferred_draft_token(gpu)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;

        // Trim rejected drafts from MTP KV cache.
        // num_drafted was recorded in the last propose() call.
        // We trim `num_drafted - num_accepted` entries.
        // e.g. K=2: drafted 1, accepted 0 → trim 1. accepted 1 → trim 0.
        // e.g. K=3: drafted 2, accepted 0 → trim 2. accepted 1 → trim 1. accepted 2 → trim 0.
        let num_drafted = mtp_state.last_num_drafted.max(1);
        let num_to_trim = num_drafted.saturating_sub(num_accepted);
        let old_sl = mtp_state.seq_len;
        if num_to_trim > 0 {
            mtp_state.seq_len = mtp_state.seq_len.saturating_sub(num_to_trim);
        }
        tracing::debug!(
            "MTP after_verify: accepted={num_accepted} drafted={num_drafted} trim={num_to_trim} mtp_seq_len: {old_sl} → {}",
            mtp_state.seq_len,
        );
        Ok(())
    }

    fn free_state(&self, state: &mut dyn ProposerState) -> Result<()> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;
        if !mtp_state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&mtp_state.block_table);
            mtp_state.block_table.clear();
        }
        mtp_state.seq_len = 0;
        Ok(())
    }

    fn prefill_last_k(
        &self,
        tokens: &[u32],
        target_hiddens: DevicePtr,
        base_position: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mtp_state = state
            .as_any_mut()
            .downcast_mut::<MtpProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP proposer state"))?;
        self.prefill_last_k_impl(
            tokens,
            target_hiddens,
            base_position,
            mtp_state,
            ctx,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_proposer_state_downcast() {
        let state: Box<dyn ProposerState> = Box::new(MtpProposerState {
            block_table: vec![0, 1, 2],
            seq_len: 42,
            last_num_drafted: 0,
        });
        let mtp = state.as_any().downcast_ref::<MtpProposerState>().unwrap();
        assert_eq!(mtp.seq_len, 42);
        assert_eq!(mtp.block_table.len(), 3);
    }
}
