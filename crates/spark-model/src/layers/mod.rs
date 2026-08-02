// SPDX-License-Identifier: AGPL-3.0-only

pub mod deepseek_v4_mtp;
pub mod dense_ffn;
pub mod dflash_head;
pub mod dspark_head;
pub mod ep_dispatch;
pub mod fp8_calibration;
pub mod moe;
pub mod mtp_head;
pub mod mtp_multi;
pub mod nemotron_mamba2;
pub mod nemotron_moe;
pub mod ops;
pub mod qwen3_attention;
pub mod qwen3_ssm;
pub mod vision_encoder;

pub use deepseek_v4_mtp::{DeepseekV4MtpHead, DeepseekV4MtpProposerState};
pub use dense_ffn::{DenseFfnLayer, DenseFfnWeights, FfnActivation};
pub use dflash_head::{
    BlockDiffusionDraftHead, DDTreePayload, DflashLayer, DflashProposerState, DflashQuantization,
};
pub use moe::MoeLayer;
pub use mtp_head::{MtpHead, MtpQuantization, mtp_drafter_prefill_enabled};
pub use nemotron_mamba2::NemotronMamba2Layer;
pub use nemotron_moe::NemotronMoeLayer;
pub use qwen3_attention::Qwen3AttentionLayer;
pub use qwen3_ssm::Qwen3SsmLayer;
pub use vision_encoder::{MergerLayer, ViTBlock, VisionEncoder};

use crate::layer::ForwardContext;
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Try to load an optional kernel, logging at debug level if it's not found.
/// Returns `KernelHandle(0)` (null) on failure — callers must check before use.
///
/// Debug (not warn) because misses are expected when a model doesn't use a
/// given feature: e.g. Qwen3-Coder-Next (GDN+attention) never calls MLA
/// kernels, but the layer builder still probes them. Warning on expected
/// misses drowned out genuine problems in startup logs.
pub fn try_kernel(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    match gpu.kernel(module, func) {
        Ok(h) => h,
        Err(_) => {
            tracing::debug!("Optional kernel '{module}::{func}' not loaded");
            KernelHandle(0)
        }
    }
}

/// FFN component: MoE (expert routing), dense SwiGLU, or None (standalone attention).
#[allow(clippy::large_enum_variant)]
pub enum FfnComponent {
    Moe(MoeLayer),
    Dense(DenseFfnLayer),
    /// No FFN — used by Nemotron-H standalone attention layers.
    None,
}

impl FfnComponent {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// True for a plain dense (SwiGLU) FFN. Wide-batch verify paths gate their
    /// `forward_prefill` fast path on this: batching reads dense weights once
    /// (big win at N=17), but on a 256-expert MoE the grouped-GEMM is a net
    /// loss at small batch (per-expert M~1 + sort/permute overhead), so MoE
    /// keeps its per-token loop.
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// ATLAS_FP32_ROUTING active for this FFN (MoE only; false otherwise).
    pub fn fp32_routing_active(&self) -> bool {
        match self {
            Self::Moe(m) => m.fp32_routing_active(),
            _ => false,
        }
    }

    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match self {
            Self::Moe(m) => m.forward(input, ctx, stream),
            Self::Dense(d) => d.forward(input, ctx, stream),
            Self::None => Ok(input),
        }
    }

    /// [`Self::forward`] with an optional fused residual tail
    /// (ATLAS_FUSED_ELEMWISE=1, serial M=1 decode). Returns `Ok((out, true))`
    /// iff `hidden += out` was folded into the final MoE blend launch (see
    /// `MoeLayer::forward_residual`); on `Ok((out, false))` the caller adds
    /// the residual itself — identical contract to `forward`. Dense/None
    /// FFNs never fuse.
    pub fn forward_residual(
        &self,
        input: DevicePtr,
        residual: Option<DevicePtr>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, bool)> {
        match self {
            Self::Moe(m) => m.forward_residual(input, residual, ctx, stream),
            Self::Dense(d) => Ok((d.forward(input, ctx, stream)?, false)),
            Self::None => Ok((input, false)),
        }
    }

    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k2(input, ctx, stream),
            Self::Dense(d) => d.forward_k2(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    /// True when [`Self::forward_k2`] would batch the two verify rows through a
    /// path that is actually faster than two single-row passes.
    ///
    /// Dense FFNs read one weight set regardless of row count, so batching is
    /// unconditionally right for them. MoE has to check — see
    /// `MoeLayer::k2_verify_ffn_is_batched`.
    pub fn k2_verify_ffn_is_batched(&self, ctx: &ForwardContext) -> bool {
        match self {
            Self::Moe(m) => m.k2_verify_ffn_is_batched(ctx),
            Self::Dense(_) => true,
            Self::None => false,
        }
    }

    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k3(input, ctx, stream),
            Self::Dense(d) => d.forward_k3(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    /// Wide DFlash verify (K=γ, e.g. 16 tokens) batched MoE — the forward_k16
    /// fix. MoE: faithful batchN_t path (falls back to per-token internally if
    /// unavailable). Dense: reuse the already-batched forward_prefill.
    pub fn forward_kn(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_kn(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_prefill(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    /// [`Self::forward_kn`] with an optional fused residual tail
    /// (ATLAS_FUSED_ELEMWISE=1). Returns `Ok(true)` iff `hidden += moe_out`
    /// was folded into the final blend launch (MoE fast batchN path only);
    /// on `Ok(false)` the output is at `moe_output()` and the caller adds
    /// the residual itself — identical contract to `forward_kn`.
    pub fn forward_kn_residual(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        residual: Option<DevicePtr>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Moe(m) => m.forward_kn_residual(input, num_tokens, residual, ctx, stream),
            Self::Dense(d) => {
                d.forward_prefill(input, num_tokens, ctx, stream)?;
                Ok(false)
            }
            Self::None => {
                let _ = (input, num_tokens);
                Ok(false)
            }
        }
    }

    /// K=4 verify FFN via batched GEMV (dense only). Returns `false` when the
    /// path is unavailable (MoE / missing batch4 kernel / non-NVFP4 weights)
    /// so the caller can fall back to `forward_prefill`.
    pub fn try_forward_k4(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Dense(d) if d.can_forward_k4() => {
                d.forward_k4(input, ctx, stream)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_prefill(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_prefill(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_batched(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_token_major_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_token_major_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_atomic_c4_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_atomic_c4_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }
}
