// SPDX-License-Identifier: AGPL-3.0-only
//! Complete attention-prefill validation before the first layer effect.
use super::super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, BatchedAttnMetadata, ForwardContext};
use crate::layers::{FfnComponent, Qwen4HyperConnection, ops, qwen4_prefill_moe};
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::KvCacheDtype;

pub(super) struct PrefillMoeSetup<'a> {
    pub(super) attn_hyper: &'a Qwen4HyperConnection,
    pub(super) mlp_hyper: &'a Qwen4HyperConnection,
    pub(super) metadata: AttnMetadataDev,
    pub(super) row_bytes: usize,
    pub(super) core_bytes: usize,
    pub(super) attn_width: usize,
    pub(super) attn_row_bytes: usize,
    pub(super) eps: f32,
    pub(super) exact_hyper: bool,
    pub(super) exact_qkv16: bool,
    pub(super) exact_o16: bool,
    pub(super) exact_attn16: bool,
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_prefill_moe_setup<'a>(
        &'a self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        seq_len_start: usize,
        block_table: &[u32],
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<PrefillMoeSetup<'a>> {
        qwen4_prefill_moe::validate(ctx, num_tokens, seq_len_start)?;
        qwen4_prefill_moe::validate_ffn(&self.ffn)?;
        ensure!(batched_meta.is_none(), "Qwen4 MoE-only prefill requires C1");
        let attn_hyper = self.qwen4_attn_hyper.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Qwen4 MoE-only prefill requires attention hyperconnection")
        })?;
        let mlp_hyper = self.qwen4_mlp_hyper.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Qwen4 MoE-only prefill requires MLP hyperconnection")
        })?;
        ensure!(
            attn_hyper.inject.as_ref().is_some_and(|w| !w.is_null())
                && mlp_hyper.inject.as_ref().is_some_and(|w| !w.is_null()),
            "Qwen4 MoE-only prefill requires both injection weights"
        );
        ensure!(
            matches!(&self.ffn, FfnComponent::Moe(_)),
            "Qwen4 MoE-only prefill requires a routed FFN"
        );
        let metadata = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("Qwen4 MoE-only prefill requires attention metadata"))?;
        ensure!(
            metadata.num_seqs == 1
                && metadata.max_blocks_per_seq as usize == block_table.len()
                && !block_table.is_empty(),
            "Qwen4 MoE-only prefill requires one complete block table"
        );
        ensure!(
            [
                metadata.positions,
                metadata.positions_h,
                metadata.positions_w,
                metadata.slot,
                metadata.block_table,
                metadata.seq_len,
            ]
            .iter()
            .all(|p| *p != DevicePtr::NULL),
            "Qwen4 MoE-only prefill requires nonnull paged metadata"
        );
        ensure!(
            hidden != DevicePtr::NULL && residual != DevicePtr::NULL && hidden != residual,
            "Qwen4 MoE-only prefill requires disjoint hidden/residual rows"
        );
        let width = ctx.config.residual_width();
        ensure!(
            attn_hyper.residual_width() == width && mlp_hyper.residual_width() == width,
            "Qwen4 MoE-only prefill hyperconnection width mismatch"
        );
        let row_bytes = width
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("row overflow"))?;
        let bytes = num_tokens
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("rows overflow"))?;
        ensure!(
            bytes <= ctx.buffers.sizes().hidden_states && bytes <= ctx.buffers.sizes().residual,
            "Qwen4 MoE-only prefill rows exceed hidden/residual arenas"
        );
        let eps = ctx.config.rms_norm_eps as f32;
        let exact_hyper = qwen4_prefill_moe::hyper_selected()?;
        let exact_qkv16 = qwen4_prefill_moe::exact_qkv16_selected()?;
        let exact_o16 = qwen4_prefill_moe::exact_o16_selected()?;
        let exact_attn16 = qwen4_prefill_moe::attn16::selected()?;
        let exact_attn32 = qwen4_prefill_moe::attn16::core32_selected()?;
        let admitted_rows = if exact_attn32 { 32 } else { 16 };
        let core_bytes = ctx
            .config
            .hidden_size
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("core row overflow"))?;
        let attn_width = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads)
            .checked_mul(self.head_dim_override.unwrap_or(ctx.config.head_dim))
            .ok_or_else(|| anyhow::anyhow!("attention row overflow"))?;
        let attn_row_bytes = attn_width
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("attention row bytes overflow"))?;
        if exact_qkv16 {
            ensure!(
                exact_hyper && seq_len_start == 0 && self.kv_dtype == KvCacheDtype::Bf16,
                "{} requires exact HC, BF16 KV, and initial-window prefill",
                qwen4_prefill_moe::EXACT_QKV16_SELECTOR
            );
            ensure!(
                self.gated
                    && self.attn.q_norm_full.is_none()
                    && self.attn.k_norm_full.is_none()
                    && self.v_norm_weight.is_none(),
                "{} requires canonical gated Qwen4 per-head norms",
                qwen4_prefill_moe::EXACT_QKV16_SELECTOR
            );
            for (name, weight) in [
                ("q", self.q_weight.as_ref().and_then(|w| w.as_nvfp4())),
                ("k", self.k_weight.as_ref().and_then(|w| w.as_nvfp4())),
                ("v", self.v_weight.as_ref().and_then(|w| w.as_nvfp4())),
            ] {
                ensure!(
                    weight.is_some_and(|weight| !weight.is_null()),
                    "{} requires ordinary {name} NVFP4 weights",
                    qwen4_prefill_moe::EXACT_QKV16_SELECTOR
                );
            }
            ensure!(
                self.w4a16_exact_qkv_kernels.qg_for_rows(16).0 != 0
                    && self.w4a16_exact_qkv_kernels.dual_kv_for_rows(16).0 != 0,
                "{} requires both exact K16 projection kernels",
                qwen4_prefill_moe::EXACT_QKV16_SELECTOR
            );
            ensure!(
                admitted_rows * core_bytes <= ctx.buffers.sizes().norm_output
                    && admitted_rows * 13_312 * 2 <= ctx.buffers.sizes().qkv_output
                    && (!exact_o16
                        || admitted_rows * attn_row_bytes <= ctx.buffers.sizes().ssm_qkvz),
                "{} scratch is undersized",
                qwen4_prefill_moe::EXACT_QKV16_SELECTOR
            );
            ensure!(
                !metadata.qwen4_qsa_required || self.qwen4_qsa.is_some(),
                "{} requires the QSA side-cache indexer",
                qwen4_prefill_moe::EXACT_QKV16_SELECTOR
            );
        }
        if exact_o16 {
            ensure!(
                self.o_weight.as_ref().and_then(|w| w.as_fp8()).is_none()
                    && self.o_dense_bf16.is_none()
                    && !self.attn.o_proj.is_null(),
                "{} requires an ordinary NVFP4 O projection",
                qwen4_prefill_moe::EXACT_O16_SELECTOR
            );
            ensure!(
                matches!(
                    self.w4a16_exact_o_proj_kernels
                        .route_for_rows(admitted_rows as u32),
                    Some(ops::ExactLmHeadRoute::Exact(_))
                ),
                "{} requires the exact M16 O-projection tier",
                qwen4_prefill_moe::EXACT_O16_SELECTOR
            );
            ensure!(
                !ops::w4a16_gemv_rt2_enabled()
                    || self
                        .w4a16_exact_o_proj_kernels
                        .rt2_for_tier(if exact_attn32 {
                            ops::ExactLmHeadTier::M32
                        } else {
                            ops::ExactLmHeadTier::M17
                        })
                        .0
                        != 0,
                "{} selected RT2 O-projection tier is unavailable",
                qwen4_prefill_moe::EXACT_O16_SELECTOR
            );
        }
        if exact_attn16 {
            ensure!(
                exact_hyper && exact_qkv16 && exact_o16 && seq_len_start == 0,
                "{} requires exact HC/QKV16/O16 initial prefill",
                qwen4_prefill_moe::attn16::SELECTOR
            );
            if exact_attn32 {
                qwen4_prefill_moe::attn16::plan_for_tile(
                    num_tokens,
                    seq_len_start,
                    block_table.len(),
                    ctx.buffers.sizes().attn_output,
                    admitted_rows,
                )?;
            } else {
                qwen4_prefill_moe::attn16::plan(
                    num_tokens,
                    seq_len_start,
                    block_table.len(),
                    ctx.buffers.sizes().attn_output,
                )?;
            }
            self.qwen4_qsa
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("attention16 requires QSA"))?
                .validate_exact_group4()?;
        }
        if exact_hyper {
            attn_hyper.validate_prefill_exact(hidden, residual, num_tokens, ctx.buffers, eps)?;
            mlp_hyper.validate_prefill_exact(hidden, residual, num_tokens, ctx.buffers, eps)?;
            attn_hyper.prepare_prefill_exact(
                hidden,
                residual,
                num_tokens,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?;
        }
        Ok(PrefillMoeSetup {
            attn_hyper,
            mlp_hyper,
            metadata,
            row_bytes,
            core_bytes,
            attn_width,
            attn_row_bytes,
            eps,
            exact_hyper,
            exact_qkv16,
            exact_o16,
            exact_attn16,
        })
    }
}
