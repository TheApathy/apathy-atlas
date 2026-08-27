// SPDX-License-Identifier: AGPL-3.0-only

//! Exact K1-order Qwen4 K=5 attention projection staging.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Projects five contiguous Qwen4 attention inputs into a strided
    /// `[Q|gate|K|V]` arena. The projection kernels preserve the ordinary K1
    /// reduction order; Q/K normalization remains the same per-head K1 call.
    ///
    /// Returns `None` unless the explicit Qwen4 K5 flag is enabled. A requested
    /// route fails closed when the layer geometry or exact kernels do not match.
    pub(in super::super) fn qwen4_k5_project_qkv_exact(
        &self,
        normed: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, usize)>> {
        if rows != 5
            || std::env::var("ATLAS_QWEN4_K5_BATCH_ATTN_QKV")
                .ok()
                .as_deref()
                != Some("1")
        {
            return Ok(None);
        }

        ensure!(self.gated, "Qwen4 exact K5 QKV requires gated attention");
        ensure!(
            self.attn.q_norm_full.is_none()
                && self.attn.k_norm_full.is_none()
                && self.v_norm_weight.is_none(),
            "Qwen4 exact K5 QKV requires per-head Q/K norms and no V norm"
        );
        let q_weight = self
            .q_weight
            .as_ref()
            .and_then(|weight| weight.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact K5 Q requires ordinary NVFP4"))?;
        let k_weight = self
            .k_weight
            .as_ref()
            .and_then(|weight| weight.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact K5 K requires ordinary NVFP4"))?;
        let v_weight = self
            .v_weight
            .as_ref()
            .and_then(|weight| weight.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact K5 V requires ordinary NVFP4"))?;

        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let q_dim = nq * hd;
        let q_proj_dim = q_dim * 2;
        let kv_dim = nkv * hd;
        let q_proj_bytes = q_proj_dim as usize * 2;
        let kv_bytes = kv_dim as usize * 2;
        let row_bytes = q_proj_bytes + 2 * kv_bytes;
        let qkv = ctx.buffers.qkv_output();
        ensure!(
            rows * row_bytes <= ctx.buffers.sizes().qkv_output,
            "Qwen4 exact K5 QKV staging exceeds QKV arena"
        );
        let row_stride_bf16 = (row_bytes / 2) as u32;
        let qg_kernel = self.w4a16_exact_qkv_kernels.qg_for_rows(rows);
        let dual_kv_kernel = self.w4a16_exact_qkv_kernels.dual_kv_for_rows(rows);
        ensure!(
            qg_kernel.0 != 0 && dual_kv_kernel.0 != 0,
            "Qwen4 exact K5 QKV kernels are unavailable"
        );

        ops::w4a16_gemv_qg_exact(
            ctx.gpu,
            qg_kernel,
            normed,
            q_weight,
            qkv,
            rows as u32,
            q_proj_dim,
            h,
            nq,
            hd,
            row_stride_bf16,
            stream,
        )?;
        let k_base = qkv.offset(q_proj_bytes);
        let v_base = k_base.offset(kv_bytes);
        ops::w4a16_gemv_dual_kv_exact(
            ctx.gpu,
            dual_kv_kernel,
            normed,
            k_weight,
            k_base,
            v_weight,
            v_base,
            rows as u32,
            kv_dim,
            h,
            row_stride_bf16,
            stream,
        )?;

        let eps = ctx.config.rms_norm_eps as f32;
        for row in 0..rows {
            let q_out = qkv.offset(row * row_bytes);
            let k_out = q_out.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    q_out,
                    &self.attn.q_norm,
                    q_out,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    k_out,
                    &self.attn.k_norm,
                    k_out,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(Some((qkv, row_bytes)))
    }
}
