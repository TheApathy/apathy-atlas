// SPDX-License-Identifier: AGPL-3.0-only
//! Default-off full-row numerical attention prefill (`qwen4_fast_proj`, family
//! `attn`): QG/K/V projections for every row in three dequant+cuBLASLt GEMMs
//! (gated-Q permutation folded into the dequant), one strided Q/K RMS-norm
//! launch, the unchanged exact 16/32-row attention core per tile (QSA, MRoPE,
//! KV write, paged attention, sigmoid gate), then one O GEMM over every row
//! and one batched HC injection.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::prefill_moe_attn16_device::DeviceKernels;
use crate::layer::ForwardContext;
use crate::layers::{Qwen4HyperConnection, ops, qwen4_fast_proj, qwen4_flash_attn, qwen4_prefill_moe};

impl Qwen3AttentionLayer {
    /// Projects all `rows` packed inputs into `qkv_output` rows of
    /// `[q (nq*hd) | gate (nq*hd) | k (nkv*hd) | v (nkv*hd)]` with Q/K norms applied.
    fn qwen4_fast_project_qkv_all(
        &self,
        normed: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, usize)> {
        ensure!(self.gated, "fast attention QKV requires gated attention");
        ensure!(
            self.attn.q_norm_full.is_none() && self.attn.k_norm_full.is_none() && self.v_norm_weight.is_none(),
            "fast attention QKV requires per-head Q/K norms and no V norm"
        );
        let q_weight = self.q_weight.as_ref().and_then(|w| w.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("fast attention Q requires ordinary NVFP4"))?;
        let k_weight = self.k_weight.as_ref().and_then(|w| w.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("fast attention K requires ordinary NVFP4"))?;
        let v_weight = self.v_weight.as_ref().and_then(|w| w.as_nvfp4())
            .ok_or_else(|| anyhow::anyhow!("fast attention V requires ordinary NVFP4"))?;
        let h = ctx.config.hidden_size;
        let nq = self.num_q_heads_override.unwrap_or(ctx.config.num_attention_heads);
        let nkv = self.num_kv_heads_override.unwrap_or(ctx.config.num_key_value_heads);
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim);
        let q_dim = nq * hd;
        let q_proj_dim = q_dim * 2;
        let kv_dim = nkv * hd;
        let row_stride = q_proj_dim + 2 * kv_dim;
        let row_bytes = row_stride * 2;
        let qkv = ctx.buffers.qkv_output();
        ensure!(
            rows * row_bytes <= ctx.buffers.sizes().qkv_output,
            "fast attention QKV staging exceeds QKV arena"
        );
        let gpu = ctx.gpu;
        qwen4_fast_proj::dequant_gemm_ex(
            gpu, normed, q_weight, qkv, rows, q_proj_dim, h, row_stride, Some((nq, hd)), stream,
        )?;
        qwen4_fast_proj::dequant_gemm_ex(
            gpu, normed, k_weight, qkv.offset(q_proj_dim * 2), rows, kv_dim, h, row_stride, None, stream,
        )?;
        qwen4_fast_proj::dequant_gemm_ex(
            gpu, normed, v_weight, qkv.offset((q_proj_dim + kv_dim) * 2), rows, kv_dim, h, row_stride, None, stream,
        )?;
        let eps = ctx.config.rms_norm_eps as f32;
        if !self.attn.q_norm.weight.is_null() {
            qwen4_fast_proj::qk_norm_strided(
                gpu, qkv, self.attn.q_norm.weight, rows, nq, hd, row_stride, eps, stream,
            )?;
        }
        if !self.attn.k_norm.weight.is_null() {
            qwen4_fast_proj::qk_norm_strided(
                gpu, qkv.offset(q_proj_dim * 2), self.attn.k_norm.weight, rows, nkv, hd, row_stride, eps, stream,
            )?;
        }
        Ok((qkv, row_bytes))
    }

    /// Full-row attention sublayer; the caller has already run HC preparation
    /// (packed inputs for every row live in `packed_inputs` = norm_output).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_moe_only_fast_attn(
        &self,
        attn_hyper: &Qwen4HyperConnection,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        packed_inputs: DevicePtr,
        raw_outputs: DevicePtr,
        core_bytes: usize,
        attn_width: usize,
        attn_row_bytes: usize,
        block_table: &mut Vec<u32>,
        kv_cache: &mut PagedKvCache,
        attn16_device: Option<DeviceKernels>,
        attn32: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            num_tokens.is_multiple_of(16) && num_tokens <= 2048,
            "fast attention requires a 16-row multiple prefill (got {num_tokens})"
        );
        ensure!(
            num_tokens * attn_row_bytes <= ctx.buffers.sizes().ssm_qkvz
                && num_tokens * core_bytes <= ctx.buffers.sizes().norm_output,
            "fast attention full-row scratch is undersized"
        );
        let (qkv, qkv_row_bytes) =
            self.qwen4_fast_project_qkv_all(packed_inputs, num_tokens, ctx, stream)?;
        // Flash core: the per-tile loop still runs QSA, MRoPE and the KV-cache
        // write (every row's K/V must be resident before any row attends), but
        // the attention core and gate are deferred to one full-prefill launch.
        let flash = qwen4_flash_attn::selected() && self.prefill_attn_k.0 != 0;
        let mut tile_start = 0;
        if flash
            && let Some(device) = attn16_device
            && super::prefill_moe_attn16::fullrow_selected()
        {
            self.prefill_moe_attn_fullrow_prep(
                packed_inputs,
                qkv,
                qkv_row_bytes,
                num_tokens,
                kv_cache,
                ctx,
                stream,
                device,
            )?;
            tile_start = num_tokens;
        }
        while tile_start < num_tokens {
            let tile_rows = if attn32 && num_tokens - tile_start >= 32 { 32 } else { 16 };
            self.prefill_moe_attn16(
                packed_inputs.offset(tile_start * core_bytes),
                qkv.offset(tile_start * qkv_row_bytes),
                qkv_row_bytes,
                raw_outputs.offset(tile_start * attn_row_bytes),
                tile_start,
                tile_rows,
                block_table,
                kv_cache,
                ctx,
                stream,
                attn16_device,
                flash,
            )?;
            tile_start += tile_rows;
        }
        if flash {
            let nq = ctx.config.num_attention_heads;
            let nkv = ctx.config.num_key_value_heads;
            let hd = ctx.config.head_dim;
            let q_dim = nq * hd;
            let row_stride = qkv_row_bytes / 2;
            qwen4_flash_attn::run(
                ctx.gpu,
                self.prefill_attn_k,
                qkv,
                row_stride,
                q_dim * 2,
                q_dim * 2 + nkv * hd,
                raw_outputs,
                attn_row_bytes / 2,
                num_tokens,
                nq,
                nkv,
                hd,
                self.effective_attn_scale(hd as u32),
                stream,
            )?;
            ops::sigmoid_gate_mul_batched(
                ctx.gpu,
                self.sigmoid_gate_mul_batched_k,
                raw_outputs,
                qkv.offset(q_dim * 2),
                raw_outputs,
                q_dim as u32,
                row_stride as u32,
                num_tokens as u32,
                stream,
            )?;
        }
        // O projection for every row lands in norm_output (the packed inputs
        // are no longer needed), then one batched saved-scale injection.
        qwen4_fast_proj::dequant_gemm(
            ctx.gpu,
            raw_outputs,
            &self.attn.o_proj,
            packed_inputs,
            num_tokens,
            ctx.config.hidden_size,
            attn_width,
            stream,
        )?;
        attn_hyper.inject_saved_batched(hidden, packed_inputs, residual, num_tokens, ctx.gpu, stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "ATTN_PREFILL_FAST_ENGAGED selector={} rows={num_tokens} projection=dequant_cublaslt_bf16_non_bit_exact",
                qwen4_fast_proj::SELECTOR
            );
        });
        let _ = qwen4_prefill_moe::attn16::SELECTOR;
        Ok(())
    }
}
