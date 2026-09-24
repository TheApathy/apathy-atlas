// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode (single-token).

use super::qwen4_k5_ssm::{
    Qwen4ExactSsmRowsRoute, qwen4_exact_ssm_rows_route, qwen4_k16_batched_verify_requested,
    qwen4_k16_exact_requested,
};
use super::*;

static QWEN4_K16_SSM_EXACT_ENGAGED: std::sync::Once = std::sync::Once::new();

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_batched_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        h_intermediate_stride: usize,
        conv_intermediate_stride: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("Qwen4 batched verify requested without hyperconnections"),
        };
        let h = ctx.config.hidden_size;
        let row_bytes = ctx.config.residual_width() * 2;
        let core_bytes = h * 2;
        let eps = ctx.config.rms_norm_eps as f32;
        let hybrid_k5 = matches!(num_tokens, 5 | 9)
            && std::env::var("ATLAS_QWEN4_K5_HYBRID").ok().as_deref() == Some("1");
        let exact_hyper_k5 =
            hybrid_k5 && std::env::var("ATLAS_QWEN4_K5_BATCH_HYPER").ok().as_deref() != Some("1");
        let legacy_batch_ssm =
            hybrid_k5 && std::env::var("ATLAS_QWEN4_K5_BATCH_SSM").ok().as_deref() == Some("1");
        let k16_requested = qwen4_k16_batched_verify_requested()?;
        let k16_exact_requested = qwen4_k16_exact_requested()?;
        anyhow::ensure!(
            h_intermediate.is_null() == conv_intermediate.is_null(),
            "Qwen4 batched SSM requires both recurrent intermediate pointers or neither"
        );
        let has_recurrent_intermediates = !h_intermediate.is_null();
        if has_recurrent_intermediates {
            anyhow::ensure!(
                h_intermediate_stride == self.h_state_bytes
                    && conv_intermediate_stride == self.conv_state_bytes,
                "Qwen4 batched SSM recurrent intermediate stride mismatch"
            );
        }
        let exact_ssm_route = qwen4_exact_ssm_rows_route(
            num_tokens,
            hybrid_k5,
            legacy_batch_ssm,
            k16_requested,
            k16_exact_requested,
            has_recurrent_intermediates,
        );
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
        if let Some(route) = exact_ssm_route {
            self.preflight_qwen4_exact_ssm_rows(
                route,
                num_tokens,
                ssm_state,
                h_intermediate,
                conv_intermediate,
                h_intermediate_stride,
                conv_intermediate_stride,
                ctx,
            )?;
        }

        // Batch the four-stream projection weights, then retain exact causal
        // ordering in the recurrent GDN core. Native K16 forces the ordered
        // sequence conv/GDN kernels and writes every post-row checkpoint.
        // The attention/recurrent projection stacks may consume split-K
        // workspace internally. MoE gate logits are idle until the later FFN.
        let attn_inputs = ctx.buffers.gate_logits();
        anyhow::ensure!(
            num_tokens * core_bytes <= ctx.buffers.sizes().gate_logits,
            "Qwen4 batched recurrent input staging exceeds gate-logit workspace"
        );
        if exact_hyper_k5 {
            for row in 0..num_tokens {
                let (mixed, _) = attn_hyper.prepare_decode(
                    hidden.offset(row * row_bytes),
                    residual.offset(row * row_bytes),
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    mixed,
                    attn_inputs.offset(row * core_bytes),
                    core_bytes,
                    stream,
                )?;
            }
        } else {
            let mixed_attn = if num_tokens <= 32 {
                attn_hyper.prepare_batched(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            } else {
                attn_hyper.prepare_prefill_exact(
                    hidden,
                    residual,
                    num_tokens,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            };
            ctx.gpu
                .copy_d2d_async(mixed_attn, attn_inputs, num_tokens * core_bytes, stream)?;
        }
        if let Some(route) = exact_ssm_route {
            let ssm_out = self.ssm_forward_qwen4_exact_rows(
                route,
                attn_inputs,
                num_tokens,
                ssm_state,
                h_intermediate,
                conv_intermediate,
                h_intermediate_stride,
                conv_intermediate_stride,
                ctx,
                stream,
            )?;
            if route == Qwen4ExactSsmRowsRoute::NativeK16 {
                crate::model::k16_route_receipt::mark_ssm_layer()?;
                QWEN4_K16_SSM_EXACT_ENGAGED.call_once(|| {
                    tracing::info!(
                        "ENGAGED ATLAS_QWEN4_K16_EXACT: ssm_exact_m16_projection_conv_gdn_sequence_enqueued"
                    );
                });
            }
            attn_hyper
                .inject_saved_batched(hidden, ssm_out, residual, num_tokens, ctx.gpu, stream)?;
        } else {
            // The per-row route writes every intermediate; a lazy commit for
            // this width would replay a retain buffer this path never fills.
            anyhow::ensure!(
                !(has_recurrent_intermediates && self.gdn_seq_lazy_engaged_inner(num_tokens)),
                "ATLAS_SSM_GDN_LAZY is not supported on the Qwen4 per-row verify route (K={num_tokens})"
            );
            for row in 0..num_tokens {
                let hidden_row = hidden.offset(row * row_bytes);
                let residual_row = residual.offset(row * row_bytes);
                let ssm_out = self.ssm_forward(
                    attn_inputs.offset(row * core_bytes),
                    ssm_state,
                    ctx,
                    stream,
                    false,
                )?;
                attn_hyper.inject_decode(
                    hidden_row,
                    ssm_out,
                    attn_hyper.saved_inject(residual_row),
                    ctx.gpu,
                    stream,
                )?;
                if !h_intermediate.is_null() {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.h_state,
                        h_intermediate.offset(row * h_intermediate_stride),
                        h_intermediate_stride,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        conv_intermediate.offset(row * conv_intermediate_stride),
                        conv_intermediate_stride,
                        stream,
                    )?;
                }
            }
        }

        let ffn_inputs = if exact_hyper_k5 {
            let staging = ctx
                .buffers
                .qkv_output()
                .offset(ctx.config.residual_width() * 2);
            anyhow::ensure!(
                ctx.config.residual_width() * 2 + num_tokens * core_bytes
                    <= ctx.buffers.sizes().qkv_output,
                "Qwen4 K=5 exact MLP staging exceeds QKV arena"
            );
            for row in 0..num_tokens {
                let (mixed, _) = mlp_hyper.prepare_decode(
                    hidden.offset(row * row_bytes),
                    residual.offset(row * row_bytes),
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    mixed,
                    staging.offset(row * core_bytes),
                    core_bytes,
                    stream,
                )?;
            }
            staging
        } else if num_tokens <= 32 {
            mlp_hyper.prepare_batched(
                hidden,
                residual,
                num_tokens,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?
        } else {
            mlp_hyper.prepare_prefill_exact(
                hidden,
                residual,
                num_tokens,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?
        };
        match num_tokens {
            2 => self.ffn.forward_k2(ffn_inputs, ctx, stream)?,
            3 => self.ffn.forward_k3(ffn_inputs, ctx, stream)?,
            5 | 9 if hybrid_k5 => self
                .ffn
                .forward_qwen4_exact_rows(ffn_inputs, num_tokens, ctx, stream)?,
            n => self.ffn.forward_prefill(ffn_inputs, n, ctx, stream)?,
        }
        let ffn_output = ctx.buffers.moe_output();
        if hybrid_k5 {
            mlp_hyper
                .inject_saved_batched(hidden, ffn_output, residual, num_tokens, ctx.gpu, stream)?;
        } else {
            for row in 0..num_tokens {
                let hidden_row = hidden.offset(row * row_bytes);
                let residual_row = residual.offset(row * row_bytes);
                mlp_hyper.inject_decode(
                    hidden_row,
                    ffn_output.offset(row * core_bytes),
                    mlp_hyper.saved_inject(residual_row),
                    ctx.gpu,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn decode_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let debug = tracing::enabled!(tracing::Level::DEBUG);
        // Stage-level synchronization is intentionally opt-in: it pinpoints
        // the producer of an asynchronous CUDA fault without imposing a
        // permanent decode synchronization tax.
        let trace = std::env::var("ATLAS_SSM_TRACE").ok().as_deref() == Some("1");

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        if let (Some(attn_hyper), Some(mlp_hyper)) = (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper)
        {
            let (mixed, inject) =
                attn_hyper.prepare_decode(hidden, residual, ctx.buffers, ctx.gpu, eps, stream)?;
            let ssm_out = self.ssm_forward(mixed, ssm_state, ctx, stream, trace)?;
            attn_hyper.inject_decode(
                hidden,
                ssm_out,
                inject.expect("Qwen4 decoder mixer has injection weights"),
                ctx.gpu,
                stream,
            )?;
            let (mixed, inject) =
                mlp_hyper.prepare_decode(hidden, residual, ctx.buffers, ctx.gpu, eps, stream)?;
            let moe_out = self.ffn.forward(mixed, ctx, stream)?;
            mlp_hyper.inject_decode(
                hidden,
                moe_out,
                inject.expect("Qwen4 decoder mixer has injection weights"),
                ctx.gpu,
                stream,
            )?;
            return Ok(());
        }

        let normed = ctx.buffers.norm_output();
        // Fused path requires: env opt-in, kernel handle present, sequential QKVZ +
        // NVFP4 weights (the only QKVZ branch that calls plain w4a16_gemv right after
        // rms_norm_residual). Falls back to the unfused sequence otherwise.
        let fuse_qkvz = self.fused_rms_qkvz_k.0 != 0
            && self.sequential_qkvz
            && self.qkvz_nvfp4.is_some()
            && !ctx.config.use_fp32_residual()
            && std::env::var("ATLAS_FUSE_SSM_QKVZ").ok().as_deref() == Some("1");

        if !fuse_qkvz {
            ops::rms_norm_residual(
                ctx.gpu,
                self.rms_norm_residual_k,
                hidden,
                &self.input_norm,
                normed,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "pre-norm", normed, 4);
        }

        let ssm_out = if fuse_qkvz {
            self.ssm_forward_with_fuse(
                normed,
                Some((hidden, residual)),
                normed,
                stream,
                ssm_state,
                ctx,
                trace,
            )?
        } else {
            self.ssm_forward(normed, ssm_state, ctx, stream, trace)?
        };
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "ssm-out", ssm_out, 4);
        }

        // Profile: time SSM vs MoE separately
        if ctx.profile {
            use std::time::Instant;
            ctx.gpu.synchronize(stream)?;
            let t0 = Instant::now();

            let normed2 = ctx.buffers.norm_output();
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                ssm_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;
            ctx.gpu.synchronize(stream)?;
            let moe_us = t0.elapsed().as_micros();
            tracing::info!("  SSM-MoE: {:.1}ms", moe_us as f64 / 1000.0);

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                h as u32,
                stream,
            )?;
            return Ok(());
        }

        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            ssm_out,
            &self.post_attn_norm,
            normed2,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "post-ssm-residual", residual, 4);
            Self::debug_bf16(ctx.gpu, "post-ssm-hidden", hidden, 4);
            Self::debug_bf16(ctx.gpu, "moe-input-normed", normed2, 4);
        }

        let moe_out = self.ffn.forward(normed2, ctx, stream)?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "moe-output", moe_out, 8);
        }
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            moe_out,
            h as u32,
            stream,
        )?;
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "final-hidden", hidden, 4);
        }

        Ok(())
    }
}
