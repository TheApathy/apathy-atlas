// SPDX-License-Identifier: AGPL-3.0-only

//! Single-token decode body for [`super::super::Qwen3AttentionLayer`],
//! split out of the trait impl for file-size budget. The trait impl
//! delegates 1:1 to [`Qwen3AttentionLayer::decode_inner`].

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::{diag_norm, diag_norm_f32, gemma4_diag_enabled};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_qwen4_batched_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        _h_intermediate: DevicePtr,
        _conv_intermediate: DevicePtr,
        _h_intermediate_stride: usize,
        _conv_intermediate_stride: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("Qwen4 batched verify requested without hyperconnections"),
        };
        let metadata = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("Qwen4 batched attention requires metadata"))?;
        let h = ctx.config.hidden_size;
        let row_bytes = ctx.config.residual_width() * 2;
        let core_bytes = h * 2;
        let eps = ctx.config.rms_norm_eps as f32;
        let mb = metadata.max_blocks_per_seq as usize;
        let hybrid_k5 =
            num_tokens == 5 && std::env::var("ATLAS_QWEN4_K5_HYBRID").ok().as_deref() == Some("1");
        let exact_hyper_k5 =
            hybrid_k5 && std::env::var("ATLAS_QWEN4_K5_BATCH_HYPER").ok().as_deref() != Some("1");

        // Batch the four-stream projection weights. KV writes and attention
        // remain token ordered to preserve exact causal semantics.
        // `attention_forward_oproj` reuses norm_output, so preserve every
        // mixed row before the first token's attention core clobbers it.
        let attn_inputs = ctx.buffers.gate_logits();
        anyhow::ensure!(
            num_tokens * core_bytes <= ctx.buffers.sizes().gate_logits,
            "Qwen4 batched attention input staging exceeds gate-logit workspace"
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
            let mixed_attn = attn_hyper.prepare_batched(
                hidden,
                residual,
                num_tokens,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?;
            ctx.gpu
                .copy_d2d_async(mixed_attn, attn_inputs, num_tokens * core_bytes, stream)?;
        }
        let exact_qkv = self.qwen4_k5_project_qkv_exact(attn_inputs, num_tokens, ctx, stream)?;
        for row in 0..num_tokens {
            let hidden_row = hidden.offset(row * row_bytes);
            let residual_row = residual.offset(row * row_bytes);
            let token_metadata = crate::layer::AttnMetadataDev {
                positions: metadata.positions.offset(row * 4),
                positions_h: metadata.positions_h.offset(row * 4),
                positions_w: metadata.positions_w.offset(row * 4),
                slot: metadata.slot.offset(row * 8),
                seq_len: metadata.seq_len.offset(row * 4),
                block_table: metadata.block_table.offset(row * mb * 4),
                num_seqs: 1,
                ..metadata
            };
            let token_ctx = ForwardContext {
                attn_metadata: Some(token_metadata),
                ..*ctx
            };
            let attn_input = attn_inputs.offset(row * core_bytes);
            let attn_out = if let Some((qkv, qkv_row_bytes)) = exact_qkv {
                self.attention_forward_preprojected(
                    attn_input,
                    qkv.offset(row * qkv_row_bytes),
                    seq_len + row,
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    kv_cache,
                    &token_ctx,
                    stream,
                )?
            } else {
                self.attention_forward(
                    attn_input,
                    seq_len + row,
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    kv_cache,
                    &token_ctx,
                    stream,
                )?
            };
            attn_hyper.inject_decode(
                hidden_row,
                attn_out,
                attn_hyper.saved_inject(residual_row),
                ctx.gpu,
                stream,
            )?;
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
        } else {
            mlp_hyper.prepare_batched(
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
            5 if hybrid_k5 => self.ffn.forward_k5_split(ffn_inputs, ctx, stream)?,
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
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        if let (Some(attn_hyper), Some(mlp_hyper)) = (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper)
        {
            let (mixed, inject) =
                attn_hyper.prepare_decode(hidden, residual, ctx.buffers, ctx.gpu, eps, stream)?;
            let attn_out = self.attention_forward(
                mixed,
                seq_len,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                kv_cache,
                ctx,
                stream,
            )?;
            attn_hyper.inject_decode(
                hidden,
                attn_out,
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
        // Disable diagnostics during CUDA graph capture — diag_norm does d2h
        // copy + sync which invalidates stream capture (status 901).
        let gemma4_diag =
            ctx.config.model_type == "gemma4" && gemma4_diag_enabled() && !ctx.graph_capture;
        // When Gemma-4 FP32 residual is active, `hidden` is a FP32 buffer;
        // use the FP32 diag reader so we don't alias 4-byte floats as two
        // 2-byte BF16s (producing bogus NaNs in the print).
        let hidden_is_fp32 = ctx.config.use_fp32_residual();
        let diag_hidden =
            |gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, stream: u64, label: &str| {
                if hidden_is_fp32 {
                    diag_norm_f32(gpu, ptr, n, stream, label);
                } else {
                    diag_norm(gpu, ptr, n, stream, label);
                }
            };

        let normed = ctx.buffers.norm_output();
        if gemma4_diag {
            diag_hidden(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{:02} hidden_in", self.attn_layer_idx),
            );
        }
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
        if gemma4_diag {
            diag_norm(
                ctx.gpu,
                normed,
                h,
                stream,
                &format!("L{:02} normed", self.attn_layer_idx),
            );
        }

        let attn_out = self.attention_forward(
            normed,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_cache,
            ctx,
            stream,
        )?;
        // TP all-reduce on attn_out after o_proj (Megatron row-parallel
        // pattern). When tp_world_size==1 this is a no-op. The o_proj GEMM
        // produced this rank's partial output on the full hidden dim; the
        // reduction across TP ranks gives the full attention output ready
        // for the residual add. Decode path: 1 token × hidden BF16.
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = h * 2; // 1 token × hidden × BF16
            comm.all_reduce_async(attn_out.0, bytes, stream)?;
        }
        if gemma4_diag {
            diag_norm(
                ctx.gpu,
                attn_out,
                h,
                stream,
                &format!("L{:02} attn_out", self.attn_layer_idx),
            );
        }

        // Gemma-4: post-attention norm (applied to attn output before residual add).
        // Weight pre-scaled by layer_scalar at load time: norm(attn) * (w * scalar).
        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                attn_out,
                post_norm,
                attn_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    attn_out,
                    h,
                    stream,
                    &format!("L{:02} post_attn_normed", self.attn_layer_idx),
                );
            }
        }

        // Standalone attention (Nemotron-H): no post-attn FFN
        if self.ffn.is_none() {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                attn_out,
                h as u32,
                stream,
            )?;
            return Ok(());
        }

        // Profile: time attention vs MoE separately
        if ctx.profile {
            use std::time::Instant;
            ctx.gpu.synchronize(stream)?;
            let t0 = Instant::now();

            let normed2 = ctx.buffers.norm_output();
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                attn_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;

            // Gemma-4: post-FFN norm
            if let Some(ref post_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    moe_out,
                    post_norm,
                    moe_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }

            ctx.gpu.synchronize(stream)?;
            let moe_us = t0.elapsed().as_micros();
            tracing::info!("  Attn-MoE: {:.1}ms", moe_us as f64 / 1000.0);

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                h as u32,
                stream,
            )?;
            // Gemma-4: hidden *= layer_scalar at end of layer
            if let Some(scalar) = self.layer_scalar {
                self.apply_layer_scalar(
                    ctx.gpu,
                    hidden,
                    h,
                    scalar,
                    stream,
                    ctx.config.use_fp32_residual(),
                )?;
            }
            return Ok(());
        }

        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            attn_out,
            &self.post_attn_norm,
            normed2,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;

        // Gemma-4 26B MoE dual FFN: run MoE FIRST (before dense FFN result is used)
        // to avoid buffer conflicts (MoE fused kernel uses attn_output internally).
        //
        // HF reference: combined = norm(norm1(mlp_out) + norm2(moe_out))
        //               hidden = residual + combined
        if let (Some(moe_ffn), Some(_pre_norm), Some(post_norm), Some(dense_norm)) = (
            &self.moe_ffn,
            &self.pre_moe_norm,
            &self.post_moe_out_norm,
            &self.post_dense_ffn_norm,
        ) {
            // 1. Run MoE on raw residual (before dense FFN output is touched).
            //    MoE writes result to moe_output buffer.
            let moe_out = moe_ffn.forward(hidden, ctx, stream)?;
            // post-MoE norm (in-place on moe_output)
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                moe_out,
                post_norm,
                moe_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
            // Save normed MoE output — dense FFN will overwrite moe_output.
            // Use logits buffer (vocab_size * 2 bytes >> h * 2) — gate_logits is too small
            let moe_saved = ctx.buffers.logits();
            ctx.gpu.copy_d2d_async(moe_out, moe_saved, h * 2, stream)?;

            // 2. Dense FFN (writes to moe_output, overwriting MoE result)
            let dense_out = self.ffn.forward(normed2, ctx, stream)?;
            // post-dense norm (layernorm_1)
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                dense_out,
                dense_norm,
                dense_out,
                1,
                h as u32,
                eps,
                stream,
            )?;

            // 3. Combine: dense_normed + moe_normed → dense_out (in-place)
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                dense_out,
                moe_saved,
                h as u32,
                stream,
            )?;

            // 4. post_feedforward_layernorm on combined
            if let Some(ref combined_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    dense_out,
                    combined_norm,
                    dense_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }

            // 5. Residual add: hidden += combined
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                h as u32,
                stream,
            )?;
        } else {
            // Non-MoE (31B dense)
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    normed2,
                    h,
                    stream,
                    &format!("L{:02} normed2", self.attn_layer_idx),
                );
            }
            let dense_out = self.ffn.forward(normed2, ctx, stream)?;
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    dense_out,
                    h,
                    stream,
                    &format!("L{:02} dense_out", self.attn_layer_idx),
                );
            }
            if let Some(ref post_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    dense_out,
                    post_norm,
                    dense_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
                if gemma4_diag {
                    diag_norm(
                        ctx.gpu,
                        dense_out,
                        h,
                        stream,
                        &format!("L{:02} post_ffn_normed", self.attn_layer_idx),
                    );
                }
            }
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                h as u32,
                stream,
            )?;
        }

        if gemma4_diag {
            diag_hidden(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{:02} post_residual", self.attn_layer_idx),
            );
        }

        // Gemma-4: hidden *= layer_scalar at end of layer
        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(
                ctx.gpu,
                hidden,
                h,
                scalar,
                stream,
                ctx.config.use_fp32_residual(),
            )?;
            if gemma4_diag {
                diag_hidden(
                    ctx.gpu,
                    hidden,
                    h,
                    stream,
                    &format!(
                        "L{:02} post_layer_scalar(scalar={:.4})",
                        self.attn_layer_idx, scalar
                    ),
                );
            }
        }

        Ok(())
    }
}
