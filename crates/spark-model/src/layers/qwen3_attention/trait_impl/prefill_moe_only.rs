// SPDX-License-Identifier: AGPL-3.0-only
//! Keep the shipping row-ordered attention core; defer only the stateless FFN.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::prefill_moe_admit::PrefillMoeSetup;
use super::prefill_moe_attn16_device::DeviceKernels;
use crate::layer::{AttnMetadataDev, BatchedAttnMetadata, ForwardContext};
use crate::layers::{ops, qwen4_prefill_moe};

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_moe_only(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let PrefillMoeSetup {
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
        } = self.prepare_prefill_moe_setup(
            hidden,
            residual,
            num_tokens,
            seq_len_start,
            block_table,
            batched_meta,
            ctx,
            stream,
        )?;

        // No MLP input is staged yet: attention and hyper preparation reuse
        // QKV/norm scratch. QSA, KV writes and attention remain row ordered;
        // exact QKV/O projections are scheduled once per complete M16 tile.
        let packed_inputs = ctx.buffers.norm_output();
        let raw_outputs = ctx.buffers.ssm_qkvz();
        let hc16 = qwen4_prefill_moe::attn16::hc_selected()?;
        if hc16 {
            attn_hyper.preflight_saved_tile(hidden, packed_inputs, residual)?;
        }
        let attn16_device = if exact_attn16 {
            DeviceKernels::load_if_selected(ctx.gpu)?
        } else {
            None
        };
        let attn32 = exact_attn16 && qwen4_prefill_moe::attn16::core32_selected()?;
        if crate::layers::qwen4_fast_proj::selection()?.attn {
            anyhow::ensure!(
                exact_hyper && exact_qkv16 && exact_o16 && exact_attn16,
                "{} attn requires HC_EXACT, ATTN_QKV16, ATTN_O16 and ATTN_CORE16",
                crate::layers::qwen4_fast_proj::SELECTOR
            );
            self.prefill_moe_only_fast_attn(
                attn_hyper,
                hidden,
                residual,
                num_tokens,
                packed_inputs,
                raw_outputs,
                core_bytes,
                attn_width,
                attn_row_bytes,
                block_table,
                kv_cache,
                attn16_device,
                attn32,
                ctx,
                stream,
            )?;
            qwen4_prefill_moe::finish(
                mlp_hyper, &self.ffn, hidden, residual, num_tokens, ctx, stream,
            )?;
            crate::model::qwen4_prefill_engagement::engage(
                crate::model::qwen4_prefill_engagement::PrefillPath::Attention,
                num_tokens,
            )?;
            return Ok(());
        }
        let mut tile_start = 0;
        while tile_start < num_tokens {
            let tile_rows = if attn32 && num_tokens - tile_start >= 32 {
                32
            } else if exact_qkv16 {
                (num_tokens - tile_start).min(16)
            } else {
                1
            };
            let exact_qkv = if exact_qkv16 && (tile_rows == 16 || tile_rows == 32) {
                for chunk in (0..tile_rows).step_by(16) {
                    let residual_chunk = residual.offset((tile_start + chunk) * row_bytes);
                    let packed_chunk = packed_inputs.offset(chunk * core_bytes);
                    attn_hyper.preflight_saved_tile(
                        hidden.offset((tile_start + chunk) * row_bytes),
                        packed_chunk,
                        residual_chunk,
                    )?;
                    attn_hyper.pack_saved_tile_or_rows(
                        hc16,
                        residual_chunk,
                        packed_chunk,
                        row_bytes,
                        core_bytes,
                        ctx.gpu,
                        stream,
                    )?;
                }
                self.qwen4_k5_project_qkv_exact(packed_inputs, tile_rows, ctx, stream)?
            } else {
                None
            };
            let batched_core = if exact_attn16 && (tile_rows == 16 || tile_rows == 32) {
                let (qkv, qkv_row_bytes) = exact_qkv.ok_or_else(|| {
                    anyhow::anyhow!("attention16 full tile lost its exact QKV route")
                })?;
                self.prefill_moe_attn16(
                    packed_inputs,
                    qkv,
                    qkv_row_bytes,
                    raw_outputs,
                    tile_start,
                    tile_rows,
                    block_table,
                    kv_cache,
                    ctx,
                    stream,
                    attn16_device,
                    false,
                )?;
                true
            } else {
                false
            };
            if !batched_core {
                for local in 0..tile_rows {
                    let row = tile_start + local;
                    let seq_len = (seq_len_start + row + 1) as u32;
                    let seq_len_bytes = seq_len.to_ne_bytes();
                    ctx.gpu.copy_h2d_group_on_stream(
                        &[HostToDeviceCopy::new(&seq_len_bytes, metadata.seq_len)],
                        stream,
                    )?;
                    let token_metadata = AttnMetadataDev {
                        positions: metadata.positions.offset(row * 4),
                        positions_h: metadata.positions_h.offset(row * 4),
                        positions_w: metadata.positions_w.offset(row * 4),
                        slot: metadata.slot.offset(row * 8),
                        seq_len: metadata.seq_len,
                        ..metadata
                    };
                    let token_ctx = ForwardContext {
                        attn_metadata: Some(token_metadata),
                        ..*ctx
                    };
                    let hidden_row = hidden.offset(row * row_bytes);
                    let residual_row = residual.offset(row * row_bytes);
                    let (mixed, inject) = if exact_hyper {
                        (residual_row, Some(attn_hyper.saved_inject(residual_row)))
                    } else {
                        attn_hyper.prepare_decode(
                            hidden_row,
                            residual_row,
                            ctx.buffers,
                            ctx.gpu,
                            eps,
                            stream,
                        )?
                    };
                    let attn_out = if exact_o16 && let Some((qkv, qkv_row_bytes)) = exact_qkv {
                        let raw = self.attention_forward_preprojected_raw(
                            packed_inputs.offset(local * core_bytes),
                            qkv.offset(local * qkv_row_bytes),
                            seq_len_start + row,
                            block_table,
                            disk_block_ids,
                            disk_last_offloaded_per_layer,
                            kv_cache,
                            &token_ctx,
                            stream,
                        )?;
                        ctx.gpu.copy_d2d_async(
                            raw,
                            raw_outputs.offset(local * attn_row_bytes),
                            attn_row_bytes,
                            stream,
                        )?;
                        None
                    } else if let Some((qkv, qkv_row_bytes)) = exact_qkv {
                        Some(self.attention_forward_preprojected(
                            packed_inputs.offset(local * core_bytes),
                            qkv.offset(local * qkv_row_bytes),
                            seq_len_start + row,
                            block_table,
                            disk_block_ids,
                            disk_last_offloaded_per_layer,
                            kv_cache,
                            &token_ctx,
                            stream,
                        )?)
                    } else {
                        Some(self.attention_forward(
                            mixed,
                            seq_len_start + row,
                            block_table,
                            disk_block_ids,
                            disk_last_offloaded_per_layer,
                            kv_cache,
                            &token_ctx,
                            stream,
                        )?)
                    };
                    if let Some(attn_out) = attn_out {
                        attn_hyper.inject_decode(
                            hidden_row,
                            attn_out,
                            inject.ok_or_else(|| {
                                anyhow::anyhow!("Qwen4 attention lost injection weights")
                            })?,
                            ctx.gpu,
                            stream,
                        )?;
                    }
                }
            }
            if exact_o16 && exact_qkv.is_some() {
                ops::w4a16_gemv_batch_logits_exact_with(
                    ctx.gpu,
                    self.w4a16_exact_o_proj_kernels,
                    raw_outputs,
                    &self.attn.o_proj,
                    packed_inputs,
                    tile_rows as u32,
                    ctx.config.hidden_size as u32,
                    attn_width as u32,
                    stream,
                    ops::w4a16_gemv_rt2_enabled(),
                )?;
                for chunk in (0..tile_rows).step_by(16) {
                    attn_hyper.inject_saved_tile_or_rows(
                        hc16,
                        hidden.offset((tile_start + chunk) * row_bytes),
                        packed_inputs.offset(chunk * core_bytes),
                        residual.offset((tile_start + chunk) * row_bytes),
                        row_bytes,
                        core_bytes,
                        ctx.gpu,
                        stream,
                    )?;
                }
            }
            tile_start += tile_rows;
        }
        qwen4_prefill_moe::finish(
            mlp_hyper, &self.ffn, hidden, residual, num_tokens, ctx, stream,
        )?;
        crate::model::qwen4_prefill_engagement::engage(
            crate::model::qwen4_prefill_engagement::PrefillPath::Attention,
            num_tokens,
        )?;
        Ok(())
    }
}
