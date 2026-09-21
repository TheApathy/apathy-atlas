// SPDX-License-Identifier: AGPL-3.0-only

//! Exact M16 grouping of the incumbent QSA and BF16 paged-decode arithmetic.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use super::prefill_moe_attn16_device::{DeviceKernels, expand_or_copy_metadata, mrope_or_scalar};
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::{ops, qwen4_prefill_moe};

impl Qwen3AttentionLayer {
    pub(super) fn preflight_moe_attn16(
        &self,
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        if !qwen4_prefill_moe::attn16::selected()? {
            return Ok(());
        }
        DeviceKernels::load_if_selected(gpu)?;
        ensure!(
            self.kv_dtype == KvCacheDtype::Bf16
                && self.gated
                && self.mrope_interleaved
                && self.rope_mrope_interleaved_k.0 != 0
                && self.paged_decode_k.0 != 0
                && self.sigmoid_gate_mul_batched_k.0 != 0,
            "attention16 requires canonical BF16 MRoPE attention kernels"
        );
        self.qwen4_qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("attention16 requires QSA"))?
            .prepare_exact_group4(gpu, kv_cache)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_moe_attn16(
        &self,
        packed_inputs: DevicePtr,
        qkv: DevicePtr,
        qkv_row_bytes: usize,
        raw_outputs: DevicePtr,
        tile_start: usize,
        rows: usize,
        block_table: &[u32],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        device: Option<DeviceKernels>,
        // When true the tile performs QSA, MRoPE and the KV-cache write only;
        // the attention core and the output gate are run once over every row by
        // the Flash-Attention path instead (ATLAS_QWEN4_PREFILL_ATTN_FLASH).
        skip_core: bool,
    ) -> Result<()> {
        ensure!(
            rows == 16 || rows == 32,
            "attention core requires 16 or 32 rows"
        );
        let metadata = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("attention16 requires paged metadata"))?;
        let plan = qwen4_prefill_moe::attn16::plan_for_tile(
            tile_start
                .checked_add(rows)
                .ok_or_else(|| anyhow::anyhow!("attention16 tile overflow"))?,
            0,
            block_table.len(),
            ctx.buffers.sizes().attn_output,
            rows,
        )?;
        ensure!(
            self.kv_dtype == KvCacheDtype::Bf16
                && metadata.num_seqs == 1
                && metadata.max_blocks_per_seq as usize == block_table.len()
                && !block_table.is_empty(),
            "attention16 requires canonical C1 BF16 paged metadata"
        );
        ensure!(
            self.gated && self.mrope_interleaved && self.rope_mrope_interleaved_k.0 != 0,
            "attention16 requires gated Qwen4 MRoPE attention"
        );
        let h = ctx.config.hidden_size;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let q_dim = nq * hd;
        let q_proj_dim = q_dim * 2;
        let kv_dim = nkv * hd;
        let row_stride = u32::try_from(qkv_row_bytes / 2)?;
        ensure!(
            row_stride == q_proj_dim + 2 * kv_dim,
            "attention16 QKV stride mismatch"
        );
        ensure!(
            rows * q_dim as usize * 2 <= ctx.buffers.sizes().ssm_qkvz,
            "attention16 output scratch is undersized"
        );

        if metadata.qwen4_qsa_required {
            let qsa = self
                .qwen4_qsa
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("attention16 requires QSA"))?;
            qsa.validate_exact_group4()?;
            static TILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let tile_sel = *TILE.get_or_init(|| std::env::var("ATLAS_QWEN4_PREFILL_QSA_TILE").ok().as_deref() == Some("1"));
            if tile_sel {
                let tile_meta = AttnMetadataDev {
                    positions: metadata.positions.offset(tile_start * 4),
                    positions_h: metadata.positions_h.offset(tile_start * 4),
                    positions_w: metadata.positions_w.offset(tile_start * 4),
                    slot: metadata.slot.offset(tile_start * 8),
                    ..metadata
                };
                qsa.update_prefill_exact_tile(
                    packed_inputs,
                    tile_start,
                    rows,
                    kv_cache,
                    tile_meta,
                    h as u32,
                    ctx.config.rms_norm_eps as f32,
                    ctx.config.rope_theta as f32,
                    ctx.config.rotary_dim() as u32,
                    ctx.gpu,
                    stream,
                )?;
            }
            for group in 0..if tile_sel { 0 } else { rows / 4 } {
                let row = tile_start + group * 4;
                let group_meta = AttnMetadataDev {
                    positions: metadata.positions.offset(row * 4),
                    positions_h: metadata.positions_h.offset(row * 4),
                    positions_w: metadata.positions_w.offset(row * 4),
                    slot: metadata.slot.offset(row * 8),
                    ..metadata
                };
                qsa.update_prefill_exact_group4(
                    packed_inputs.offset(group * 4 * h * 2),
                    row,
                    kv_cache,
                    group_meta,
                    h as u32,
                    ctx.config.rms_norm_eps as f32,
                    ctx.config.rope_theta as f32,
                    ctx.config.rotary_dim() as u32,
                    ctx.gpu,
                    stream,
                )?;
            }
        }

        let q_proj_bytes = q_proj_dim as usize * 2;
        let kv_bytes = kv_dim as usize * 2;
        mrope_or_scalar(
            device,
            ctx.gpu,
            self.rope_mrope_interleaved_k,
            qkv,
            qkv_row_bytes,
            metadata.positions.offset(tile_start * 4),
            metadata.positions_h.offset(tile_start * 4),
            metadata.positions_w.offset(tile_start * 4),
            row_stride,
            q_proj_dim,
            nq,
            nkv,
            hd,
            ctx.config.rotary_dim() as u32,
            ctx.config.rope_theta as f32,
            rows as u32,
            stream,
        )?;
        let k = qkv.offset(q_proj_bytes);
        let v = k.offset(kv_bytes);
        self.write_kv_cache(
            ctx.gpu,
            k,
            v,
            kv_cache,
            metadata.slot.offset(tile_start * 8),
            rows as u32,
            nkv,
            hd,
            kv_cache.block_size() as u32,
            row_stride,
            row_stride,
            stream,
            false,
        )?;

        let meta_scratch = ctx.buffers.attn_output();
        let seq_lens = meta_scratch.offset(plan.seq_lens_offset);
        expand_or_copy_metadata(
            device,
            ctx.gpu,
            metadata.block_table,
            meta_scratch,
            seq_lens,
            metadata.seq_len,
            block_table,
            tile_start,
            rows as u32,
            stream,
        )?;
        if skip_core {
            return Ok(());
        }
        self.run_paged_decode(
            ctx.gpu,
            qkv,
            kv_cache,
            raw_outputs,
            meta_scratch,
            seq_lens,
            block_table.len() as u32,
            rows as u32,
            nq,
            nkv,
            hd,
            kv_cache.block_size() as u32,
            self.effective_attn_scale(hd),
            row_stride,
            ctx.buffers.splitk_workspace(),
            DevicePtr::NULL,
            DevicePtr::NULL,
            0,
            tile_start as u32 + rows as u32,
            stream,
        )?;
        ops::sigmoid_gate_mul_batched(
            ctx.gpu,
            self.sigmoid_gate_mul_batched_k,
            raw_outputs,
            qkv.offset(q_dim as usize * 2),
            raw_outputs,
            q_dim,
            row_stride,
            rows as u32,
            stream,
        )?;
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::warn!(
                "ENGAGED ATLAS_QWEN4_PREFILL_ATTN_CORE16: exact QSA4 + BF16 paged attention16"
            )
        });
        if device.is_some() {
            static DEVICE_ENGAGED: std::sync::Once = std::sync::Once::new();
            DEVICE_ENGAGED.call_once(|| {
                tracing::warn!(
                    "ENGAGED ATLAS_QWEN4_PREFILL_ATTN_DEVICE16: exact device MRoPE + metadata"
                )
            });
        }
        if rows == 32 {
            static CORE32_ENGAGED: std::sync::Once = std::sync::Once::new();
            CORE32_ENGAGED.call_once(|| {
                tracing::warn!(
                    "ENGAGED ATLAS_QWEN4_PREFILL_ATTN_CORE32: ordered QSA4 + BF16 paged attention32"
                )
            });
        }
        Ok(())
    }
}
