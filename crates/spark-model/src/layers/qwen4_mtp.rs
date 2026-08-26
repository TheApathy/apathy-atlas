// SPDX-License-Identifier: AGPL-3.0-only

//! Native Qwen3.8-Flash-Next multi-token predictor.

use std::any::Any;

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::weights::WeightStore;

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState, TransformerLayer};
use crate::layers::{Qwen4HyperConnection, ops};
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{
    DenseWeight, QuantizedWeight, dense, dense_auto, quantize_to_nvfp4_cached,
};

pub struct Qwen4MtpState {
    layer_state: Box<dyn LayerState>,
    block_table: Vec<u32>,
    disk_block_ids: Vec<u32>,
    disk_last_offloaded_per_layer: Vec<u32>,
    seq_len: usize,
    last_num_drafted: usize,
}

impl ProposerState for Qwen4MtpState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct Qwen4MtpHead {
    pre_fc_norm_embedding: DenseWeight,
    pre_fc_norm_hidden: DenseWeight,
    fc_embedding: QuantizedWeight,
    fc_hidden: QuantizedWeight,
    layer: Box<dyn TransformerLayer>,
    final_mixer: Qwen4HyperConnection,
    embed_tokens: DenseWeight,
    lm_head_nvfp4: QuantizedWeight,
    mtp_vocab_size: u32,
    kv_cache: Mutex<PagedKvCache>,
    rms_norm_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemv_exact_m4_k: KernelHandle,
    residual_add_k: KernelHandle,
    argmax_k: KernelHandle,
}

impl Qwen4MtpHead {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: &WeightStore,
        config: &atlas_core::config::ModelConfig,
        layer: Box<dyn TransformerLayer>,
        final_mixer: Qwen4HyperConnection,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: QuantizedWeight,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
        max_batch_size: usize,
    ) -> Result<Self> {
        ensure!(
            config.is_qwen4_exp(),
            "Qwen4MtpHead requires qwen4_exp target"
        );
        let h = config.hidden_size;
        let stream = gpu.default_stream();
        let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let fc_embedding_dense = dense_auto(store, "mtp.fc_embedding.weight", gpu)?;
        let fc_hidden_dense = dense_auto(store, "mtp.fc_hidden.weight", gpu)?;
        let fc_embedding = quantize_to_nvfp4_cached(
            &fc_embedding_dense,
            h,
            h,
            gpu,
            absmax,
            quantize,
            stream,
            "mtp.fc_embedding.nvfp4",
        )?;
        let fc_hidden = quantize_to_nvfp4_cached(
            &fc_hidden_dense,
            h,
            h,
            gpu,
            absmax,
            quantize,
            stream,
            "mtp.fc_hidden.nvfp4",
        )?;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let blocks_per_seq = max_seq_len.div_ceil(kv_config.block_size) + 1;
        let kv_cache = PagedKvCache::new(kv_config, blocks_per_seq * max_batch_size.max(1), gpu)?;
        let effective_vocab = if mtp_vocab_size == 0 {
            config.vocab_size
        } else {
            (mtp_vocab_size as usize).min(config.vocab_size)
        };
        tracing::info!(
            hidden = h,
            residual = config.residual_width(),
            experts = config.num_experts,
            vocab = effective_vocab,
            "Qwen4 native MTP head constructed"
        );
        Ok(Self {
            pre_fc_norm_embedding: dense(store, "mtp.pre_fc_norm_embedding.weight")?,
            pre_fc_norm_hidden: dense(store, "mtp.pre_fc_norm_hidden.weight")?,
            fc_embedding,
            fc_hidden,
            layer,
            final_mixer,
            embed_tokens,
            lm_head_nvfp4,
            mtp_vocab_size,
            kv_cache: Mutex::new(kv_cache),
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_exact_m4_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m4")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut Qwen4MtpState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;

        let embed = ctx.buffers.ssm_qkvz();
        let embed_src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu
            .copy_d2d_async(embed_src, embed, row_bytes, stream)?;
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed,
            &self.pre_fc_norm_embedding,
            normed_embed,
            1,
            h as u32,
            eps,
            stream,
        )?;
        let normed_hidden = ctx.buffers.residual();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.pre_fc_norm_hidden,
            normed_hidden,
            1,
            r as u32,
            eps,
            stream,
        )?;

        let embed_proj = ctx.buffers.qkv_output();
        ops::w4a16_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            normed_embed,
            &self.fc_embedding,
            embed_proj,
            h as u32,
            h as u32,
            stream,
        )?;
        let hidden = ctx.buffers.hidden_states();
        ensure!(hc == 4, "Qwen4 native MTP expects four hidden streams");
        ops::w4a16_gemv_batch_logits_exact_with(
            ctx.gpu,
            ops::W4a16ExactLmHeadKernels::new(
                self.w4a16_gemv_exact_m4_k,
                KernelHandle(0),
                KernelHandle(0),
                KernelHandle(0),
            ),
            normed_hidden,
            &self.fc_hidden,
            hidden,
            hc as u32,
            h as u32,
            h as u32,
            stream,
            false,
        )?;
        for branch in 0..hc {
            let hidden_branch = hidden.offset(branch * row_bytes);
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden_branch,
                embed_proj,
                h as u32,
                stream,
            )?;
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = state.seq_len / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let physical = state.block_table[state.seq_len / bs];
        let slot = physical as i64 * bs as i64 + (state.seq_len % bs) as i64;
        let meta_base = ctx.buffers.scratch().offset(32768);
        let bt_bytes = state.block_table.len() * 4;
        let mut packed = vec![0u8; 768 + bt_bytes];
        packed[0..4].copy_from_slice(&(position as u32).to_le_bytes());
        packed[256..264].copy_from_slice(&slot.to_le_bytes());
        packed[512..516].copy_from_slice(&((state.seq_len + 1) as i32).to_le_bytes());
        for (i, block) in state.block_table.iter().enumerate() {
            packed[768 + i * 4..772 + i * 4].copy_from_slice(&(*block as i32).to_le_bytes());
        }
        ctx.gpu
            .copy_h2d_group_on_stream(&[HostToDeviceCopy::new(&packed, meta_base)], stream)?;
        let metadata = AttnMetadataDev {
            qwen4_qsa_required: false,
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: state.block_table.len() as u32,
            num_seqs: 1,
        };
        let layer_ctx = ForwardContext {
            attn_metadata: Some(metadata),
            comm: None,
            graph_capture: false,
            ..*ctx
        };
        self.layer.decode(
            hidden,
            ctx.buffers.residual(),
            state.layer_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut state.disk_block_ids,
            &mut state.disk_last_offloaded_per_layer,
            &layer_ctx,
            stream,
        )?;

        let (sample_hidden, inject) = self.final_mixer.prepare_decode(
            hidden,
            ctx.buffers.residual(),
            ctx.buffers,
            ctx.gpu,
            eps,
            stream,
        )?;
        debug_assert!(inject.is_none());
        let vocab = if self.mtp_vocab_size == 0 {
            ctx.config.vocab_size
        } else {
            (self.mtp_vocab_size as usize).min(ctx.config.vocab_size)
        };
        let logits = ctx.buffers.logits();
        ops::w4a16_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            sample_hidden,
            &self.lm_head_nvfp4,
            logits,
            vocab as u32,
            h as u32,
            stream,
        )?;
        let out = ctx.buffers.scratch();
        ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out, vocab as u32, stream)?;
        let token_id = if let Some(mask) = grammar_bitmask {
            let mut bytes = vec![0u8; vocab * 2];
            ctx.gpu.copy_d2h(logits, &mut bytes)?;
            let mut best = None::<(u32, f32)>;
            for candidate in 0..vocab {
                if candidate / 32 >= mask.len()
                    || (mask[candidate / 32] & (1i32 << (candidate % 32))) == 0
                {
                    continue;
                }
                let bits = u16::from_le_bytes([bytes[candidate * 2], bytes[candidate * 2 + 1]]);
                let value = f32::from_bits((bits as u32) << 16);
                if best.is_none_or(|(_, old)| value > old) {
                    best = Some((candidate as u32, value));
                }
            }
            best.map_or(0, |(candidate, _)| candidate)
        } else {
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h(out, &mut bytes)?;
            u32::from_le_bytes(bytes)
        };
        state.seq_len += 1;
        Ok(token_id)
    }
}

impl DraftProposer for Qwen4MtpHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(Qwen4MtpState {
            layer_state: self.layer.alloc_state(gpu)?,
            block_table: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: vec![u32::MAX],
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
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        let mut hidden = target_hidden;
        for step in 0..num_drafts {
            token = self.forward_one(
                token,
                hidden,
                position + step,
                state,
                ctx,
                stream,
                grammar_bitmask,
            )?;
            drafts.push(token);
            hidden = ctx.buffers.hidden_states();
        }
        state.last_num_drafted = drafts.len();
        Ok(drafts)
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        let trim = state.last_num_drafted.saturating_sub(num_accepted);
        state.seq_len = state.seq_len.saturating_sub(trim);
        state.last_num_drafted = 0;
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
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        if tokens.is_empty() {
            return Ok(());
        }
        ensure!(
            state.seq_len == 0,
            "Qwen4 MTP prompt replay requires an empty proposer cache"
        );
        let stride = ctx.config.residual_width() * 2;
        let start_position = base_position + 1 - tokens.len();
        for (index, token) in tokens.iter().copied().enumerate() {
            let _ = self.forward_one(
                token,
                target_hiddens.offset(index * stride),
                start_position + index + 1,
                state,
                ctx,
                stream,
                None,
            )?;
        }
        state.last_num_drafted = 0;
        Ok(())
    }

    fn free_state(&self, state: &mut dyn ProposerState) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        if !state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&state.block_table);
            state.block_table.clear();
        }
        state.seq_len = 0;
        Ok(())
    }
}
