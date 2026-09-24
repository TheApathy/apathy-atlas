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
    embed_from_argmax_k: KernelHandle,
}

/// Scratch layout of the sync-free draft chain. The single-token MoE router
/// writes the first bytes of scratch and the per-step attention metadata
/// starts at `CHAIN_META_OFFSET` (the same base the one-step path uses).
const CHAIN_ARGMAX_OFFSET: usize = 16384;
const CHAIN_TOKEN_OFFSET: usize = CHAIN_ARGMAX_OFFSET + 1024;
const CHAIN_META_OFFSET: usize = 32768;

/// Where one MTP step takes its inputs from and leaves its draft.
#[derive(Clone, Copy)]
struct ChainStep {
    /// Pre-uploaded attention metadata for this step.
    meta: DevicePtr,
    /// Device u32 holding the previous step's argmax (None: host token).
    device_token: Option<DevicePtr>,
    /// Device u32 receiving this step's argmax; no host readback.
    argmax_out: DevicePtr,
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
            embed_from_argmax_k: gpu
                .kernel("embed_from_argmax", "embed_from_argmax")
                .unwrap_or(KernelHandle(0)),
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
        emit_draft: bool,
        chain: Option<ChainStep>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;

        let embed = ctx.buffers.ssm_qkvz();
        if let Some(device_token) = chain.and_then(|c| c.device_token) {
            ops::embed_from_argmax(
                ctx.gpu,
                self.embed_from_argmax_k,
                device_token,
                self.embed_tokens.weight,
                embed,
                ctx.buffers.scratch().offset(CHAIN_TOKEN_OFFSET),
                h as u32,
                stream,
            )?;
        } else {
            let embed_src = self.embed_tokens.weight.offset(token as usize * row_bytes);
            ctx.gpu
                .copy_d2d_async(embed_src, embed, row_bytes, stream)?;
        }
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
        let meta_base = if let Some(c) = chain {
            c.meta
        } else {
            let meta_base = ctx.buffers.scratch().offset(CHAIN_META_OFFSET);
            let packed = self.step_metadata(&mut kv_cache, state, state.seq_len, position)?;
            ctx.gpu
                .copy_h2d_group_on_stream(&[HostToDeviceCopy::new(&packed, meta_base)], stream)?;
            meta_base
        };
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

        // Prompt replay only fills the MTP KV cache. It must not run the head:
        // `buffers.logits()` still holds the target's prefill logits, which the
        // scheduler samples the first output token from after this returns.
        if !emit_draft {
            state.seq_len += 1;
            return Ok(0);
        }

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
        if let Some(c) = chain {
            ops::argmax_bf16(
                ctx.gpu,
                self.argmax_k,
                logits,
                c.argmax_out,
                vocab as u32,
                stream,
            )?;
            state.seq_len += 1;
            return Ok(0);
        }
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

    /// Attention metadata for the MTP step writing KV row `seq_len` at rotary
    /// `position`, allocating its block if needed: position at +0, KV slot at
    /// +256, sequence length at +512, block table at +768.
    fn step_metadata(
        &self,
        kv_cache: &mut PagedKvCache,
        state: &mut Qwen4MtpState,
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<u8>> {
        let bs = kv_cache.block_size();
        let blocks_needed = seq_len / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let physical = state.block_table[seq_len / bs];
        let slot = physical as i64 * bs as i64 + (seq_len % bs) as i64;
        let bt_bytes = state.block_table.len() * 4;
        let mut packed = vec![0u8; 768 + bt_bytes];
        packed[0..4].copy_from_slice(&(position as u32).to_le_bytes());
        packed[256..264].copy_from_slice(&slot.to_le_bytes());
        packed[512..516].copy_from_slice(&((seq_len + 1) as i32).to_le_bytes());
        for (i, block) in state.block_table.iter().enumerate() {
            packed[768 + i * 4..772 + i * 4].copy_from_slice(&(*block as i32).to_le_bytes());
        }
        Ok(packed)
    }

    /// Greedy draft chain with one metadata upload and one readback: each
    /// step's argmax stays on the device and feeds the next step's embedding
    /// gather, so the host never waits between drafts. Same arithmetic per
    /// step as the one-step path.
    #[allow(clippy::too_many_arguments)]
    fn propose_chain(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut Qwen4MtpState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let scratch = ctx.buffers.scratch();
        let first_seq_len = state.seq_len;
        let mut uploads: Vec<Vec<u8>> = Vec::with_capacity(num_drafts);
        {
            let mut kv_cache = self.kv_cache.lock();
            // Allocate every step's block first so all steps share one table.
            let last = first_seq_len + num_drafts - 1;
            self.step_metadata(&mut kv_cache, state, last, position + num_drafts - 1)?;
            for step in 0..num_drafts {
                uploads.push(self.step_metadata(
                    &mut kv_cache,
                    state,
                    first_seq_len + step,
                    position + step,
                )?);
            }
        }
        let stride = uploads[0].len().next_multiple_of(256);
        ensure!(
            CHAIN_META_OFFSET + num_drafts * stride <= ctx.buffers.sizes().scratch
                && CHAIN_ARGMAX_OFFSET + num_drafts * 4 <= CHAIN_TOKEN_OFFSET,
            "Qwen4 MTP draft chain of {num_drafts} steps does not fit the scratch layout"
        );
        let mut packed = vec![0u8; num_drafts * stride];
        for (step, upload) in uploads.iter().enumerate() {
            packed[step * stride..step * stride + upload.len()].copy_from_slice(upload);
        }
        let meta_base = scratch.offset(CHAIN_META_OFFSET);
        ctx.gpu
            .copy_h2d_group_on_stream(&[HostToDeviceCopy::new(&packed, meta_base)], stream)?;

        let argmax_base = scratch.offset(CHAIN_ARGMAX_OFFSET);
        let mut hidden = target_hidden;
        for step in 0..num_drafts {
            let chain = ChainStep {
                meta: meta_base.offset(step * stride),
                device_token: (step > 0).then(|| argmax_base.offset((step - 1) * 4)),
                argmax_out: argmax_base.offset(step * 4),
            };
            self.forward_one(
                last_token,
                hidden,
                position + step,
                state,
                ctx,
                stream,
                None,
                true,
                Some(chain),
            )?;
            hidden = ctx.buffers.hidden_states();
        }
        ctx.gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; num_drafts * 4];
        ctx.gpu.copy_d2h(argmax_base, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
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
        if grammar_bitmask.is_none()
            && num_drafts > 0
            && self.embed_from_argmax_k.0 != 0
            && std::env::var("ATLAS_QWEN4_MTP_CHAIN").ok().as_deref() == Some("1")
        {
            let drafts = self.propose_chain(
                last_token,
                target_hidden,
                position,
                num_drafts,
                state,
                ctx,
                stream,
            )?;
            state.last_num_drafted = drafts.len();
            return Ok(drafts);
        }
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
                true,
                None,
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
                false,
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
