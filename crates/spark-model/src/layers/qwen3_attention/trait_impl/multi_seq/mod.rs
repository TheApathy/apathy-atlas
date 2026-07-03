// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence batched-decode body for [`super::super::Qwen3AttentionLayer`].
//!
//! Split into phase modules under the `_inner` delegation pattern:
//! - `ctx`  — `MultiSeqCtx` shared scalars + buffer pointers
//! - `qkv`  — phase 2: per-token Q/K/V projections (batch3/batch2/seq)
//! - `attn` — phases 3-6: RoPE → cache write → paged decode → O proj
//! - `ffn`  — phase 7: residual + post-norm + MoE/dense FFN
//!
//! The trait impl in `super::trait_impl` calls
//! [`Qwen3AttentionLayer::decode_multi_seq_inner`] which simply builds
//! the ctx, runs phase 1 inline (RMS norm), and dispatches the rest.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

mod attn;
mod ctx;
mod ffn;
mod qkv;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = states; // Attention layers use EmptyLayerState — no per-seq state.
        let bs = kv_cache.block_size() as u32;
        // Max host-side seq_len drives the split-K KV partition. Safe upper
        // bound across all sequences in the batch (verify dispatch uploads
        // monotonically increasing seq_len per K slot, so `max()` matches the
        // longest KV scan any CTA will perform).
        let max_seq_len_host = seq_lens
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_add(1) as u32;
        let c = ctx::MultiSeqCtx::new(
            self,
            ctx,
            hidden,
            residual,
            num_seqs,
            bs,
            stream,
            max_seq_len_host,
        );

        // ── Phase 1: RMS norm + residual for N tokens ──
        crate::kprof!(ctx.gpu, stream, "attn_rms_norm_residual", {
            ops::rms_norm_residual(
                ctx.gpu,
                self.rms_norm_residual_k,
                c.hidden,
                &self.input_norm,
                c.normed,
                c.residual,
                c.n as u32,
                c.h as u32,
                c.eps,
                c.stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        // ── Phase 2: QKV projections (batch3 / batch2 / sequential) ──
        crate::kprof!(ctx.gpu, stream, "attn_qkv_proj", {
            self.ms_phase_qkv(&c)?;
            anyhow::Result::<()>::Ok(())
        })?;

        // ── Phase 3: RoPE per-sequence ──
        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");
        crate::kprof!(ctx.gpu, stream, "attn_rope", {
            self.ms_phase_rope(&c, meta)?;
            anyhow::Result::<()>::Ok(())
        })?;

        // ── Phase 4: KV cache write ──
        crate::kprof!(ctx.gpu, stream, "attn_cache_write", {
            self.ms_phase_cache_write(&c, kv_cache, meta)?;
            anyhow::Result::<()>::Ok(())
        })?;

        // ── Phase 5: paged decode attention (batched) ──
        let attn_out = crate::kprof!(ctx.gpu, stream, "attn_paged_decode", {
            self.ms_phase_paged_decode(&c, kv_cache, meta)?
        });

        // ── Phase 6: gate multiply + O projection ──
        let o_out = crate::kprof!(ctx.gpu, stream, "attn_o_proj", {
            self.ms_phase_o_proj(&c, attn_out)?
        });

        // TP all-reduce on o_out after o_proj (Megatron row-parallel
        // pattern). Mirrors decode_inner.rs and prefill_inner.rs. Without
        // this, multi-token decode (K=2 / K=3 / K=γ verify) under
        // tp_world_size>1 reads a partial attention output from each
        // rank, corrupting the FFN/MoE input and producing degenerate
        // logits — observed as `/`/`,` repetition spirals on
        // Qwen3.6 FP8 + TP=2 + MTP for HTML/code prompts.
        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        // ── Phase 7: residual + post-norm + MoE ──
        crate::kprof!(ctx.gpu, stream, "attn_ffn", {
            self.ms_phase_ffn(&c, o_out)?;
            anyhow::Result::<()>::Ok(())
        })?;

        Ok(())
    }
}
