// SPDX-License-Identifier: AGPL-3.0-only

//! DDTree M2b — batched tree-verify layer forward with a SPLIT KV
//! cache-write.
//!
//! One batched K_t-row forward per layer, identical to
//! [`super::Qwen3AttentionLayer::decode_multi_seq_inner`] (qkv GEMM →
//! rope → cache write → paged decode → o-proj → ffn) with ONE change:
//! the cache-write phase runs as two row-range invocations with the
//! branch re-seed d2d copies between them:
//!
//! 1. cache-write rows `[0, spine_rows)` — bonus + spine — into their
//!    canonical blocks;
//! 2. for each `(canonical_block, scratch_block)` pair in `reseed`,
//!    d2d-copy THIS layer's K/V block canonical → scratch (the copy is
//!    stream-ordered after step 1, so the scratch captures this step's
//!    spine K/V — the intra-step ancestor-visibility requirement);
//! 3. cache-write rows `[spine_rows, num_seqs)` — the branch rows —
//!    through their per-row slots (which point at scratch).
//!
//! All other phases run single-launch over all K_t rows. That is safe
//! because the batched paged decode reads KV ONLY through the per-row
//! metadata views (per-row block tables / seq_lens), and by the time it
//! launches, every write for this layer (spine → canonical, seeded
//! canonical → scratch, branch → scratch) is already queued on the same
//! stream.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::{MultiSeqCtx, TreeSplit};
use crate::layer::{ForwardContext, TreeReseed};
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// Copy THIS layer's `(canonical → scratch)` KV blocks for every branch.
    ///
    /// Shared by the generic batched tree body below and by the V4-Flash
    /// MLA cache-write split in `decode::rows_rope_cache`, so both orderings
    /// re-seed identically. Must be queued AFTER the spine rows' cache write
    /// and BEFORE the branch rows': that is what gives a branch row intra-step
    /// visibility of the spine K/V at depths shallower than its fork.
    pub(in crate::layers::qwen3_attention) fn tree_reseed_blocks(
        &self,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        reseed: &TreeReseed<'_>,
        stream: u64,
    ) -> Result<()> {
        let li = self.attn_layer_idx;
        let kb = kv_cache.config().k_block_bytes_for_layer(li);
        let vb = kv_cache.config().v_block_bytes_for_layer(li);
        match reseed {
            TreeReseed::HostPairs(pairs) => {
                for &(canonical, scratch) in *pairs {
                    ctx.gpu.copy_d2d_async(
                        kv_cache.k_cache_ptr(li, canonical),
                        kv_cache.k_cache_ptr(li, scratch),
                        kb,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        kv_cache.v_cache_ptr(li, canonical),
                        kv_cache.v_cache_ptr(li, scratch),
                        vb,
                        stream,
                    )?;
                }
            }
            // M5 graphed path: one indirect launch per layer — the (src,
            // dst) block ids AND the pair count ride in the device buffer,
            // so this launch is CUDA-graph-capturable (no per-step host
            // pointers). Stream-ordered exactly like the memcpys above.
            TreeReseed::Indirect {
                kernel,
                meta,
                max_pairs,
            } => {
                ops::kv_block_indirect_copy(
                    ctx.gpu,
                    *kernel,
                    kv_cache.k_cache_ptr(li, 0),
                    kv_cache.v_cache_ptr(li, 0),
                    kv_cache.k_block_stride(li),
                    kv_cache.v_block_stride(li),
                    kb,
                    vb,
                    *meta,
                    *max_pairs,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Batched tree-verify decode (DDTree M2b). Returns `Ok(false)` —
    /// with NO work launched — when this layer has no batched tree path;
    /// the caller falls back to per-row sequential decode for this layer.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_multi_seq_tree_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        kv_cache: &mut PagedKvCache,
        spine_rows: usize,
        reseed: &TreeReseed<'_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        debug_assert!(spine_rows <= num_seqs);

        // DeepSeek-V4-Flash: MLA + mHC. The generic phase sequence below does
        // not apply (mHC owns the residual highway, and MLA's cache write sits
        // inside `ms_mla_decode`), but the batched flat body already handles
        // both — the ONLY tree-specific difference is the split cache write,
        // which `mla_rows_rope_and_cache` performs when `c.tree` is armed. So
        // run the flat body verbatim with the split armed, instead of forking
        // a second copy of a 400-line phase sequence that would drift.
        if self.mla.is_some() && self.hc.is_some() {
            let bs = kv_cache.block_size() as u32;
            let mut c = MultiSeqCtx::new(self, ctx, hidden, residual, num_seqs, bs, stream);
            // Eligibility must be decided BEFORE any launch (the trait
            // contract: `Ok(false)` means nothing was queued).
            let capable = self.ms_mla_v4_tree_capable(&c);
            {
                static ROUTE_ONCE: std::sync::Once = std::sync::Once::new();
                ROUTE_ONCE.call_once(|| {
                    tracing::info!(
                        "V4-tree route: {} (n={num_seqs} spine_rows={spine_rows})",
                        if capable {
                            "BATCHED split-cache-write"
                        } else {
                            "per-row fallback"
                        },
                    );
                });
            }
            if !capable {
                return Ok(false);
            }
            if let Some(m) = ctx.attn_metadata.as_ref() {
                c.seq_slot = m.seq_slot;
            }
            c.tree = Some(TreeSplit {
                spine_rows,
                reseed,
            });
            self.decode_multi_seq_inner_hc(c, kv_cache, ctx, stream)?;
            return Ok(true);
        }

        // MLA-without-mHC / mHC-without-MLA are not wired for the split.
        if self.mla.is_some() || self.hc.is_some() {
            return Ok(false);
        }

        let bs = kv_cache.block_size() as u32;
        let mut c = MultiSeqCtx::new(self, ctx, hidden, residual, num_seqs, bs, stream);
        if let Some(m) = ctx.attn_metadata.as_ref() {
            c.seq_slot = m.seq_slot;
        }

        // ── Phase 1: RMS norm + residual for all K_t rows ──
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

        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");

        // ── Phase 2: QKV projections (all rows, one launch set) ──
        self.ms_phase_qkv(&c)?;

        // ── Phase 3: RoPE (per-row positions from metadata) ──
        self.ms_phase_rope(&c, meta)?;

        // ── Phase 4a: cache-write bonus + spine rows → canonical blocks ──
        self.ms_phase_cache_write_range(&c, kv_cache, meta, 0, spine_rows)?;

        // ── Phase 4b: branch scratch re-seed (canonical → scratch, THIS
        // layer). Ordered after the spine writes above, before the branch
        // writes below — the scratch holds the committed prefix AND this
        // step's bonus+spine K/V at depths shallower than the fork.
        self.tree_reseed_blocks(ctx, kv_cache, reseed, stream)?;

        // ── Phase 4c: cache-write branch rows → their scratch slots ──
        self.ms_phase_cache_write_range(&c, kv_cache, meta, spine_rows, num_seqs)?;

        // ── Phase 5: batched paged decode over all K_t rows (reads KV
        // through the per-row block tables / seq_lens only) ──
        let attn_out = self.ms_phase_paged_decode(&c, kv_cache, meta)?;

        // ── Phase 6: gate multiply + O projection ──
        let o_out = self.ms_phase_o_proj(&c, attn_out)?;

        // TP all-reduce parity with decode_multi_seq_inner. The tree path
        // gates on comm == None, so this is dead on Laguna — kept so the
        // phase sequence stays a faithful mirror of the flat path.
        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        // ── Phase 7: residual + post-norm + MoE/dense FFN (all rows) ──
        self.ms_phase_ffn(&c, o_out)?;

        Ok(true)
    }
}
