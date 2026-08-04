// SPDX-License-Identifier: AGPL-3.0-only

//! DDTree M2/M2b — tree-shaped K=γ (DFlash) verify execution.
//!
//! Executes ONE verify forward over K_t tree rows (row 0 = bonus, rows
//! 1..=S = spine, then contiguous branch runs). Bonus+spine rows are
//! byte-identical to the flat `verify_d.rs` path (canonical KV blocks, same
//! positions/slots); branch rows use the SAME position arithmetic but their
//! touched blocks are remapped to per-branch copy-on-write scratch blocks.
//!
//! Intra-step ancestor visibility (the one subtle part): a branch row at
//! depth d must see THIS STEP's bonus+spine K/V written at depths 0..d-1
//! into the canonical block.
//!
//! M2b (batched execution, mirrors the flat path's layer gating): layers
//! the flat verify batches through `decode_multi_seq` (FullAttention
//! always; sliding too under `ATLAS_DFLASH_ALL_MULTISEQ=1`) run ONE
//! batched K_t-row forward via `decode_multi_seq_tree`, whose cache-write
//! phase is split into two row ranges — rows `[0, spine_end)` → canonical,
//! then the branch scratch re-seed d2d copies (canonical → scratch, current
//! layer), then rows `[spine_end, K_t)` → scratch. All other phases
//! (qkv/rope/paged-decode/oproj/ffn) are single-launch over all K_t rows;
//! the paged decode reads KV only through the per-row metadata views, and
//! all of this layer's writes are stream-ordered before it. This removes
//! the per-row GEMV weight re-reads that made M2's all-per-row execution
//! 3.6× the flat verify cost.
//!
//! Layers the flat path runs per-row (sliding layers without ALL_MULTISEQ;
//! `ATLAS_DFLASH_ALL_BATCHED=1` debug; MLA/mHC) keep M2's per-row
//! sequential decode: rows in ascending order, and immediately BEFORE each
//! branch's first row the touched canonical K/V block(s) are d2d-copied
//! into that branch's scratch block(s) for the CURRENT layer — the copy is
//! ordered on the same stream after the spine rows' writes, so it captures
//! this step's spine K/V. Either way, tree cost tracks the flat path's
//! layer-for-layer execution mode.
//!
//! CUDA graphs: THIS module's execution is EAGER — it never touches
//! `begin_capture`/`launch_graph`, so cached flat-step graphs are
//! undisturbed and flat steps stay bit-identical. M5
//! (ATLAS_DFLASH_TREE_GRAPH=1, default off): eligible tree steps are
//! routed to the CUDA-graphed variant in `verify_d_tree_graph.rs` FIRST;
//! any gate failure there falls back to this eager path unchanged.
//!
//! ## Safety
//!
//! `unsafe { from_raw_parts(...) }` blocks reinterpret `Vec`s of POD
//! integers (`u32`, `i32`, `i64`) as byte slices for H2D upload — same
//! contract as `verify_d.rs` / `verify_c.rs`.

#![allow(clippy::too_many_arguments)]

use anyhow::Result;

use super::super::block_mgmt::ensure_blocks_through_decode;
use super::super::types::TransformerModel;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::dflash_head::ddtree;
use crate::layers::ops;
use crate::traits::SequenceState;

/// K_t arena cap — the DFlash K=γ buffers (`sizes.rs` dflash_k) hold 20 rows.
pub(super) const TREE_MAX_ROWS: usize = 20;

/// M5 env gate: `ATLAS_DFLASH_TREE_GRAPH=1` routes eligible tree steps
/// through the CUDA-graphed verify (`verify_d_tree_graph.rs`). Default off
/// — this module's eager execution, byte-identical to before.
pub(super) fn tree_graph_env() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_DFLASH_TREE_GRAPH").ok().as_deref() == Some("1"))
}

impl TransformerModel {
    /// Try to execute the verify as a tree (DDTree M2).
    ///
    /// Returns `Ok(None)` when a gate fails — the caller MUST degrade to the
    /// flat path (nothing was executed, no state was touched beyond freeing
    /// any partially-allocated scratch). Returns `Ok(Some(argmax_rows))` on
    /// success: K_t = 1 + payload.len() rows in the TREE frame (row 0 +
    /// spine rows are exactly the flat frame; branch rows follow).
    ///
    /// Gates (fall back to flat): HSS engaged, TP>1, LoRA-eager, K_t > 20,
    /// malformed/stale payload (spine ≠ drafts), no branches, scratch alloc
    /// failure.
    pub(super) fn try_decode_verify_tree(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        payload: &crate::layers::DDTreePayload,
    ) -> Result<Option<Vec<u32>>> {
        // ── M5 (ATLAS_DFLASH_TREE_GRAPH=1): try the CUDA-graphed tree
        // verify first. `Ok(None)` from it (kernel missing, HSS, unusual
        // shape, shape-cache full, scratch-pool exhaustion, …) falls
        // through to THIS eager path — the graphed fn mutates no sequence
        // state before its commit point, so the eager execution below
        // starts from the same state as an env-off step.
        if tree_graph_env()
            && let Some(out) = self.try_decode_verify_tree_graphed(tokens, seq, payload)?
        {
            return Ok(Some(out));
        }

        let k = tokens.len();
        let kt = 1 + payload.tree_token_ids.len();
        if k == 0 || kt > TREE_MAX_ROWS || kt <= k {
            // kt <= k ⇒ no branch rows beyond the flat frame — pointless.
            return Ok(None);
        }
        if self.comm.is_some() {
            return Ok(None); // TP/EP > 1
        }
        if self.lora.is_some() && crate::lora::lora_eager_env() {
            return Ok(None);
        }

        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = 2usize;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();
        if kv_cache.config().cache_blocks_per_seq.is_some() {
            return Ok(None); // HSS engaged — host I/O + window offsets
        }
        let bs = kv_cache.block_size();
        let base = seq.seq_len;

        let Some(plan) = ddtree::build_tree_verify_plan(payload, base, bs) else {
            return Ok(None);
        };
        // Staleness guard: the payload's spine must be exactly this step's
        // draft chain (tokens[1..]); otherwise the tree frame and the
        // scheduler's flat walk would disagree.
        if plan.spine_len != k - 1 || payload.tree_token_ids[..plan.spine_len] != tokens[1..] {
            return Ok(None);
        }
        if plan.branches.is_empty() {
            return Ok(None); // pure flat payload — flat path is strictly better
        }

        // ── Canonical block allocation (identical to the flat path, extended
        // through the deepest row position — branch depths never exceed the
        // spine by more than one block's worth in practice, but stay exact).
        let max_depth = plan.max_depth();
        for d in 0..=max_depth {
            let pos = base + d;
            let blocks_needed = (pos / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // ── Per-branch copy-on-write scratch allocation. Sibling branches get
        // DISTINCT scratch blocks. On alloc failure: free what we got and
        // fall back to flat (never panic on the tree path).
        if !seq.tree_branch_scratch.is_empty() {
            // Leftover scratch from a previous errored step — reclaim it
            // rather than leak (never panic on the tree path). M5: entries
            // referencing the PERSISTENT pool are only dropped, never freed
            // — the pool owns those blocks (free_sequence reclaims them).
            if seq.tree_scratch_persistent {
                seq.tree_branch_scratch.clear();
                seq.tree_scratch_persistent = false;
            } else {
                for branch in std::mem::take(&mut seq.tree_branch_scratch) {
                    for (_, s) in branch {
                        kv_cache.free_block(s);
                    }
                }
            }
        }
        let mut branch_scratch: Vec<Vec<(usize, u32)>> = Vec::with_capacity(plan.branches.len());
        let mut alloc_failed = false;
        'alloc: for b in &plan.branches {
            let mut v = Vec::with_capacity(b.touched_hi - b.touched_lo + 1);
            for ab in b.touched_lo..=b.touched_hi {
                if seq.physical_block_for(ab).is_none() {
                    alloc_failed = true; // canonical block missing — bail
                }
                match kv_cache.alloc_block() {
                    Ok(s) => v.push((ab, s)),
                    Err(_) => alloc_failed = true,
                }
                if alloc_failed {
                    branch_scratch.push(v);
                    break 'alloc;
                }
            }
            branch_scratch.push(v);
        }
        if alloc_failed {
            for branch in branch_scratch {
                for (_, s) in branch {
                    kv_cache.free_block(s);
                }
            }
            return Ok(None);
        }
        // Record for the scheduler's dflash_adopt_tree_branch and the
        // free_sequence leak backstop. NOTE: scratch is NOT seeded here —
        // the per-layer loop below seeds each branch right before its first
        // row so the copy captures THIS step's bonus+spine writes.
        seq.tree_branch_scratch = branch_scratch;
        // The tree supersedes the doc-16 block fork for this call.
        seq.block_fork = None;

        // ── From here on, errors propagate (GPU state is being mutated);
        // the scheduler marks the seq finished and free_sequence reclaims
        // the scratch blocks.

        // Embed all K_t rows: row 0 = bonus anchor, rows 1.. = payload nodes.
        self.embed(tokens[0], hidden, stream)?;
        for (i, &tok) in payload.tree_token_ids.iter().enumerate() {
            self.embed(tok, hidden.offset((i + 1) * h * fp32), stream)?;
        }

        // ── Per-row metadata (host-built), uploaded to the flat path's
        // meta_base scratch layout: positions | seq_slot(+128) | slots(+256)
        // | seq_lens(+512) | block_table(+768).
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;
        let mb = max_blocks as usize;
        let md = ddtree::build_tree_row_metadata(
            &plan,
            base,
            bs,
            &seq.block_table,
            &seq.tree_branch_scratch,
        );

        let pos_bytes =
            unsafe { std::slice::from_raw_parts(md.positions.as_ptr() as *const u8, kt * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let slot_bytes =
            unsafe { std::slice::from_raw_parts(md.slots.as_ptr() as *const u8, kt * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        let sl_bytes =
            unsafe { std::slice::from_raw_parts(md.seq_lens.as_ptr() as *const u8, kt * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        let mut bt_buf = vec![0i32; kt * mb];
        for (row, table) in md.block_tables.iter().enumerate() {
            for (j, &block) in table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, kt * mb * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        debug_assert!(kt <= 32, "tree verify seq_slot +128 gap holds K ≤ 32");
        let seq_slot =
            self.upload_seq_slot_uniform(seq.adapter_slot, kt, meta_base.offset(128), stream)?;

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: kt as u32,
            seq_slot,
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: std::env::var("ATLAS_PROFILE").is_ok(),
            comm: self.comm_ref(),
            graph_capture: false, // tree steps are ALWAYS eager
            gdn_exact_replay: false,
            // Hash-MoE routing reads `tid2eid[token_id]` per row, so the
            // K_t row tokens must be resident exactly as the flat path
            // uploads its `tokens_all` (verify_d.rs). Leaving this None made
            // DeepSeek-V4 abort the whole verify with "hash-MoE layer
            // requires ForwardContext.token_ids (decode)" — the tree had
            // simply never run on a hash-routed model.
            token_ids: Some(self.buffers.token_ids()),
            routed_lora_layers: None,
            midchunk_capture: None,
        };
        // Row 0 = bonus anchor, rows 1.. = payload nodes — the same order the
        // embed loop above uses, so row t's id sits at offset t.
        let tid_bytes: Vec<u8> = std::iter::once(tokens[0])
            .chain(payload.tree_token_ids.iter().copied())
            .flat_map(|t| t.to_le_bytes())
            .collect();
        self.gpu
            .copy_h2d_async(&tid_bytes, self.buffers.token_ids(), stream)?;

        let capture_all = std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_UNIFIED_CTX").ok().as_deref() == Some("1");

        // ── M2b batched-execution gating: mirror the flat path's per-layer
        // choice (verify_d.rs) so tree cost ≈ flat cost. HSS is already
        // gated off above; all_batched is the flat path's per-row debug
        // hatch.
        let all_multiseq = std::env::var("ATLAS_DFLASH_ALL_MULTISEQ").ok().as_deref() == Some("1");
        let all_batched = std::env::var("ATLAS_DFLASH_ALL_BATCHED").ok().as_deref() == Some("1");
        // Rows [0, spine_end) = bonus + spine (canonical blocks); rows
        // [spine_end, kt) = branch rows (scratch blocks).
        let spine_end = plan.spine_len + 1;
        // Flattened (canonical_phys, scratch) block pairs for the batched
        // path's per-layer re-seed. The block table is final here (all
        // positions were ensured above), so compute once.
        let reseed_pairs: Vec<(u32, u32)> = seq
            .tree_branch_scratch
            .iter()
            .flat_map(|branch| {
                branch
                    .iter()
                    .map(|&(ab, scratch)| (seq.physical_block_for(ab).unwrap_or(0), scratch))
            })
            .collect();

        // ── Layer loop. Batched layers take ONE K_t-row forward with the
        // split cache write (spine → seed → branch); the rest keep the M2
        // per-row sequential decode, rows in ascending row-major order so
        // ancestors' K/V lands before descendants read it.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(layer_idx);
            let use_multiseq = !all_batched
                && (layer_type == atlas_core::config::LayerType::FullAttention || all_multiseq);
            let batched = use_multiseq
                && layer.decode_multi_seq_tree(
                    hidden,
                    residual,
                    kt,
                    &mut kv_cache,
                    spine_end,
                    &crate::layer::TreeReseed::HostPairs(&reseed_pairs),
                    &ctx,
                    stream,
                )?;
            if !batched {
                // Per-row sequential fallback (M2 core) — sliding layers
                // without ALL_MULTISEQ, ALL_BATCHED debug, MLA/mHC.
                for t in 0..kt {
                    // Re-seed a branch's scratch from canonical for the
                    // CURRENT layer immediately before its first row: the
                    // copy is queued after this layer's bonus+spine rows
                    // (0..t) wrote their K/V, so the scratch holds the
                    // committed prefix AND this step's spine writes at
                    // depths shallower than the fork.
                    if let Some(bidx) = plan.branch_starting_at_row(t) {
                        let kb = kv_cache.config().k_block_bytes_for_layer(layer_idx);
                        let vb = kv_cache.config().v_block_bytes_for_layer(layer_idx);
                        for &(ab, scratch) in &seq.tree_branch_scratch[bidx] {
                            let phys = seq.physical_block_for(ab).unwrap_or(0);
                            self.gpu.copy_d2d_async(
                                kv_cache.k_cache_ptr(layer_idx, phys),
                                kv_cache.k_cache_ptr(layer_idx, scratch),
                                kb,
                                stream,
                            )?;
                            self.gpu.copy_d2d_async(
                                kv_cache.v_cache_ptr(layer_idx, phys),
                                kv_cache.v_cache_ptr(layer_idx, scratch),
                                vb,
                                stream,
                            )?;
                        }
                    }
                    // Per-row metadata view (see verify_d.rs ROOT-CAUSE FIX:
                    // the single-token decode consumes ctx.attn_metadata, so
                    // each row needs its own device-offset view).
                    let meta_t = AttnMetadataDev {
                        positions: metadata.positions.offset(t * 4),
                        positions_h: metadata.positions_h.offset(t * 4),
                        positions_w: metadata.positions_w.offset(t * 4),
                        slot: metadata.slot.offset(t * 8),
                        seq_len: metadata.seq_len.offset(t * 4),
                        block_table: metadata.block_table.offset(t * mb * 4),
                        max_blocks_per_seq: metadata.max_blocks_per_seq,
                        num_seqs: 1,
                        seq_slot: if metadata.seq_slot.0 != 0 {
                            metadata.seq_slot.offset(t * 4)
                        } else {
                            metadata.seq_slot
                        },
                    };
                    let ctx_t = ForwardContext {
                        buffers: ctx.buffers,
                        gpu: ctx.gpu,
                        config: ctx.config,
                        attn_metadata: Some(meta_t),
                        profile: ctx.profile,
                        comm: ctx.comm,
                        graph_capture: false,
                        gdn_exact_replay: ctx.gdn_exact_replay,
                        // Per-row view, same reason as the metadata offsets
                        // above: this decode sees num_seqs=1, so hash-MoE
                        // would route EVERY row on row 0's token without it.
                        token_ids: ctx.token_ids.map(|p| p.offset(t * 4)),
                        routed_lora_layers: ctx.routed_lora_layers,
                        midchunk_capture: None,
                    };
                    layer.decode(
                        hidden.offset(t * h * bf16),
                        residual.offset(t * h * bf16),
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        base + plan.row_depths[t],
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx_t,
                        stream,
                    )?;
                }
            }
            // DFlash intermediate hidden capture — mirrors verify_d.rs. Rows
            // 0..k-1 (bonus+spine) are the flat frame the scheduler's ctx
            // append consumes; extra branch rows past the arena cap are
            // clamped inside try_dflash_capture_all.
            if capture_all {
                self.try_dflash_capture_all(layer_idx, kt, stream)?;
            } else {
                self.try_dflash_capture(layer_idx, k - 1, stream)?;
            }
            // DSpark capture: the FLAT frame's k rows at their sequence
            // positions, exactly as verify_d.rs does it. Omitting this was the
            // tree's accept collapse (0.50 vs the flat path's 1.33 tok/step):
            // the DSpark drafter seeds its ring from `dspark_dump_buf` at
            // `base..base+k`, and a tree step that never writes those rows
            // leaves the next propose reading whatever the last flat step (or
            // prefill) left there. Every step builds a payload under
            // TREE_DEGEN, so the ring went stale immediately and the drafts
            // were near-garbage from the first tree step on. Rows past k are
            // branch rows — they live at DUPLICATE depths, so writing them
            // here would clobber the spine's hiddens at the same positions.
            // `base`, not `seq.seq_len`: the tail below advances seq_len by k.
            self.try_dspark_capture(layer_idx, k, base, false, stream)?;
        }

        // Final norm over ALL K_t rows.
        let normed = self.buffers.norm_output();
        ops::rms_norm(
            self.gpu.as_ref(),
            self.rms_norm_kernel,
            hidden,
            &self.final_norm,
            normed,
            kt as u32,
            h as u32,
            self.config.rms_norm_eps as f32,
            stream,
        )?;

        // LM head + per-row argmax over all K_t rows.
        self.lm_head_batched(normed, kt as u32, self.buffers.logits(), stream)?;
        let vocab = self.config.vocab_size;
        let argmax_out = self.buffers.scratch();
        for t in 0..kt {
            let logits_t = self.buffers.logits().offset(t * vocab * bf16);
            ops::argmax_bf16(
                self.gpu.as_ref(),
                self.argmax_kernel,
                logits_t,
                argmax_out.offset(t * 4),
                vocab as u32,
                stream,
            )?;
        }

        // D2H.
        let mut buf = vec![0u8; kt * 4];
        self.gpu.copy_d2h(argmax_out, &mut buf)?;
        let mut out = Vec::with_capacity(kt);
        for t in 0..kt {
            let off = t * 4;
            out.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        // Flat bookkeeping: only the bonus+spine chain advances the sequence
        // (branch rows live purely in scratch and are trimmed by the
        // scheduler). Identical to the flat path's tail.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        static TREE_DBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = TREE_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 4 || n % 256 == 0 {
            tracing::info!(
                "DFLASH_TREE verify #{n}: K_t={} (k={} + {} branch rows, {} branches) eager \
                 M2b batched (all_multiseq={all_multiseq}, all_batched={all_batched})",
                kt,
                k,
                kt - k,
                plan.branches.len(),
            );
        }

        Ok(Some(out))
    }
}
