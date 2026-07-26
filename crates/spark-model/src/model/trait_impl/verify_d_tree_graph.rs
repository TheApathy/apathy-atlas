// SPDX-License-Identifier: AGPL-3.0-only

//! DDTree M5 — CUDA-graph-captured tree verify (ATLAS_DFLASH_TREE_GRAPH=1).
//!
//! Same tree execution as `verify_d_tree.rs` (M2b batched layer forwards
//! with the split cache-write), restructured so the whole layer loop +
//! final norm + LM head + argmax is captured ONCE per
//! `(slot_idx, K_t, shape_id)` and REPLAYED on subsequent steps — tree
//! steps then cost the same as graphed flat steps instead of paying the
//! eager launch overhead every step.
//!
//! ## What's captured vs pre-graph
//!
//! Pre-graph (per step, host-interactive or per-step values):
//!   * gates + plan build + staleness checks (no state touched on bail),
//!   * `ensure_blocks_through_decode` for the canonical blocks,
//!   * persistent scratch-pool reservation / branch assignment,
//!   * K_t row embeds,
//!   * metadata upload (positions / slots / seq_lens / per-row block
//!     tables / seq_slot) to the flat path's `meta_base` scratch layout,
//!   * indirect re-seed args upload (`[n_pairs, src, dst, ...]`) to the
//!     model's fixed `tree_reseed_buf`.
//!
//! Captured (replayed as one graph):
//!   * per-layer `decode_multi_seq_tree` with `TreeReseed::Indirect` — the
//!     per-layer canonical→scratch re-seed runs as ONE
//!     `kv_block_indirect_copy` launch that reads src/dst block IDs (and
//!     the pair count) from `tree_reseed_buf`, stream-ordered between the
//!     spine cache-writes and the branch cache-writes exactly like the
//!     eager path's d2d copies,
//!   * the DFlash hidden captures, final norm, LM head, per-row argmax.
//!
//! ## Shape keying
//!
//! Graph cache key = `(slot_idx, K_t)` → up to `TREE_GRAPH_SHAPES_PER_KEY`
//! `(shape_id, graph)` entries, where `shape_id` (`ddtree::tree_shape_id`)
//! encodes spine_len + per-branch (fork depth, row count). Everything the
//! captured launch sequence bakes (row ranges, launch counts, buffer
//! addresses) is fixed by `(K_t, shape_id)`; per-step values (positions,
//! block IDs, reseed pairs) ride in device buffers uploaded pre-replay.
//! A shape missing from a FULL entry falls back to the eager tree path.
//!
//! ## Scratch lifecycle
//!
//! `seq.tree_scratch_pool` blocks are reserved at the first graphed tree
//! step, REUSED every tree step (contents re-seeded per layer inside the
//! graph) and freed at sequence end (`free_sequence`). Adoption of a
//! branch win must NOT donate pool blocks to the block table — see
//! `dflash_adopt_tree_branch`'s persistent path (copy scratch→canonical).

#![allow(clippy::too_many_arguments)]

use anyhow::{Result, bail};

use super::super::block_mgmt::ensure_blocks_through_decode;
use super::super::types::{TREE_GRAPH_SHAPES_PER_KEY, TransformerModel};
use super::verify_d_tree::TREE_MAX_ROWS;
use crate::layer::{AttnMetadataDev, ForwardContext, TreeReseed};
use crate::layers::dflash_head::ddtree;
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Try to execute the verify as a CUDA-graphed tree step.
    ///
    /// Returns `Ok(None)` when a gate fails — NOTHING was executed and no
    /// sequence state was touched beyond idempotent canonical-block
    /// ensures and (possibly) growing the persistent scratch pool, so the
    /// caller can run the EAGER tree path (or flat) from a clean slate.
    /// Past the commit point (branch assignment + embeds), errors
    /// propagate exactly like the eager path.
    pub(super) fn try_decode_verify_tree_graphed(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        payload: &crate::layers::DDTreePayload,
    ) -> Result<Option<Vec<u32>>> {
        let k = tokens.len();
        let kt = 1 + payload.tree_token_ids.len();
        if k == 0 || kt > TREE_MAX_ROWS || kt <= k {
            return Ok(None);
        }
        // Graph prerequisites (mirror the flat path's use_graphs gates).
        if self.kv_block_copy_kernel.0 == 0 || self.tree_reseed_buf.is_null() {
            return Ok(None); // kernel set without kv_block_indirect_copy
        }
        if self.comm.is_some() {
            return Ok(None); // TP/EP > 1
        }
        if self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(None);
        }
        if self.lora.is_some() && crate::lora::lora_eager_env() {
            return Ok(None);
        }
        if std::env::var("ATLAS_DFLASH_DEBUG_NO_GRAPH").ok().as_deref() == Some("1") {
            return Ok(None);
        }
        // ALL_BATCHED forces the per-row debug loop — host block pointers
        // per row, not capturable. PROFILE syncs mid-forward — illegal in
        // a capture.
        if std::env::var("ATLAS_DFLASH_ALL_BATCHED").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_PROFILE").is_ok()
        {
            return Ok(None);
        }
        // EVERY layer must take the batched tree path: a mid-capture
        // per-row fallback would bake per-step host block pointers into
        // the graph. (Laguna: all layers FullAttention → always true.)
        let all_multiseq = std::env::var("ATLAS_DFLASH_ALL_MULTISEQ").ok().as_deref() == Some("1");
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(layer_idx);
            let multiseq =
                layer_type == atlas_core::config::LayerType::FullAttention || all_multiseq;
            if !multiseq || !layer.tree_graph_capable() {
                return Ok(None);
            }
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
        if plan.spine_len != k - 1 || payload.tree_token_ids[..plan.spine_len] != tokens[1..] {
            return Ok(None);
        }
        if plan.branches.is_empty() {
            return Ok(None);
        }
        let Some(shape_id) = ddtree::tree_shape_id(&plan) else {
            return Ok(None); // rare shape (>4 branches / wide fields) → eager
        };

        // ── Graph-cache admission: replay a cached shape, capture a new
        // one while under the per-(slot,kt) cap, or fall back to eager for
        // shapes beyond the cap (graphs are never destroyed — a hard cap
        // beats leaky LRU eviction). Decided BEFORE any state mutation.
        let cache_key = (seq.slot_idx, kt);
        let cached_graph = {
            let cache = self.verify_tree_graph.lock();
            match cache.get(&cache_key) {
                Some(shapes) => match shapes.iter().find(|(s, _)| *s == shape_id) {
                    Some(&(_, g)) => Some(g),
                    None if shapes.len() >= TREE_GRAPH_SHAPES_PER_KEY => {
                        return Ok(None); // rare shape — eager fallback
                    }
                    None => None,
                },
                None => None,
            }
        };

        // ── Canonical block allocation (identical to the eager tree path;
        // idempotent, so an eager fallback after this is unaffected).
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

        // ── Persistent scratch pool: reserve once, grow only if a later
        // step's plan touches more blocks; REUSED every tree step and
        // freed at sequence end. Growth failure → eager fallback (the
        // pool keeps what it already holds).
        let needed: usize = plan
            .branches
            .iter()
            .map(|b| b.touched_hi - b.touched_lo + 1)
            .sum();
        if needed == 0 || needed > ddtree::TREE_RESEED_MAX_PAIRS {
            return Ok(None);
        }
        while seq.tree_scratch_pool.len() < needed {
            match kv_cache.alloc_block() {
                Ok(b) => seq.tree_scratch_pool.push(b),
                Err(_) => return Ok(None),
            }
        }

        // Reclaim leftover EAGER per-step scratch from a previous errored
        // step (persistent leftovers are pool-owned: drop only).
        if !seq.tree_branch_scratch.is_empty() {
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

        // ── Assign pool blocks to branches + build the re-seed pairs
        // (canonical_phys → scratch). Pure bookkeeping; the device sees
        // these only through per-step uploads below.
        let mut branch_scratch: Vec<Vec<(usize, u32)>> = Vec::with_capacity(plan.branches.len());
        let mut reseed_pairs: Vec<(u32, u32)> = Vec::with_capacity(needed);
        let mut cursor = 0usize;
        for b in &plan.branches {
            let mut v = Vec::with_capacity(b.touched_hi - b.touched_lo + 1);
            for ab in b.touched_lo..=b.touched_hi {
                let Some(phys) = seq.physical_block_for(ab) else {
                    return Ok(None); // canonical block missing — bail clean
                };
                let scratch = seq.tree_scratch_pool[cursor];
                cursor += 1;
                v.push((ab, scratch));
                reseed_pairs.push((phys, scratch));
            }
            branch_scratch.push(v);
        }
        let Some(reseed_meta) =
            ddtree::build_reseed_meta(&reseed_pairs, ddtree::TREE_RESEED_MAX_PAIRS)
        else {
            return Ok(None);
        };

        // ── COMMIT POINT: record the branch→scratch mapping for the
        // scheduler's walk/adopt; from here on errors propagate (the
        // scheduler marks the seq finished; free_sequence reclaims the
        // pool and drops the persistent-flagged references).
        seq.tree_branch_scratch = branch_scratch;
        seq.tree_scratch_persistent = true;
        seq.block_fork = None; // the tree supersedes the doc-16 fork

        // ── Embeds: all K_t rows (row 0 = bonus anchor, rows 1.. = nodes).
        self.embed(tokens[0], hidden, stream)?;
        for (i, &tok) in payload.tree_token_ids.iter().enumerate() {
            self.embed(tok, hidden.offset((i + 1) * h * fp32), stream)?;
        }

        // ── Per-row metadata upload — identical layout to the eager tree
        // path (meta_base: positions | seq_slot(+128) | slots(+256) |
        // seq_lens(+512) | block_table(+768)). Contents change per step;
        // the graph bakes only the ADDRESSES.
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

        // SAFETY: POD integer Vecs reinterpreted as byte slices for H2D
        // upload — same contract as verify_d.rs / verify_d_tree.rs.
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

        // ── Indirect re-seed args upload (pair count + src/dst block ids)
        // to the model's fixed buffer — the ONE thing that replaces the
        // eager path's per-layer host-pointer d2d copies.
        let rm_bytes = unsafe {
            std::slice::from_raw_parts(reseed_meta.as_ptr() as *const u8, reseed_meta.len() * 4)
        };
        self.gpu
            .copy_h2d_async(rm_bytes, self.tree_reseed_buf, stream)?;

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
            profile: false, // prof! syncs are illegal mid-capture
            comm: self.comm_ref(),
            graph_capture: true,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        };

        let capture_all = std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_DFLASH_UNIFIED_CTX").ok().as_deref() == Some("1");
        let spine_end = plan.spine_len + 1;
        let reseed = TreeReseed::Indirect {
            kernel: self.kv_block_copy_kernel,
            meta: self.tree_reseed_buf,
            max_pairs: ddtree::TREE_RESEED_MAX_PAIRS as u32,
        };

        // ── Capture / replay ──
        if let Some(graph) = cached_graph {
            self.gpu.launch_graph(graph, stream)?;
        } else {
            self.gpu.begin_capture(stream)?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let batched = layer.decode_multi_seq_tree(
                    hidden,
                    residual,
                    kt,
                    &mut kv_cache,
                    spine_end,
                    &reseed,
                    &ctx,
                    stream,
                )?;
                if !batched {
                    // Pre-checked above — reaching here means the layer's
                    // capability predicate lied; the capture is poisoned.
                    bail!(
                        "tree graph capture: layer {layer_idx} lost its batched tree path \
                         mid-capture"
                    );
                }
                // DFlash intermediate hidden capture — inside the graph,
                // mirroring the flat path (verify_d.rs).
                if capture_all {
                    self.try_dflash_capture_all(layer_idx, kt, stream)?;
                } else {
                    self.try_dflash_capture(layer_idx, k - 1, stream)?;
                }
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

            // LM head + per-row argmax (fixed scratch addresses — graph-safe).
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

            let graph = self.gpu.end_capture(stream)?;
            if graph.0 != 0 {
                tracing::info!(
                    "Captured CUDA graph for DFlash TREE verify (slot={} K_t={} shape={:#x})",
                    seq.slot_idx,
                    kt,
                    shape_id,
                );
                self.verify_tree_graph
                    .lock()
                    .entry(cache_key)
                    .or_default()
                    .push((shape_id, graph));
                self.gpu.launch_graph(graph, stream)?;
            }
            // graph.0 == 0: backend without capture (mock) already executed
            // the work eagerly during "capture" — results are valid, no
            // caching (mirrors verify_d.rs).
        }

        // ── Post-graph: D2H + flat bookkeeping (identical to eager tail).
        let argmax_out = self.buffers.scratch();
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

        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        static TREE_GRAPH_DBG: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let n = TREE_GRAPH_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n <= 4 || n % 256 == 0 {
            tracing::info!(
                "DFLASH_TREE_GRAPH verify #{n}: K_t={} (k={} + {} branch rows, {} branches) \
                 shape={:#x} {}",
                kt,
                k,
                kt - k,
                plan.branches.len(),
                shape_id,
                if cached_graph.is_some() {
                    "replay"
                } else {
                    "capture"
                },
            );
        }

        Ok(Some(out))
    }
}
