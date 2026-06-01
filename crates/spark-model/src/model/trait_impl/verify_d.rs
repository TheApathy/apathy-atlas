// SPDX-License-Identifier: AGPL-3.0-only

//! K=γ (DFlash) verify path.
//!
//! ## Safety
//!
//! `unsafe { from_raw_parts(...) }` blocks reinterpret stack arrays
//! / `Vec`s of POD integers (`u32`, `i32`, `i64`, `usize`) as byte
//! slices for H2D upload. See `verify_c.rs` module docs for the full
//! safety contract — same pattern, same invariants here.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn decode_verify_graphed_kgamma_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let k = tokens.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        // F62 (2026-04-27): SpecMamba dual-buffer pre-verify copy.
        self.pre_verify_copy_async(seq)?;

        // ATLAS_FULL_PROFILE=1: bump per-step counter so per-kernel kprof!
        // accumulators fire on K=γ verify. begin_step() early-returns when
        // env var is unset — see full_profile.rs:103-111 for the safety
        // contract (must NOT flip ACTIVE while CUDA graph capture is live).
        crate::full_profile::begin_step();

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1: Pre-graph (varies per step, NOT captured) ──
        //
        // DFS reorder (option C, ATLAS_DDTREE_DFS_REORDER=1):
        // Permute the K kernel-frame tokens into DFS pre-order so each
        // ancestor chain is contiguous in slot order. The paged-decode
        // attention kernel iterates [0..seq_lens[t]) sequentially — DFS
        // reorder ensures that the deepest ancestor chain (the one
        // greedy_sample_ddtree will walk) reads its TRUE ancestors,
        // not sibling/cousin slots. Sibling root children at the end of
        // the DFS order get short seq_lens and only read the bonus +
        // already-visited subtree — wrong but acceptable because the
        // greedy walker only ever commits one chain.
        //
        // `dfs_perm[i] = j` means "DFS slot i contains kernel slot j".
        // `dfs_inv_perm[j] = i` means "kernel slot j is at DFS slot i".
        // Empty when DFS reorder is disabled or no tree payload is active.
        let dfs_enabled =
            std::env::var("ATLAS_DDTREE_DFS_REORDER").ok().as_deref() == Some("1");
        let (dfs_perm, dfs_inv_perm, dfs_depths): (Vec<usize>, Vec<usize>, Vec<usize>) = {
            let host_parents_lock = self.ddtree_parent_ids_host.lock();
            // Allow payload shorter than k (we pad with linear-chain tail to
            // match k below). Only require non-empty payload.
            if dfs_enabled && !host_parents_lock.is_empty() && host_parents_lock.len() <= k {
                use crate::layers::dflash_head::ddtree::dfs_reorder;
                // Pad up to k if payload shorter (mirror the linear-chain
                // padding logic from the existing depth derivation).
                let mut hp = host_parents_lock.clone();
                drop(host_parents_lock);
                while hp.len() < k {
                    let next_chain_parent = hp.len() as i32 - 1;
                    hp.push(next_chain_parent);
                }
                let (perm, inv, depths) = dfs_reorder(&hp);
                // If perm is identity (flat chain or trivially-flat payload),
                // skip the reorder machinery — it's a no-op and the existing
                // chain path is already optimal.
                let is_identity = perm.iter().enumerate().all(|(i, &j)| i == j);
                if is_identity {
                    (Vec::new(), Vec::new(), Vec::new())
                } else {
                    static DFS_DBG_DONE: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !DFS_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        tracing::info!(
                            "ATLAS_DDTREE_DFS_REORDER: k={k} perm={:?} depths={:?}",
                            perm, depths
                        );
                    }
                    (perm, inv, depths)
                }
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            }
        };
        let dfs_active = !dfs_perm.is_empty();

        // ATLAS_DDTREE_DFS_REORDER_METADATA_ONLY=1: skip the token permutation
        // (embedding stays in linear-drafts order) and only permute attention
        // metadata + parent_ids. Use case: when the verify input `tokens`
        // is the LINEAR top-1 drafts (current Atlas plumbing), not the
        // tree's tree_token_ids. Permuting linear drafts into DFS slots
        // places semantically wrong tokens (e.g. drafts[5] at the chain-child
        // slot of drafts[0]) and the model output becomes incoherent.
        //
        // With METADATA_ONLY=1 we keep token-at-slot consistency for
        // attention (preserves the existing chain accept behavior) while
        // letting depth-based RoPE positions still fire. This is a
        // research-only mode used to A/B the metadata permutation alone.
        let dfs_permute_tokens =
            std::env::var("ATLAS_DDTREE_DFS_REORDER_METADATA_ONLY").ok().as_deref()
                != Some("1");

        // 1a. Embed K tokens — in DFS order if reorder active AND
        // METADATA_ONLY=0 (default).
        crate::kprof!(self.gpu.as_ref(), stream, "embed", {
            for t in 0..k {
                let src_token = if dfs_active && dfs_permute_tokens {
                    tokens[dfs_perm[t]]
                } else {
                    tokens[t]
                };
                self.embed(src_token, hidden.offset(t * h * fp32), stream)?;
            }
            anyhow::Result::<()>::Ok(())
        })?;

        // 1b. Allocate KV blocks for all K positions
        let bs = kv_cache.block_size();
        for t in 0..k {
            let pos = seq.seq_len + t;
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

        // 1c. Upload K-entry attention metadata. Layout in scratch (after
        // mtp metadata reservation): positions[K*4] | slots[K*8] | seq_lens[K*4]
        // | block_table[K*max_blocks*4]. Need K*16 + K*max_blocks*4 bytes per
        // call — at K=17 max_blocks=512 that's ~36 KB which fits comfortably
        // in the scratch arena (offset 32768).
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        // Tree-aware metadata (ATLAS_DDTREE_TREE_AWARE_VERIFY=1):
        //
        // The chain-mode meta below treats the γ+1 verify tokens as a FLAT
        // chain (positions = seq.seq_len + t, seq_lens = seq.seq_len + t + 1).
        // That is wrong for the real top-K tree topology produced by M4B v2:
        //
        //   - Sibling root children (e.g. compact 0..4 with parent=-1) get
        //     DIFFERENT RoPE positions (1, 2, 3, 4, 5) when they are all at
        //     tree depth 1 and should share position 1. RoPE mis-alignment
        //     skews query directions per sibling and degrades the per-position
        //     argmax accuracy.
        //   - The per-row seq_lens upper bound also grows with compact index,
        //     so a sibling at compact index 4 reads positions [seq.seq_len ..
        //     seq.seq_len + 4] = bonus + 4 sibling KVs (not its ancestor chain).
        //
        // The full fix (per-row KV indirection so each query attends only to
        // its ancestors) requires modifying the paged-decode attention kernel
        // — see paged_decode_attn_nvfp4.cu / paged_decode_attn_fp8.cu, which
        // today iterate positions [0..seq_lens[t]) and read in-block offset
        // `pos % bs` with no slot-indirection knob. Until that lands,
        // ATLAS_DDTREE_TREE_AWARE_VERIFY=1 implements a *partial* fix that
        // produces coherent output without changing tok/s vs flat chain:
        //
        //   - positions[t] = seq.seq_len + depth[t]   (depth-based RoPE so
        //     sibling root children share RoPE position 1 instead of 1..N).
        //   - seq_lens[t]  unchanged (still compact-index-based). Setting it
        //     to depth-based caused deep chain queries to skip legitimate
        //     prior chain context and drop accept rate even further.
        //
        // Slots are still written at compact-index physical positions so the
        // KV cache stays consistent for the post-verify commit path. Block-
        // tables are unchanged (the same full sequence row replicated per t).
        //
        // RESULT: B (tree-aware) text == A (flat chain) text on essay prompt
        //         but tok/s does NOT clear A — needs the kernel-level KV
        //         indirection to land for a real win on non-flat topologies.
        //
        // depth[t] is derived from the kernel-frame parent_ids stashed by
        // set_ddtree_parent_ids — index 0 is the bonus (depth 0), index i+1
        // is draft i. parent[i] = -1 means "child of pre-tree state" (= bonus,
        // depth 1); parent[i] = k means depth[t] = 1 + depth[k].
        let tree_aware_enabled =
            std::env::var("ATLAS_DDTREE_TREE_AWARE_VERIFY").ok().as_deref() == Some("1");
        let tree_depths: Option<Vec<usize>> = if tree_aware_enabled {
            // host_parents may be shorter than k when the tree payload has
            // fewer nodes than γ_eff (e.g. budget < γ_eff caps the tree at
            // budget). The persistent device buffer is padded by
            // clear_ddtree_parent_ids' linear-chain default — replicate the
            // same padding here so depth lookups don't OOB.
            let mut host_parents = self.ddtree_parent_ids_host.lock().clone();
            if !host_parents.is_empty() && host_parents.len() < k {
                let last_payload_idx = host_parents.len() as i32 - 1;
                while host_parents.len() < k {
                    let next_chain_parent = host_parents.len() as i32 - 1;
                    let _ = last_payload_idx;
                    host_parents.push(next_chain_parent);
                }
            }
            if host_parents.len() == k && !host_parents.is_empty() {
                // Build kernel-frame depths. Bonus (index 0) is depth 0.
                let mut depths = vec![0usize; k];
                // For index i >= 1: parent < 0 means "attaches to pre-tree
                // state" — treat as a child of the bonus (depth 1). Otherwise
                // parent kernel slot p (0 = bonus) means depth = 1 + depth[p].
                for i in 1..k {
                    let p = host_parents[i];
                    if p < 0 {
                        depths[i] = 1;
                    } else {
                        let pi = p as usize;
                        if pi >= i {
                            // Defensive: malformed payload → fall back to
                            // chain depth so we don't panic. Set to compact
                            // index so downstream stays self-consistent.
                            depths[i] = i;
                        } else {
                            depths[i] = depths[pi].saturating_add(1);
                        }
                    }
                }
                // Refuse to apply when the payload is degenerate (flat chain)
                // — chain mode is already optimal.
                let is_flat_chain = (1..k).all(|i| depths[i] == i);
                if is_flat_chain {
                    None
                } else {
                    static TREE_AWARE_DBG_DONE: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !TREE_AWARE_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        tracing::info!(
                            "ATLAS_DDTREE_TREE_AWARE_VERIFY: k={k} depths={:?}",
                            depths
                        );
                    }
                    Some(depths)
                }
            } else {
                None
            }
        } else {
            None
        };

        // ATLAS_TREE_AWARE_ATTN: full per-row KV indirection. Activated
        // when the env var is set AND tree mode is producing non-flat
        // depths AND DFS reorder is OFF (DFS already provides contiguous
        // ancestor reads, so the two should not be combined).
        //
        // STATUS (2026-05-21): correctness verified end-to-end (coherent
        // outputs, ancestor chains built correctly per log) but the kernel
        // single-position fallback in the tree window is ~3.6x slower than
        // the batched chain-mode path on the qwen3.6-27b config (K=γ=17,
        // depth up to 14, 16 q_heads × 17 rows = 272 CTAs each walking
        // 1-14 single-position iterations). The original chain path uses
        // BC=4 batched loads which dominate even though tree mode reads
        // fewer total KV positions.
        //
        // Default OFF until the kernel single-position path is batched
        // (e.g. by re-introducing BC=4 over indirected ancestors, which
        // requires same-block packing — non-trivial because ancestor slots
        // are scattered across blocks for deep trees).
        let tree_aware_attn_enabled =
            std::env::var("ATLAS_TREE_AWARE_ATTN").ok().as_deref() == Some("1");
        let tree_kv_active = tree_aware_attn_enabled
            && !dfs_active
            && tree_depths.is_some()
            && self.tree_kv_indir_stride > 0
            && self.tree_kv_indir_persistent.0 != 0
            && k <= self.tree_kv_indir_stride;

        let positions: Vec<u32> = if dfs_active {
            // DFS slot i contains kernel slot dfs_perm[i]; its tree depth
            // (kernel frame) is dfs_depths[dfs_perm[i]]. RoPE position is
            // seq.seq_len + depth so siblings at the same depth share
            // a position.
            (0..k)
                .map(|t| (seq.seq_len + dfs_depths[dfs_perm[t]]) as u32)
                .collect()
        } else if let Some(ref depths) = tree_depths {
            // Kernel slot 0 (bonus) is depth 0 → position seq.seq_len (the
            // last_token's slot). For drafts at depth d, position is
            // seq.seq_len + d.
            (0..k)
                .map(|t| (seq.seq_len + depths[t]) as u32)
                .collect()
        } else {
            (0..k).map(|t| (seq.seq_len + t) as u32).collect()
        };
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        let mut slots = vec![0i64; k];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        // 256-byte gap mirrors K=4 layout for ABI compatibility with
        // attention kernels that index meta_base + fixed offsets.
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        // seq_lens stays compact-index-based even in tree-aware mode. Reasoning:
        // setting seq_lens to depth+1 makes a depth-d query read positions
        // [0..seq.seq_len+d] which still are compact-ordered slots — those
        // are sibling/cousin slots, not actual ancestors, for any non-flat
        // tree. Without per-row block_table indirection (next TODO), the
        // chain-based seq_len is no worse than depth-based and avoids
        // dropping legitimate chain context inside ancestor-only subtrees.
        //
        // DFS reorder changes this: in DFS pre-order, when we visit a node at
        // depth d, the immediate ancestor chain (root → … → us) occupies the
        // d slots immediately preceding the current one (their stack frames).
        // So `seq_lens[t] = seq.seq_len + depth[dfs_perm[t]] + 1` gives a
        // depth-d query exactly its ancestor chain (the bonus + d ancestors),
        // *contiguously* in slot order. Sibling subtrees that were already
        // visited (and would otherwise contaminate the read) are skipped
        // because seq_len cuts off the read before reaching them.
        let seq_lens: Vec<i32> = if dfs_active {
            (0..k)
                .map(|t| (seq.seq_len + dfs_depths[dfs_perm[t]] + 1) as i32)
                .collect()
        } else if tree_kv_active {
            // ATLAS_TREE_AWARE_ATTN: kernel only walks the ancestor chain
            // for the tree window, so seq_len = prior_context + depth[t] + 1.
            // The kernel remaps positions ≥ seq.seq_len via the indirection
            // table so they read the right ancestors, not sibling slots.
            let depths = tree_depths.as_ref().unwrap();
            (0..k)
                .map(|t| (seq.seq_len + depths[t] + 1) as i32)
                .collect()
        } else {
            (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect()
        };
        let sl_bytes = unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
        self.gpu
            .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

        let mb = max_blocks as usize;
        let needed = k * mb;
        let mut bt_buf = vec![0i32; needed];
        for row in 0..k {
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[row * mb + j] = block as i32;
            }
        }
        let bt_bytes =
            unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
        self.gpu
            .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

        // ATLAS_TREE_AWARE_ATTN CUDA graph fix: upload `seq.seq_len as i32`
        // into the persistent 1×i32 device buffer that backs the kernel's
        // `kv_indir_base_ptr` / `abs_base_ptr` args. The captured K=γ verify
        // graph reads this via pointer indirection so the fresh value lands
        // on every replay (previously the scalar was baked into the kernel-
        // launch node at capture time and went stale → wrong `local_off`).
        // Upload happens BEFORE `begin_capture` (same pattern as `seq_lens`),
        // so it stays outside the graph.
        // Stamp the per-step base value into the persistent pinned-host
        // shadow then fire the async H2D copy. The kernels read this via
        // `kv_indir_base_ptr` / `abs_base_ptr` so captured CUDA graphs
        // pick up the fresh value on each replay (the prior scalar arg was
        // baked into the kernel-launch node at capture time).
        if tree_kv_active
            && !self.tree_kv_indir_base_persistent.is_null()
            && !self.tree_kv_indir_base_host_pinned.is_null()
        {
            // Write the new base into the persistent pinned-host shadow
            // (in-place, host-side, no GPU involved) THEN fire the async
            // H2D copy from the pinned shadow. Pinned-host sources establish
            // a proper stream-ordered dependency for captured graph kernels;
            // pageable Vec sources do not (small pageable copies can race
            // ahead of subsequent graph launches → stale `kv_indir_base`).
            unsafe {
                let dst = self.tree_kv_indir_base_host_pinned as *mut i32;
                *dst = seq.seq_len as i32;
            }
            let base_bytes = unsafe {
                std::slice::from_raw_parts(
                    self.tree_kv_indir_base_host_pinned,
                    std::mem::size_of::<i32>(),
                )
            };
            self.gpu.copy_h2d_async(
                base_bytes,
                self.tree_kv_indir_base_persistent,
                stream,
            )?;
        }

        // ATLAS_TREE_AWARE_ATTN: build + upload per-row KV indirection table.
        // Row t = ancestor chain of compact slot t in top-down (bonus → t)
        // order, indexed by depth in [0..depth[t]+1). Kernel reads
        // indirection[seq_idx][i] for i in [0..seq_lens[t]-kv_indir_base).
        if tree_kv_active {
            let stride = self.tree_kv_indir_stride;
            let depths_ref = tree_depths.as_ref().unwrap();
            let host_parents = self.ddtree_parent_ids_host.lock().clone();
            let mut indir = vec![0i32; stride * stride];
            for t in 0..k {
                let mut chain: Vec<usize> = Vec::with_capacity(depths_ref[t] + 1);
                chain.push(t);
                let mut cur = t;
                while cur != 0 {
                    let p = if cur < host_parents.len() {
                        host_parents[cur]
                    } else {
                        cur as i32 - 1 // padded linear chain tail
                    };
                    let parent_slot: usize = if p < 0 { 0 } else { (p as usize).min(k - 1) };
                    if parent_slot == cur {
                        break;
                    }
                    chain.push(parent_slot);
                    cur = parent_slot;
                }
                chain.reverse();
                let want = depths_ref[t] + 1;
                while chain.len() < want {
                    chain.push(*chain.last().unwrap_or(&t));
                }
                for (j, &anc) in chain.iter().enumerate().take(stride) {
                    indir[t * stride + j] = anc as i32;
                }
                let last_anc = *chain.last().unwrap_or(&t) as i32;
                for j in chain.len()..stride {
                    indir[t * stride + j] = last_anc;
                }
            }
            let indir_bytes_view = unsafe {
                std::slice::from_raw_parts(
                    indir.as_ptr() as *const u8,
                    indir.len() * std::mem::size_of::<i32>(),
                )
            };
            self.gpu.copy_h2d_async(
                indir_bytes_view,
                self.tree_kv_indir_persistent,
                stream,
            )?;
            static TREE_KV_DBG_DONE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !TREE_KV_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let preview: Vec<Vec<i32>> = (0..k)
                    .map(|t| {
                        let d = (depths_ref[t] + 1).min(stride);
                        indir[t * stride..t * stride + d].to_vec()
                    })
                    .collect();
                tracing::info!(
                    "ATLAS_TREE_AWARE_ATTN: k={k} indir_stride={stride} chains={:?}",
                    preview
                );
            }
        }

        let metadata = AttnMetadataDev {
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: max_blocks,
            num_seqs: k as u32,
        };

        // Phase 6.2.c — HSS host I/O is illegal under CUDA graph capture.
        let hss_engaged = kv_cache.config().cache_blocks_per_seq.is_some();
        // ATLAS_DFLASH_DEBUG_NO_GRAPH=1 forces eager (no graph capture) so
        // CUDA_LAUNCH_BLOCKING=1 reports the exact failing kernel — used
        // to localize K=γ illegal-address crashes downstream of SSM.
        let force_eager = std::env::var("ATLAS_DFLASH_DEBUG_NO_GRAPH").ok().as_deref() == Some("1");
        // Mirror verify_b.rs (K=2) auto-unsuppress logic. Without this the
        // K=γ verify path stays EAGER forever once FP8/turbo KV calibration
        // sets suppress_graphs=true at startup, and never gets recaptured
        // into a CUDA graph. Eager K=γ verify is ~5x slower than graphed
        // (observed: K=9 eager 600ms/step vs K=4 graphed 80ms/step). Once
        // seq_len passes the calibration window, FP8 scales are frozen and
        // graph capture is safe.
        if self
            .suppress_graphs
            .load(std::sync::atomic::Ordering::Relaxed)
            && seq.seq_len > self.config.fp8_kv_calibration_tokens + 10
            && std::env::var("ATLAS_DUMP_HIDDEN").is_err()
        {
            self.suppress_graphs
                .store(false, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("FP8 calibration frozen — re-enabling CUDA graphs (K=γ verify)");
        }
        // ATLAS_FULL_PROFILE=1 disables graph capture so per-kernel sync is
        // legal (CUDA graph capture forbids host syncs / D2H copies inside).
        let full_profile = crate::full_profile::is_enabled();
        let use_graphs = self.comm.is_none()
            && !self
                .suppress_graphs
                .load(std::sync::atomic::Ordering::Relaxed)
            && !hss_engaged
            && !force_eager
            && !full_profile;

        // M8A: upload DDTree parent_ids (when stashed) so the GDN dispatch
        // can fire the tree-aware kernel. Stash lives on the model so we
        // don't have to thread it through every call site here.
        // verify_dflash_step.rs sets ddtree_parent_ids_dev before calling
        // this verify entry (via model.set_ddtree_parent_ids) so we just
        // read here.
        //
        // CUDA-graph-safe path: when graphs are enabled AND the verify K
        // matches the configured γ+1 (= dflash_kgamma = persistent buffer
        // capacity), substitute the persistent linear-chain buffer when
        // no tree payload was stashed. This forces every K=γ verify
        // through the `gdn_tree_wy_k` branch in the SSM dispatch, so the
        // captured graph always references the same kernel and the same
        // device pointer. Linear-chain parents (= `[-1, 0, 1, ..., γ-1]`)
        // make `gated_delta_rule_tree_wy` bit-equivalent to the old wy17
        // path (verified cos=1.0 across all 17 tokens in a prior session),
        // so output text is identical to the pre-fix eager path.
        let mut ddtree_parent_ids_dev = *self.ddtree_parent_ids_dev.lock();
        if use_graphs
            && ddtree_parent_ids_dev.is_none()
            && self.ddtree_parent_ids_capacity > 0
            && k == self.ddtree_parent_ids_capacity
        {
            ddtree_parent_ids_dev = Some(self.ddtree_parent_ids_persistent);
        }

        // DFS reorder: re-stamp parent_ids in DFS frame so the SSM (GDN
        // tree-WY) kernel processes tokens in DFS order, matching the
        // attention KV layout. After this, h_state_inter[i] holds the SSM
        // state AFTER DFS slot i. The commit path must map original-compact
        // indices → DFS slots via dfs_inv_perm to read the right inter slot.
        if dfs_active && self.ddtree_parent_ids_capacity > 0 {
            use crate::layers::dflash_head::ddtree::permute_parent_ids;
            // host_parents is the current kernel-frame parent_ids (pre-DFS).
            // Pad with linear-chain tail to match k, mirroring the depth
            // derivation logic above.
            let mut host_parents = self.ddtree_parent_ids_host.lock().clone();
            while host_parents.len() < k {
                let next_chain_parent = host_parents.len() as i32 - 1;
                host_parents.push(next_chain_parent);
            }
            if host_parents.len() == k {
                let permuted = permute_parent_ids(&host_parents, &dfs_perm, &dfs_inv_perm);
                let bytes: Vec<u8> = permuted.iter().flat_map(|p| p.to_le_bytes()).collect();
                self.gpu.copy_h2d_async(
                    &bytes,
                    self.ddtree_parent_ids_persistent,
                    stream,
                )?;
                // Make sure dispatch uses persistent buffer (which now holds
                // DFS-frame parents).
                ddtree_parent_ids_dev = Some(self.ddtree_parent_ids_persistent);
            }
        }

        // Stash dfs_inv_perm so commit_verify_state_async_with_slot can map
        // original-compact `last_inter_slot` (from greedy_sample_ddtree) →
        // DFS slot (where the SSM kernel actually wrote the canonical state).
        // Cleared by clear_ddtree_parent_ids() after commit finishes reading.
        //
        // METADATA_ONLY=1: SSM still writes h_state_inter in kernel-original
        // slot order (because parent_ids re-stamping happens above either
        // way), so we still need the inv_perm mapping for commit.
        if dfs_active {
            *self.ddtree_dfs_inv_perm.lock() = dfs_inv_perm.clone();
        } else {
            self.ddtree_dfs_inv_perm.lock().clear();
        }

        // ATLAS_TREE_AWARE_ATTN: wire the per-row KV indirection plumbing
        // into the forward context so paged-decode attention can remap
        // tree-window positions to true ancestors. None when the env var
        // is off or DFS reorder is active.
        //
        // ATLAS_TREE_KV_PACK (additional): also pack ancestor KV into a
        // contiguous per-layer scratch pool and upload `seq_lens =
        // depth[t]+1` so the consumer kernel can run with NULL indirection
        // over the scratch — restores the fast BC=4 batched attention path
        // that the per-position fallback gives up.
        let tree_aware_attn = if tree_kv_active {
            let pack = if self.tree_kv_pack_active
                && !self.tree_kv_pack_scratch_k.is_empty()
                && !self.tree_kv_pack_scratch_v.is_empty()
            {
                // Upload per-step `seq_lens = depth[t]+1` into the dedicated
                // packed-KV seq_lens buffer (separate from the main attn
                // metadata `seq_len` which still holds `seq.seq_len + d + 1`
                // for the unpacked path / fallback).
                let depths_ref = tree_depths.as_ref().unwrap();
                let mut packed_seq_lens = vec![0i32; self.tree_kv_indir_stride];
                for t in 0..k {
                    packed_seq_lens[t] = (depths_ref[t] + 1) as i32;
                }
                let sl_bytes_view = unsafe {
                    std::slice::from_raw_parts(
                        packed_seq_lens.as_ptr() as *const u8,
                        packed_seq_lens.len() * std::mem::size_of::<i32>(),
                    )
                };
                self.gpu.copy_h2d_async(
                    sl_bytes_view,
                    self.tree_kv_pack_seq_lens,
                    stream,
                )?;
                // NOTE: kv_cache mutex is already held earlier in this fn
                // (`let mut kv_cache = self.kv_cache.lock();`), so we must
                // NOT re-lock it. Read block_size via the already-held
                // borrow (`kv_cache.block_size()`).
                let cache_block_size = kv_cache.block_size() as u32;
                static PACK_DBG_DONE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !PACK_DBG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!(
                        "ATLAS_TREE_KV_PACK: k={k} chain_lens={:?} stride={} cache_bs={}",
                        &packed_seq_lens[..k],
                        self.tree_kv_indir_stride,
                        cache_block_size
                    );
                }
                Some(crate::layer::TreeKvPack {
                    num_attn_layers: self.tree_kv_pack_scratch_k.len() as u32,
                    scratch_k_ptrs: self.tree_kv_pack_scratch_k.as_ptr(),
                    scratch_v_ptrs: self.tree_kv_pack_scratch_v.as_ptr(),
                    identity_block_table: self.tree_kv_pack_block_table,
                    seq_lens: self.tree_kv_pack_seq_lens,
                    block_stride_bytes: self.tree_kv_pack_block_stride_bytes,
                    data_section_bytes: self.tree_kv_pack_data_section_bytes,
                    block_size: self.tree_kv_indir_stride as u32,
                    scatter_fp8_kernel: self.tree_kv_pack_scatter_fp8_kernel,
                    scatter_nvfp4_kernel: self.tree_kv_pack_scatter_nvfp4_kernel,
                    cache_block_size,
                    cache_max_blocks_per_seq: max_blocks,
                    // CUDA graph fix: read abs_base via device buffer so a
                    // captured graph picks up the fresh `seq.seq_len` on
                    // each replay. Buffer is updated above (before
                    // begin_capture).
                    abs_base_ptr: self.tree_kv_indir_base_persistent,
                })
            } else {
                None
            };
            Some(crate::layer::TreeAwareAttn {
                kv_indir: self.tree_kv_indir_persistent,
                // CUDA graph fix: see `abs_base_ptr` above.
                kv_indir_base_ptr: self.tree_kv_indir_base_persistent,
                kv_indir_stride: self.tree_kv_indir_stride as u32,
                pack,
            })
        } else {
            None
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: full_profile,
            comm: self.comm_ref(),
            graph_capture: use_graphs,
            ddtree_parent_ids_dev,
            tree_aware_attn,
            ssm_multi_seq_ptr_table_override: None,
        };

        // ── Phase 2: CUDA graph capture / replay ──

        let mut graph_cache = if use_graphs {
            Some(self.verify_kgamma_graph.lock())
        } else {
            None
        };

        // Cache key includes (slot, k, pack_active_flag). The pack flag is
        // critical: when ATLAS_TREE_KV_PACK is enabled, the first K=γ verify
        // (often pre-tree, flat-chain) captures a graph WITHOUT pack-pool
        // kernel arguments. Later verifies with non-flat trees can't reuse
        // that graph — its kernel sequence has stale arg pointers. Treating
        // pack as a separate cache key forces a fresh capture for the
        // pack-active path.
        let pack_key = ctx
            .tree_aware_attn
            .and_then(|t| t.pack)
            .is_some() as u32;
        let cache_key = (seq.slot_idx, k, pack_key);
        let cached_for_slot = graph_cache
            .as_ref()
            .and_then(|c| c.get(&cache_key).copied());
        if let Some(graph) = cached_for_slot
            && graph.0 != 0
        {
            self.gpu.launch_graph(graph, stream)?;
        }
        let need_run = cached_for_slot.is_none();
        if need_run {
            let seq_lens_vec: Vec<usize> = (0..k).map(|t| seq.seq_len + t).collect();
            let block_tables_vec: Vec<Vec<u32>> = vec![seq.block_table.clone(); k];

            if use_graphs {
                self.gpu.begin_capture(stream)?;
            }

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_type = self.config.layer_type(layer_idx);

                if layer_type == LayerType::FullAttention {
                    let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                        .map(|_| layer.alloc_state(self.gpu.as_ref()))
                        .collect::<Result<_>>()?;
                    let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                        dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                    layer.decode_multi_seq(
                        hidden,
                        residual,
                        k,
                        &mut refs,
                        &mut kv_cache,
                        &seq_lens_vec,
                        &block_tables_vec,
                        &ctx,
                        stream,
                    )?;
                } else {
                    layer.decode_batched(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        &ctx,
                        stream,
                    )?;
                }
                // DFlash hidden capture for ctx conditioning. Save ALL k
                // tokens so the scheduler can pick the correct one
                // (num_accepted) after verify. Layout:
                // [token_idx, capture_layer, hidden] in dflash_hidden_save.
                for t in 0..k {
                    self.try_dflash_capture(layer_idx, t, stream)?;
                }
                // ATLAS_KGAMMA_DEBUG_DUMP=1: dump per-position hidden state
                // for K=γ verify so we can diff against HF reference
                // (modelforge inspect-batched). Writes
                // /tmp/atlas_kgamma_layer{L}_pos{t}.bin one-shot per pair.
                // Dump only when NOT capturing a CUDA graph (sync ops illegal
                // during capture). Set ATLAS_DFLASH_DEBUG_NO_GRAPH=1 too.
                if std::env::var("ATLAS_KGAMMA_DEBUG_DUMP").is_ok() && !use_graphs {
                    let h_bytes = h * 2;
                    for t in 0..k {
                        let path = format!(
                            "/tmp/atlas_kgamma_layer{}_pos{}.bin",
                            layer_idx, t
                        );
                        if !std::path::Path::new(&path).exists() {
                            let mut buf = vec![0u8; h_bytes];
                            self.gpu.synchronize(stream)?;
                            self.gpu
                                .copy_d2h(hidden.offset(t * h_bytes), &mut buf)?;
                            let _ = std::fs::write(&path, &buf);
                        }
                    }
                }
            }

            // Final norm [K, H]
            let normed = self.buffers.norm_output();
            crate::kprof!(self.gpu.as_ref(), stream, "final_norm", {
                ops::rms_norm(
                    self.gpu.as_ref(),
                    self.rms_norm_kernel,
                    hidden,
                    &self.final_norm,
                    normed,
                    k as u32,
                    h as u32,
                    self.config.rms_norm_eps as f32,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;

            // DEBUG: dump post-final-norm hidden states (K positions)
            if std::env::var("ATLAS_KGAMMA_DEBUG_DUMP").is_ok() && !use_graphs {
                let h_bytes = h * 2;
                for t in 0..k {
                    let path = format!("/tmp/atlas_kgamma_final_norm_pos{}.bin", t);
                    if !std::path::Path::new(&path).exists() {
                        let mut buf = vec![0u8; h_bytes];
                        self.gpu.synchronize(stream)?;
                        self.gpu.copy_d2h(normed.offset(t * h_bytes), &mut buf)?;
                        let _ = std::fs::write(&path, &buf);
                    }
                }
            }

            // LM head for K tokens
            crate::kprof!(self.gpu.as_ref(), stream, "lm_head", {
                self.lm_head_batched(normed, k as u32, stream)?;
                anyhow::Result::<()>::Ok(())
            })?;

            // DEBUG: dump logits for first 100 vocab entries per position
            if std::env::var("ATLAS_KGAMMA_DEBUG_DUMP").is_ok() && !use_graphs {
                let vocab = self.config.vocab_size;
                let bf16 = 2usize;
                let dump_n = 100.min(vocab);
                for t in 0..k {
                    let path = format!("/tmp/atlas_kgamma_logits_pos{}.bin", t);
                    if !std::path::Path::new(&path).exists() {
                        let mut buf = vec![0u8; dump_n * bf16];
                        self.gpu.synchronize(stream)?;
                        self.gpu.copy_d2h(
                            self.buffers.logits().offset(t * vocab * bf16),
                            &mut buf,
                        )?;
                        let _ = std::fs::write(&path, &buf);
                    }
                }
            }

            // Argmax inside graph (fixed scratch addresses — graph-safe)
            let vocab = self.config.vocab_size;
            let argmax_out = self.buffers.scratch();
            crate::kprof!(self.gpu.as_ref(), stream, "argmax", {
                for t in 0..k {
                    let logits_t = self.buffers.logits().offset(t * vocab * bf16);
                    let out_t = argmax_out.offset(t * 4);
                    ops::argmax_bf16(
                        self.gpu.as_ref(),
                        self.argmax_kernel,
                        logits_t,
                        out_t,
                        vocab as u32,
                        stream,
                    )?;
                }
                anyhow::Result::<()>::Ok(())
            })?;

            if use_graphs {
                let graph = self.gpu.end_capture(stream)?;
                if graph.0 != 0 {
                    tracing::info!(
                        "Captured CUDA graph for K=γ verify (slot={} K={})",
                        seq.slot_idx,
                        k
                    );
                    if let Some(ref mut cache) = graph_cache {
                        cache.insert(cache_key, graph);
                    }
                    self.gpu.launch_graph(graph, stream)?;
                }
            }
        }

        // ── Phase 3: Post-graph (D2H copy only) ──

        // ATLAS_DUMP_HIDDEN: flush captured layer hiddens to file. See
        // verify_b.rs for the eager-mode safety contract. On the DFlash
        // path try_dflash_capture is already called per layer (line 790
        // above), so by Phase 3 dflash_hidden_save holds k tokens' worth
        // of (n_capture, hidden) records ready to flush.
        self.flush_hidden_dump(k)?;

        let out_ptr = self.buffers.scratch();
        let mut buf = vec![0u8; k * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut out = Vec::with_capacity(k);
        for t in 0..k {
            let off = t * 4;
            out.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        // DFS reorder: un-permute verified outputs back to original-compact
        // frame so the caller (verify_dflash_step.rs::greedy_sample_ddtree)
        // sees outputs indexed by the ORIGINAL compact (kernel) frame.
        //
        // `out[i]` (DFS frame) is the argmax-after-`tokens[dfs_perm[i]]`.
        // We want `out_orig[j]` = argmax-after-`tokens[j]` = `out[dfs_inv_perm[j]]`.
        //
        // Un-permute is only needed when token permutation was actually
        // applied (METADATA_ONLY=0). With METADATA_ONLY=1, tokens stayed
        // in linear order so out[i] already corresponds to drafts[i-1]'s
        // next-token prediction in the same kernel slot.
        if dfs_active && dfs_permute_tokens {
            let mut out_orig = vec![0u32; k];
            for j in 0..k {
                out_orig[j] = out[dfs_inv_perm[j]];
            }
            out = out_orig;
        }

        // See decode_verify_graphed for rationale on `seq_len += k` fix.
        for &t in tokens {
            seq.tokens.push(t);
        }
        seq.seq_len += k;

        Ok(out)
    }
}
