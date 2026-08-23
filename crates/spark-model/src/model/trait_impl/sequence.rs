// SPDX-License-Identifier: AGPL-3.0-only

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
    pub(super) fn cache_sequence_dispatch(&self, seq: &SequenceState) {
        let bs = self.kv_cache.lock().block_size();
        // Only cache if the sequence has block-aligned content worth caching.
        // Sequences shorter than one block have no reusable KV blocks.
        if seq.tokens.len() >= bs && !seq.block_table.is_empty() {
            // Prompt tokens were already inserted + ref-bumped by prefill.
            // Only generated tokens past `prompt_len` are "newly seq-owned"
            // at this point — pass prompt_len as matched_tokens so insert
            // skips re-bumping the prompt portion.
            //
            // Phase 6.3 sliding-window: when HSS has slid older blocks out,
            // `block_table` no longer parallels `tokens` from index 0 — the
            // physical IDs at the front of block_table now hold WRITES for
            // recent positions, not the historical prompt. Skip cache_sequence
            // insert in that case to avoid populating the radix tree with
            // mis-correlated entries. (Disk-side ref counting via
            // `apply_evicted_blocks` keeps the disk_block_ids alive
            // independently when the prefix cache later evicts.)
            // Skip when the prefix cache is a no-op (`--enable-prefix-caching`
            // off): the manual inc_ref below would never get a paired dec_ref
            // from cache eviction, leaking the seq's blocks every request.
            // Also skip when HSS sliding has occurred (front of block_table no
            // longer parallels tokens) and on vision prompts.
            if self.prefix_cache.is_active()
                && !self.tokens_have_vision_pad(&seq.tokens)
                && seq.hss_window_start() == 0
            {
                let acquired = self.prefix_cache.insert(
                    &seq.tokens,
                    &seq.block_table,
                    &seq.disk_block_ids,
                    bs,
                    seq.prompt_len,
                );
                super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
                // Bump KV block ref_counts so the prefix cache "owns" a reference.
                // This keeps blocks alive after free_sequence drops the sequence's ref.
                // Eviction (return_evicted_block) releases these refs when nodes are removed.
                let mut kv = self.kv_cache.lock();
                let num_cached_blocks = (seq.tokens.len() / bs).min(seq.block_table.len());
                for &block_idx in &seq.block_table[..num_cached_blocks] {
                    kv.inc_ref(block_idx);
                }
            }
        }
    }

    /// Release one proposer state's DFlash per-sequence device buffers, if it
    /// is a DFlash state at all. Extracted from `free_sequence_dispatch` so
    /// both proposer arms can be freed by the same code — see the call sites.
    ///
    /// Takes `&mut Option<Box<dyn ProposerState>>` — the FIELD's own type —
    /// rather than the `Option<&mut dyn ProposerState>` that reads more
    /// naturally. That is a borrow-checker requirement, not a style choice.
    /// `Box<dyn ProposerState>` carries the default object lifetime bound
    /// `+ 'static`, while an elided `&'a mut dyn ProposerState` parameter
    /// means `&'a mut (dyn ProposerState + 'a)`. `&mut` is INVARIANT in its
    /// pointee, so handing it a `&'a mut (dyn ProposerState + 'static)`
    /// derived from `seq: &'1 mut SequenceState` forces `'1: 'static` — the
    /// borrow is then inferred to live forever and every later use of `seq`
    /// in this method conflicts with it. Passing the field by reference keeps
    /// both sides at `+ 'static` and never forms the mismatched `&mut dyn`.
    fn free_dflash_state_buffers(
        &self,
        state: &mut Option<Box<dyn crate::speculative::ProposerState>>,
    ) {
        let Some(ps) = state.as_deref_mut() else {
            return;
        };
        let Some(ds) = ps
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
        else {
            return;
        };
        // The multi-GB ctx accumulator is pooled across requests (see
        // `ctx_acc_pool` in dflash_head.rs) — returning it to the pool
        // instead of cuMemFree avoids the ~200-950 ms UMA first-touch
        // page-fault cost of re-allocating it on the next request. The
        // smaller fc/K/V caches are still freed normally.
        let ctx_acc_bytes = ds.max_ctx_len * ds.ctx_slot_bytes;
        crate::layers::dflash_head::ctx_acc_pool_return(ctx_acc_bytes, ds.ctx_hidden_acc);
        ds.ctx_hidden_acc = spark_runtime::gpu::DevicePtr(0);
        let mut ptrs: Vec<spark_runtime::gpu::DevicePtr> = vec![ds.ctx_fc_cache];
        ptrs.append(&mut ds.ctx_k_cache);
        ptrs.append(&mut ds.ctx_v_cache);
        for p in ptrs {
            if p.0 != 0
                && let Err(e) = self.gpu.free(p)
            {
                tracing::warn!("free_sequence: dflash buffer free failed: {e:#}");
            }
        }
        ds.ctx_fc_cache = spark_runtime::gpu::DevicePtr(0);
    }

    pub(super) fn free_sequence_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        // A verify-state commit may still be writing this slot on the
        // secondary stream. Chain that event onto the default stream before
        // zero_slot and its synchronize below; otherwise a newly allocated
        // sequence can reuse the slot while the retired request is still
        // overwriting its recurrent state.
        self.sync_secondary_dispatch()?;

        // ATLAS_DFLASH_ASYNC: an in-flight second-stream propose reads this
        // sequence's ctx accumulator + fc/K/V caches — resolve (sync +
        // discard) BEFORE those buffers are freed below (use-after-free
        // guard). No-op when nothing is in flight.
        //
        // Both arms are resolved and freed. A two-arm build allocated a
        // proposer state per arm in `alloc_sequence`, and only ONE of them is
        // in `seq.proposer_state` at any moment — freeing just that one would
        // leak the parked arm's ctx accumulator + fc/K/V caches on every
        // request, which is the same unbounded-UMA failure the single-arm
        // free below was written to fix.
        if let Some(proposer) = self.active_proposer() {
            let res = match seq.proposer_state.as_deref_mut() {
                Some(ps) => proposer.resolve_async_inflight(self.gpu.as_ref(), Some(ps)),
                None => proposer.resolve_async_inflight(self.gpu.as_ref(), None),
            };
            if let Err(e) = res {
                tracing::warn!("free_sequence: resolve_async_inflight failed: {e:#}");
            }
        }
        if let Some(proposer) = self.inactive_proposer() {
            let res = match seq.proposer_state_alt.as_deref_mut() {
                Some(ps) => proposer.resolve_async_inflight(self.gpu.as_ref(), Some(ps)),
                None => proposer.resolve_async_inflight(self.gpu.as_ref(), None),
            };
            if let Err(e) = res {
                tracing::warn!("free_sequence: resolve_async_inflight (parked arm) failed: {e:#}");
            }
        }

        // Free the DFlash proposer state's per-sequence device buffers.
        // free_state can't (no GpuBackend in scope there); without this,
        // every request leaked the ctx accumulator + fc/K/V caches
        // (hundreds of MB per sequence — unbounded UMA growth while
        // serving).
        self.free_dflash_state_buffers(&mut seq.proposer_state);
        self.free_dflash_state_buffers(&mut seq.proposer_state_alt);

        // Release prefix cache refs before freeing blocks.
        // dec_ref will only actually free blocks whose ref_count hits 0
        // CRITICAL: release SSM slot FIRST to prevent slot leak if later
        // operations fail (e.g. after sticky CUDA error 700). The slot is a
        // CPU-side resource; its release must not be gated on GPU success.
        let slot_to_release = if seq.slot_idx < self.ssm_pool.max_slots {
            Some(seq.slot_idx)
        } else {
            None
        };
        if let Some(slot) = slot_to_release {
            let stream = self.gpu.default_stream();
            if let Err(e) = self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream) {
                tracing::error!("free_sequence: ssm_pool.zero_slot({slot}): {e:#}");
            }
            if let Err(e) = self.gpu.synchronize(stream) {
                tracing::error!("free_sequence: gpu.synchronize after zero_slot({slot}): {e:#}");
            }
            self.ssm_pool.release_slot(slot);
        }

        // Release prefix cache refs before freeing blocks.
        // (i.e., blocks not shared with the prefix cache).
        self.prefix_cache
            .release(&seq.tokens, self.kv_cache.lock().block_size());
        if !seq.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&seq.block_table);
            seq.block_table.clear();
        }

        // --high-speed-swap: release disk-side refs for every block this
        // sequence ever held (Phase 6.1.c). disk_block_ids are layer-
        // agnostic (each ID indexes a slot in *every* layer's file), so
        // one dec_disk_ref per ID covers all layers' data simultaneously.
        // The orchestrator's free list only reclaims an ID when its
        // refcount hits 0, so sequences sharing a prefix correctly keep
        // each other's disk blocks alive via ref-counting.
        if !seq.disk_block_ids.is_empty() {
            // with_local returns Option<Result>: None when HSS isn't engaged
            // (no-op, fine), Some(Err) when the closure failed (advisory).
            if let Some(Err(e)) = spark_storage::with_local(|hss| {
                for &disk_id in &seq.disk_block_ids {
                    hss.dec_disk_ref(disk_id);
                }
                Ok(())
            }) {
                tracing::error!("free_sequence: spark_storage dec_disk_ref batch: {e:#}");
            }
            seq.disk_block_ids.clear();
            for v in seq.disk_last_offloaded_per_layer.iter_mut() {
                *v = 0;
            }
        }

        // All SSM buffers (h_state, conv_state, checkpoints, intermediates) belong
        // to the pool — do NOT gpu.free() them. Just clear the references.
        for state in &mut seq.layer_states {
            if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                ssm.h_state = DevicePtr(0);
                ssm.conv_state = DevicePtr(0);
                ssm.h_state_checkpoint = None;
                ssm.conv_state_checkpoint = None;
                ssm.h_state_intermediates.clear();
                ssm.conv_state_intermediates.clear();
            }
        }

        // Invalidate cached CUDA graphs that reference this sequence's slot
        // — the graph was captured with this slot's KV/SSM pointers baked in,
        // and replaying after the slot is freed would read stale data.
        // decode_graph is keyed by slot, so drop only this slot's entry.
        // (parking_lot::Mutex::lock() never poisons, so the previous `if let
        // Ok(...) = .lock()` graceful-recovery branch is unreachable.)
        if let Some(graph) = self.decode_graph.lock().remove(&seq.slot_idx)
            && let Err(e) = self.gpu.destroy_graph(graph)
        {
            tracing::error!(
                "free_sequence: destroy_graph(decode_graph[{}]): {e:#}",
                seq.slot_idx
            );
        }
        // batch_decode_graphs is keyed by (sorted_slot_ids, padded_n) when
        // ATLAS_SSM_MULTI_SEQ_GRAPH is on, else by ((empty), padded_n). In
        // graph mode, drop only entries that include this freed slot — other
        // slot-set graphs remain valid and avoid unnecessary recapture. In the
        // legacy off mode the cache is empty (never inserted), so the drain
        // is a no-op.
        {
            let mut cache = self.batch_decode_graphs.lock();
            let stale_keys: Vec<(Vec<usize>, usize)> = cache
                .keys()
                .filter(|(slots, _)| slots.is_empty() || slots.contains(&seq.slot_idx))
                .cloned()
                .collect();
            for key in stale_keys {
                if let Some(graph) = cache.remove(&key)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "free_sequence: destroy_graph(batch_decode_graphs[{:?}]): {e:#}",
                        key
                    );
                }
            }
        }
        // Verify graphs are now slot-keyed (sibling of decode_graph fix).
        // Drop only this slot's entry to preserve other concurrent seqs' graphs.
        for graph_mutex in [
            &self.verify2_graph,
            &self.verify3_graph,
            &self.verify4_graph,
        ] {
            if let Some(graph) = graph_mutex.lock().remove(&seq.slot_idx)
                && let Err(e) = self.gpu.destroy_graph(graph)
            {
                tracing::error!(
                    "free_sequence: destroy_graph(verify[{}]): {e:#}",
                    seq.slot_idx
                );
            }
        }
        {
            let mut cache = self.verify_kgamma_graph.lock();
            let stale_keys = super::sequence_graph_cleanup::verify_kgamma_keys_for_slots(
                &cache,
                &[seq.slot_idx],
            );
            for key in stale_keys {
                if let Some(graph) = cache.remove(&key)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "free_sequence: destroy_graph(verify_kgamma_graph[{key:?}]): {e:#}"
                    );
                }
            }
        }

        // Free MTP proposer state (KV cache blocks) — for both arms, each
        // through the proposer that allocated it.
        if let Some(proposer) = self.active_proposer()
            && let Some(ref mut pstate) = seq.proposer_state
        {
            proposer.free_state(pstate.as_mut())?;
        }
        if let Some(proposer) = self.inactive_proposer()
            && let Some(ref mut pstate) = seq.proposer_state_alt
        {
            proposer.free_state(pstate.as_mut())?;
        }

        self.free_chunked_prefill_meta(seq)?;

        Ok(())
    }

    pub(super) fn compact_sequence_dispatch(
        &self,
        seq: &mut SequenceState,
        new_slot: usize,
    ) -> Result<()> {
        let old_slot = seq.slot_idx;
        if old_slot == new_slot {
            return Ok(());
        }

        let stream = self.gpu.default_stream();
        self.ssm_pool
            .copy_slot(old_slot, new_slot, self.gpu.as_ref(), stream)?;

        // Update ALL SsmLayerState pool pointers to point at the new slot.
        // BUG FIX: previously only h_state and conv_state were repointed, leaving
        // the MTP checkpoint and intermediate pointers aimed at the OLD slot.
        // After release_slot, that old slot is reallocatable to a NEW sequence,
        // and any subsequent MTP save_hidden / start_checkpoint_async on this seq
        // would write into the new occupant's pool memory — cross-seq corruption.
        let has_mtp = self.ssm_pool.has_mtp;
        let num_intermediates = self.ssm_pool.num_intermediates;
        let mut ssm_layer_idx = 0usize;
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                    ssm.h_state = self.ssm_pool.h_state(ssm_layer_idx, new_slot);
                    ssm.conv_state = self.ssm_pool.conv_state(ssm_layer_idx, new_slot);
                    if has_mtp {
                        if ssm.h_state_checkpoint.is_some() {
                            ssm.h_state_checkpoint =
                                Some(self.ssm_pool.h_checkpoint(ssm_layer_idx, new_slot));
                        }
                        if ssm.conv_state_checkpoint.is_some() {
                            ssm.conv_state_checkpoint =
                                Some(self.ssm_pool.conv_checkpoint(ssm_layer_idx, new_slot));
                        }
                        if !ssm.h_state_intermediates.is_empty() {
                            ssm.h_state_intermediates.clear();
                            for t in 0..num_intermediates {
                                ssm.h_state_intermediates.push(self.ssm_pool.h_intermediate(
                                    ssm_layer_idx,
                                    new_slot,
                                    t,
                                ));
                            }
                        }
                        if !ssm.conv_state_intermediates.is_empty() {
                            ssm.conv_state_intermediates.clear();
                            for t in 0..num_intermediates {
                                ssm.conv_state_intermediates
                                    .push(self.ssm_pool.conv_intermediate(
                                        ssm_layer_idx,
                                        new_slot,
                                        t,
                                    ));
                            }
                        }
                        // WY17 LAZY-commit retention pointers follow the slot.
                        if ssm.wy17_kv_retain.is_some() {
                            ssm.wy17_kv_retain =
                                self.ssm_pool.wy17_kv_retain(ssm_layer_idx, new_slot);
                        }
                        if ssm.wy17_gate_retain.is_some() {
                            ssm.wy17_gate_retain =
                                self.ssm_pool.wy17_gate_retain(ssm_layer_idx, new_slot);
                        }
                    }
                }
                ssm_layer_idx += 1;
            }
        }

        seq.slot_idx = new_slot;
        // BUG FIX: synchronize before releasing the old slot. copy_slot is async
        // (queued D2D), so without this barrier, claim_slot() in the next request
        // could hand the old_slot back to a new sequence while the copy's source
        // reads are still in flight — cross-seq race that produces partial data.
        self.gpu.synchronize(stream)?;
        self.ssm_pool.release_slot(old_slot);

        // Invalidate every graph bound to either side of the slot swap. Graphs
        // for old_slot reference the moved sequence's former buffers. Graphs
        // for new_slot reference the retired sequence that occupied the
        // destination; its slot is set to usize::MAX before free_sequence, so
        // compaction is the only place that can remove those entries.
        for stale_slot in [old_slot, new_slot] {
            if let Some(graph) = self.decode_graph.lock().remove(&stale_slot)
                && let Err(e) = self.gpu.destroy_graph(graph)
            {
                tracing::error!(
                    "compact_sequence: destroy_graph(decode_graph[{stale_slot}]): {e:#}"
                );
            }
        }
        {
            let mut cache = self.batch_decode_graphs.lock();
            let stale_keys: Vec<(Vec<usize>, usize)> = cache
                .keys()
                .filter(|(slots, _)| slots.contains(&old_slot) || slots.contains(&new_slot))
                .cloned()
                .collect();
            for key in stale_keys {
                if let Some(graph) = cache.remove(&key)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "compact_sequence: destroy_graph(batch_decode_graphs[{:?}]): {e:#}",
                        key
                    );
                }
            }
        }
        for graph_mutex in [
            &self.verify2_graph,
            &self.verify3_graph,
            &self.verify4_graph,
        ] {
            for stale_slot in [old_slot, new_slot] {
                if let Some(graph) = graph_mutex.lock().remove(&stale_slot)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!("compact_sequence: destroy_graph(verify[{stale_slot}]): {e:#}");
                }
            }
        }
        {
            let mut cache = self.verify_kgamma_graph.lock();
            let stale_keys = super::sequence_graph_cleanup::verify_kgamma_keys_for_slots(
                &cache,
                &[old_slot, new_slot],
            );
            for key in stale_keys {
                if let Some(graph) = cache.remove(&key)
                    && let Err(e) = self.gpu.destroy_graph(graph)
                {
                    tracing::error!(
                        "compact_sequence: destroy_graph(verify_kgamma_graph[{key:?}]): {e:#}"
                    );
                }
            }
        }
        // NOTE (ATLAS_MULTISEQ_GRAPHS): `piecewise_decode_graphs` is
        // intentionally NOT invalidated on compaction. Those segment graphs
        // are keyed by `(padded_n, seg_id)` and bake NO per-slot address —
        // the SSM h_state/conv_state pointers are read from the layer-stable
        // ptr scratch, which the piecewise dispatcher refreshes from each
        // sequence's (post-compaction) `layer_states` before every replay.
        // So a compaction that changes a seq's slot is transparent: the next
        // pre-replay `multiseq_refresh_ptr_table` uploads the new slot's
        // pointers and the same cached graph replays correctly.
        Ok(())
    }

    pub(super) fn save_sequence_state_dispatch(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();

        // Phase 1: Copy all KV block data from GPU to host buffers under the lock.
        let kv_buffers = {
            let kv = self.kv_cache.lock();
            let mut bufs = Vec::with_capacity(seq.block_table.len() * kv.num_layers());
            for &block_idx in &seq.block_table {
                for layer_idx in 0..kv.num_layers() {
                    bufs.push(kv.read_block(layer_idx, block_idx, gpu)?);
                }
            }
            bufs
        }; // Lock released here.

        // Phase 2: Write KV data to disk (no lock held).
        for (k_data, v_data) in &kv_buffers {
            writer.write_all(k_data)?;
            writer.write_all(v_data)?;
        }

        // Phase 3: Copy SSM states from GPU to host, then write to disk.
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let mut h_buf = vec![0u8; self.ssm_pool.h_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                // The swap FILE is always FP32, so a resumed sequence can be
                // read back without carrying a dtype tag. Under
                // ATLAS_SSM_H_FP16 a decoding slot holds FP16 packed into the
                // first half of h_bytes; serializing that raw would resume into
                // a freshly-allocated SequenceState whose h_is_f16 is false,
                // and the next decode step would narrow already-narrow bytes.
                // Widen here instead — the mirror of ssm_snapshot::save().
                if ssm.h_is_f16 {
                    let src = self.widen_h_to_f32_scratch(gpu, ssm.h_state)?;
                    gpu.copy_d2h(src, &mut h_buf)?;
                } else {
                    gpu.copy_d2h(ssm.h_state, &mut h_buf)?;
                }
                gpu.copy_d2h(ssm.conv_state, &mut c_buf)?;
                writer.write_all(&h_buf)?;
                writer.write_all(&c_buf)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    pub(super) fn restore_sequence_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();

        // Phase 1: Read all KV block data from disk into host buffers.
        let (num_layers, layer_strides) = {
            let kv = self.kv_cache.lock();
            let n = kv.num_layers();
            let strides: Vec<usize> = (0..n).map(|i| kv.block_stride_bytes_for_layer(i)).collect();
            (n, strides)
        };

        let mut kv_buffers = Vec::with_capacity(num_blocks * num_layers);
        for _ in 0..num_blocks {
            for layer_idx in 0..num_layers {
                let stride = layer_strides[layer_idx];
                let mut k_data = vec![0u8; stride];
                let mut v_data = vec![0u8; stride];
                reader.read_exact(&mut k_data)?;
                reader.read_exact(&mut v_data)?;
                kv_buffers.push((k_data, v_data));
            }
        }

        // Phase 2: Allocate blocks and write data under the lock.
        {
            let mut kv = self.kv_cache.lock();
            let mut new_block_table = Vec::with_capacity(num_blocks);
            let mut buf_idx = 0;
            for _ in 0..num_blocks {
                let block_idx = kv.alloc_block()?;
                for layer_idx in 0..num_layers {
                    let (ref k_data, ref v_data) = kv_buffers[buf_idx];
                    kv.write_block(layer_idx, block_idx, k_data, v_data, gpu)?;
                    buf_idx += 1;
                }
                new_block_table.push(block_idx);
            }
            seq.block_table = new_block_table;
        } // Lock released here.

        // Phase 3: Read SSM state data from disk and upload to GPU.
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let mut h_buf = vec![0u8; self.ssm_pool.h_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                reader.read_exact(&mut h_buf)?;
                reader.read_exact(&mut c_buf)?;
                gpu.copy_h2d(&h_buf, ssm.h_state)?;
                gpu.copy_h2d(&c_buf, ssm.conv_state)?;
            }
        }

        Ok(())
    }

    pub(super) fn num_free_blocks_dispatch(&self) -> usize {
        self.kv_cache.lock().num_free_blocks()
    }
}
