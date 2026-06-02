// SPDX-License-Identifier: AGPL-3.0-only

//! `impl Model for TransformerModel` — thin trait impl that delegates to
//! `<method>_dispatch` helpers split across sibling files for the ≤500
//! LoC cap. Each sibling adds methods to the `TransformerModel`
//! inherent impl. The trait impl below is purely one-line delegators.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{AttnMetadataDev, LayerState};
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, PrefillSlice, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights};

mod async_chkpt;
mod decode_a;
mod decode_a2;
mod decode_b;
mod decode_b2;
mod ep_misc;
mod meta;
mod prefill_a;
mod prefill_b;
mod prefill_c;
mod prefill_d;
mod sequence;
mod speculative;
mod verify_a;
mod verify_b;
mod verify_c;
mod verify_c2;
mod verify_csk;
mod verify_d;

impl Model for TransformerModel {
    fn prepare_vision_embed(&self, images: &[(Vec<f32>, usize, usize)]) -> Result<()> {
        self.prepare_vision_embed_dispatch(images)
    }
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.prefill_dispatch(tokens, seq, stream)
    }
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_chunk_dispatch(tokens, seq, chunk_start, chunk_len, is_last_chunk, stream)
    }
    fn prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_twophase_dispatch(tokens, seq, chunk_size, stream)
    }
    fn decode(&self, token: u32, seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        self.decode_dispatch(token, seq, _stream)
    }
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.decode_batch_dispatch(tokens, seqs, stream)
    }
    fn mixed_forward(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_chunk_start: usize,
        prefill_chunk_len: usize,
        prefill_is_last: bool,
        stream: u64,
    ) -> Result<crate::traits::MixedForwardResult> {
        self.mixed_forward_dispatch(
            decode_tokens,
            decode_seqs,
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            prefill_chunk_len,
            prefill_is_last,
            stream,
        )
    }

    /// Q12 Phase 4b override: try the model-level batched dispatch
    /// (`prefill_batch_chunk_dispatch`) first; on the not-yet-implemented
    /// stub failure, fall back to the trait's default per-stream loop.
    /// This keeps callers correct while the per-layer-batched body is
    /// staged in subsequent commits.
    fn prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        // Try the concrete dispatch. The Phase 4b stub returns Err for the
        // "not-yet-implemented" path so we transparently downgrade to the
        // single-stream-loop default impl. Once Phase 2b/3 land, the
        // dispatch returns Ok with logits and this fallback becomes dead
        // code that we can drop.
        match self.prefill_batch_chunk_dispatch(streams, stream) {
            Ok(v) => Ok(v),
            Err(e) => {
                // Log at debug — under expected for this stub. Promotes to
                // info if a real error is encountered (future Phase 4b body).
                tracing::debug!(
                    "prefill_batch_chunk_dispatch unavailable, falling back to \
                     per-stream loop: {e}"
                );
                let mut out = Vec::with_capacity(streams.len());
                for slice in streams.iter_mut() {
                    let logits = self.prefill_chunk(
                        slice.prompt_tokens,
                        slice.seq,
                        slice.chunk_start,
                        slice.chunk_len,
                        slice.is_last_chunk,
                        stream,
                    )?;
                    out.push(logits);
                }
                Ok(out)
            }
        }
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size_dispatch()
    }
    fn high_speed_swap_dims(&self) -> Option<spark_storage::ModelDims> {
        self.high_speed_swap_dims_dispatch()
    }
    fn normalize_ssm_states(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        self.normalize_ssm_states_dispatch(seq, stream)
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.bind_gpu_to_thread_dispatch()
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        self.alloc_sequence_dispatch()
    }
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.copy_logits_to_host_dispatch(logits_ptr, dst)
    }
    fn logits_ptr_is_fp32(&self, logits_ptr: DevicePtr) -> bool {
        self.logits_ptr_is_fp32_dispatch(logits_ptr)
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.logits_buffer_ptr_dispatch()
    }
    fn argmax_on_device(&self, logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        self.argmax_on_device_dispatch(logits_ptr, _stream)
    }
    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, _stream: u64) -> Result<Vec<u32>> {
        self.argmax_batch_dispatch(logits_ptr, n, _stream)
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        self.hidden_after_norm_dispatch()
    }
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_dispatch(tokens, seq, stream)
    }
    fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()> {
        self.checkpoint_ssm_states_dispatch(seq)
    }
    fn rollback_ssm_states(&self, seq: &mut SequenceState, num_accepted: usize) -> Result<()> {
        self.rollback_ssm_states_dispatch(seq, num_accepted)
    }
    fn has_ssm_layers(&self) -> bool {
        self.ssm_pool.num_ssm_layers > 0
    }
    fn decode_rollback_ring_slots(&self) -> usize {
        if self.ssm_snapshots.decode_rollback_enabled() {
            self.ssm_snapshots.decode_ring_slots
        } else {
            0
        }
    }
    fn save_decode_ssm_snapshot(&self, seq: &SequenceState, ring_slot: usize) -> Result<()> {
        self.save_decode_ssm_snapshot_dispatch(seq, ring_slot)
    }
    fn restore_decode_ssm_snapshot(&self, seq: &SequenceState, ring_slot: usize) -> Result<()> {
        self.restore_decode_ssm_snapshot_dispatch(seq, ring_slot)
    }
    fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        self.generate_speculative_dispatch(prompt_tokens, params, num_drafts)
    }
    fn has_proposer(&self) -> bool {
        self.has_proposer_dispatch()
    }
    fn has_self_speculative(&self) -> bool {
        self.has_self_speculative_dispatch()
    }
    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.decode_draft_dispatch(token, seq, stream)
    }
    fn cache_sequence(&self, seq: &SequenceState) {
        self.cache_sequence_dispatch(seq)
    }
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        self.free_sequence_dispatch(seq)
    }
    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 2]> {
        self.decode_verify_graphed_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 3]> {
        self.decode_verify_graphed_k3_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_k3_batched_csk(
        &self,
        tokens_per_seq: &[[u32; 3]],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<[u32; 3]>> {
        self.decode_verify_k3_batched_csk_dispatch(tokens_per_seq, seqs)
    }
    fn decode_verify_k2_batched_csk(
        &self,
        tokens_per_seq: &[[u32; 2]],
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<[u32; 2]>> {
        self.decode_verify_k2_batched_csk_dispatch(tokens_per_seq, seqs)
    }
    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 4]> {
        self.decode_verify_graphed_k4_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_graphed_kgamma(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_graphed_kgamma_dispatch(tokens, seq, _stream)
    }
    fn save_hidden_for_mtp(&self, token_idx: usize, _stream: u64) -> Result<()> {
        self.save_hidden_for_mtp_dispatch(token_idx, _stream)
    }
    fn save_hidden_for_dflash(&self, token: u32, seq: &mut SequenceState, _stream: u64) -> Result<()> {
        self.save_hidden_for_dflash_dispatch(token, seq, _stream)
    }
    fn run_mtp_propose(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        self.run_mtp_propose_dispatch(token, position, seq, _stream)
    }
    fn run_mtp_propose_multi(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        _stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.run_mtp_propose_multi_dispatch(
            token,
            position,
            num_drafts,
            seq,
            _stream,
            grammar_bitmask,
        )
    }
    fn read_deferred_draft_token(&self) -> Result<u32> {
        self.read_deferred_draft_token_dispatch()
    }
    fn take_pending_tree_payload(
        &self,
        seq: &mut SequenceState,
    ) -> Option<crate::layers::DDTreePayload> {
        let proposer = self.proposer.as_ref()?;
        let state = seq.proposer_state.as_mut()?;
        proposer.take_pending_tree_payload(state.as_mut())
    }

    /// M8A: upload payload.parent_indices to the per-model scratch slot.
    ///
    /// Writes into the persistent `ddtree_parent_ids_persistent` device
    /// buffer (allocated once at init) so the device address stays stable
    /// across CUDA graph capture and replay. Older revisions of this
    /// function allocated a fresh scratch buffer per call, which left the
    /// captured graph reading a freed/stale pointer on replay.
    fn set_ddtree_parent_ids(&self, payload: &crate::layers::DDTreePayload) -> Result<bool> {
        use crate::layers::dflash_head::ddtree_gdn_dispatch::requires_tree_kernel;
        static SETDBG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = SETDBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let needs_tree = requires_tree_kernel(Some(payload));
        // ATLAS_FORCE_TREEWY=1: stash even for flat chain payloads so the
        // M8A v2 kernel fires on them. Used for A/B precision diff vs wy17
        // (without the bypass, flat chains skip M8A and dispatch wy17).
        let force_treewy =
            std::env::var("ATLAS_FORCE_TREEWY").ok().as_deref() == Some("1");
        if n < 3 {
            tracing::info!(
                "M8A set_ddtree_parent_ids #{n}: payload.len={} needs_tree={} force_treewy={} parents={:?}",
                payload.tree_token_ids.len(), needs_tree, force_treewy,
                payload.parent_indices.iter().take(8).collect::<Vec<_>>()
            );
        }
        if !needs_tree && !force_treewy {
            // Flat chain → kernel falls through to wy_k. Nothing to stash.
            return Ok(false);
        }
        // Convert payload (compact-tree frame) to kernel frame.
        //
        // The K=γ verify kernel processes T = γ+1 tokens in order:
        //   token 0    = bonus (= last verified token from previous step)
        //   token i+1  = drafts[i] for i in 0..γ
        //
        // Each kernel token needs parent_ids[i]:
        //   -1  → load state from h_state (the pre-tree commit point)
        //   k≥0 → load state from h_state_inter[k]   (must satisfy k < i)
        //
        // payload.parent_indices[j] is in COMPACT-TREE frame for draft j:
        //   -1  → draft j attaches to the synthetic root (a.k.a. the bonus)
        //   k≥0 → draft j's parent is draft k
        //
        // Mapping compact-tree → kernel frame:
        //   kernel_parents[0]     = -1                                 (bonus loads from h_state)
        //   kernel_parents[j+1]   = if payload.parent_indices[j] < 0:
        //                              0                               (root-children load post-bonus state)
        //                          else:
        //                              payload.parent_indices[j] + 1   (+1 skips bonus slot)
        let stream = self.gpu.default_stream();
        let mut kernel_parents: Vec<i32> = Vec::with_capacity(1 + payload.parent_indices.len());
        kernel_parents.push(-1i32);
        for &p in &payload.parent_indices {
            kernel_parents.push(if p < 0 { 0 } else { p + 1 });
        }
        // Refuse to overrun the persistent buffer (defensive — payload
        // should never exceed dflash_kgamma).
        if self.ddtree_parent_ids_capacity == 0 {
            anyhow::bail!(
                "set_ddtree_parent_ids: persistent buffer not allocated (dflash_kgamma=0)"
            );
        }
        if kernel_parents.len() > self.ddtree_parent_ids_capacity {
            anyhow::bail!(
                "set_ddtree_parent_ids: payload has {} tokens > capacity {}",
                kernel_parents.len(),
                self.ddtree_parent_ids_capacity
            );
        }
        let bytes: Vec<u8> = kernel_parents
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect();
        // Write into the persistent buffer (NOT a fresh alloc). The device
        // pointer never changes — CUDA graph replays see the new payload.
        self.gpu.copy_h2d_async(
            &bytes,
            self.ddtree_parent_ids_persistent,
            stream,
        )?;
        *self.ddtree_parent_ids_dev.lock() = Some(self.ddtree_parent_ids_persistent);
        *self.ddtree_num_tree_tokens.lock() = kernel_parents.len();
        // Stash the host-side mirror so verify_d.rs can derive per-token
        // depths for tree-aware RoPE/seq_lens without a D2H copy.
        *self.ddtree_parent_ids_host.lock() = kernel_parents;
        Ok(true)
    }

    fn clear_ddtree_parent_ids(&self) {
        // Restore the persistent buffer to the linear-chain default so the
        // graph-safe verify path (which keeps `Some(persistent)` to always
        // hit tree_wy) reads bit-equivalent-to-wy17 parents on the next
        // non-tree call. We don't release the buffer — its address must
        // remain stable for captured graphs.
        if self.ddtree_parent_ids_capacity > 0 {
            let cap = self.ddtree_parent_ids_capacity;
            let mut chain = Vec::<i32>::with_capacity(cap);
            chain.push(-1);
            for i in 1..cap {
                chain.push((i - 1) as i32);
            }
            let bytes: Vec<u8> = chain
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect();
            let stream = self.gpu.default_stream();
            // Best-effort restamp. A failed copy here is non-fatal: the next
            // set_ddtree_parent_ids call will overwrite, and the graph-safe
            // path always re-stamps before replay-only invocations.
            if let Err(e) = self
                .gpu
                .copy_h2d_async(&bytes, self.ddtree_parent_ids_persistent, stream)
            {
                tracing::warn!("clear_ddtree_parent_ids: failed to restamp linear chain: {e:#}");
            }
        }
        *self.ddtree_parent_ids_dev.lock() = None;
        *self.ddtree_num_tree_tokens.lock() = 0;
        self.ddtree_parent_ids_host.lock().clear();
        // DFS reorder state lives only for the duration of one verify+commit.
        self.ddtree_dfs_inv_perm.lock().clear();
    }
    fn trim_proposer_state(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        self.trim_proposer_state_dispatch(seq, num_accepted, _stream)
    }
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        self.compact_sequence_dispatch(seq, new_slot)
    }
    fn save_sequence_state(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        self.save_sequence_state_dispatch(seq, writer)
    }
    fn restore_sequence_state(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        self.restore_sequence_state_dispatch(seq, num_blocks, reader)
    }
    fn num_free_blocks(&self) -> usize {
        self.num_free_blocks_dispatch()
    }
    fn start_checkpoint_async(&self, seq: &mut SequenceState) -> Result<()> {
        self.start_checkpoint_async_dispatch(seq)
    }
    fn start_rollback_and_checkpoint_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        self.start_rollback_and_checkpoint_async_dispatch(seq, num_accepted)
    }
    fn sync_secondary(&self) -> Result<()> {
        self.sync_secondary_dispatch()
    }
    fn pre_verify_copy_async(&self, seq: &mut SequenceState) -> Result<()> {
        self.pre_verify_copy_async_dispatch(seq)
    }
    fn commit_verify_state_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        // Legacy callers (K=2/K=3/K=4 chain verifies) → derive the kernel
        // slot as `num_accepted - 1` (chain-contiguous). DDTree callers
        // must use `commit_verify_state_async_with_slot` to pass the
        // explicit non-contiguous slot.
        let last_inter_slot = num_accepted.saturating_sub(1);
        self.commit_verify_state_async_dispatch(seq, num_accepted, k, last_inter_slot)
    }
    fn commit_verify_state_async_with_slot(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
        last_inter_slot: usize,
    ) -> Result<()> {
        // DFS reorder (option C): when the K=γ verify ran with
        // ATLAS_DDTREE_DFS_REORDER=1, the SSM kernel wrote h_state_inter in
        // DFS slot order — not original-compact order. greedy_sample_ddtree
        // returns indices in the ORIGINAL compact frame, so we must map
        // them via dfs_inv_perm[orig] = dfs_slot before reading the inter
        // pool. Mapping is a no-op when the buffer is empty (chain mode or
        // DFS-disabled).
        let inv_perm = self.ddtree_dfs_inv_perm.lock();
        let mapped_slot = if !inv_perm.is_empty() && last_inter_slot < inv_perm.len() {
            inv_perm[last_inter_slot]
        } else {
            last_inter_slot
        };
        drop(inv_perm);
        self.commit_verify_state_async_dispatch(seq, num_accepted, k, mapped_slot)
    }
    fn ep_worker_step(&self, seq: &mut SequenceState) -> Result<bool> {
        self.ep_worker_step_dispatch(seq)
    }
    fn is_ep(&self) -> bool {
        self.is_ep_dispatch()
    }
    fn is_mla(&self) -> bool {
        self.is_mla_dispatch()
    }
    fn decode_logits_fp32(&self) -> bool {
        self.decode_logits_fp32_dispatch()
    }
    fn decode_logits_ptr(&self) -> DevicePtr {
        self.decode_logits_ptr_dispatch()
    }
    fn ep_broadcast_cmd(&self, cmd: u32) -> Result<()> {
        self.ep_broadcast_cmd_dispatch(cmd)
    }
    fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        self.ep_broadcast_tokens_dispatch(tokens)
    }
    fn default_stream(&self) -> u64 {
        self.default_stream_dispatch()
    }
    fn create_stream(&self) -> Result<u64> {
        self.create_stream_dispatch()
    }
    fn create_event(&self) -> Result<u64> {
        self.create_event_dispatch()
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event_dispatch(event, stream)
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_dispatch(stream, event)
    }
}
