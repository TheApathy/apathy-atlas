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
    pub(super) fn generate_speculative_dispatch(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        // Self-speculative mode: draft via layer-skipping (no MTP weights needed)
        if self.self_speculative {
            let mut seq = self.alloc_sequence()?;
            let stream = self.gpu.default_stream();
            let result = self.generate_self_speculative_inner(
                prompt_tokens,
                params,
                num_drafts,
                &mut seq,
                stream,
            );
            self.free_sequence(&mut seq)?;
            return result;
        }

        let proposer = match self.active_proposer() {
            Some(p) => p.clone(),
            None => {
                // Fallback to regular generation
                return crate::engine::generate(self, prompt_tokens, params);
            }
        };

        let mut seq = self.alloc_sequence()?;
        let stream = self.gpu.default_stream();

        let result = self.generate_speculative_inner(
            prompt_tokens,
            params,
            num_drafts,
            &proposer,
            &mut seq,
            stream,
        );

        self.free_sequence(&mut seq)?;

        result
    }

    pub(super) fn has_proposer_dispatch(&self) -> bool {
        self.any_proposer() || self.self_speculative
    }

    pub(super) fn has_self_speculative_dispatch(&self) -> bool {
        self.self_speculative
    }

    pub(super) fn decode_draft_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        TransformerModel::decode_draft(self, token, seq, stream)
    }

    pub(super) fn decode_draft_sparse_dispatch(
        &self,
        token: u32,
        thresh_frac: f32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        TransformerModel::decode_draft_sparse(self, token, thresh_frac, seq, stream)
    }

    pub(super) fn save_hidden_for_mtp_dispatch(
        &self,
        token_idx: usize,
        _stream: u64,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let _bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        // Save the RAW hidden state (before final_norm), not norm_output.
        // The MTP head applies its own pre_fc_norm_hidden — passing norm_output
        // would double-normalize and degrade prediction accuracy.
        let src = self.buffers.hidden_states().offset(token_idx * h * fp32);
        self.gpu
            .copy_d2d_async(src, self.mtp_hidden_save, h * fp32, stream)?;
        self.last_mtp_hidden_idx
            .store(token_idx, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub(super) fn run_mtp_propose_dispatch(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        let drafts = self.run_mtp_propose_multi(token, position, 1, seq, 0, None)?;
        Ok(drafts.into_iter().next())
    }

    pub(super) fn run_mtp_propose_multi_dispatch(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        _stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        // MTP loads ALL experts on every rank — no EP all_reduce needed.
        // Rank 1 does not participate in MTP propose.
        self.run_mtp_propose_inner(token, position, num_drafts, seq, grammar_bitmask)
    }

    pub(super) fn read_deferred_draft_token_dispatch(&self) -> Result<u32> {
        let proposer = match self.active_proposer() {
            Some(p) => p.as_ref(),
            None => return Ok(0),
        };
        proposer.read_deferred_draft_token(self.gpu.as_ref())
    }

    /// ATLAS_DFLASH_ASYNC: collect a deferred async propose's drafts for
    /// this sequence (no-op `None` when nothing is pending). See
    /// `dflash_head/async_propose.rs`.
    pub(super) fn dflash_collect_async_drafts_dispatch(
        &self,
        seq: &mut SequenceState,
    ) -> Result<Option<Vec<u32>>> {
        let Some(proposer) = self.active_proposer() else {
            return Ok(None);
        };
        let Some(ref mut state) = seq.proposer_state else {
            return Ok(None);
        };
        proposer.collect_async_drafts(self.gpu.as_ref(), state.as_mut())
    }

    pub(super) fn trim_proposer_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        let proposer = match self.active_proposer() {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let stream = self.gpu.default_stream();
        if let Some(ref mut state) = seq.proposer_state {
            proposer.after_verify(num_accepted, state.as_mut(), stream)?;
        }
        Ok(())
    }

    /// FIX 1 (ATLAS_DFLASH_TREE_COMMIT): stamp the accepted-path capture rows
    /// onto the DFlash proposer state so the next propose's ctx-hidden append
    /// reads the scattered (fork) capture rows instead of the contiguous
    /// prefix. Must be called AFTER `trim_proposer_state` (`after_verify`
    /// clears the path). No-op for non-DFlash proposers.
    ///
    /// TASK #29 frame fix: `accepted_compact` is in the COMPACT frame, but
    /// `dflash_hidden_save` (which these rows index) was written by
    /// `try_dflash_capture(layer, t)` in the KERNEL (DFS-reordered) frame —
    /// capture row `t` holds kernel slot `t`. So compact index `c`'s captured
    /// hidden lives at row `dfs_inv_perm[c]`. We map here through the SAME
    /// `ddtree_dfs_inv_perm` the KV gather and SSM commit use, then store the
    /// KERNEL-frame rows the append will read directly. Empty inv_perm (chain
    /// mode / DFS off) ⇒ identity ⇒ flat path unchanged.
    pub(super) fn set_dflash_accepted_compact_dispatch(
        &self,
        seq: &mut SequenceState,
        accepted_compact: &[usize],
    ) {
        let Some(ref mut state) = seq.proposer_state else {
            return;
        };
        if let Some(ds) = state
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
        {
            use crate::layers::dflash_head::ddtree::map_compact_path_to_kernel;
            let inv_perm = self.ddtree_dfs_inv_perm.lock();
            let kernel_rows = map_compact_path_to_kernel(accepted_compact, &inv_perm);
            drop(inv_perm);
            ds.last_accepted_compact.clear();
            ds.last_accepted_compact.extend_from_slice(&kernel_rows);
        }
    }
}
