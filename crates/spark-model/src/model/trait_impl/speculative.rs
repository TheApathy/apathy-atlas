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

        let proposer = match &self.proposer {
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
        self.proposer.is_some() || self.self_speculative
    }

    /// Whether self-speculative drafting can actually do anything on THIS
    /// model. Requesting it is not the same as being capable of it.
    ///
    /// The mechanism is layer-*type* skipping, not depth truncation:
    /// [`TransformerModel::decode_draft`] runs the full layer loop and
    /// `continue`s only on `LayerType::LinearAttention` (impl_b1.rs). On a
    /// model with zero SSM layers it therefore skips NOTHING — the "draft" is
    /// a complete forward pass, so a step costs `num_drafts` full forwards
    /// plus a verify.
    ///
    /// On DeepSeek-V4-Flash (43/43 `FullAttention`, see
    /// `config/parsers/deepseek_v4.rs`) that arms a path whose ceiling AT
    /// PERFECT ACCEPTANCE is 13.6–18.1 tok/s against plain decode's 21.9 —
    /// i.e. it cannot win at any acceptance rate. Full derivation in
    /// `docs/SELF-SPECULATION-ANALYSIS.md` §0.
    ///
    /// So the predicate is derived from the config, never from the request —
    /// the same SSOT idiom as `ModelConfig::has_recurrent_state`
    /// (`atlas-core/src/config/methods.rs`). `serve.rs` already falls back to
    /// plain decode when this returns false, so the degradation is graceful.
    /// Hybrid SSM/attention models (Qwen3-Next, Nemotron-H) are unaffected.
    pub(super) fn has_self_speculative_dispatch(&self) -> bool {
        self.self_speculative && self.config.num_ssm_layers() > 0
    }

    pub(super) fn decode_draft_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        TransformerModel::decode_draft(self, token, seq, stream)
    }

    /// ATLAS_MTP_DRAFTER_PREFILL: copy this prefill chunk's final-layer
    /// hiddens (`[proc_count, h]` BF16, contiguous at the head of the hidden
    /// buffer) into the whole-prompt capture at row `chunk_start`.
    ///
    /// Contiguity-tracked: `chunk_start == 0` (re)starts the capture; a chunk
    /// extending the current range appends; anything else (prefix-cache
    /// reuse, Marconi warm restore — rows whose hiddens were never computed)
    /// leaves the tracked length short, which safely disables the drafter
    /// prefill for that sequence via the coverage check at the propose site.
    pub(super) fn try_mtp_prefill_capture(
        &self,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        if self.mtp_prefill_hidden.is_null() || proc_count == 0 {
            return Ok(());
        }
        use std::sync::atomic::Ordering;
        let len = self.mtp_prefill_capture_len.load(Ordering::Relaxed);
        let new_len = if chunk_start == 0 {
            proc_count
        } else if chunk_start == len {
            len + proc_count
        } else {
            return Ok(());
        };
        if chunk_start + proc_count > self.mtp_prefill_capacity {
            return Ok(());
        }
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        self.gpu.copy_d2d_async(
            self.buffers.hidden_states(),
            self.mtp_prefill_hidden.offset(chunk_start * h * bf16),
            proc_count * h * bf16,
            stream,
        )?;
        self.mtp_prefill_capture_len
            .store(new_len, Ordering::Relaxed);
        Ok(())
    }

    pub(super) fn save_hidden_for_mtp_dispatch(
        &self,
        token_idx: usize,
        _stream: u64,
    ) -> Result<()> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        // Residual stream is always BF16, so the saved hidden is BF16.
        let fp32 = 2usize;
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

    /// ATLAS_MTP_CATCHUP: ring-capture the final hidden of a serially
    /// decoded token (position `pos`), keeping the ring's position range
    /// contiguous (a gap resets the range to just this row).
    pub(super) fn save_hidden_for_catchup_dispatch(
        &self,
        token_idx: usize,
        pos: usize,
    ) -> Result<()> {
        if self.mtp_catchup_ring.is_null() {
            return Ok(());
        }
        let ring_rows = super::super::types::MTP_CATCHUP_RING_ROWS;
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let src = self.buffers.hidden_states().offset(token_idx * h * bf16);
        let dst = self.mtp_catchup_ring.offset((pos % ring_rows) * h * bf16);
        self.gpu.copy_d2d_async(src, dst, h * bf16, stream)?;
        let mut meta = self.mtp_catchup_meta.lock();
        let (start, count) = *meta;
        *meta = if count > 0 && pos == start + count {
            // Contiguous append; cap the range at ring capacity by advancing
            // the start once the ring wraps (oldest row overwritten).
            if count == ring_rows {
                (start + 1, ring_rows)
            } else {
                (start, count + 1)
            }
        } else {
            (pos, 1)
        };
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
        let out = self.run_mtp_propose_inner(token, position, num_drafts, seq, grammar_bitmask);
        // ── Task 1 DIAG (ATLAS_DFLASH_DIAG=1): the drafts.len() that actually
        // leaves the model layer for the scheduler. Differs from propose.rs
        // DIAG (c) ONLY when the confidence-tau trim in run_mtp_propose_inner
        // (impl_b3.rs) fired and returned an empty Vec — the one place <4 (in
        // fact 0) can appear that propose.rs's own DIAG doesn't show.
        // (Env read cached once — this fired on EVERY propose.)
        static DIAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let diag =
            *DIAG.get_or_init(|| std::env::var("ATLAS_DFLASH_DIAG").ok().as_deref() == Some("1"));
        if diag {
            match &out {
                Ok(d) => tracing::info!(
                    "DFLASH DIAG multi_dispatch: requested num_drafts={} → returned len={} \
                     (token={} position={}); scheduler >=4 dispatch sees THIS",
                    num_drafts,
                    d.len(),
                    token,
                    position,
                ),
                Err(e) => tracing::info!(
                    "DFLASH DIAG multi_dispatch: propose ERRORED (num_drafts={} token={} \
                     position={}): {e:#} → scheduler gets empty → plain decode",
                    num_drafts,
                    token,
                    position,
                ),
            }
        }
        out
    }

    pub(super) fn read_deferred_draft_token_dispatch(&self) -> Result<u32> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(0),
        };
        proposer.read_deferred_draft_token(self.gpu.as_ref())
    }

    pub(super) fn trim_proposer_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let stream = self.gpu.default_stream();
        if let Some(ref mut state) = seq.proposer_state {
            proposer.after_verify(num_accepted, state.as_mut(), stream)?;
        }
        Ok(())
    }
}
