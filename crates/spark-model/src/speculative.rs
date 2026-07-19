// SPDX-License-Identifier: AGPL-3.0-only

//! Speculative decoding abstraction (SDD).
//!
//! Defines the [`DraftProposer`] trait for speculative decoding strategies.
//! MTP implements this first; EAGLE-3 can implement later without engine changes.

use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layer::ForwardContext;

/// Per-sequence state owned by a [`DraftProposer`].
///
/// Stores KV cache, hidden states, or whatever the proposer needs
/// across decode steps. Follows the same downcasting pattern as `LayerState`.
pub trait ProposerState: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A draft token proposer for speculative decoding.
///
/// The engine calls `propose()` after each target decode to get draft tokens,
/// then verifies them with the target model. `after_verify()` lets the
/// proposer trim state (e.g., KV cache) based on how many drafts were accepted.
pub trait DraftProposer: Send + Sync {
    /// Allocate per-sequence proposer state.
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>>;

    /// True when this proposer is the DFlash γ-block drafter.
    ///
    /// DFlash's block argmax is GPU-side and ignores the `grammar_bitmask`
    /// argument to [`Self::propose`], so the scheduler must gate (or
    /// verify-side mask) speculative drafting for grammar-constrained
    /// sequences — see `ATLAS_DFLASH_GRAMMAR_MODE` in spark-server. MTP and
    /// ngram proposers honor the mask and keep the default `false`.
    fn is_dflash(&self) -> bool {
        false
    }

    /// Propose up to `num_drafts` tokens autoregressively.
    ///
    /// # Arguments
    /// * `last_token` - The last verified token (target model output)
    /// * `target_hidden` - Target model's hidden states after final norm [1, hidden_size] BF16
    /// * `position` - Current sequence position (for RoPE)
    /// * `num_drafts` - Maximum number of draft tokens to produce
    /// * `state` - Per-sequence proposer state
    /// * `ctx` - Shared forward context (buffers, gpu, config)
    /// * `stream` - CUDA stream handle
    /// * `grammar_bitmask` - Optional XGrammar bitmask (ceil(vocab_size/32) i32
    ///   words). When `Some`, drafts are constrained to tokens the grammar
    ///   accepts at the current matcher position; bit `tok` set ⇒ allowed.
    ///   `None` preserves the unconstrained fast path.
    /// * `target_hidden_stack` - Optional pointer to a contiguous buffer of
    ///   `5 × target_hidden × bf16` containing the most-recently-decoded
    ///   token's hidden states captured at the drafter's `target_layer_ids`
    ///   (DFlash uses this; MTP ignores). Layout matches vLLM's
    ///   `combine_hidden_states` input: shallow-to-deep concatenation along
    ///   the feature axis.
    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>>;

    /// Read the draft token ID stored on GPU by the last `propose()` call
    /// that used `draft_embed_target = Some(...)`. Returns 0 if not supported.
    fn read_deferred_draft_token(&self, gpu: &dyn GpuBackend) -> Result<u32> {
        let _ = gpu;
        Ok(0)
    }

    /// Called after target verification to trim proposer state.
    ///
    /// `num_accepted` indicates how many draft tokens were accepted.
    /// The proposer should trim its KV cache / state to match.
    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        stream: u64,
    ) -> Result<()>;

    /// Free per-sequence proposer state (KV cache blocks, etc.).
    ///
    /// Must be called when a sequence is finished to avoid resource leaks.
    fn free_state(&self, state: &mut dyn ProposerState) -> Result<()> {
        let _ = state;
        Ok(())
    }

    /// ATLAS_DFLASH_ASYNC (task #20): collect the real drafts of an async
    /// (second-stream) propose previously launched for this sequence.
    ///
    /// Returns `Ok(None)` when nothing is pending (the universal default);
    /// `Ok(Some(drafts))` with the real chain to replace the placeholder;
    /// `Ok(Some(vec![]))` when the placeholder was orphaned — the caller
    /// clears its pending drafts and falls back to the bootstrap decode
    /// (lossless: drafts only ever propose).
    fn collect_async_drafts(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut dyn ProposerState,
    ) -> Result<Option<Vec<u32>>> {
        let _ = (gpu, state);
        Ok(None)
    }

    /// ATLAS_DFLASH_ASYNC: resolve (sync + discard) any in-flight async
    /// propose. Must be called before per-sequence device buffers the
    /// in-flight kernels may read are freed (`free_sequence`). Default no-op.
    fn resolve_async_inflight(
        &self,
        gpu: &dyn GpuBackend,
        state: Option<&mut dyn ProposerState>,
    ) -> Result<()> {
        let _ = (gpu, state);
        Ok(())
    }

    /// ATLAS_DFLASH_FUSED=1: record the propose-ordering CUDA event immediately
    /// after the target verify returns (before commit kernels are enqueued).
    /// Default no-op; only `BlockDiffusionDraftHead` overrides.
    fn arm_propose_overlap(&self, gpu: &dyn GpuBackend, default_stream: u64) -> Result<()> {
        let _ = (gpu, default_stream);
        Ok(())
    }

    /// DDTree M6: drain any pending tree payload built by the most recent
    /// `propose()` call. Default returns `None` (flat MTP / ngram / flat
    /// DFlash). DDTree-capable drafters override to return + clear the
    /// payload stashed on per-seq state. Caller assigns to
    /// `ActiveSeq.pending_tree_payload` for the next-step verifier.
    fn take_pending_tree_payload(
        &self,
        state: &mut dyn ProposerState,
    ) -> Option<crate::layers::DDTreePayload> {
        let _ = state;
        None
    }

    /// Sequentially populate the proposer's per-sequence KV cache for the last
    /// `k` prompt-tail positions before the first decode step.
    ///
    /// Without this, MTP's self-attention starts with an empty KV cache and
    /// predicts long-context drafts off zero historical context — empirically
    /// drops draft accept rate 1.83 → 0.92 (target_seq=5085 vs mtp_seq=423 at
    /// 4K prompt + 1K decode). Calling this after target prefill lets the
    /// drafter see the recent prompt tail (where attention mass concentrates),
    /// restoring most of the lost accept.
    ///
    /// # Arguments
    /// * `tokens` — slice of `k` prompt token IDs at absolute positions
    ///   `[base_position - k + 1 .. base_position]`. `tokens[i]` is the token
    ///   at absolute position `base_position - k + 1 + i`.
    /// * `target_hiddens` — device pointer to a contiguous BF16/FP32 buffer
    ///   of shape `[k, hidden_size]`. Row `i` is the target's last-layer
    ///   hidden state for the input token at the same absolute position as
    ///   `tokens[i]`. The proposer reads `i`-th row as input to step `i`.
    /// * `base_position` — absolute position of the LAST captured token
    ///   (`prompt_len - 1`). Step `i` uses position `base_position - k + 2 + i`
    ///   for RoPE (the position the proposer is predicting INTO).
    /// * `state` — per-sequence proposer state; KV cache is grown in-place.
    /// * `ctx` — shared forward context (buffers, gpu, config).
    /// * `stream` — CUDA stream handle.
    ///
    /// Default: no-op (proposers without prefill support skip silently).
    fn prefill_last_k(
        &self,
        tokens: &[u32],
        target_hiddens: DevicePtr,
        base_position: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = (tokens, target_hiddens, base_position, state, ctx, stream);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProposerState {
        tokens_proposed: Vec<u32>,
    }

    impl ProposerState for MockProposerState {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn test_proposer_state_downcast() {
        let state: Box<dyn ProposerState> = Box::new(MockProposerState {
            tokens_proposed: vec![42, 99],
        });
        let mock = state.as_any().downcast_ref::<MockProposerState>().unwrap();
        assert_eq!(mock.tokens_proposed, vec![42, 99]);
    }

    #[test]
    fn test_proposer_state_downcast_mut() {
        let mut state: Box<dyn ProposerState> = Box::new(MockProposerState {
            tokens_proposed: vec![],
        });
        let mock = state
            .as_any_mut()
            .downcast_mut::<MockProposerState>()
            .unwrap();
        mock.tokens_proposed.push(7);
        assert_eq!(mock.tokens_proposed, vec![7]);
    }
}
