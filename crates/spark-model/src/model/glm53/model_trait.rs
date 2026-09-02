// SPDX-License-Identifier: AGPL-3.0-only

//! `Model` for the GLM-5.3 target.
//!
//! Only the single-sequence, non-speculative decode path is implemented. Every
//! other seam **refuses by name** rather than returning a plausible value: a
//! stub that returned `Ok` would let the scheduler drive speculation or
//! batching against a model that silently ignored it, which is the
//! fluent-wrong-output failure this port keeps guarding against.
//!
//! What is deliberately a no-op rather than a refusal:
//!
//! * `checkpoint_ssm_states` / `rollback_ssm_states` — this path never
//!   speculates, so there is nothing to check point or roll back. The staging
//!   split in `kda_state_binding` is what actually protects recurrent state.
//! * `cache_sequence`, `save_hidden_for_mtp`, `trim_proposer_state` — no prefix
//!   cache and no proposer are wired, so there is nothing to record.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use crate::traits::{Model, SequenceState};

use super::target_model::Glm53Model;

/// Slot for the one sequence this model serves.
const ONLY_SLOT: usize = 0;

impl Model for Glm53Model {
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        let logits = self.prefill_tokens(tokens, stream)?;
        seq.tokens = tokens.to_vec();
        seq.seq_len = tokens.len();
        seq.prompt_len = tokens.len();
        seq.kv_valid_tokens = tokens.len();
        Ok(logits)
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
        // The schedule is built for exactly one token, so the prompt is walked
        // sequentially. Chunking would only be meaningful once a batched
        // prefill schedule exists.
        if chunk_start != 0 || chunk_len != tokens.len() || !is_last_chunk {
            bail!(
                "GLM-5.3 has no chunked prefill: the target schedule is one token wide, so the \
                 whole prompt must arrive as a single chunk"
            );
        }
        self.prefill(tokens, seq, stream)
    }

    fn decode(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        let logits = self.decode_token(token, stream)?;
        seq.tokens.push(token);
        seq.seq_len += 1;
        seq.kv_valid_tokens = seq.seq_len;
        Ok(logits)
    }

    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        // Batching needs per-sequence recurrent and latent state; the arena is
        // planned for batch 1 (`exact_full_1m_b1`).
        if tokens.len() != 1 || seqs.len() != 1 {
            bail!(
                "GLM-5.3 serves one sequence: the arena is planned for batch 1, so a {}-token \
                 batch has nowhere to keep its state",
                tokens.len()
            );
        }
        self.decode(tokens[0], seqs[0], stream)
    }

    fn vocab_size(&self) -> usize {
        self.vocab()
    }

    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.bind_thread()
    }

    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState {
            adapter_id: 0,
            adapter_slot: -1,
            acquired_adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            layer_states: Vec::new(),
            proposer_state: None,
            slot_idx: ONLY_SLOT,
            ssm_slot: None,
            marconi_skip_to: 0,
            marconi_exact_snap: None,
            session_hash: 0,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            kv_valid_tokens: 0,
            last_decode_ckpt_block: 0,
            prompt_len: 0,
            collect_prompt_logprobs: None,
            prompt_logprobs: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: Vec::new(),
        })
    }

    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.copy_logits(logits_ptr, dst)
    }

    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.logits_ptr()
    }

    fn argmax_on_device(&self, logits_ptr: DevicePtr, stream: u64) -> Result<u32> {
        self.argmax_host(logits_ptr, stream)
    }

    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>> {
        if n != 1 {
            bail!("GLM-5.3 serves one sequence; argmax_batch was asked for {n}");
        }
        Ok(vec![self.argmax_on_device(logits_ptr, stream)?])
    }

    fn hidden_after_norm(&self) -> DevicePtr {
        // The NextN/MTP head is not wired, so there is no hidden state to hand
        // a drafter. A non-null pointer here would be read as a valid capture.
        DevicePtr(0)
    }

    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        // The one sequence's state lives in the arena for the model's lifetime;
        // there is no per-sequence allocation to release.
        Ok(())
    }

    fn compact_sequence(&self, _seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        if new_slot == ONLY_SLOT {
            return Ok(());
        }
        bail!("GLM-5.3 has a single sequence slot; cannot compact into slot {new_slot}")
    }

    fn detach_slot_for_reuse(&self, seq: &mut SequenceState) {
        seq.slot_idx = usize::MAX;
    }

    fn cache_sequence(&self, _seq: &SequenceState) {}

    fn decode_marconi_checkpoint(&self, _seq: &mut SequenceState) {}

    fn checkpoint_ssm_states(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    fn rollback_ssm_states(&self, _seq: &mut SequenceState, num_accepted: usize) -> Result<()> {
        // A rollback request means something above believed it could speculate.
        // Accepting it silently would leave recurrent state one step ahead of
        // the sequence with no way back.
        if num_accepted == 0 {
            bail!(
                "GLM-5.3 cannot roll back recurrent state: the accept path publishes without \
                 completion receipts, so a rejected step has already advanced it"
            );
        }
        Ok(())
    }

    fn has_proposer(&self) -> bool {
        false
    }

    fn has_self_speculative(&self) -> bool {
        false
    }

    fn save_hidden_for_mtp(&self, _token_idx: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn trim_proposer_state(
        &self,
        _seq: &mut SequenceState,
        _num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        Ok(())
    }

    fn run_mtp_propose(
        &self,
        _token: u32,
        _position: usize,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        Ok(None)
    }

    fn run_mtp_propose_multi(
        &self,
        _token: u32,
        _position: usize,
        _num_drafts: usize,
        _seq: &mut SequenceState,
        _stream: u64,
        _grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }

    fn decode_draft(
        &self,
        _token: u32,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        bail!("GLM-5.3 has no drafter wired")
    }

    fn decode_verify(
        &self,
        _tokens: &[u32],
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        bail!("GLM-5.3 speculative verify is not implemented")
    }

    fn decode_verify_graphed(
        &self,
        _tokens: &[u32; 2],
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 2]> {
        bail!("GLM-5.3 graphed verify is not implemented")
    }

    fn decode_verify_graphed_k3(
        &self,
        _tokens: &[u32; 3],
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 3]> {
        bail!("GLM-5.3 graphed verify is not implemented")
    }

    fn decode_verify_graphed_k4(
        &self,
        _tokens: &[u32; 4],
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 4]> {
        bail!("GLM-5.3 graphed verify is not implemented")
    }

    fn generate_speculative(
        &self,
        _prompt_tokens: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        bail!("GLM-5.3 speculative generation is not implemented")
    }

    fn has_ssm_layers(&self) -> bool {
        // 34 of 45 layers carry recurrent KDA state.
        true
    }

    fn is_mla(&self) -> bool {
        // The 11 DSA layers are absorbed-MLA, which forces single-chunk prefill
        // — the same constraint `prefill_chunk` enforces above.
        true
    }

    fn num_free_blocks(&self) -> usize {
        // State lives in the arena, not in the scheduler's paged cache, so
        // admission must not reject on block count.
        1 << 20
    }

    fn num_total_blocks(&self) -> usize {
        1 << 20
    }
}
