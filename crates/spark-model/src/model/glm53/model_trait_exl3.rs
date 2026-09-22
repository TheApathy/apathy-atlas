// SPDX-License-Identifier: AGPL-3.0-only

//! Production `Model` boundary for the single-sequence GLM-5.3 EXL3 target.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use crate::traits::{Model, SequenceState};

use super::target_model_exl3::Glm53Exl3Model;

const ONLY_SLOT: usize = 0;
use super::verify_policy_binding::BoundVerifyRequest;
use super::verify_policy_transaction::{VerifyOutcome, VerifyPolicy};

impl Model for Glm53Exl3Model {
    fn requires_verify_policy(&self) -> bool {
        true
    }

    fn bind_verify_policy_request(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        stream: u64,
    ) -> Result<BoundVerifyRequest> {
        self.bind_policy_request(tokens, seq, stream)
    }

    fn decode_verify_with_policy(
        &self,
        request: BoundVerifyRequest,
        policy: &mut dyn VerifyPolicy,
    ) -> Result<VerifyOutcome> {
        self.verify_dflash2_with_policy(request, policy)
    }

    fn poison_verify_policy(&self, stream: u64) {
        self.poison_verify(stream);
    }

    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        let logits = self.prefill_request(tokens, stream)?;
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
        if chunk_start != 0 || chunk_len != tokens.len() || !is_last_chunk {
            bail!("GLM-5.3 EXL3 has no chunked prefill: the whole prompt must arrive as one chunk");
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
        if tokens.len() != 1 || seqs.len() != 1 {
            bail!(
                "GLM-5.3 EXL3 serves one sequence; decode batch has {} tokens and {} sequences",
                tokens.len(),
                seqs.len()
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
        self.claim_sequence()?;
        Ok(SequenceState::for_model_owned_state(ONLY_SLOT))
    }

    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.copy_logits(logits_ptr, dst)
    }

    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.logits_ptr()
    }

    fn argmax_on_device(&self, logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        Glm53Exl3Model::argmax_host(self, logits_ptr)
    }

    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>> {
        if n != 1 {
            bail!("GLM-5.3 EXL3 serves one sequence; argmax batch was asked for {n}");
        }
        Ok(vec![self.argmax_on_device(logits_ptr, stream)?])
    }

    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }

    fn free_sequence(&self, _seq: &mut SequenceState) -> Result<()> {
        self.release_sequence();
        Ok(())
    }

    fn compact_sequence(&self, _seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        if new_slot == ONLY_SLOT {
            Ok(())
        } else {
            bail!("GLM-5.3 EXL3 cannot compact its only sequence into slot {new_slot}")
        }
    }

    fn cache_sequence(&self, _seq: &SequenceState) {}

    fn checkpoint_ssm_states(&self, _seq: &mut SequenceState) -> Result<()> {
        Ok(())
    }

    fn rollback_ssm_states(&self, _seq: &mut SequenceState, _num_accepted: usize) -> Result<()> {
        // The GLM verifier restores and serially replays a partial prefix before
        // returning, so the scheduler-level rollback hook has no remaining work.
        Ok(())
    }

    fn has_proposer(&self) -> bool {
        self.has_dflash2()
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
        token: u32,
        _position: usize,
        _seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Option<u32>> {
        Ok(self.propose_dflash2(token, 1, stream)?.into_iter().next())
    }

    fn run_mtp_propose_multi(
        &self,
        token: u32,
        _position: usize,
        num_drafts: usize,
        _seq: &mut SequenceState,
        stream: u64,
        _grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.propose_dflash2(token, num_drafts, stream)
    }

    fn decode_draft(
        &self,
        _token: u32,
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<DevicePtr> {
        bail!("GLM-5.3 EXL3 has no trait-level drafter wired")
    }

    fn decode_verify(
        &self,
        _tokens: &[u32],
        _seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        bail!(
            "GLM EXL3 requires bound policy-before-commit verification; raw Model verifier refused"
        )
    }

    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 2]> {
        self.decode_verify(tokens, seq, stream)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("GLM DFlash2 K2 verifier returned the wrong row count"))
    }

    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 3]> {
        self.decode_verify(tokens, seq, stream)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("GLM DFlash2 K3 verifier returned the wrong row count"))
    }

    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 4]> {
        self.decode_verify(tokens, seq, stream)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("GLM DFlash2 K4 verifier returned the wrong row count"))
    }

    fn decode_verify_graphed_kgamma(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify(tokens, seq, stream)
    }

    fn generate_speculative(
        &self,
        _prompt_tokens: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        bail!("GLM-5.3 EXL3 speculative generation is not exposed through Model")
    }

    fn prepare_vision_embed(&self, images: &[(Vec<f32>, usize, usize)]) -> Result<()> {
        self.prepare_vision_images(images)
    }

    fn has_ssm_layers(&self) -> bool {
        true
    }

    fn is_mla(&self) -> bool {
        true
    }

    fn num_free_blocks(&self) -> usize {
        1 << 20
    }

    fn num_total_blocks(&self) -> usize {
        1 << 20
    }

    fn default_stream(&self) -> u64 {
        self.gpu().default_stream()
    }

}
