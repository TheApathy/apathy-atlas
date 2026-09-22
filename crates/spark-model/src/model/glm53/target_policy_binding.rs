// SPDX-License-Identifier: AGPL-3.0-only

//! Bind real target state before the scheduler borrows its policy view.

use super::*;
use crate::model::glm53::verify_policy_binding::{BoundVerifyRequest, VerifyFrame};
use crate::model::glm53::verify_policy_transaction::VerifyRequest;
use crate::traits::SequenceState;

impl Glm53Exl3Model {
    fn verify_frame(&self, state: &WalkState, stream: u64) -> Result<VerifyFrame> {
        ensure!(
            self.live_sequence.load(Ordering::Acquire),
            "GLM policy verify has no live sequence"
        );
        ensure!(
            state.poisoned_stream.is_none(),
            "GLM policy verify sequence is poisoned"
        );
        ensure!(
            stream == self.gpu.default_stream(),
            "GLM policy verify requires the model stream"
        );
        Ok(VerifyFrame {
            generation: state.generation,
            nonce: state.nonce,
            position: state.position as usize,
            capacity: self.capacity as usize,
            vocab: VOCAB as usize,
            stream,
        })
    }

    pub(in crate::model::glm53) fn bind_policy_request(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        stream: u64,
    ) -> Result<BoundVerifyRequest> {
        let mut state = self.state.lock().unwrap();
        ensure!(
            seq.slot_idx == 0
                && seq.tokens.len() == seq.seq_len
                && seq.kv_valid_tokens == seq.seq_len
                && seq.seq_len == state.position as usize,
            "GLM policy binding requires the actual model-owned host prefix"
        );
        let mut frame = self.verify_frame(&state, stream)?;
        frame.nonce = frame
            .nonce
            .checked_add(1)
            .context("GLM verify nonce exhausted")?;
        let bound = self.verify_binding_owner.bind(frame, &seq.tokens, tokens)?;
        state.nonce = frame.nonce;
        Ok(bound)
    }

    pub(super) fn consume_policy_request(
        &self,
        bound: BoundVerifyRequest,
    ) -> Result<(VerifyRequest, u64)> {
        let stream = bound.stream();
        let mut state = self.state.lock().unwrap();
        let frame = self.verify_frame(&state, stream)?;
        let next = state
            .nonce
            .checked_add(1)
            .context("GLM verify nonce exhausted")?;
        let request = self.verify_binding_owner.consume(bound, frame)?;
        state.nonce = next;
        Ok((request, stream))
    }
}
