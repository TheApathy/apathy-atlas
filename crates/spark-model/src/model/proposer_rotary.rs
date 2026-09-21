// SPDX-License-Identifier: AGPL-3.0-only

//! Target MRoPE belongs to the native head, not the qwen3 DFlash2 drafter.

use crate::model::types::TransformerModel;
use crate::traits::SequenceState;
use anyhow::{Result, ensure};

impl TransformerModel {
    pub(super) fn sync_proposer_rotary(&self, seq: &mut SequenceState) -> Result<()> {
        let Some(state) = seq.proposer_state.as_mut() else {
            return Ok(());
        };
        if let Some(mtp) = state
            .as_any_mut()
            .downcast_mut::<crate::layers::mtp_head::MtpProposerState>()
        {
            mtp.rotary_positions = seq.rotary_positions.clone();
        } else if state.as_any().is::<crate::layers::DflashProposerState>() {
            // Trained qwen3/default-RoPE DFlash2 uses chronological PHYSICAL
            // context and noise indices. Do not change any cached/ring index.
        } else {
            ensure!(
                seq.rotary_positions.is_identity(),
                "this proposer does not admit target image rotary state"
            );
        }
        Ok(())
    }
}
