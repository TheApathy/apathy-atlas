// SPDX-License-Identifier: AGPL-3.0-only

use crate::model::types::TransformerModel;
use crate::traits::{RotaryPositions, SequenceState};
use anyhow::{Result, ensure};

impl TransformerModel {
    /// Admit the full prompt before any prefill device writes. Later chunks
    /// must retain the exact same image geometry and physical span mapping.
    pub(in crate::model) fn admit_rotary_prompt(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
    ) -> Result<()> {
        let candidate = if self.config.mrope_interleaved && self.vision_prompt_present(tokens) {
            let config = self
                .config
                .vision
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("image rotary configuration missing"))?;
            let state = self.vision_embeddings.lock();
            let plan = state.plan(tokens, config.image_pad_token_id, 0, 0)?;
            RotaryPositions::from_image_spans(tokens.len(), state.max_rows, &plan.image_spans)?
        } else {
            RotaryPositions::identity()
        };
        if !candidate.is_identity() {
            ensure!(
                self.kv_cache.lock().config().cache_blocks_per_seq.is_none(),
                "image rotary positions are not supported by high-speed sliding swap"
            );
        }
        if chunk_start == 0 {
            if crate::model::rotary_graphs::layout_changes(&seq.rotary_positions, &candidate) {
                self.invalidate_rotary_graphs(seq.slot_idx)?;
            }
            seq.rotary_positions = candidate;
        } else {
            ensure!(
                seq.rotary_positions == candidate,
                "rotary prompt geometry changed across prefill chunks"
            );
        }
        Ok(())
    }
}
