// SPDX-License-Identifier: AGPL-3.0-only
//! Host metadata for models whose recurrent/cache resources live on the model.
use super::SequenceState;

impl SequenceState {
    /// Construct empty host metadata for an explicitly selected model-owned
    /// slot. This does not acquire a slot, allocate device state, initialize
    /// caches, or authorize inference. The model must perform its own claim
    /// before returning this metadata from `Model::alloc_sequence`.
    pub fn for_model_owned_state(slot_idx: usize) -> Self {
        Self {
            tokens: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            rotary_positions: super::RotaryPositions::identity(),
            qwen4_qsa_required: false,
            layer_states: Vec::new(),
            proposer_state: None,
            proposer_state_alt: None,
            slot_idx,
            marconi_skip_to: 0,
            session_hash: 0,
            chunked_prefill_meta: None,
            cached_prefix_tokens: 0,
            kv_valid_tokens: 0,
            prompt_len: 0,
            disk_block_ids: Vec::new(),
            mtp_lastk_host_buf: Vec::new(),
            mtp_lastk_host_filled: 0,
            mtp_lastk_end_abs: 0,
            disk_last_offloaded_per_layer: Vec::new(),
        }
    }
}
