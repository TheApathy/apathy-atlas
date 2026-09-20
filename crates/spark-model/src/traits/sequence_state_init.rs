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
            slot_idx,
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
        }
    }
}
