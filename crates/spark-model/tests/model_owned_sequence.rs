// SPDX-License-Identifier: AGPL-3.0-only
use spark_model::traits::SequenceState;

#[test]
fn model_owned_sequence_has_no_pool_or_cache_authority() {
    let s = SequenceState::for_model_owned_state(7);
    assert_eq!(s.slot_idx, 7);
    assert_eq!(s.ssm_slot_idx(), None);
    assert!(s.tokens.is_empty() && s.block_table.is_empty() && s.layer_states.is_empty());
    assert!(s.proposer_state.is_none() && s.chunked_prefill_meta.is_none());
    assert_eq!(
        (
            s.seq_len,
            s.kv_valid_tokens,
            s.prompt_len,
            s.cached_prefix_tokens
        ),
        (0, 0, 0, 0)
    );
    assert_eq!(
        (s.adapter_slot, s.acquired_adapter_slot, s.adapter_id),
        (-1, -1, 0)
    );
    assert_eq!(
        (s.marconi_skip_to, s.last_decode_ckpt_block, s.session_hash),
        (0, 0, 0)
    );
    assert!(s.marconi_exact_snap.is_none() && s.collect_prompt_logprobs.is_none());
    assert!(s.prompt_logprobs.is_empty() && s.disk_block_ids.is_empty());
    assert!(s.disk_last_offloaded_per_layer.is_empty());
    assert_eq!((s.src_lang_id, s.tgt_lang_id, s.num_beams), (0, 0, 1));
    assert_eq!(s.length_penalty, 1.0);
    assert!(!s.early_stopping);
}

#[test]
fn both_glm_allocators_claim_before_constructing_host_metadata() {
    for source in [
        include_str!("../src/model/glm53/model_trait_exl3.rs"),
        include_str!("../src/model/glm53/model_trait.rs"),
    ] {
        let tail = &source[source.find("fn alloc_sequence(").unwrap()..];
        let body = &tail[..tail.find("fn copy_logits_to_host(").unwrap()];
        let claim = body.find("self.claim_sequence()?").unwrap();
        let construct = body
            .find("SequenceState::for_model_owned_state(ONLY_SLOT)")
            .expect("model-owned state construction is shared");
        assert!(
            claim < construct,
            "metadata must not replace model ownership admission"
        );
        assert!(!body.contains("SequenceState {"));
    }
}
