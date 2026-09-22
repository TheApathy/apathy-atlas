// SPDX-License-Identifier: AGPL-3.0-only
use spark_model::traits::SequenceState;

#[test]
fn model_owned_sequence_has_no_pool_or_cache_authority() {
    // This engine's SequenceState has no adapter/NLLB/beam/Marconi-snapshot
    // fields; the upstream assertions on those are dropped, the rest kept.
    let s = SequenceState::for_model_owned_state(7);
    assert_eq!(s.slot_idx, 7);
    assert!(s.tokens.is_empty() && s.block_table.is_empty() && s.layer_states.is_empty());
    assert!(s.proposer_state.is_none() && s.proposer_state_alt.is_none());
    assert!(s.chunked_prefill_meta.is_none());
    assert_eq!(
        (
            s.seq_len,
            s.kv_valid_tokens,
            s.prompt_len,
            s.cached_prefix_tokens
        ),
        (0, 0, 0, 0)
    );
    assert_eq!((s.marconi_skip_to, s.session_hash), (0, 0));
    assert!(s.disk_block_ids.is_empty() && s.disk_last_offloaded_per_layer.is_empty());
    assert!(s.mtp_lastk_host_buf.is_empty());
    assert!(!s.qwen4_qsa_required);
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
