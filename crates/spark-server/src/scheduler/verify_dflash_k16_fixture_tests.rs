// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn every_accepted_length_has_one_never_emitted_mismatch() {
    let chain: Vec<u32> = (0..DRAFTS as u32).collect();
    for accepted in 0..=DRAFTS {
        let drafts = forced_drafts(&chain, accepted, LOGICAL_VOCAB).unwrap();
        let observed = drafts
            .iter()
            .zip(&chain)
            .take_while(|(a, b)| a == b)
            .count();
        assert_eq!(observed, accepted);
    }
    let mut edge = chain;
    edge[0] = LOGICAL_VOCAB as u32 - 1;
    assert_eq!(forced_drafts(&edge, 0, LOGICAL_VOCAB).unwrap()[0], 0);
    edge[1] = LOGICAL_VOCAB as u32;
    assert!(forced_drafts(&edge, 1, LOGICAL_VOCAB).is_err());
    edge[1] = 1;
    assert!(forced_drafts(&edge[..DRAFTS - 1], 1, LOGICAL_VOCAB).is_err());
    assert!(forced_drafts(&edge, K16, LOGICAL_VOCAB).is_err());
}

#[test]
fn scheduler_observes_before_emission_and_receipts_after_commit() {
    let source = include_str!("verify_dflash_step.rs");
    let prepare = source.find("k16_acceptance_fixture::prepare").unwrap();
    let observe = source.find("fixture.observe").unwrap();
    let accept = source
        .find("let (num_accepted, tree_last_inter_slot)")
        .unwrap();
    let emit = source.find("let emit_take = num_accepted").unwrap();
    let serial_oracle = source.find("let serial_oracle =").unwrap();
    let evidence_begin = source.find("EvidenceGuard::begin").unwrap();
    let production_verify = source
        .find("let mut verified = match model.decode_verify_dflash")
        .unwrap();
    let commit = source.find("let commit_res =").unwrap();
    let finish = source.find("fixture.finish").unwrap();
    assert!(prepare < observe && observe < accept);
    assert!(accept < emit && emit < commit && commit < finish);
    assert!(serial_oracle < evidence_begin && evidence_begin < production_verify);
    assert!(production_verify < observe);

    let fixture = include_str!("verify_dflash_k16_fixture.rs");
    let query = fixture
        .find("evidence.confirm(pre_verify_len, inputs)?")
        .unwrap();
    let success = fixture
        .find("DFLASH_K16_ACCEPTANCE_FIXTURE match=true")
        .unwrap();
    assert!(query < success);
}

#[test]
fn production_route_markers_are_at_the_exact_k16_call_sites() {
    let verify = include_str!("../../../spark-model/src/model/trait_impl/verify_d.rs");
    let attention = include_str!(
        "../../../spark-model/src/layers/qwen3_attention/decode/qwen4_k5_projection.rs"
    );
    let ssm = include_str!("../../../spark-model/src/layers/qwen3_ssm/trait_decode.rs");
    assert!(verify.contains("k16_route_receipt::mark_batched_entry(seq.seq_len, tokens)?"));
    assert!(attention.contains("k16_route_receipt::mark_qkv_layer()?"));
    assert!(ssm.contains("k16_route_receipt::mark_ssm_layer()?"));
}
