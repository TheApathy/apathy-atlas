// SPDX-License-Identifier: AGPL-3.0-only

//! Production-seam RED gates, supplementary to the actual scheduler tests.
//! These checks do not establish arithmetic, model, or GPU correctness.

use std::path::PathBuf;

fn source(name: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    std::fs::read_to_string(root.join(name)).expect("required production source")
}

fn compact(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn legacy_width_controls_remain_available_for_other_models() {
    let text = source("mtp_step.rs");
    for method in [
        "step_verify_k2(",
        "step_verify_k3(",
        "step_verify_k4(",
        "step_verify_dflash(",
    ] {
        assert_eq!(text.matches(method).count(), 2, "{method}");
    }
}

#[test]
fn typed_driver_precedes_bootstrap_and_legacy_grammar_mutation() {
    let text = source("mtp_step.rs");
    let body = &text[text.find("pub fn step_mtp(").unwrap()..];
    let route = body
        .find("model.requires_verify_policy()")
        .expect("typed route");
    let legacy = body.find("let mut bootstrap_idxs").unwrap();
    assert!(route < legacy, "all GLM N1 and K2..K8 paths require policy");
    let branch = &body[route..legacy];
    assert!(branch.contains("glm53_policy_driver::step("));
    assert!(branch.contains("adaptive_sampling"));
    assert!(branch.contains("code_fence_token"));
    assert!(
        branch.contains("return;"),
        "no fallthrough into raw acceptance"
    );
}

#[test]
fn policy_driver_uses_owned_binding_and_commit_before_publication() {
    let text = compact(&source("glm53_policy_driver.rs"));
    let bind = text
        .find("bind_verify_policy_request(")
        .expect("owned binding");
    let policy = text
        .find("OrdinaryVerifyPolicy::new(")
        .expect("actual adapter");
    let verify = text
        .find("decode_verify_with_policy(")
        .expect("policy transaction");
    let publish = text.find(".publish(").expect("consumed commit receipt");
    let install = text.find(".install(").expect("retained ordinary state");
    assert!(bind < policy && policy < verify && verify < publish && publish < install);
    for forbidden in [
        "decode_verify_dflash(",
        "decode_and_verify_fused(",
        "verify_pick_all_with_pipeline(",
        "emit_token(",
        "commit_accepted_prefix(",
        "trim_proposer_state(",
        "commit_ctx(",
        "SequenceState::for_model_owned_state",
        "std::mem::take(&mut a.seq",
    ] {
        assert!(
            !text.contains(forbidden),
            "policy driver cannot use {forbidden}"
        );
    }
    assert!(text.contains("poison_verify_policy("));
}

#[test]
fn ordinary_replay_and_normal_decode_share_the_actual_logits_entry() {
    let text = source("decode_logits_step.rs");
    assert!(text.contains("fn process_decode_logits_slice("));
    assert_eq!(text.matches("ordinary_greedy::try_pick_batch(").count(), 1);
    assert_eq!(text.matches("process_seq_logits(").count(), 1);
    assert_eq!(text.matches("advance_ordinary(").count(), 1);
    let driver = source("glm53_policy_driver.rs");
    let replay = source("glm53_policy_replay.rs");
    assert!(driver.contains("RestoredForOrdinaryReplay"));
    assert!(driver.contains("validate_host_prefix(") || replay.contains("validate_host_prefix("));
    assert!(replay.contains("process_decode_logits_slice("));
    assert!(replay.contains("std::slice::from_mut("));
    assert!(replay.contains("decode_batch("));
    assert!(!replay.contains("emit_token("));
    assert!(!replay.contains("sample_token_with_grammar("));
}

#[test]
fn zero_drafts_suspension_and_serial_seam_keep_real_ordinary_entry() {
    let text = source("glm53_policy_driver.rs");
    assert!(text.contains("num_drafts == 0"));
    assert!(text.contains("adaptive_spec::spec_allowed(a)"));
    assert!(text.contains("dflash_seam_serial_enabled()"));
    assert!(text.contains("replay::ordinary("));
    assert!(text.contains("adaptive_spec::tick_serial(a)"));
}
