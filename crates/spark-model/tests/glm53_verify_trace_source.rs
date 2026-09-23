// SPDX-License-Identifier: AGPL-3.0-only

const TRANSACTION: &str = include_str!("../src/model/glm53/dsa_verify_execution.rs");
const TARGET: &str = include_str!("../src/model/glm53/target_model_exl3.rs");
const CUDA: &str = include_str!("../../../kernels/gb10/glm5.3-flash/iq3/glm53_kda.cu");

#[test]
fn trace_configuration_is_admitted_before_state_snapshot_effects() {
    let trace = TRANSACTION
        .find("verify_trace::selection_from_env()?")
        .expect("trace selector admission");
    let snapshot = TRANSACTION.find("self.bind_dsa_snapshot").unwrap();
    assert!(trace < snapshot);
}

#[test]
fn wide_trace_observes_selected_oracle_inside_precommit_error_boundary() {
    let staged = TRANSACTION.find("let staged = (||").unwrap();
    let observe = TRANSACTION
        .find("TracePhase::Wide")
        .expect("wide trace hook");
    let returned = TRANSACTION.find("Ok(oracle)").unwrap();
    let acceptance = TRANSACTION.find("let accepted = drafts").unwrap();
    assert!(staged < observe && observe < returned && returned < acceptance);
    let hook = &TRANSACTION[staged..returned];
    assert!(hook.contains("oracle[0]"));
    assert!(hook.contains("verify_trace::capture_row("));
}

#[test]
fn replay_trace_follows_existing_walk_inside_commit_failure_boundary() {
    let restored = TRANSACTION.find("if accepted != drafts.len()").unwrap();
    let committed = TRANSACTION.find("let committed = (||").unwrap();
    let replay = TRANSACTION[committed..]
        .find("self.walk(token, stream)?")
        .unwrap()
        + committed;
    let observe = TRANSACTION
        .find("TracePhase::Replay")
        .expect("replay trace hook");
    let failure = TRANSACTION.find("if let Err(error) = committed").unwrap();
    assert!(restored < committed && committed < replay && replay < observe && observe < failure);
    assert_eq!(
        TRANSACTION.matches("self.walk(token, stream)?").count(),
        1,
        "trace must not add a forward"
    );
}

#[test]
fn diagnostic_does_not_change_any_existing_pick_or_scope_policy() {
    assert!(TRANSACTION.contains("self.argmax_rows_device(logits, tokens.len(), stream)?"));
    assert!(TRANSACTION.contains("self.argmax_rows_host(logits, tokens.len())?"));
    assert!(TRANSACTION.contains("!glm53_exact_wide_prefill_active()"));
    assert!(TRANSACTION.contains("!glm53_layer_major_prefill_active()"));
    assert!(TARGET.contains("if value > best.1"));
    assert!(CUDA.contains("candidate == best && candidate_index < best_index"));
}
