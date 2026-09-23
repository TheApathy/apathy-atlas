// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn verifier_faults_do_not_masquerade_as_successful_length_completions() {
    let verify = include_str!("verify_dflash_step.rs");
    for stage in [
        "sync_secondary",
        "decode_verify_dflash",
        "commit_accepted_prefix (dflash)",
    ] {
        assert!(verify.contains(&format!("mark_sequence_error(a, \"{stage}\"")));
    }
    let lifecycle = include_str!("lifecycle.rs");
    let finish = lifecycle.split("pub fn finish_sequence(").nth(1).unwrap();
    let error = finish
        .find("if let Some(error) = a.terminal_error.take()")
        .unwrap();
    let report = finish.find("send_error(model, a, &error)").unwrap();
    let done = finish.find("StreamEvent::Done").unwrap();
    assert!(error < report && report < done);
    assert!(finish[report..done].contains("return;"));
    assert!(report < finish.find("model.cache_sequence(").unwrap());
}

#[test]
fn failure_is_retired_once_and_never_reproposed() {
    let source = include_str!("sequence_error.rs");
    let compact: String = source.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(compact.contains("a.terminal_error.get_or_insert_with"));
    assert!(source.contains("a.pending_drafts.clear()"));
    assert!(source.contains("a.finished = true"));
    assert!(!source.contains("free_sequence("));
}
