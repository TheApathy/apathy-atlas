// SPDX-License-Identifier: AGPL-3.0-only
//! RED seams: numerical qualification still requires the actual GPU example.

use std::{fs, path::PathBuf};
fn read(relative: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)).unwrap_or_default()
}
fn compact(value: &str) -> String {
    value.split_whitespace().collect()
}

#[test]
fn explicit_diagnostic_modes_are_registered_without_an_environment_bypass() {
    let module = read("src/model/glm53/mod.rs");
    assert!(module.contains("mod dflash2_probe_contract;"));
    assert!(module.contains("mod dflash2_probe_capture;"));
    let runtime = read("src/model/glm53/dflash2_runtime.rs");
    assert!(runtime.contains("mod probe;"));
    let probe = read("src/model/glm53/dflash2_probe_runtime.rs");
    assert!(probe.contains("pub fn propose_diagnostic("));
    assert!(probe.contains("stream_is_capturing(stream)"));
    assert!(!probe.contains("set_var("));
}

#[test]
fn full_reference_calls_original_branch_and_does_not_publish_cache_receipts() {
    let probe = compact(&read("src/model/glm53/dflash2_probe_runtime.rs"));
    assert!(probe.contains("Dflash2ProbeMode::FullRecompute"));
    assert!(probe.contains("Dflash2ProbeMode::CachedPrefix"));
    assert!(probe.contains("enqueue_proposal(target,anchor,stream,None,Some(observer))"));
    assert!(probe.contains("reset_kv_prefix("));
    assert!(probe.contains("read_proposal("));
    assert!(probe.contains(".reset("));
}

#[test]
fn real_shared_arithmetic_observes_attention_before_reuse_and_head_before_selection() {
    let proposal = read("src/model/glm53/dflash2_proposal.rs");
    assert!(proposal.contains("Glm53Dflash2ProbeObserver"));
    let attention = proposal
        .find("observe_layer(")
        .expect("actual attention observation");
    let reuse = proposal[attention..]
        .find("layer.output.weight")
        .expect("next output projection");
    assert!(reuse > 0);
    let hidden = proposal
        .find("ProbeStage::SelectedHidden")
        .expect("selected hidden before head");
    let head = proposal
        .find("self.head.launch(")
        .expect("same actual EXL3 head");
    let logits = proposal
        .find("ProbeStage::HeadLogits")
        .expect("actual head output");
    let topk = proposal
        .find("self.topk.launch(")
        .expect("original selector path");
    assert!(hidden < head && head < logits && logits < topk);
    assert!(proposal.contains("ProbeStage::DraftIds"));
    let runtime = compact(&read("src/model/glm53/dflash2_runtime.rs"));
    assert!(runtime.contains("enqueue_proposal(target,anchor,stream,None,None)"));
}

#[test]
fn executable_owns_two_real_runtimes_and_target_across_completion_failure() {
    let example = read("examples/glm53_dflash2_kv_parity.rs");
    let session = read("examples/glm53_dflash2_kv_parity/session.rs");
    assert!(example.contains("Dflash2ProbeMode::FullRecompute"));
    assert!(example.contains("Dflash2ProbeMode::CachedPrefix"));
    assert!(example.contains("propose_diagnostic("));
    assert_eq!(session.matches("Glm53Dflash2Runtime::load(").count(), 1);
    assert_eq!(session.matches("install_dflash2(").count(), 1);
    assert!(example.contains("propose_installed_diagnostic("));
    assert!(session.contains("ManuallyDrop"));
    assert!(session.contains("catch_unwind"));
    assert!(session.contains("synchronize("));
    assert!(session.contains("ProbeCapture"));
    assert!(!example.contains("set_var("));
}

#[test]
fn installed_candidate_diagnostic_delegates_without_policy_guard_bypass() {
    let target = read("src/model/glm53/target_model_exl3.rs");
    assert!(target.contains("mod target_dflash2_probe;"));
    let seam = read("src/model/glm53/target_dflash2_probe.rs");
    assert!(seam.contains("pub fn propose_installed_diagnostic("));
    assert!(seam.contains("pub fn installed_diagnostic_state("));
    assert!(seam.contains("ensure_dflash2_proposal_ready()?"));
    assert!(seam.contains("Dflash2ProbeMode::CachedPrefix"));
    assert!(seam.contains("runtime.propose_diagnostic("));
    assert!(seam.contains("runtime.context_tokens()"));
    assert!(seam.contains("runtime.probe_layout()?"));
    assert!(!seam.contains("decode_token("));
    assert!(!seam.contains("decode_verify_with_policy("));
    assert!(!seam.contains("reset_sequence("));
}
