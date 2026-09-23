// SPDX-License-Identifier: AGPL-3.0-only
use std::{fs, path::PathBuf};
fn read(path: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap_or_default()
}
#[test]
fn three_arms_are_explicit_and_original_full_runs_before_stable_full() {
    let source = read("examples/glm53_dflash2_kv_parity/three_way.rs");
    let calls: String = source.split_whitespace().collect();
    assert!(calls.contains("let[original_mode,full_mode,cached_mode]=choice.modes();"));
    let original = calls
        .find("letoriginal=ifcandidate.is_ok(){reference.propose_diagnostic(&session.model,anchor,session.stream,original_mode,")
        .expect("original arm retained");
    let stable = calls
        .find("letstable=iforiginal.is_ok(){reference.propose_diagnostic(&session.model,anchor,session.stream,full_mode,")
        .expect("stable full arm");
    assert!(original < stable);
    assert!(calls.contains("propose_installed_diagnostic_mode(anchor,session.stream,cached_mode,"));
    assert!(source.contains("compare_three("));
    assert!(source.contains("original_capture"));
    assert!(source.contains("stable_cache_exact"));
    assert!(!source.contains("set_var("));
    let main = read("examples/glm53_dflash2_kv_parity.rs");
    assert!(main.contains("--stable-projection"));
    assert!(main.contains("three_way::triple("));
    assert!(main.contains("propose_installed_diagnostic("));
}
#[test]
fn original_observer_is_session_owned_and_drained_before_backend_release() {
    let session = read("examples/glm53_dflash2_kv_parity/session.rs");
    assert!(session.contains("pub original_capture: Option<ProbeCapture>"));
    let drain = session
        .find("&mut self.original_capture")
        .expect("original observer completion");
    let free = session
        .find("ManuallyDrop::into_inner(reference).free")
        .unwrap();
    assert!(drain < free);
    assert_eq!(session.matches("Glm53Dflash2Runtime::load(").count(), 1);
    assert_eq!(session.matches("install_dflash2(").count(), 1);
}
