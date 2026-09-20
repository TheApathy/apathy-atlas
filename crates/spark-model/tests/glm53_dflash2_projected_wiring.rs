// SPDX-License-Identifier: AGPL-3.0-only
//! P11 RED seams only; real input equality requires the retained GPU artifacts.
use std::{fs, path::PathBuf};
fn read(relative: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)).unwrap_or_default()
}
fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

#[test]
fn optional_input_callbacks_wrap_both_modes_inside_admitted_unwind_boundary() {
    let source = read("src/model/glm53/dflash2_probe_runtime.rs");
    assert!(source.contains("fn wants_projected_target("));
    assert!(source.contains("with_projected_target("));
    let start = source.find("pub fn propose_diagnostic(").unwrap();
    let end = source[start..].find("fn propose_reference(").unwrap() + start;
    let body = &source[start..end];
    let admitted = body.find("observer.admit(").unwrap();
    let catch = body.find("catch_unwind(").unwrap();
    let before = body
        .find("ProbeStage::ProjectedTargetBefore")
        .expect("actual before callback");
    // Admission now also matches the modes before any callback. Verify the
    // actual execution dispatch inside the before/after observation envelope.
    let full = before
        + body[before..]
            .find("Dflash2ProbeMode::FullRecompute")
            .unwrap();
    let cached = before
        + body[before..]
            .find("Dflash2ProbeMode::CachedPrefix")
            .unwrap();
    assert!(body.find("ensure_projection_ready(").unwrap() < admitted);
    assert!(body.find("CommittedProjection::for_probe(").unwrap() < admitted);
    let after = body
        .find("ProbeStage::ProjectedTargetAfter")
        .expect("actual after callback");
    assert!(admitted < catch && catch < before && before < full && full < after);
    assert!(before < cached && cached < after);
    assert!(body[after..].contains("poison_verify(stream)"));
    assert!(source.contains("self.plan.projected_target"));
    assert!(!source.contains("set_var("));
}

#[test]
fn default_production_and_original_full_reference_branches_remain_unobserved() {
    let runtime = compact(&read("src/model/glm53/dflash2_runtime.rs"));
    assert!(runtime.contains("enqueue_proposal(target,anchor,stream,None,None)"));
    let probe = compact(&read("src/model/glm53/dflash2_probe_runtime.rs"));
    assert!(probe.contains("enqueue_proposal(target,anchor,stream,None,Some(observer))"));
    assert!(probe.contains("fnwants_projected_target(&self)->bool{false}"));
    let target = read("src/model/glm53/target_dflash2_probe.rs");
    assert!(target.contains("runtime.propose_diagnostic("));
    assert!(!target.contains("set_var("));
}

#[test]
fn executable_explicitly_extracts_both_modes_and_records_cross_context_checks() {
    let source = read("examples/glm53_dflash2_kv_parity.rs");
    assert!(source.contains("--with-projected-target"));
    assert!(source.contains("with_projected_target("));
    assert!(source.contains("propose_installed_diagnostic("));
    assert!(source.contains("Dflash2ProbeMode::FullRecompute"));
    assert!(source.contains("projected::compare_inputs("));
    assert!(source.contains("projected-inputs.json"));
    assert!(source.contains("previous_projected_target"));
    assert!(source.contains("\"with_projected_target\""));
    let artifacts = read("examples/glm53_dflash2_kv_parity/artifacts.rs");
    assert!(artifacts.contains("frame.stage.name()"));
    assert!(artifacts.contains("sha256(&path)?"));
    let session = read("examples/glm53_dflash2_kv_parity/session.rs");
    assert!(session.contains("ManuallyDrop"));
    assert!(session.contains("candidate_capture"));
    assert!(session.contains("reference_capture"));
}
