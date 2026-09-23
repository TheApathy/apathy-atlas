// SPDX-License-Identifier: AGPL-3.0-only
//! P12 RED integration seams; source routing is not numerical qualification.
use std::{fs, path::PathBuf};
fn read(relative: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)).unwrap_or_default()
}
fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

#[test]
fn stable_modes_are_explicit_and_only_the_diagnostic_resolves_registered_tc() {
    let contract = read("src/model/glm53/dflash2_probe_contract.rs");
    for mode in [
        "FullRecompute",
        "CachedPrefix",
        "StableFullProjection",
        "StableCachedProjection",
    ] {
        assert!(contract.contains(mode), "missing explicit mode {mode}");
    }
    let projection = compact(&read("src/model/glm53/dflash2_committed_projection.rs"));
    let registry = compact(&read("../../kernels/gb10/common/KERNEL.toml"));
    assert!(registry.contains("dense_gemm_tc=\"gemm_tc\""));
    assert!(projection.contains("gpu.kernel(\"gemm_tc\",\"dense_gemm_tc\")"));
    assert!(projection.contains("ops::dense_gemm_tc("));
    assert!(!projection.contains("set_var("));
    assert!(!projection.contains("gpu.alloc("));
    let runtime = read("src/model/glm53/dflash2_runtime.rs");
    assert!(!runtime.contains("gpu.kernel(\"gemm_tc\""));
    let probe = read("src/model/glm53/dflash2_probe_runtime.rs");
    let start = probe.find("pub fn propose_diagnostic(").unwrap();
    let body = &probe[start..];
    assert!(
        body.find("CommittedProjection::for_probe(")
            .expect("typed diagnostic admission")
            < body.find("observer.admit(").unwrap()
    );
}

#[test]
fn exactly_the_committed_kv_calls_use_the_typed_projector_noise_stays_original() {
    let proposal = read("src/model/glm53/dflash2_proposal.rs");
    let cached = read("src/model/glm53/dflash2_kv_prefix_runtime.rs");
    assert_eq!(proposal.matches("project_committed(").count(), 2);
    assert_eq!(cached.matches("project_committed(").count(), 2);
    // Committed calls take the original DenseWeight by reference. These raw
    // weight-pointer calls are the two unchanged cuBLAS noise projections.
    for source in [&proposal, &cached] {
        assert_eq!(source.matches("layer.k_proj.weight").count(), 1);
        assert_eq!(source.matches("layer.v_proj.weight").count(), 1);
        assert!(source.contains("layer.k_proj"));
        assert!(source.contains("layer.v_proj"));
    }
    assert!(proposal.contains("enqueue_proposal_with_projection("));
    assert!(proposal.contains("CommittedProjection::Original"));
    let runtime = compact(&read("src/model/glm53/dflash2_runtime.rs"));
    assert!(runtime.contains("enqueue_proposal(target,anchor,stream,None,None)"));
    let probe = compact(&read("src/model/glm53/dflash2_probe_runtime.rs"));
    assert!(probe.contains("enqueue_proposal(target,anchor,stream,None,Some(observer))"));
}

#[test]
fn actual_cached_owner_selects_family_before_begin_and_resets_via_same_contract() {
    let owner = compact(&read("src/model/glm53/dflash2_kv_prefix_owner.rs"));
    assert!(owner.contains("projection:ProjectionBinding"));
    assert!(owner.contains(".projection.select("));
    assert!(owner.contains(".projection.reset("));
    let cached = compact(&read("src/model/glm53/dflash2_kv_prefix_runtime.rs"));
    let selected = cached
        .find("state.select_projection(")
        .expect("real owner family binding");
    let begin = cached[selected..]
        .find("state.prefix.begin(")
        .expect("same cursor begin");
    let upload = cached[selected..].find("copy_h2d_async(").unwrap();
    assert!(begin < upload);
    assert!(cached.contains("state.reset(&mutDrainIo(gpu))"));
    assert!(cached.contains("CommittedProjection::Original"));
}

#[test]
fn installed_mode_entrypoint_preserves_legacy_wrapper_and_only_admits_cached_modes() {
    let source = read("src/model/glm53/target_dflash2_probe.rs");
    assert!(source.contains("pub fn propose_installed_diagnostic_mode("));
    assert!(source.contains("pub fn propose_installed_diagnostic("));
    assert!(source.contains("Dflash2ProbeMode::CachedPrefix"));
    assert!(source.contains("Dflash2ProbeMode::StableCachedProjection"));
    assert!(source.contains("runtime.propose_diagnostic("));
    assert!(source.contains("ensure_dflash2_proposal_ready()?"));
    assert!(!source.contains("set_var("));
    assert!(!source.contains("decode_verify_with_policy("));
    assert!(!source.contains("reset_sequence("));
}
