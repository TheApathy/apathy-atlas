// SPDX-License-Identifier: AGPL-3.0-only
//! Production routing checks; actual GPU raw parity remains an independent gate.
fn read(path: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_default()
}
fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

#[test]
fn new_explicit_family_and_modes_preserve_original_and_tc_names() {
    let family = read("src/model/glm53/dflash2_projection_contract.rs");
    assert!(family.contains("StableGemv"));
    let modes = read("src/model/glm53/dflash2_probe_contract.rs");
    for name in [
        "FullRecompute",
        "CachedPrefix",
        "StableFullProjection",
        "StableCachedProjection",
        "StableGemvFullProjection",
        "StableGemvCachedProjection",
    ] {
        assert!(modes.contains(name), "missing mode {name}");
    }
    for label in [
        "stable-gemv-full-projection",
        "stable-gemv-cached-projection",
    ] {
        assert!(modes.contains(label));
    }
}

#[test]
fn metadata_free_launcher_uses_checked_plan_and_only_existing_gemv_kernels() {
    let modules = read("src/model/glm53/dflash2_runtime.rs");
    assert!(modules.contains("dflash2_gemv_plan.rs"));
    assert!(modules.contains("dflash2_gemv_projection.rs"));
    let launch = compact(&read("src/model/glm53/dflash2_gemv_projection.rs"));
    assert!(launch.contains("gpu.kernel(\"gemv\",\"dense_gemv_bf16\")?"));
    assert!(launch.contains("gpu.kernel(\"dense_gemv_bf16_batch2\",\"dense_gemv_bf16_batch2\")?"));
    assert!(launch.contains("GemvPlan::new("));
    assert!(launch.contains(".execute("));
    assert!(launch.contains("ops::dense_gemv("));
    assert!(launch.contains("ops::dense_gemv_batch2("));
    for forbidden in [
        "gpu.alloc(",
        "copy_h2d",
        "copy_d2h",
        "lora_bgmv",
        "dense_gemm_tc",
        "cublas",
    ] {
        assert!(
            !launch.contains(forbidden),
            "unscoped operation {forbidden}"
        );
    }
}

#[test]
fn new_projection_preflight_precedes_observers_and_both_modes_use_shared_body() {
    let probe = read("src/model/glm53/dflash2_probe_runtime.rs");
    let body = probe.split("pub fn propose_diagnostic(").nth(1).unwrap();
    let preflight = body
        .find("preflight_gemv_projection(")
        .expect("missing all-layer GEMV admission");
    assert!(preflight < body.find("observer.admit(").unwrap());
    assert!(body[..preflight].contains("CommittedProjection::for_probe("));
    assert!(body.contains("StableGemvFullProjection"));
    assert!(body.contains("StableGemvCachedProjection"));
    assert!(body.contains("propose_reference_with_projection("));
    assert!(body.contains("propose_cached_observed_with_projection("));
    let projection = read("src/model/glm53/dflash2_committed_projection.rs");
    assert!(projection.contains("StableGemv"));
    assert!(projection.contains("ops::dense_gemm_tc("));
    assert!(projection.contains("Self::Original"));
}

#[test]
fn installed_admits_new_cached_mode_but_default_serving_and_noise_stay_original() {
    let installed = read("src/model/glm53/target_dflash2_probe.rs");
    assert!(installed.contains("StableGemvCachedProjection"));
    assert!(!installed.contains("StableGemvFullProjection"));
    let runtime = compact(&read("src/model/glm53/dflash2_runtime.rs"));
    assert!(runtime.contains("enqueue_proposal(target,anchor,stream,None,None)"));
    for path in [
        "src/model/glm53/dflash2_proposal.rs",
        "src/model/glm53/dflash2_kv_prefix_runtime.rs",
    ] {
        let source = read(path);
        assert_eq!(source.matches("project_committed(").count(), 2);
        assert_eq!(source.matches("layer.k_proj.weight").count(), 1);
        assert_eq!(source.matches("layer.v_proj.weight").count(), 1);
    }
}
