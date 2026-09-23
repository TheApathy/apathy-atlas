// SPDX-License-Identifier: AGPL-3.0-only
//! GPU launch seams supplement the behavioral plan; they are not GPU evidence.
fn source() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/glm53_projection_probe/gemv.rs"
    ))
    .unwrap()
}
fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn resolves_real_registered_symbols_and_rejects_null_handles() {
    let raw = source();
    let source = compact(&raw);
    for lookup in [
        "gpu.kernel(\"gemv\",\"dense_gemv_bf16\")?",
        "gpu.kernel(\"lora_bgmv\",\"lora_bgmv_shrink\")?",
        "gpu.kernel(\"dense_gemv_bf16_batch2\",\"dense_gemv_bf16_batch2\")?",
    ] {
        assert!(
            source.contains(lookup),
            "missing registered lookup {lookup}"
        );
    }
    assert!(source.contains("validate_handles("));
    let launch = raw.find("pub fn launch(").unwrap();
    assert!(
        !raw[launch..].contains("gpu.kernel("),
        "lookup inside projection"
    );
}

#[test]
fn direct_gather_launch_uses_only_shrink_and_bound_row_geometry() {
    let source = compact(&source());
    assert!(source.contains(".grid([OUTPUTS.div_ceil(4),plan.rows(),1])"));
    assert!(source.contains(".block([256,1,1])"));
    for pointer in ["input", "slots", "table", "output"] {
        assert!(source.contains(&format!(".arg_ptr(DevicePtr({pointer}.ptr))")));
    }
    assert!(
        source.contains(".arg_u32(plan.rows()).arg_u32(OUTPUTS).arg_u32(HIDDEN).arg_u32(HIDDEN)")
    );
    assert!(!source.contains("expand_fold"));
    assert!(!source.contains("apply_lora"));
}

#[test]
fn dual_row_and_single_row_operators_remain_explicit_new_family() {
    let source = source();
    assert!(source.contains("ops::dense_gemv("));
    assert!(source.contains("ops::dense_gemv_batch2("));
    assert!(compact(&source).contains("GemvMode::Batch2ifplan.rows()==2"));
    assert!(source.contains("batch2-with-m1-gemv"));
    for forbidden in [
        "cublas",
        "dense_gemm_tc",
        "copy_h2d",
        "copy_d2h",
        "gpu.alloc",
        "unsafe",
        "std::fs",
    ] {
        assert!(
            !source.contains(forbidden),
            "new GEMV helper contains {forbidden}"
        );
    }
}
