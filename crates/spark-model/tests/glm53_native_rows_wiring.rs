// SPDX-License-Identifier: AGPL-3.0-only
//! Source-wiring gates, not FFI execution or numerical qualification.
const TARGET: &str = include_str!("../src/model/glm53/target_model_exl3.rs");
const POLICY: &str = include_str!("../src/model/glm53/dsa_policy_execution.rs");
const VERIFY: &str = include_str!("../src/model/glm53/dsa_verify_execution.rs");
const REPLAY: &str = include_str!("../src/model/glm53/target_partial_replay.rs");
const PROJECTION: &str = include_str!("../src/layers/ops/glm53_exl3_projection.rs");
const FFI: &str = include_str!("../../spark-runtime/src/cublaslt/serial_rows_ffi.rs");
const RUNTIME: &str = include_str!("../../spark-runtime/src/cublaslt.rs");

#[test]
fn model_latches_explicit_policy_before_its_first_owned_allocation() {
    let constructor = TARGET.split("pub fn from_loaded").last().unwrap_or(TARGET);
    // Formatting may split a method chain without changing startup admission.
    let constructor = constructor.split_whitespace().collect::<String>();
    let parse = constructor
        .find("NativeRowsSetting::parse_with_batched(")
        .expect("typed startup selection");
    let validate = constructor
        .find("native_rows.validate_exact_mode(")
        .expect("explicit exact mode admission");
    let allocate = constructor
        .find("build_moe_pointer_tables(")
        .expect("first owned allocation");
    assert!(parse < validate && validate < allocate);
    assert!(TARGET.contains("native_rows: NativeRowsSetting"));
    assert!(TARGET.contains("native_rows,"));
    assert!(constructor.contains("ATLAS_GLM53_NATIVE_ROW_PREPARED"));
    assert!(constructor.contains("ATLAS_GLM53_NATIVE_ROW_BATCHED"));
}

#[test]
fn all_three_verifier_entrypoints_use_owned_scope_without_rereading_the_new_flag() {
    for source in [POLICY, VERIFY, REPLAY] {
        assert!(source.contains("with_glm53_native_rows("));
        assert!(source.contains("native_rows, ||"));
        assert!(source.contains("with_glm53_exact_verify("));
        assert!(!source.contains("ATLAS_GLM53_NATIVE_ROW_PREPARED"));
        assert!(!source.contains("ATLAS_GLM53_NATIVE_ROW_BATCHED"));
    }
}

#[test]
fn selected_native_dispatch_checks_capture_and_preserves_original_control_calls() {
    let select = PROJECTION
        .find("if native_rows_selected(")
        .expect("typed dispatch selection");
    let capture = PROJECTION[select..]
        .find("stream_is_capturing(stream)")
        .unwrap()
        + select;
    let call = PROJECTION[select..].find("bf16_gemm_serial_rows(").unwrap() + select;
    let original = PROJECTION
        .find("if plan.rows > 1 && exact_arithmetic")
        .unwrap();
    assert!(select < capture && capture < call && call < original);
    assert!(!PROJECTION.contains("ATLAS_GLM53_NATIVE_ROW_PREPARED"));
    assert!(!PROJECTION.contains("ATLAS_GLM53_NATIVE_ROW_BATCHED"));
    assert!(PROJECTION.contains("glm53_native_rows_batched_active()"));
    assert!(PROJECTION.contains("bf16_gemm_batched_rows("));
    for name in ["bf16_gemm_act_weight_t(", "bf16_gemm_act_weight("] {
        assert!(PROJECTION.contains(name));
    }
}

#[test]
fn ffi_preflights_real_operands_before_cold_context_then_uses_the_tested_driver() {
    let function = FFI
        .split("pub fn bf16_gemm_serial_rows(")
        .nth(1)
        .expect("additive API");
    let preflight = function.find("SerialRowsPlan::new(request)").unwrap();
    let context = function.find("ctx()?").unwrap();
    let run = function.find("run_serial_rows(").unwrap();
    assert!(preflight < context && context < run);
    assert!(RUNTIME.contains("mod serial_rows_ffi;"));
    assert!(RUNTIME.contains("pub use serial_rows_ffi::bf16_gemm_serial_rows;"));
    for original in [
        "pub fn bf16_gemm_act_weight_t(",
        "pub fn bf16_gemm_act_weight(",
        "pub fn bf16_gemm_act_weight_t_diagnostic(",
    ] {
        assert!(RUNTIME.contains(original));
    }
}
