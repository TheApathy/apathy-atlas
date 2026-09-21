// SPDX-License-Identifier: AGPL-3.0-only

use std::{fs, path::PathBuf};

fn source(name: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)).unwrap()
}

fn ordered(code: &str, markers: &[&str]) {
    let mut tail = code;
    for marker in markers {
        let index = tail
            .find(marker)
            .unwrap_or_else(|| panic!("missing {marker}"));
        tail = &tail[index + marker.len()..];
    }
}

#[test]
fn malformed_and_conflicting_modes_are_admitted_before_effects() {
    let code = source("src/layers/qwen3_ssm/qwen4_prefill_exact.rs");
    ordered(
        &code,
        &[
            "pub(crate) fn admit_request(",
            "qwen4_prefill_gemm::admit(",
            "if !exact {",
        ],
    );
    ordered(
        &code,
        &[
            "fn preflight_qwen4_prefill_exact(",
            "qwen4_prefill_gemm::validate_handle(",
            "attn.validate_prefill_exact(",
        ],
    );
    let module = source("src/layers/qwen3_ssm/qwen4_prefill_gemm.rs");
    assert!(module.contains("ATLAS_QWEN4_PREFILL_SSM_GEMM"));
    assert!(module.contains("pub(crate) fn mode() -> Result<Mode>"));
}

#[test]
fn only_two_projection_sites_change_and_exact_else_paths_remain() {
    let code = source("src/layers/qwen3_ssm/qwen4_prefill_exact_forward.rs");
    ordered(
        &code,
        &[
            "Projection::Qkvz",
            "self.project_prefill_gemm(",
            "} else {",
            "self.project_prefill_exact_tiles(",
            "ops::dense_gemv_ba_gates_batchn(",
            "ops::conv1d_update_l2norm_f32_sequence(",
            "ops::gdn_decode_f32_sequence(",
            "ops::gated_rms_norm_f32_multi_seq(",
            "Projection::Output",
            "self.project_prefill_gemm(",
            "} else {",
            "self.project_prefill_exact_tiles(",
            "attn.inject_saved_batched(",
            "check.verify(",
        ],
    );
    assert_eq!(code.matches("self.project_prefill_gemm(").count(), 2);
    assert_eq!(code.matches("self.project_prefill_exact_tiles(").count(), 2);
    assert!(
        source("src/layers/qwen3_ssm/trait_prefill.rs").contains("selected()? && num_tokens > 1")
    );
}

#[test]
fn gemm_uses_original_weights_and_existing_handle_only() {
    let code = source("src/layers/qwen3_ssm/qwen4_prefill_gemm.rs");
    for required in [
        "ops::w4a16_gemm(",
        "self.w4a16_gemm_k",
        "self.qkvz_nvfp4",
        "&self.ssm.out_proj",
        "Plan::QKVZ",
        "Plan::H",
        "Plan::VALUE_DIM",
    ] {
        assert!(code.contains(required), "missing {required}");
    }
    for forbidden in [
        "fp8_gemm",
        "_nvfp4_t",
        "transpose",
        "gpu.alloc(",
        "copy_d2d",
        "prefill_out_proj_dispatch(",
        "prefill_qkvz_proj(",
        "w4a16_gemm_pipe(",
    ] {
        assert!(!code.contains(forbidden), "unexpected {forbidden}");
    }
}
