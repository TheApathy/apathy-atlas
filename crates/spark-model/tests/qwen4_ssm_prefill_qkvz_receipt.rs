// SPDX-License-Identifier: AGPL-3.0-only

use std::{fs, path::PathBuf};

fn source(path: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap()
}

fn compact(code: &str) -> String {
    code.chars().filter(|c| !c.is_whitespace()).collect()
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
fn qkvz_receipt_labels_only_qkvz_as_mma_and_does_not_claim_qualification() {
    let receipt = source("src/model/qwen4_prefill_engagement.rs");
    let receipt_compact = compact(&receipt);
    assert!(receipt_compact.contains("Mode::Qkvz=>\"bf16_mma_qkvz_exact_output_sequence_nosnap\""));
    for existing in [
        "Mode::Off=>\"exact_projection_sequence_nosnap\"",
        "Mode::Out=>\"exact_qkvz_bf16_mma_output_sequence_nosnap\"",
        "Mode::All=>\"bf16_mma_projections_sequence_nosnap\"",
    ] {
        assert!(
            receipt_compact.contains(existing),
            "lost existing label {existing}"
        );
    }
    assert!(receipt.contains("path_success=enqueued"));
    for forbidden in [
        "parity=pass",
        "qualified=true",
        "performance_qualified=true",
        "quality=pass",
    ] {
        assert!(
            !receipt_compact.contains(forbidden),
            "unsupported receipt claim {forbidden}"
        );
    }
}

#[test]
fn receipt_uses_the_same_selected_production_mode_as_dispatch() {
    let selectors = compact(&source("src/model/qwen4_prefill_engagement/selectors.rs"));
    assert!(selectors.contains("selectors.ssm_gemm=qwen4_prefill_gemm::mode()?;"));
    let receipt = compact(&source("src/model/qwen4_prefill_engagement.rs"));
    assert!(receipt.contains("matchself.selectors.ssm_gemm{"));
    assert!(receipt.contains("ssm_projection_gemm={}"));
    assert!(receipt.contains("self.selectors.ssm_gemm.as_str()"));
    let forward = source("src/layers/qwen3_ssm/qwen4_prefill_exact_forward.rs");
    ordered(
        &forward,
        &[
            "let projection_mode = qwen4_prefill_gemm::mode()?;",
            "if projection_mode.uses_gemm(Projection::Qkvz)",
            "self.project_prefill_gemm(Projection::Qkvz,",
            "} else {",
            "self.project_prefill_exact_tiles(",
            "if projection_mode.uses_gemm(Projection::Output)",
            "self.project_prefill_gemm(Projection::Output,",
            "} else {",
            "self.project_prefill_exact_tiles(",
        ],
    );
}

#[test]
fn shared_admission_still_precedes_early_return_and_layer_effects() {
    let admission = source("src/layers/qwen3_ssm/qwen4_prefill_exact.rs");
    ordered(
        &admission,
        &[
            "pub(crate) fn admit_request(",
            "qwen4_prefill_gemm::admit(exact, check)?;",
            "if !exact {",
        ],
    );
    ordered(
        &admission,
        &[
            "fn preflight_qwen4_prefill_exact(",
            "qwen4_prefill_gemm::validate_handle(self.w4a16_gemm_k.0)?;",
            "attn.validate_prefill_exact(",
        ],
    );
}
