// SPDX-License-Identifier: AGPL-3.0-only

use std::{fs, path::PathBuf};

fn source(name: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)).unwrap()
}

#[test]
fn projection_receipt_reads_the_production_selector() {
    let selectors = source("src/model/qwen4_prefill_engagement/selectors.rs");
    assert!(selectors.contains("ssm_gemm: Mode"));
    assert!(selectors.contains("selectors.ssm_gemm = qwen4_prefill_gemm::mode()?"));
    assert!(selectors.contains("ssm_gemm: Mode::Off"));
}

#[test]
fn tensorcore_modes_have_distinct_nonexact_labels() {
    let receipt = source("src/model/qwen4_prefill_engagement.rs");
    for label in [
        "Mode::Off => \"exact_projection_sequence_nosnap\"",
        "Mode::Out => \"exact_qkvz_bf16_mma_output_sequence_nosnap\"",
        "Mode::All => \"bf16_mma_projections_sequence_nosnap\"",
        "ssm_projection_gemm={}",
        "self.selectors.ssm_gemm.as_str()",
    ] {
        assert!(receipt.contains(label), "missing {label}");
    }
    assert!(!receipt.contains("parity=pass"));
}
