// SPDX-License-Identifier: AGPL-3.0-only
//! Source linkage, not a numerical test or a replacement for GPU evidence.
use std::{fs, path::PathBuf};
fn source(path: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap_or_default()
}
#[test]
fn production_and_explicit_diagnostic_use_one_core_without_env_switches() {
    let code = source("src/cublaslt.rs");
    let compact: String = code.split_whitespace().collect();
    assert!(code.contains("pub fn bf16_gemm_act_weight_t_diagnostic("));
    assert!(compact.contains("bf16_gemm_impl(act,weight,out,m,n,k,stream,weight_is_nk,None)"));
    assert!(compact.contains("bf16_gemm_impl(act,weight,out,m,n,k,stream,true,Some(policy))"));
    assert!(!code.contains("std::env::"));
}
#[test]
fn diagnostic_receipts_are_bound_to_the_selected_algorithm() {
    let code = source("src/cublaslt/diagnostic.rs");
    for needle in [
        "cublasLtMatmulAlgoConfigGetAttribute",
        "cublasLtGetVersion",
        "cublasLtGetCudartVersion",
        "validate_selected",
        "size_written",
    ] {
        assert!(code.contains(needle), "missing receipt gate {needle}");
    }
}
