// SPDX-License-Identifier: AGPL-3.0-only
//! Check the actual registry module, not an old microtest's filename guess.
use std::{fs, path::PathBuf};
#[test]
fn tensor_core_operator_uses_registered_module_and_existing_wrapper() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest = fs::read_to_string(root.join("../../kernels/gb10/common/KERNEL.toml")).unwrap();
    let module = manifest
        .lines()
        .find_map(|line| {
            line.strip_prefix("dense_gemm_tc = \"")
                .and_then(|value| value.strip_suffix('"'))
        })
        .unwrap();
    let source = fs::read_to_string(root.join("examples/glm53_projection_probe.rs")).unwrap();
    let compact: String = source.split_whitespace().collect();
    assert!(compact.contains(&format!("kernel(\"{module}\",\"dense_gemm_tc\")")));
    assert!(source.contains("ops::dense_gemm_tc("));
}
