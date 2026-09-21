// SPDX-License-Identifier: AGPL-3.0-only

use std::{fs, path::PathBuf};

fn source(path: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap()
}

#[test]
fn selector_is_strict_default_off_and_nested_under_exact_rt2() {
    let admission = source("src/layers/qwen3_ssm/qwen4_prefill_exact.rs");
    for marker in [
        "ATLAS_QWEN4_PREFILL_SSM_GRID32",
        "grid32_selected()?",
        "w4a16_gemv_rt2_enabled()",
        "rt2_m32_grid",
    ] {
        assert!(admission.contains(marker), "missing {marker}");
    }
}

#[test]
fn exact_forward_launches_complete_grid_once_and_preserves_tail_route() {
    let forward = source("src/layers/qwen3_ssm/qwen4_prefill_exact_forward.rs");
    for marker in [
        "grid32_partition(plan.rows)",
        "w4a16_gemv_batch_logits_exact_rt2_m32_grid(",
        "for (start, count) in plan.tiles_from(full_rows)",
        "ops::w4a16_decode_gemv(",
        "ops::w4a16_gemv_batch_logits_exact_with(",
    ] {
        assert!(forward.contains(marker), "missing {marker}");
    }
}

#[test]
fn cuda_wrapper_uses_y_only_for_disjoint_row_tiles() {
    let cuda = source("../../kernels/gb10/common/w4a16_gemv_rt.cu");
    for marker in [
        "w4a16_gemv_batch_logits_exact_rt2_m32_grid",
        "const unsigned int row_start = blockIdx.y * 32u;",
        "A + (unsigned long long)row_start * K",
        "C + (unsigned long long)row_start * N",
        "w4a16_gemv_batch_logits_exact_rt_body<32, 2>(",
    ] {
        assert!(cuda.contains(marker), "missing {marker}");
    }
}
