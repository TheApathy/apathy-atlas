// SPDX-License-Identifier: AGPL-3.0-only

use super::{CUDA, HOST};

const HOST_MODULE: &str = "\"glm53_dsa_current_visibility\",";
const HOST_FUNCTION: &str = "\"atlas_glm53_dsa_current_visibility_bf16\",";
const CUDA_ENTRY: &str = "extern \"C\" __global__ void __launch_bounds__(GLM53_DSA_VIS_THREADS, 1)\natlas_glm53_dsa_current_visibility_bf16(";
const LAYER_LOOP: &str = "for (unsigned int layer = lane; layer < GLM53_DSA_VIS_LAYERS;";
const LATENT_LOOP: &str = "for (unsigned long long item = lane; item < latent_items;";
const POOL_LOOP: &str = "for (unsigned long long item = lane; item < pool_items;";
const POOL_BRANCH: &str = "if (writes_pool) {";
const HOST_LAUNCH: &str = ".launch(stream)";
const POOL_ROW: &str =
    "const unsigned long long pool_row =\n            logical_position / GLM53_DSA_VIS_KPOOL;";
const RECEIPT: &str = "if (staged_ends[layer] != end_position ||\n            staged_generations[layer] != generation ||\n            staged_nonces[layer] != transaction_nonce ||\n            staged_statuses[layer] != 0U ||\n            (writes_pool && pool_overlay_validity[layer] != 1U)) {";
const INVALID_RETURN: &str = "if (invalid_transaction != 0U) {\n        return;\n    }";
const PREVALIDATION: &str = r#"__shared__ unsigned int invalid_transaction;
    if (lane == 0U) {
        invalid_transaction = 0U;
    }
    __syncthreads();
    for (unsigned int layer = lane; layer < GLM53_DSA_VIS_LAYERS;
         layer += GLM53_DSA_VIS_THREADS) {
        if (staged_ends[layer] != end_position ||
            staged_generations[layer] != generation ||
            staged_nonces[layer] != transaction_nonce ||
            staged_statuses[layer] != 0U ||
            (writes_pool && pool_overlay_validity[layer] != 1U)) {
            atomicExch(&invalid_transaction, 1U);
        }
    }
    __syncthreads();
    if (invalid_transaction != 0U) {
        return;
    }"#;

fn contract(host: &str, cuda: &str) -> bool {
    host.matches(HOST_MODULE).count() == 1
        && host.matches(HOST_FUNCTION).count() == 1
        && host.matches(HOST_LAUNCH).count() == 1
        && cuda.matches(CUDA_ENTRY).count() == 1
        && cuda.matches(LAYER_LOOP).count() == 4
        && cuda.matches(LATENT_LOOP).count() == 1
        && cuda.matches(POOL_LOOP).count() == 1
        && cuda.matches(POOL_BRANCH).count() == 1
        && cuda.matches(POOL_ROW).count() == 1
        && cuda.matches(RECEIPT).count() == 1
        && cuda.matches(INVALID_RETURN).count() == 1
        && cuda.matches(PREVALIDATION).count() == 1
}

fn replace_nth(source: &str, from: &str, to: &str, occurrence: usize) -> String {
    let start = source
        .match_indices(from)
        .nth(occurrence)
        .map(|(offset, _)| offset)
        .expect("mutation source occurrence must exist");
    let mut mutant = source.to_owned();
    mutant.replace_range(start..start + from.len(), to);
    mutant
}

#[test]
fn registration_symbols_loops_receipts_and_pool_branch_are_exact() {
    assert!(contract(HOST, CUDA));

    for (from, to) in [
        (HOST_MODULE, "\"glm53_dsa_current_visibility_bad\","),
        (HOST_FUNCTION, "\"atlas_glm53_dsa_current_visibility_bad\","),
    ] {
        assert!(!contract(&HOST.replacen(from, to, 1), CUDA));
    }
    assert!(!contract(
        HOST,
        &CUDA.replacen(
            "atlas_glm53_dsa_current_visibility_bf16(",
            "atlas_glm53_dsa_current_visibility_bad(",
            1,
        ),
    ));

    assert!(!contract(
        &HOST.replacen(HOST_LAUNCH, ".launch(0)", 1),
        CUDA
    ));
    for (from, to) in [
        (
            POOL_ROW,
            "const unsigned long long pool_row =\n            logical_position % GLM53_DSA_VIS_KPOOL;",
        ),
        ("if (lane == 0U) {", "if (lane != 0U) {"),
        (
            "atomicExch(&invalid_transaction, 1U);",
            "atomicExch(&invalid_transaction, 0U);",
        ),
    ] {
        assert!(!contract(HOST, &CUDA.replacen(from, to, 1)));
    }
    for occurrence in 0..2 {
        assert!(!contract(
            HOST,
            &replace_nth(CUDA, "__syncthreads();", "", occurrence),
        ));
    }

    for occurrence in 0..4 {
        assert!(!contract(
            HOST,
            &replace_nth(
                CUDA,
                LAYER_LOOP,
                "for (unsigned int layer = lane; layer <= GLM53_DSA_VIS_LAYERS;",
                occurrence,
            ),
        ));
    }
    for (from, to) in [
        (
            LATENT_LOOP,
            "for (unsigned long long item = lane; item <= latent_items;",
        ),
        (
            POOL_LOOP,
            "for (unsigned long long item = lane; item <= pool_items;",
        ),
        (POOL_BRANCH, "if (!writes_pool) {"),
        (
            "staged_ends[layer] != end_position ||",
            "staged_ends[layer] != end_position &&",
        ),
        (INVALID_RETURN, "if (invalid_transaction != 0U) {\n    }"),
    ] {
        let mutant = CUDA.replacen(from, to, 1);
        assert_ne!(mutant, CUDA, "mutation source must exist: {from}");
        assert!(!contract(HOST, &mutant));
    }
}
