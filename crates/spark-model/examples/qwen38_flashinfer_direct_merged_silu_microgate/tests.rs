// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const ENTRY: &str = include_str!("../qwen38_flashinfer_direct_merged_silu_microgate.rs");
const AUTHORITY: &str = include_str!("authority.rs");
const CONTRACT: &str = include_str!("contract.rs");
const GUARDED: &str = include_str!("guarded.rs");
const HOSTILE_TESTS: &str = include_str!("hostile_tests.rs");
const LAUNCH: &str = include_str!("launch.rs");
const OWNER: &str = include_str!("owner.rs");
const OWNER_TESTS: &str = include_str!("owner_tests.rs");
const PROVENANCE: &str = include_str!("provenance.rs");
const RUNTIME: &str = include_str!("runtime.rs");
const RUNTIME_HELPERS: &str = include_str!("runtime_helpers.inc.rs");
const SCHEDULER: &str = include_str!("scheduler_authority.rs");
const SCHEDULER_TESTS: &str = include_str!("scheduler_authority_tests.rs");
const SCHEDULER_BUILD: &str = include_str!("scheduler_build.rs");
const SCHEDULER_BUILD_TESTS: &str = include_str!("scheduler_build_tests.rs");
const SCHEDULER_TRUST: &str = include_str!("scheduler_trust.rs");
const TESTS: &str = include_str!("tests.rs");
const TIMING: &str = include_str!("timing.rs");

#[test]
fn exact_extents() {
    let small = Plan::checked(2079, K, 21, SCALE2).unwrap();
    assert_eq!(
        (small.padded, small.merged_bytes, small.scales, small.tail),
        (2176, 144_764_928, 2_367_488, 105_536)
    );
    let large = Plan::checked(8192, K, 31, SCALE2).unwrap();
    assert_eq!(
        (large.padded, large.merged_bytes, large.scales, large.tail),
        (8192, 570_425_344, 8_912_896, 0)
    );
}

#[test]
fn hostile_preflight() {
    for result in [
        Plan::checked(2048, K, 21, SCALE2),
        Plan::checked(2079, K - 64, 21, SCALE2),
        Plan::checked(2079, K, 20, SCALE2),
        Plan::checked(8192, K, 30, SCALE2),
        Plan::checked(2079, K, 21, 0.0),
        Plan::checked(2079, K, 21, f32::NAN),
    ] {
        assert!(result.is_err());
    }
}

#[test]
fn split_contract() {
    let plan = Plan::checked(2079, K, 21, SCALE2).unwrap();
    let merged = merged_input(1, K);
    let side = K as usize * 2;
    assert!(
        exact_split(
            &merged,
            &merged[..side],
            &merged[side..],
            Plan { m: 1, ..plan }
        )
        .is_ok()
    );
    let mut bad = merged[side..].to_vec();
    bad[0] ^= 1;
    assert!(exact_split(&merged, &merged[..side], &bad, Plan { m: 1, ..plan }).is_err());
}

#[test]
fn source_contract() {
    let specs = source_specs();
    assert_eq!(specs.len(), 6);
    assert!(specs.contains(&("crates/spark-model/src/layers/dense_ffn.rs", ROUTE_SHA)));
    let quant = include_str!("../../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu");
    let split = include_str!("../../../../kernels/gb10/common/flashinfer_projection_split.cu");
    let rust = include_str!("../../src/layers/ops/gemm_dense.rs");
    assert!(quant.contains("void quantize_silu_mul_bf16_to_nvfp4_atlas_128x4("));
    assert!(quant.contains("void quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4("));
    assert!(split.contains("void flashinfer_projection_split_ffn_gate_up("));
    assert!(rust.contains("pub fn quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4("));
    assert!(check_hash("hostile", QUANT_CUDA_SHA, SPLIT_CUDA_SHA).is_err());
    assert!(check_hash("hostile", "A", "A").is_err());
}

#[test]
fn release_authority_is_deliberately_unreleased() {
    let path = scheduler_authority::RELEASE_SCHEDULER_MANIFEST_PATH;
    assert!(path.starts_with("UNRELEASED-"));
    assert!(!std::path::Path::new(path).is_absolute());
}

#[test]
fn hostile_bundle_drift() {
    let base = [
        (DIRECT_MODULES[0], "a"),
        (DIRECT_MODULES[1], "b"),
        (DIRECT_MODULES[2], "c"),
    ];
    let identity = bundle_identity("sm_121/qwen3.8-27b/nvfp4", &base).unwrap();
    let mut drift = base;
    drift[1].1 = "changed";
    assert_ne!(
        identity.sha256,
        bundle_identity(&identity.target, &drift).unwrap().sha256
    );
    assert!(bundle_identity(&identity.target, &base[..2]).is_err());
    let duplicate = [
        (DIRECT_MODULES[0], "a"),
        (DIRECT_MODULES[0], "b"),
        (DIRECT_MODULES[1], "c"),
        (DIRECT_MODULES[2], "d"),
    ];
    assert!(bundle_identity(&identity.target, &duplicate).is_err());
}

#[test]
fn every_source_respects_file_cap() {
    for (name, source) in [
        ("entry", ENTRY),
        ("authority", AUTHORITY),
        ("contract", CONTRACT),
        ("guarded", GUARDED),
        ("hostile-tests", HOSTILE_TESTS),
        ("launch", LAUNCH),
        ("owner", OWNER),
        ("owner-tests", OWNER_TESTS),
        ("provenance", PROVENANCE),
        ("runtime", RUNTIME),
        ("runtime-helpers", RUNTIME_HELPERS),
        ("scheduler", SCHEDULER),
        ("scheduler-tests", SCHEDULER_TESTS),
        ("scheduler-build", SCHEDULER_BUILD),
        ("scheduler-build-tests", SCHEDULER_BUILD_TESTS),
        ("scheduler-trust", SCHEDULER_TRUST),
        ("tests", TESTS),
        ("timing", TIMING),
    ] {
        assert!(source.lines().count() <= 250, "{name} exceeds 250 lines");
        assert!(source.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
    }
}

#[test]
fn receipt_and_timing_contract() {
    let source = [
        ENTRY,
        AUTHORITY,
        CONTRACT,
        GUARDED,
        HOSTILE_TESTS,
        LAUNCH,
        OWNER,
        OWNER_TESTS,
        PROVENANCE,
        RUNTIME,
        RUNTIME_HELPERS,
        SCHEDULER,
        SCHEDULER_TESTS,
        SCHEDULER_BUILD,
        SCHEDULER_BUILD_TESTS,
        SCHEDULER_TRUST,
        TESTS,
        TIMING,
    ]
    .concat();
    for required in [
        SCHEMA,
        "tail_zero_bytes",
        "scale2_bits",
        "merged_input_hash",
        "down_bf16_hash",
        "paired_median_ms",
        "paired_lower95_ms",
        "positive_pair_count",
        "parent_samples_ms",
        "raw_ciic_ms",
        "minimum_relative_effect",
        "same_nondefault_stream",
        "C/I/I/C",
        QUANT_CUDA_SHA,
        SPLIT_CUDA_SHA,
        DOWN_CUDA_SHA,
        WRAPPER_SHA,
        ROUTE_SHA,
        MANIFEST_SHA,
        "running_executable",
        "embedded_bundle",
        "build-attestation-v1",
        "qwen38-direct-merged-silu-scheduler-v1",
        "scheduler_session_id",
        "scheduler_authority",
        "build_receipt",
        "build_environment",
        "runtime_environment",
        "gate_source_sha256",
        "scheduler ticket has trailing bytes",
        "performance_claim\":false",
    ] {
        assert!(source.contains(required), "missing {required}");
    }
    assert!(TIMING.contains("[true, false, false, true]"));
    assert!(TIMING.contains("timing gate failed"));
    assert!(
        RUNTIME.find("let timing = measure_balanced(").unwrap()
            < RUNTIME.find("\"schema\":SCHEMA").unwrap()
    );
}
