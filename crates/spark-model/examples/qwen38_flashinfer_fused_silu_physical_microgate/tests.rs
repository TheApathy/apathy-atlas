// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const ENTRY: &str = include_str!("../qwen38_flashinfer_fused_silu_physical_microgate.rs");
const CONTRACT: &str = include_str!("contract.rs");
const GUARDED: &str = include_str!("guarded.rs");
const LAUNCH: &str = include_str!("launch.rs");
const PROVENANCE: &str = include_str!("provenance.rs");
const RUNTIME: &str = include_str!("runtime.rs");
const TESTS: &str = include_str!("tests.rs");
const TIMING: &str = include_str!("timing.rs");

#[test]
fn exact_extents() {
    let small = Plan::checked(2079, K, 21, SCALE2).unwrap();
    assert_eq!(
        (small.padded, small.scales, small.tail),
        (2176, 2_367_488, 105_536)
    );
    let large = Plan::checked(8192, K, 31, SCALE2).unwrap();
    assert_eq!(
        (large.padded, large.scales, large.tail),
        (8192, 8_912_896, 0)
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
fn source_contract() {
    require_sources().unwrap();
    let cu = include_str!("../../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu");
    let silu = include_str!("../../../../kernels/gb10/common/moe_silu_mul.cu");
    let physical = include_str!(
        "../../../../kernels/gb10/qwen3.8-27b/nvfp4/quantize_bf16_to_nvfp4_cutlass.cu"
    );
    let down = include_str!("../../../../kernels/gb10/qwen3.8-27b/nvfp4/cutlass_nvfp4_gemm.cu");
    let rust = include_str!("../../src/layers/ops/gemm_dense.rs");
    assert!(cu.contains("void quantize_silu_mul_bf16_to_nvfp4_atlas_128x4("));
    assert!(silu.contains("void moe_silu_mul("));
    assert!(physical.contains("void quantize_bf16_to_nvfp4_atlas_128x4("));
    assert!(down.contains("void nvfp4_nvfp4_gemm_kmajor_m256("));
    assert!(rust.contains("pub fn quantize_silu_mul_bf16_to_nvfp4_atlas_128x4("));
    assert!(rust.contains("row padding overflow"));
    assert!(check_hash("hostile", CUDA_SHA, SILU_SHA).is_err());
    assert!(check_hash("hostile", "A", "A").is_err());
}

#[test]
fn hostile_binary_drift() {
    let release = BinaryIdentity {
        path: "/sealed/gate".into(),
        sha256: "a".repeat(64),
        profile: "release",
    };
    assert!(stable_binary(&release, &release, true).is_ok());
    let debug = BinaryIdentity {
        profile: "debug",
        ..release.clone()
    };
    assert!(stable_binary(&debug, &debug, true).is_err());
    let changed = BinaryIdentity {
        sha256: "b".repeat(64),
        ..release.clone()
    };
    assert!(stable_binary(&release, &changed, true).is_err());
    let replaced = BinaryIdentity {
        path: "/replaced/gate".into(),
        ..release.clone()
    };
    assert!(stable_binary(&release, &replaced, true).is_err());
}

#[test]
fn hostile_bundle_drift() {
    let base = [
        (DIRECT_MODULES[0], "a"),
        (DIRECT_MODULES[1], "b"),
        (DIRECT_MODULES[2], "c"),
        (DIRECT_MODULES[3], "d"),
    ];
    let identity = bundle_identity("sm_121/qwen3.8-27b/nvfp4", &base).unwrap();
    let mut drift = base;
    drift[2].1 = "changed";
    assert_ne!(
        identity.sha256,
        bundle_identity(&identity.target, &drift).unwrap().sha256
    );
    assert!(bundle_identity(&identity.target, &base[..3]).is_err());
    let duplicate = [
        (DIRECT_MODULES[0], "a"),
        (DIRECT_MODULES[0], "b"),
        (DIRECT_MODULES[1], "c"),
        (DIRECT_MODULES[2], "d"),
        (DIRECT_MODULES[3], "e"),
    ];
    assert!(bundle_identity(&identity.target, &duplicate).is_err());
}

#[test]
fn every_source_respects_file_cap() {
    for (name, source) in [
        ("entry", ENTRY),
        ("contract", CONTRACT),
        ("guarded", GUARDED),
        ("launch", LAUNCH),
        ("provenance", PROVENANCE),
        ("runtime", RUNTIME),
        ("tests", TESTS),
        ("timing", TIMING),
    ] {
        assert!(source.lines().count() <= 250, "{name} exceeds 250 lines");
        assert!(source.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
    }
}

#[test]
fn receipt_contract() {
    let source = [
        ENTRY, CONTRACT, GUARDED, LAUNCH, PROVENANCE, RUNTIME, TESTS, TIMING,
    ]
    .concat();
    for required in [
        SCHEMA,
        "tail_zero_bytes",
        "scale2_bits",
        "gate_input_hash",
        "down_weight_hash",
        "paired_median_ms",
        "same_nondefault_stream",
        CUDA_SHA,
        WRAPPER_SHA,
        ROUTE_SHA,
        SILU_SHA,
        PHYSICAL_SHA,
        DOWN_SHA,
        "running_executable",
        "embedded_bundle",
        "build-attestation-v1",
        "performance_claim\":false",
    ] {
        assert!(source.contains(required));
    }
    assert!(TIMING.contains("timing gate failed"));
    assert!(
        RUNTIME.find("let timing = measure_balanced(").unwrap()
            < RUNTIME.find("let receipt = json!").unwrap()
    );
}
