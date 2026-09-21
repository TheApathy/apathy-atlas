// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/moe/native_shared_verify.rs"]
mod native_shared_verify;

use native_shared_verify::NativeSharedVerifyPlan;
use spark_runtime::gpu::DevicePtr;

fn pointers() -> [DevicePtr; 4] {
    [
        DevicePtr(0x100000),
        DevicePtr(0x200000),
        DevicePtr(0x300000),
        DevicePtr(0x400000),
    ]
}

#[test]
fn every_supported_verify_width_has_exact_batched_geometry() {
    for rows in 2..=8 {
        let p = pointers();
        NativeSharedVerifyPlan::new(rows, 8, 4096, 2048, p, [32768, 32768, 65536]).unwrap();
    }
}

#[test]
fn rejects_empty_over_capacity_wrong_shape_and_short_output_buffers() {
    let p = pointers();
    let caps = [24576, 24576, 49152];
    for rows in [0, 1, 7, 9, u32::MAX] {
        assert!(NativeSharedVerifyPlan::new(rows, 6, 4096, 2048, p, caps).is_err());
    }
    for (h, inter) in [(0, 2048), (4095, 2048), (4096, 0), (4096, 2049)] {
        assert!(NativeSharedVerifyPlan::new(6, 6, h, inter, p, caps).is_err());
    }
    for i in 0..3 {
        let mut short = caps;
        short[i] -= 1;
        assert!(NativeSharedVerifyPlan::new(6, 6, 4096, 2048, p, short).is_err());
    }
}

#[test]
fn rejects_null_misaligned_overflow_and_any_input_output_alias() {
    let caps = [24576, 24576, 49152];
    for i in 0..4 {
        for bad in [0, 3, 0x500002, 0x500004, 0x500008, u64::MAX - 15] {
            let mut p = pointers();
            p[i] = DevicePtr(bad);
            assert!(NativeSharedVerifyPlan::new(6, 6, 4096, 2048, p, caps).is_err());
        }
        for j in 0..4 {
            if i == j {
                continue;
            }
            let mut p = pointers();
            p[i] = p[j].offset(2);
            assert!(NativeSharedVerifyPlan::new(6, 6, 4096, 2048, p, caps).is_err());
        }
    }
}

#[test]
fn verify_executor_uses_exact_batched_native_projection_not_gemm() {
    let source = include_str!("../src/layers/moe/native_shared_fp8.rs");
    let body = source
        .split("fn run_native_fp8_shared_verify(")
        .nth(1)
        .unwrap()
        .split("pub(super) fn run_native_fp8_shared_expert(")
        .next()
        .unwrap();
    assert!(body.contains("NativeSharedVerifyPlan::new("));
    assert_eq!(body.matches("ops::w8a16_gemv_batchm_exact(").count(), 1);
    assert_eq!(body.matches("project(").count(), 3);
    assert!(body.contains("ops::silu_mul("));
    assert!(!body.contains("w8a16_gemm"));
    assert!(body.contains("sizes.logits"));
    assert!(body.contains("sizes.ssm_qkvz"));
    assert!(body.contains("sizes.attn_output"));
}
