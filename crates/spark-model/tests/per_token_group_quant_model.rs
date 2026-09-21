// SPDX-License-Identifier: AGPL-3.0-only

use half::bf16;

const KERNEL: &str = include_str!("../../../kernels/gb10/common/per_token_group_quant_fp8.cu");
const DISPATCH: &str = include_str!("../src/layers/ops/gemm_quant.rs");

const FP8_E4M3_MAX: f32 = 448.0;
const MIN_SCALE: f32 = 1.0e-12;

fn cpu_quant_params(group: &[bf16]) -> (f32, Vec<f32>) {
    let absmax = group
        .iter()
        .map(|value| value.to_f32().abs())
        .fold(0.0_f32, f32::max);
    let scale = (absmax / FP8_E4M3_MAX).max(MIN_SCALE);
    let normalized = group
        .iter()
        .map(|value| (value.to_f32() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX))
        .collect();
    (scale, normalized)
}

#[test]
fn cpu_reference_covers_scale_floor_and_saturation() {
    let zeros = vec![bf16::ZERO; 128];
    let (zero_scale, zero_normalized) = cpu_quant_params(&zeros);
    assert_eq!(zero_scale, MIN_SCALE);
    assert!(zero_normalized.iter().all(|value| *value == 0.0));

    let mut mixed = vec![bf16::from_f32(1.0); 128];
    mixed[0] = bf16::from_f32(-1024.0);
    mixed[127] = bf16::from_f32(512.0);
    let (scale, normalized) = cpu_quant_params(&mixed);
    assert_eq!(scale, 1024.0 / FP8_E4M3_MAX);
    assert!((normalized[0] + FP8_E4M3_MAX).abs() < 1.0e-3);
    assert!((normalized[127] - 224.0).abs() < 1.0e-3);
}

#[test]
fn kernel_caches_each_bf16_value_across_absmax_reduction() {
    let global_read = "A[m * K + k_start + tid]";
    assert_eq!(
        KERNEL.matches(global_read).count(),
        1,
        "each lane must issue one BF16 global read, not reread after reduction"
    );
    assert!(KERNEL.contains("float my_value = 0.0f;"));
    assert!(KERNEL.contains("my_abs = fabsf(my_value);"));
    assert!(KERNEL.contains("float v = my_value / smem_scale;"));
}

#[test]
fn dispatch_rejects_partial_fp8_groups() {
    assert!(DISPATCH.contains("k.is_multiple_of(FP8_GROUP_K)"));
    assert!(DISPATCH.contains("const FP8_GROUP_K: u32 = 128;"));
    assert!(DISPATCH.contains(".grid([m, k / FP8_GROUP_K, 1])"));
}
