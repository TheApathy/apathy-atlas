// SPDX-License-Identifier: AGPL-3.0-only

use super::geometry::{Geometry, bf16_bytes, rope_angles};
use super::weights::{WeightSpec, validate_tensor};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightTensor};

#[test]
fn deepseek_vision_geometry_pads_incomplete_three_by_three_edges() {
    let g = Geometry::new(1024, 2816, 16, 14, 3, 384, 4096).unwrap();
    assert_eq!(g.grid(4, 5, 20 * 588).unwrap().1, 4);
    assert_eq!(g.grid(3, 3, 9 * 588).unwrap().1, 1);
    assert!(g.grid(0, 3, 0).is_err());
    assert!(g.grid(usize::MAX, 3, 0).is_err());
    assert!(g.grid(100, 100, 10_000 * 588).is_err());
    assert!(g.grid(3, 3, 9 * 588 - 1).is_err());
    assert!(g.scratch_bytes().unwrap() < 256 * 1024 * 1024);
}

#[test]
fn deepseek_vision_rotary_is_half_split_with_height_then_width_frequencies() {
    let a = rope_angles(2, 2, 64, 10_000.0).unwrap();
    assert_eq!(a.len(), 4 * 64);
    // Each patch carries cos[32],sin[32]; first 16 angles encode height.
    assert!(a[..32].iter().all(|&x| x == 1.0));
    assert!(a[32..64].iter().all(|&x| x == 0.0));
    assert_eq!(a[64], 1.0); // row 0, col 1: height angle is zero.
    assert!((a[64 + 16] - 1.0_f32.cos()).abs() < 1e-6);
    assert!((a[128] - 1.0_f32.cos()).abs() < 1e-6);
    assert_eq!(a[128 + 16], 1.0); // row 1, col 0.
    assert!(rope_angles(1, 1, 62, 10_000.0).is_err());
    assert!(rope_angles(1, 1, 64, f64::NAN).is_err());
    assert!(rope_angles(1, 1, 64, f64::MIN_POSITIVE).is_err());
    assert!(rope_angles(1, 1, 64, f64::MAX).is_err());
}

#[test]
fn deepseek_vision_bf16_input_uses_nearest_even_and_rejects_nonfinite() {
    let a = f32::from_bits(0x3f80_8000);
    let b = f32::from_bits(0x3f81_8000);
    assert_eq!(bf16_bytes(&[a, b]).unwrap(), [0x80, 0x3f, 0x82, 0x3f]);
    assert!(bf16_bytes(&[f32::NAN]).is_err());
    assert!(bf16_bytes(&[f32::INFINITY]).is_err());
}

#[test]
fn deepseek_vision_native_weight_contract_rejects_wrong_shape_and_dtype() {
    let spec = WeightSpec {
        name: "vision.patch_embed.proj.weight".into(),
        shape: vec![1024, 588],
    };
    let mut tensor = WeightTensor {
        ptr: DevicePtr(256),
        dtype: WeightDtype::BF16,
        shape: vec![1024, 588],
    };
    validate_tensor(&spec, &tensor).unwrap();
    tensor.shape = vec![588, 1024];
    assert!(validate_tensor(&spec, &tensor).is_err());
    tensor.shape = spec.shape.clone();
    tensor.dtype = WeightDtype::FP8E4M3;
    assert!(validate_tensor(&spec, &tensor).is_err());
}

#[test]
fn deepseek_vision_missing_weights_fail_before_scratch_allocation() {
    use atlas_core::config::DeepSeekVisionConfig;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::WeightStore;
    let config = DeepSeekVisionConfig {
        hidden_size: 1024,
        intermediate_size: 2816,
        num_hidden_layers: 32,
        num_attention_heads: 16,
        patch_size: 14,
        downsample_ratio: 3,
        max_tokens: 384,
        max_wh_ratio: Some(8.0),
        min_pixels: 147456,
        rope_theta: 10_000.0,
    };
    let gpu = MockGpuBackend::new();
    assert!(
        super::DeepSeekVisionEncoder::load(&WeightStore::empty(), &config, 4096, &gpu).is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn deepseek_vision_scratch_regions_are_aligned_and_disjoint() {
    let g = Geometry::new(1024, 2816, 16, 14, 3, 384, 4096).unwrap();
    let (offsets, total) = g.scratch_layout().unwrap();
    assert_eq!(offsets.len(), 13);
    assert_eq!(offsets[0], 0);
    assert!(offsets.iter().all(|x| x.is_multiple_of(256)));
    assert!(offsets.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(offsets[12] + 384 * 4096 * 2, total);
}

#[test]
fn deepseek_vision_attention_probabilities_keep_fp32_precision() {
    let g = Geometry::new(1024, 2816, 16, 14, 3, 384, 4096).unwrap();
    let (offsets, total) = g.scratch_layout().unwrap();
    let attention_bytes = g.max_patches * g.max_patches * size_of::<f32>();
    assert_eq!(offsets[10] - offsets[9], attention_bytes);
    assert_eq!(offsets[11] - offsets[10], attention_bytes);
    assert_eq!(total, 206_275_584);
}

#[test]
fn v41_geometry_admits_5120_and_1024_tokens_only() {
    assert!(Geometry::new(1024, 2816, 16, 14, 3, 1024, 5120).is_ok());
    assert!(Geometry::new(1024, 2816, 16, 14, 3, 1025, 5120).is_err());
    // the V4 limits are unchanged
    assert!(Geometry::new(1024, 2816, 16, 14, 3, 385, 4096).is_err());
    assert!(Geometry::new(1024, 2816, 16, 14, 3, 384, 4096).is_ok());
    assert!(Geometry::new(1024, 2816, 16, 14, 3, 384, 7168).is_err());
}
