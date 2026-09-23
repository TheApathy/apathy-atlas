// SPDX-License-Identifier: AGPL-3.0-only
//! Admission and wiring regressions only. CPU/source tests are not CUDA
//! numerical proof; the retained same-QKV GPU hashes are the numerical gate.

use super::{DeepSeekVisionEncoder, geometry::Geometry};
use atlas_core::config::DeepSeekVisionConfig;
use spark_runtime::{gpu::mock::MockGpuBackend, weights::WeightStore};

fn config(theta: f64) -> DeepSeekVisionConfig {
    DeepSeekVisionConfig {
        hidden_size: 1024,
        intermediate_size: 2816,
        num_hidden_layers: 32,
        num_attention_heads: 16,
        patch_size: 14,
        downsample_ratio: 3,
        max_tokens: 384,
        max_wh_ratio: Some(8.0),
        min_pixels: 147456,
        rope_theta: theta,
    }
}

#[test]
fn deepseek_vision_angles_reject_noncanonical_theta_before_weights_or_gpu_effects() {
    for theta in [
        1.0,
        f64::from_bits(10_000.0f64.to_bits() - 1),
        f64::from_bits(10_000.0f64.to_bits() + 1),
        f64::MIN_POSITIVE,
        f64::MAX,
    ] {
        let gpu = MockGpuBackend::new();
        let error = DeepSeekVisionEncoder::load(&WeightStore::empty(), &config(theta), 4096, &gpu)
            .err()
            .expect("noncanonical theta must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("theta") && message.contains("10000"),
            "theta={theta}: {message}"
        );
        assert!(
            !message.contains("missing DeepSeek vision tensor"),
            "theta guard must precede weights"
        );
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.launch_count(), 0);
    }
}

#[test]
fn deepseek_vision_angles_exact_theta_reaches_existing_weight_validation() {
    let gpu = MockGpuBackend::new();
    let error = DeepSeekVisionEncoder::load(&WeightStore::empty(), &config(10_000.0), 4096, &gpu)
        .err()
        .expect("empty weights must still fail");
    assert!(error.to_string().contains("missing DeepSeek vision tensor"));
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn deepseek_vision_angles_use_existing_head_and_padded_grid_bounds() {
    let g = Geometry::new(1024, 2816, 16, 14, 3, 384, 4096).unwrap();
    assert_eq!(g.head_dim, 64);
    for (h, w, rows) in [(3, 3, 1), (4, 5, 4), (54, 54, 324), (48, 72, 384)] {
        assert_eq!(g.grid(h, w, h * w * 588).unwrap(), (h * w, rows));
        assert!(h * w * 64 * 4 <= g.max_patches * g.head_dim * 4);
    }
    // Patch count alone is insufficient: ceil-to-three padding must fit too.
    assert!(g.grid(54, 64, 54 * 64 * 588).is_err());
    for (h, w, pixels) in [
        (0, 3, 0),
        (usize::MAX, 3, 0),
        (49, 72, 49 * 72 * 588),
        (3, 3, 1),
    ] {
        assert!(g.grid(h, w, pixels).is_err());
    }
    assert!(Geometry::new(1024, 2816, 8, 14, 3, 384, 4096).is_err());
    assert!(Geometry::new(2048, 2816, 16, 14, 3, 384, 4096).is_err());
}

fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

#[test]
fn deepseek_vision_angles_kernel_is_required_before_scratch_allocation() {
    let source = compact(include_str!("mod.rs"));
    assert!(source.contains("angles:KernelHandle"));
    let load = source
        .find("angles:gpu.kernel(\"deepseek_vision_angles\",\"deepseek_vision_angles\")?")
        .expect("new angle kernel must be a required handle, not an optional fallback");
    // The scratch arena is allocated through the (owned or unowned) DeviceAllocs guard.
    assert!(load < source.find("letarena=allocs.alloc(gpu,bytes)?").unwrap());
}

#[test]
fn deepseek_vision_angles_launch_has_three_arguments_on_the_consumer_stream() {
    let source = compact(include_str!("forward.rs"));
    assert!(
        !source.contains("rope_angles"),
        "production must not construct host-libm angles"
    );
    assert!(
        !source.contains("angle_bytes"),
        "production host-angle upload must be removed"
    );
    let run = source
        .split_once("fnrun(")
        .unwrap()
        .1
        .split_once("fnlinear(")
        .unwrap()
        .0;
    let stream = run.find("letstream=gpu.default_stream();").unwrap();
    let producer = run
        .find("KernelLaunch::new(gpu,self.kernels.angles)")
        .expect("device angle producer missing");
    let first_consumer = run.find("self.linear(gpu,s.pixels").unwrap();
    assert!(stream < producer && producer < first_consumer);
    let launch = run[producer..].split_once(".launch(stream)?;").unwrap().0;
    assert!(launch.contains(".grid([div_ceil((p*32)asu32,256),1,1])"));
    assert!(launch.contains(".block([256,1,1])"));
    assert!(launch.contains(".arg_ptr(s.angles).arg_u32(grid_hasu32).arg_u32(grid_wasu32)"));
    assert_eq!(launch.matches(".arg_ptr(").count(), 1);
    assert_eq!(launch.matches(".arg_u32(").count(), 2);
    assert!(!launch.contains(".arg_f32("));
    assert_eq!(
        run.matches("KernelLaunch::new(gpu,self.kernels.angles)")
            .count(),
        1
    );
}

#[test]
fn deepseek_vision_angles_cuda_entry_preserves_the_proven_abi_and_bounds() {
    // Real copy on this port, not a symlink into deepseek-v4-flash (out of scope; see
    // pv_tests.rs's fp32_attention_source_cannot_silently_restore_bf16_probability_rounding).
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/deepseek-v4.1/cb3/deepseek_vision_angles.cu");
    let source = compact(&std::fs::read_to_string(path).expect("model-local angle kernel missing"));
    assert!(source.contains(
        "extern\"C\"__global__voiddeepseek_vision_angles(float*out,unsignedgh,unsignedgw)"
    ));
    assert!(source.contains("if(i>=gh*gw*32)return;"));
    assert!(source.contains("out[row*64+col]=cosf(angle);"));
    assert!(source.contains("out[row*64+32+col]=sinf(angle);"));
}
