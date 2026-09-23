// SPDX-License-Identifier: AGPL-3.0-only

//! CPU layout/tail proofs; GPU numerical qualification remains separate.

#[test]
fn fp32_pv_tile_covers_each_output_once_and_masks_patch_tails() {
    for patches in [1_usize, 9, 20, 33, 65] {
        let p: Vec<f32> = (0..patches * patches)
            .map(|i| ((i * 13 % 97) as f32 + 0.375) / 97.0)
            .collect();
        let v: Vec<f32> = (0..64 * patches)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 16.0)
            .collect();
        let mut output = vec![f32::NAN; patches * 96];
        let mut writes = vec![0; patches * 96];
        for first in (0..patches).step_by(16) {
            let mut accum = [[0.0_f32; 4]; 256];
            for kb in (0..patches).step_by(32) {
                let mut sp = [0.0_f32; 16 * 32];
                let mut sv = [0.0_f32; 32 * 64];
                for tid in 0..256 {
                    for i in (tid..16 * 32).step_by(256) {
                        let (r, k) = (i / 32, i % 32);
                        if first + r < patches && kb + k < patches {
                            sp[i] = p[(first + r) * patches + kb + k];
                        }
                    }
                    for i in (tid..64 * 32).step_by(256) {
                        let (c, k) = (i / 32, i % 32);
                        if kb + k < patches {
                            sv[k * 64 + c] = v[c * patches + kb + k];
                        }
                    }
                }
                for (tid, sums) in accum.iter_mut().enumerate() {
                    let (col, row) = (tid % 64, tid / 64);
                    for k in 0..32 {
                        for (r, sum) in sums.iter_mut().enumerate() {
                            *sum = sp[(row + r * 4) * 32 + k].mul_add(sv[k * 64 + col], *sum);
                        }
                    }
                }
            }
            for (tid, sums) in accum.iter().enumerate() {
                let (col, row) = (tid % 64, tid / 64);
                for (r, &sum) in sums.iter().enumerate() {
                    let output_row = first + row + r * 4;
                    if output_row < patches {
                        let index = output_row * 96 + col;
                        output[index] = sum;
                        writes[index] += 1;
                    }
                }
            }
        }
        for row in 0..patches {
            for col in 0..96 {
                let index = row * 96 + col;
                if col < 64 {
                    let expected = (0..patches).fold(0.0, |sum, k| {
                        p[row * patches + k].mul_add(v[col * patches + k], sum)
                    });
                    assert_eq!(output[index], expected);
                    assert_eq!(writes[index], 1);
                } else {
                    assert!(output[index].is_nan());
                    assert_eq!(writes[index], 0);
                }
            }
        }
    }
}

#[test]
fn fp32_attention_source_cannot_silently_restore_bf16_probability_rounding() {
    // D symlinks deepseek-v4.1/cb3's vision kernels from deepseek-v4-flash/nvfp4 (the towers
    // are identical); that target doesn't exist on this port (out of scope, see
    // weight_loader/deepseek_v41.rs's doc comment), so these read the real copies this port
    // keeps directly under deepseek-v4.1/cb3 instead -- same file contents either way.
    const SOFTMAX: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/deepseek-v4.1/cb3/deepseek_vision.cu"
    ));
    const PV: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/deepseek-v4.1/cb3/deepseek_vision_pv.cu"
    ));
    assert!(SOFTMAX.contains("const float* scores, float* probs"));
    assert!(!SOFTMAX.contains("probs[row + c] = __float2bfloat16"));
    assert!(PV.contains("const float* probabilities"));
    assert!(PV.contains("accum[r] = fmaf("));
    assert!(!PV.contains("wmma::"));
    assert_eq!(PV.matches("__syncthreads();").count(), 2);
}

#[test]
fn explicit_observer_is_optional_and_propagates_callback_failure() {
    use super::observer::{VisionStageDtype, observe};
    use spark_runtime::gpu::{DevicePtr, mock::MockGpuBackend};
    let gpu = MockGpuBackend::new();
    observe(
        &gpu,
        &mut None,
        "unused",
        DevicePtr(1),
        [1, 1],
        VisionStageDtype::Bf16,
    )
    .unwrap();
    let mut calls = 0;
    let mut callback = |name: &str, ptr, shape, dtype| {
        calls += 1;
        assert_eq!(
            (name, ptr, shape, dtype),
            ("patch", DevicePtr(256), [9, 1024], VisionStageDtype::Bf16)
        );
        anyhow::bail!("diagnostic stop")
    };
    let error = observe(
        &gpu,
        &mut Some(&mut callback),
        "patch",
        DevicePtr(256),
        [9, 1024],
        VisionStageDtype::Bf16,
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "diagnostic stop");
    assert_eq!(calls, 1);
    assert_eq!(gpu.launch_count(), 0);
}
