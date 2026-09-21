// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::config::DeepSeekVisionConfig;
use spark_runtime::gpu::{DevicePtr, mock::MockGpuBackend};

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "deepseek_v4".into();
    c.hidden_size = 4096;
    c.hc_mult = 4;
    c.ep_world_size = 1;
    c.tp_world_size = 1;
    c.deepseek_vision = Some(DeepSeekVisionConfig {
        hidden_size: 1024,
        intermediate_size: 2816,
        num_hidden_layers: 32,
        num_attention_heads: 16,
        patch_size: 14,
        downsample_ratio: 3,
        max_tokens: 384,
        max_wh_ratio: None,
        min_pixels: 147456,
        rope_theta: 10000.0,
    });
    c
}

fn hc() -> HcWeights {
    let site = || super::super::HcSiteWeights {
        hc_fn: DevicePtr(1),
        hc_base: DevicePtr(2),
        hc_scale: DevicePtr(3),
    };
    HcWeights {
        attn: site(),
        ffn: site(),
        head: None,
        hc_mult: 4,
        sinkhorn_iters: 20,
        hc_eps: 1e-6,
    }
}

#[test]
fn absent_and_zero_leave_every_model_and_kernel_untouched() {
    for c in [config(), ModelConfig::qwen3_next_80b_nvfp4()] {
        for flag in [None, Some(OsStr::new("0"))] {
            let policy = VisionHcBf16::parse(flag, &c).unwrap();
            assert_eq!(policy, VisionHcBf16(false));
            policy.validate_execution(99, false, false).unwrap();
            policy.validate_hc(None).unwrap();
            assert!(
                policy
                    .resolve(|_, _| panic!("disabled lookup"))
                    .unwrap()
                    .is_none()
            );
        }
    }
}

#[test]
fn only_exact_one_and_actual_geometry_are_admitted() {
    let mut c = config();
    assert_eq!(
        VisionHcBf16::parse(Some(OsStr::new("1")), &c).unwrap(),
        VisionHcBf16(true)
    );
    for bad in ["", "true", "01", " 1", "1 ", "2", "-1"] {
        assert!(VisionHcBf16::parse(Some(OsStr::new(bad)), &c).is_err());
    }
    for field in 0..6 {
        c = config();
        match field {
            0 => c.deepseek_vision = None,
            1 => c.model_type = "qwen3_next".into(),
            2 => c.hidden_size = 8192,
            3 => c.hc_mult = 8,
            4 => c.ep_world_size = 2,
            _ => c.tp_world_size = 2,
        }
        assert!(VisionHcBf16::parse(Some(OsStr::new("1")), &c).is_err());
    }
}

#[cfg(unix)]
#[test]
fn invalid_utf8_is_not_silently_disabled() {
    use std::os::unix::ffi::OsStrExt;
    assert!(VisionHcBf16::parse(Some(OsStr::from_bytes(&[255])), &config()).is_err());
}

#[test]
fn c1_single_gpu_target_only_are_required() {
    let policy = VisionHcBf16(true);
    policy.validate_execution(1, true, true).unwrap();
    for n in [0, 2, 32] {
        assert!(policy.validate_execution(n, true, true).is_err());
    }
    assert!(policy.validate_execution(1, false, true).is_err());
    assert!(policy.validate_execution(1, true, false).is_err());
}

#[test]
fn actual_hc_state_and_each_site_pointer_are_checked() {
    let policy = VisionHcBf16(true);
    assert!(policy.validate_hc(None).is_err());
    policy.validate_hc(Some(&hc())).unwrap();
    for field in 0..7 {
        let mut h = hc();
        match field {
            0 => h.hc_mult = 8,
            1 => h.attn.hc_fn = DevicePtr::NULL,
            2 => h.attn.hc_base = DevicePtr::NULL,
            3 => h.attn.hc_scale = DevicePtr::NULL,
            4 => h.ffn.hc_fn = DevicePtr::NULL,
            5 => h.ffn.hc_base = DevicePtr::NULL,
            _ => h.ffn.hc_scale = DevicePtr::NULL,
        }
        assert!(policy.validate_hc(Some(&h)).is_err());
    }
}

#[test]
fn kernel_lookup_is_strict_and_does_not_allocate_or_launch() {
    let policy = VisionHcBf16(true);
    assert!(
        policy
            .resolve(|_, _| anyhow::bail!("missing entry"))
            .is_err()
    );
    assert!(policy.resolve(|_, _| Ok(KernelHandle(0))).is_err());
    assert_eq!(
        policy
            .resolve(|module, name| {
                assert_eq!(module, "deepseek_vision_hc_post_bf16");
                assert_eq!(name, module);
                Ok(KernelHandle(123))
            })
            .unwrap()
            .unwrap()
            .0,
        123
    );
    let gpu = MockGpuBackend::new();
    policy.validate_kernel(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(gpu.launch_count(), 0);
}
