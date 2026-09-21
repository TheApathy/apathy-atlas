// SPDX-License-Identifier: AGPL-3.0-only

//! CPU admission tests: Vision must not enter generic non-V4 KV decode paths.

use atlas_core::config::{DeepSeekVisionConfig, ModelConfig};
use spark_model::factory::vision_kv::{
    validate_deepseek_vision_kv, validate_deepseek_vision_kv_request,
};
use spark_runtime::kv_cache::KvCacheDtype as D;

fn config(vision: bool) -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".into();
    config.deepseek_vision = vision.then_some(DeepSeekVisionConfig {
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
    config
}

#[test]
fn default_and_explicit_fp8_with_no_bf16_boundary_are_admitted() {
    for target_default in ["", "fp8"] {
        for boundary in ["0", "00"] {
            validate_deepseek_vision_kv_request(&config(true), "fp8", target_default, boundary)
                .unwrap();
        }
    }
    validate_deepseek_vision_kv(&config(true), D::Fp8, &[]).unwrap();
    let config = config(true);
    validate_deepseek_vision_kv(
        &config,
        D::Fp8,
        &vec![D::Fp8; config.num_attention_layers()],
    )
    .unwrap();
}

#[test]
fn requested_bf16_nvfp4_and_turbo_are_rejected_without_silent_override() {
    for requested in ["bf16", "nvfp4", "turbo4", "fp8k_turbo2v", "unknown"] {
        let error =
            validate_deepseek_vision_kv_request(&config(true), requested, "fp8", "0").unwrap_err();
        assert!(error.to_string().contains("--kv-cache-dtype fp8"));
    }
}

#[test]
fn target_default_cannot_silently_resolve_fp8_to_an_unsupported_dtype() {
    for target_default in ["bf16", "nvfp4", "turbo8", "unknown"] {
        let error = validate_deepseek_vision_kv_request(&config(true), "fp8", target_default, "0")
            .unwrap_err();
        assert!(error.to_string().contains("MODEL.toml"));
    }
}

#[test]
fn bf16_boundary_requests_and_invalid_counts_fail_closed() {
    for boundary in ["1", "2", "auto", "max", "all", "-1", "invalid", ""] {
        let error =
            validate_deepseek_vision_kv_request(&config(true), "fp8", "fp8", boundary).unwrap_err();
        assert!(error.to_string().contains("--kv-high-precision-layers 0"));
    }
}

#[test]
fn typed_effective_dtype_and_every_boundary_layer_are_checked() {
    let config = config(true);
    let mut layers = vec![D::Fp8; config.num_attention_layers()];
    for dtype in [D::Bf16, D::Nvfp4, D::Turbo8, D::Fp8KTurbo2V] {
        assert!(validate_deepseek_vision_kv(&config, dtype, &[]).is_err());
        assert!(validate_deepseek_vision_kv(&config, dtype, &layers).is_err());
        for index in 0..layers.len() {
            layers[index] = dtype;
            assert!(validate_deepseek_vision_kv(&config, D::Fp8, &layers).is_err());
            layers[index] = D::Fp8;
        }
    }
    layers.pop();
    assert!(validate_deepseek_vision_kv(&config, D::Fp8, &layers).is_err());
}

#[test]
fn same_deepseek_model_type_without_typed_vision_is_unchanged() {
    let config = config(false);
    for dtype in [D::Fp8, D::Bf16, D::Nvfp4, D::Turbo8] {
        validate_deepseek_vision_kv(&config, dtype, &[D::Bf16]).unwrap();
        validate_deepseek_vision_kv_request(&config, &dtype.to_string(), "bf16", "auto").unwrap();
    }
}

#[test]
fn actual_factory_rejects_unsupported_kv_before_any_weights_or_gpu_work() {
    use spark_runtime::{gpu::mock::MockGpuBackend, prefix_cache::NoPrefixCaching};
    let result = spark_model::factory::build_model(
        config(true),
        &spark_runtime::weights::WeightStore::empty(),
        Box::new(MockGpuBackend::new()),
        2048,
        16,
        12288,
        1,
        spark_model::layers::MtpQuantization::Nvfp4,
        false,
        Box::new(NoPrefixCaching),
        0,
        None,
        false,
        1,
        D::Bf16,
        512 * 1024 * 1024,
        0.84,
        Some(12304),
        0,
        vec![],
        0,
        None,
        None,
        None,
        None,
        None,
    );
    let error = result.err().expect("BF16 Vision must fail before loading");
    assert!(
        error
            .to_string()
            .contains("DeepSeek Vision requires FP8 KV")
    );
}

#[test]
fn serving_checks_requests_before_gpu_and_rechecks_resolved_layer_types() {
    let serve = include_str!("../../spark-server/src/main_modules/serve.rs");
    assert!(
        serve.find("validate_deepseek_vision_kv_request(").unwrap()
            < serve.find("serve_phases::init_gpu_backend(").unwrap()
    );
    let resolve = include_str!("../../spark-server/src/main_modules/serve_phases/kv_cache.rs");
    assert!(
        resolve.find("let layer_dtypes =").unwrap()
            < resolve.find("validate_deepseek_vision_kv(").unwrap()
    );
}

#[test]
fn actual_vision_cannot_capture_or_replay_a_decode_graph() {
    let decode = include_str!("../src/model/trait_impl/decode_a.rs");
    let eligibility = decode
        .split("let use_graphs =")
        .nth(1)
        .unwrap()
        .split(';')
        .next()
        .unwrap();
    assert!(eligibility.contains("self.config.deepseek_vision.is_none()"));
    assert!(decode.contains("let mut graph_cache = if use_graphs"));
    assert!(decode.contains("if use_graphs {\n            tracing::info!("));
}
