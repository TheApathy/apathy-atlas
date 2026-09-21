// SPDX-License-Identifier: AGPL-3.0-only
use atlas_core::config::{DeepSeekVisionConfig, ModelConfig};

#[path = "../src/api/chat/image_admission.rs"]
mod admission;
#[path = "../src/main_modules/serve_phases/vision_context.rs"]
mod context;

fn native() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".into();
    config.yarn_factor = 16.0;
    config.yarn_original_max_position_embeddings = 65536;
    config.deepseek_vision = Some(DeepSeekVisionConfig {
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
fn native_compressed_yarn_does_not_reject_ordinary_vision_context() {
    for limit in [1, 12288, 65536] {
        assert!(!context::text_only_yarn_context(&native(), limit));
        admission::validate(
            2,
            true,
            1,
            context::text_only_yarn_context(&native(), limit),
        )
        .unwrap();
    }
}

#[test]
fn unqualified_extension_or_missing_native_ceiling_stays_text_only() {
    for limit in [0, 65537, usize::MAX] {
        assert!(context::text_only_yarn_context(&native(), limit));
    }
    let mut config = native();
    config.yarn_original_max_position_embeddings = 0;
    assert!(context::text_only_yarn_context(&config, 12288));
}

#[test]
fn other_yarn_architectures_and_untyped_deepseek_are_not_exempted() {
    let mut config = native();
    config.model_type = "qwen3_next".into();
    assert!(context::text_only_yarn_context(&config, 12288));
    config.model_type = "deepseek_v4".into();
    config.deepseek_vision = None;
    assert!(context::text_only_yarn_context(&config, 12288));
    config.yarn_factor = 0.0;
    assert!(!context::text_only_yarn_context(&config, 12288));
}

#[test]
fn serve_uses_the_effective_model_and_served_context_classification() {
    let source = include_str!("../src/main_modules/serve.rs");
    assert!(
        source.contains(
            "yarn_context: serve_phases::text_only_yarn_context(&config, args.max_seq_len)"
        )
    );
    assert!(!source.contains("yarn_context: config.yarn_factor > 0.0"));
}

#[test]
fn image_choices_fail_before_pixels_can_be_dropped() {
    assert!(admission::validate_choices(1, 1).is_ok());
    assert!(admission::validate_choices(0, 4).is_ok());
    for choices in [0, 2, usize::MAX] {
        assert!(admission::validate_choices(1, choices).is_err());
    }
}

#[test]
fn image_rejection_balances_active_request_metrics() {
    let source = include_str!("../src/api/chat/mod.rs");
    let guard = source.find("image_admission::validate(").unwrap();
    let body = &source[guard..source[guard..].find("// Tool-parser").unwrap() + guard];
    assert!(body.contains("image_admission::validate_choices(image_count, req.n)"));
    assert!(
        body.find("REQUESTS_ACTIVE.dec()").unwrap()
            < body.find("return ChatOutcome::Http").unwrap()
    );
}
