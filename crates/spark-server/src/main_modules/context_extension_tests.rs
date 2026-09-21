// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    ContextExtension, ContextRuntimeMode, apply_context_extension,
    validate_context_extension_runtime,
};
use atlas_core::config::ModelConfig;

fn qwen38() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen3_5".into();
    config.hidden_size = 5120;
    config.num_hidden_layers = 64;
    config.num_attention_heads = 24;
    config.num_key_value_heads = 4;
    config.head_dim = 256;
    config.num_experts = 0;
    config.mrope_interleaved = true;
    config.mrope_section = [11, 11, 10];
    config.max_position_embeddings = 262_144;
    config.partial_rotary_factor = 0.25;
    config
}

fn million_extension() -> ContextExtension {
    let mut config = qwen38();
    apply_context_extension(&mut config, 1_000_000, None, Some(4.0))
        .unwrap()
        .unwrap()
}

fn theta_extension() -> ContextExtension {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.max_position_embeddings = 262_144;
    apply_context_extension(&mut config, 300_000, Some(20_000_000.0), None)
        .unwrap()
        .unwrap()
}

fn runtime_mode(max_seq_len: usize) -> ContextRuntimeMode {
    ContextRuntimeMode {
        max_seq_len,
        config_capacity: max_seq_len,
        ..Default::default()
    }
}

#[test]
fn native_context_needs_no_override() {
    let mut config = qwen38();
    assert!(
        apply_context_extension(&mut config, 262_144, None, None)
            .unwrap()
            .is_none()
    );
    assert_eq!(config.rope_theta, 10_000_000.0);
}

#[test]
fn official_one_million_yarn_keeps_theta() {
    let mut config = qwen38();
    let extension = apply_context_extension(&mut config, 1_000_000, None, Some(4.0))
        .unwrap()
        .unwrap();
    assert!(matches!(
        extension,
        ContextExtension::Yarn { factor: 4.0, .. }
    ));
    assert_eq!(config.max_position_embeddings, 1_000_000);
    assert_eq!(config.rope_theta, 10_000_000.0);
    assert_eq!(config.yarn_original_max_position_embeddings, 262_144);
}

#[test]
fn qwen_theta_and_every_over_million_boundary_fail_closed() {
    let mut config = qwen38();
    assert!(apply_context_extension(&mut config, 1_000_000, Some(41_800_000.0), None).is_err());
    for requested in [1_000_001, 1_048_576] {
        let mut config = qwen38();
        assert!(apply_context_extension(&mut config, requested, None, Some(4.0)).is_err());
        assert_eq!(config.max_position_embeddings, 262_144);
    }
}

#[test]
fn missing_small_or_conflicting_scale_fails_closed() {
    let mut config = qwen38();
    assert!(apply_context_extension(&mut config, 1_000_000, None, None).is_err());
    let mut config = qwen38();
    assert!(apply_context_extension(&mut config, 1_000_000, None, Some(2.0)).is_err());
    assert_eq!(config.yarn_factor, 0.0);
    assert_eq!(config.max_position_embeddings, 262_144);
    let mut config = qwen38();
    assert!(
        apply_context_extension(&mut config, 1_000_000, Some(41_800_000.0), Some(4.0)).is_err()
    );

    let mut non_qwen38 = ModelConfig::qwen3_next_80b_nvfp4();
    let before = (non_qwen38.yarn_factor, non_qwen38.max_position_embeddings);
    assert!(apply_context_extension(&mut non_qwen38, 1_000_000, None, Some(4.0)).is_err());
    assert_eq!(
        (non_qwen38.yarn_factor, non_qwen38.max_position_embeddings,),
        before
    );

    let mut non_interleaved = qwen38();
    non_interleaved.mrope_interleaved = false;
    assert!(apply_context_extension(&mut non_interleaved, 1_000_000, None, Some(4.0)).is_err());
    assert_eq!(non_interleaved.yarn_factor, 0.0);
    assert_eq!(non_interleaved.max_position_embeddings, 262_144);
}

#[test]
fn compatible_checkpoint_yarn_is_honored() {
    let mut config = qwen38();
    config.yarn_factor = 4.0;
    config.yarn_beta_fast = 32.0;
    config.yarn_beta_slow = 1.0;
    config.yarn_original_max_position_embeddings = 262_144;
    assert!(matches!(
        apply_context_extension(&mut config, 1_000_000, None, None).unwrap(),
        Some(ContextExtension::Yarn { .. })
    ));
    assert!(matches!(
        // The preceding call extended this mutable config to one million.
        // Its next receipt must name that current capacity, not the old one.
        apply_context_extension(&mut config, 262_144, None, None).unwrap(),
        Some(ContextExtension::Yarn {
            native_tokens: 1_000_000,
            requested_tokens: 262_144,
            ..
        })
    ));
}

#[test]
fn extended_context_rejects_every_speculative_mode() {
    for mode in [
        ContextRuntimeMode {
            speculative: true,
            ..runtime_mode(1_000_000)
        },
        ContextRuntimeMode {
            dflash: true,
            ..runtime_mode(1_000_000)
        },
        ContextRuntimeMode {
            self_speculative: true,
            ..runtime_mode(1_000_000)
        },
        ContextRuntimeMode {
            ngram_speculative: true,
            ..runtime_mode(1_000_000)
        },
    ] {
        assert!(validate_context_extension_runtime(Some(million_extension()), mode).is_err());
    }
    assert!(
        validate_context_extension_runtime(Some(million_extension()), runtime_mode(1_000_000))
            .is_ok()
    );
}

#[test]
fn extended_context_hss_requires_full_nonoverflowing_residency() {
    let hss = |cache_blocks_per_seq, block_size| ContextRuntimeMode {
        high_speed_swap: true,
        hss_cache_blocks_per_seq: cache_blocks_per_seq,
        block_size,
        ..runtime_mode(1_000_000)
    };
    assert!(validate_context_extension_runtime(Some(million_extension()), hss(64, 16)).is_err());
    assert!(
        validate_context_extension_runtime(Some(million_extension()), hss(62_499, 16)).is_err()
    );
    assert!(validate_context_extension_runtime(Some(million_extension()), hss(62_500, 16)).is_ok());
    assert!(
        validate_context_extension_runtime(Some(million_extension()), hss(2, usize::MAX)).is_err()
    );
}

#[test]
fn nonextended_context_does_not_change_existing_mode_admission() {
    let mode = ContextRuntimeMode {
        dflash: true,
        high_speed_swap: true,
        hss_cache_blocks_per_seq: 1,
        block_size: 0,
        max_seq_len: 262_144,
        config_capacity: 262_144,
        ..Default::default()
    };
    assert!(validate_context_extension_runtime(None, mode).is_ok());
    assert!(
        validate_context_extension_runtime(
            Some(ContextExtension::Yarn {
                native_tokens: 262_144,
                requested_tokens: 262_144,
                factor: 4.0,
                original_tokens: 262_144,
                attention_factor: 1.0,
            }),
            mode,
        )
        .is_ok()
    );
}

#[test]
fn extended_theta_uses_the_same_target_only_and_hss_boundary() {
    let extension = theta_extension();
    assert!(matches!(
        extension,
        ContextExtension::Theta {
            native_tokens: 262_144,
            requested_tokens: 300_000,
            rope_theta: 20_000_000.0,
        }
    ));
    assert!(validate_context_extension_runtime(Some(extension), runtime_mode(300_000)).is_ok());
    for mode in [
        ContextRuntimeMode {
            speculative: true,
            ..runtime_mode(300_000)
        },
        ContextRuntimeMode {
            dflash: true,
            ..runtime_mode(300_000)
        },
        ContextRuntimeMode {
            self_speculative: true,
            ..runtime_mode(300_000)
        },
        ContextRuntimeMode {
            ngram_speculative: true,
            ..runtime_mode(300_000)
        },
    ] {
        assert!(validate_context_extension_runtime(Some(extension), mode).is_err());
    }
    let hss = |blocks| ContextRuntimeMode {
        high_speed_swap: true,
        hss_cache_blocks_per_seq: blocks,
        block_size: 16,
        ..runtime_mode(300_000)
    };
    assert!(validate_context_extension_runtime(Some(extension), hss(18_749)).is_err());
    assert!(validate_context_extension_runtime(Some(extension), hss(18_750)).is_ok());
}
