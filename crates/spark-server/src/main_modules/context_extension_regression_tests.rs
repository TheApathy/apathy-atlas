// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    ContextExtension, ContextRuntimeMode, apply_context_extension,
    validate_context_extension_runtime,
};
use atlas_core::config::ModelConfig;

fn prescaled_qwen38_yarn() -> ContextExtension {
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
    config.max_position_embeddings = 1_000_000;
    config.partial_rotary_factor = 0.25;
    config.yarn_factor = 4.0;
    config.yarn_beta_fast = 32.0;
    config.yarn_beta_slow = 1.0;
    config.yarn_original_max_position_embeddings = 262_144;
    apply_context_extension(&mut config, 1_000_000, None, None)
        .unwrap()
        .unwrap()
}

fn target_only_c1() -> ContextRuntimeMode {
    ContextRuntimeMode {
        max_batch_size: 1,
        max_seq_len: 1_000_000,
        config_capacity: 1_000_000,
        ..Default::default()
    }
}

#[test]
fn prescaled_checkpoint_yarn_remains_an_extension_from_its_origin() {
    let extension = prescaled_qwen38_yarn();
    assert!(matches!(
        extension,
        ContextExtension::Yarn {
            native_tokens: 1_000_000,
            requested_tokens: 1_000_000,
            original_tokens: 262_144,
            ..
        }
    ));
    assert!(validate_context_extension_runtime(Some(extension), target_only_c1()).is_ok());
    for mode in [
        ContextRuntimeMode {
            speculative: true,
            ..target_only_c1()
        },
        ContextRuntimeMode {
            dflash: true,
            ..target_only_c1()
        },
        ContextRuntimeMode {
            self_speculative: true,
            ..target_only_c1()
        },
        ContextRuntimeMode {
            ngram_speculative: true,
            ..target_only_c1()
        },
    ] {
        assert!(validate_context_extension_runtime(Some(extension), mode).is_err());
    }
}

#[test]
fn prescaled_checkpoint_yarn_keeps_the_full_resident_hss_boundary() {
    let extension = prescaled_qwen38_yarn();
    let hss = |blocks| ContextRuntimeMode {
        high_speed_swap: true,
        hss_cache_blocks_per_seq: blocks,
        block_size: 16,
        ..target_only_c1()
    };
    assert!(validate_context_extension_runtime(Some(extension), hss(64)).is_err());
    assert!(validate_context_extension_runtime(Some(extension), hss(62_499)).is_err());
    assert!(validate_context_extension_runtime(Some(extension), hss(62_500)).is_ok());
}

#[test]
fn extended_context_requires_exactly_one_decode_batch_slot() {
    let extension = prescaled_qwen38_yarn();
    for max_batch_size in [0, 2, 8, usize::MAX] {
        let mode = ContextRuntimeMode {
            max_batch_size,
            ..target_only_c1()
        };
        assert!(validate_context_extension_runtime(Some(extension), mode).is_err());
    }
    assert!(validate_context_extension_runtime(Some(extension), target_only_c1()).is_ok());
}

#[test]
fn native_context_preserves_existing_batch_admission() {
    for max_batch_size in [0, 1, 8, usize::MAX] {
        let mode = ContextRuntimeMode {
            max_batch_size,
            max_seq_len: 262_144,
            config_capacity: 262_144,
            ..Default::default()
        };
        assert!(validate_context_extension_runtime(None, mode).is_ok());
    }
}

#[test]
fn forged_extension_identity_never_mints_an_admission_receipt() {
    let wrong_target = ContextExtension::Yarn {
        native_tokens: 1_000_000,
        requested_tokens: 999_999,
        factor: 4.0,
        original_tokens: 262_144,
        attention_factor: 1.0,
    };
    assert!(validate_context_extension_runtime(Some(wrong_target), target_only_c1()).is_err());
    let impossible_native = ContextExtension::Yarn {
        native_tokens: 1_048_576,
        requested_tokens: 1_000_000,
        factor: 4.0,
        original_tokens: 262_144,
        attention_factor: 1.0,
    };
    assert!(validate_context_extension_runtime(Some(impossible_native), target_only_c1()).is_err());
}
