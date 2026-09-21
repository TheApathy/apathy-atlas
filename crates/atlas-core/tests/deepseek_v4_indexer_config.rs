// SPDX-License-Identifier: AGPL-3.0-only
//! Parser regressions for the checkpoint-native indexer contract only.
use atlas_core::config::{ModelConfig, parse_config};
use serde_json::{Value, json};

fn fixture() -> Value {
    json!({
        "model_type":"deepseek_v4", "hidden_size":4096,
        "num_hidden_layers":3, "num_attention_heads":64,
        "num_key_value_heads":1, "head_dim":512, "vocab_size":129280,
        "max_position_embeddings":4096, "q_lora_rank":1024,
        "o_lora_rank":1024, "qk_rope_head_dim":64,
        "rms_norm_eps":1e-20, "compress_ratios":[0,0,4],
        "index_n_heads":64, "index_head_dim":128, "index_topk":512
    })
}

#[test]
fn native_indexer_fields_are_preserved_without_changing_norm_or_schedule() {
    let config = parse_config(&fixture().to_string()).unwrap();
    let indexer = config.deepseek_v4_indexer.as_ref().unwrap();
    assert_eq!(
        (indexer.num_heads, indexer.head_dim, indexer.top_k),
        (64, 128, 512)
    );
    assert_eq!(config.rms_norm_eps, 1e-20);
    assert_eq!(config.compress_ratios, [0, 0, 4]);
    indexer.validate_model(&config).unwrap();
}

#[test]
fn absent_contract_preserves_legacy_and_unrelated_config() {
    let mut raw = fixture();
    for key in ["index_n_heads", "index_head_dim", "index_topk"] {
        raw.as_object_mut().unwrap().remove(key);
    }
    assert!(
        parse_config(&raw.to_string())
            .unwrap()
            .deepseek_v4_indexer
            .is_none()
    );
    assert!(
        ModelConfig::qwen3_next_80b_nvfp4()
            .deepseek_v4_indexer
            .is_none()
    );
}

#[test]
fn partial_or_malformed_indexer_never_becomes_legacy_absence() {
    for key in ["index_n_heads", "index_head_dim", "index_topk"] {
        let mut raw = fixture();
        raw.as_object_mut().unwrap().remove(key);
        assert!(parse_config(&raw.to_string()).is_err(), "missing {key}");
        for bad in [
            json!(null),
            json!(0),
            json!(-1),
            json!(true),
            json!("64"),
            json!(1.5),
            json!(u64::MAX),
        ] {
            let mut raw = fixture();
            raw[key] = bad;
            assert!(
                parse_config(&raw.to_string()).is_err(),
                "{key}: {}",
                raw[key]
            );
        }
    }
}

#[test]
fn stage_one_rejects_unimplemented_indexer_geometries() {
    for (key, value) in [
        ("index_n_heads", 32),
        ("index_head_dim", 64),
        ("index_topk", 1024),
    ] {
        let mut raw = fixture();
        raw[key] = json!(value);
        assert!(parse_config(&raw.to_string()).is_err(), "{key}");
    }
}

#[test]
fn indexer_model_dimensions_are_required_before_legacy_sanitization() {
    for key in [
        "hidden_size",
        "q_lora_rank",
        "qk_rope_head_dim",
        "num_hidden_layers",
    ] {
        for bad in [
            json!(null),
            json!(0),
            json!(-1),
            json!(true),
            json!("64"),
            json!(u64::MAX),
        ] {
            let mut raw = fixture();
            raw[key] = bad;
            assert!(
                parse_config(&raw.to_string()).is_err(),
                "{key}: {}",
                raw[key]
            );
        }
        let mut raw = fixture();
        raw.as_object_mut().unwrap().remove(key);
        assert!(parse_config(&raw.to_string()).is_err(), "missing {key}");
    }
}

#[test]
fn malformed_schedule_is_not_filter_mapped_into_another_layer() {
    for bad in [
        json!(null),
        json!([]),
        json!([0, 4]),
        json!([0, null, 4]),
        json!([0, -1, 4]),
        json!([0, "0", 4]),
        json!([0, 8, 4]),
    ] {
        let mut raw = fixture();
        raw["compress_ratios"] = bad;
        assert!(parse_config(&raw.to_string()).is_err());
    }
    let mut raw = fixture();
    raw["compress_ratios"] = json!([0, 0, 4, 0, 0, 0]);
    assert_eq!(
        parse_config(&raw.to_string()).unwrap().compress_ratios,
        [0, 0, 4, 0, 0, 0]
    );
}

#[test]
fn loader_revalidation_rejects_mutated_public_config() {
    let base = parse_config(&fixture().to_string()).unwrap();
    for changed in 0..7 {
        let mut config = base.clone();
        match changed {
            0 => config.hidden_size = 2048,
            1 => config.q_lora_rank = 0,
            2 => config.qk_rope_head_dim = 128,
            3 => config.compress_ratios.clear(),
            4 => config.num_hidden_layers = usize::MAX,
            5 => config.model_type = "qwen3_next".into(),
            _ => config.deepseek_v4_indexer.as_mut().unwrap().top_k = 0,
        }
        assert!(
            config
                .deepseek_v4_indexer
                .as_ref()
                .unwrap()
                .validate_model(&config)
                .is_err()
        );
    }
}
