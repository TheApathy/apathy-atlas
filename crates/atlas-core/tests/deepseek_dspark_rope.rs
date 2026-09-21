// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark must retain the declared uncompressed base independently of the target.

use atlas_core::config::{ModelConfig, parse_config};
use serde_json::{Value, json};

fn fixture() -> Value {
    json!({
        "model_type":"deepseek_v4", "hidden_size":4096,
        "num_hidden_layers":43, "num_attention_heads":64,
        "num_key_value_heads":1, "head_dim":512, "vocab_size":129280,
        "max_position_embeddings":4096, "q_lora_rank":1024,
        "o_lora_rank":1024, "qk_rope_head_dim":64,
        "rope_theta":10000, "compress_rope_theta":160000,
        "dspark_block_size":5
    })
}

#[test]
fn native_dspark_retains_both_declared_thetas() {
    let config = parse_config(&fixture().to_string()).unwrap();
    assert_eq!(config.deepseek_main_rope_theta, Some(10000.0));
    assert_eq!(config.rope_theta, 160000.0);
}

#[test]
fn base_is_checkpoint_data_not_a_new_hardcoded_default() {
    let mut raw = fixture();
    raw["rope_theta"] = json!(25000.0);
    let config = parse_config(&raw.to_string()).unwrap();
    assert_eq!(config.deepseek_main_rope_theta, Some(25000.0));
    assert_eq!(config.rope_theta, 160000.0);
}

#[test]
fn declared_dspark_requires_a_finite_positive_base_before_sanitizing() {
    for bad in [
        json!(null),
        json!(0),
        json!(-1),
        json!(true),
        json!("10000"),
    ] {
        let mut raw = fixture();
        raw["rope_theta"] = bad;
        assert!(parse_config(&raw.to_string()).is_err(), "{raw}");
    }
    let mut raw = fixture();
    raw.as_object_mut().unwrap().remove("rope_theta");
    assert!(parse_config(&raw.to_string()).is_err());
    let overflowing = fixture()
        .to_string()
        .replace("\"rope_theta\":10000", "\"rope_theta\":1e999");
    assert!(parse_config(&overflowing).is_err());
}

#[test]
fn absent_dspark_preserves_legacy_missing_and_null_base_behavior() {
    let mut raw = fixture();
    raw.as_object_mut().unwrap().remove("dspark_block_size");
    for explicit_null in [false, true] {
        raw.as_object_mut().unwrap().remove("rope_theta");
        if explicit_null {
            raw["rope_theta"] = Value::Null;
        }
        let config = parse_config(&raw.to_string()).unwrap();
        assert_eq!(config.deepseek_main_rope_theta, None);
        assert_eq!(config.rope_theta, 160000.0);
    }
    raw["dspark_block_size"] = json!(0);
    assert!(parse_config(&raw.to_string()).is_ok());
    assert_eq!(
        ModelConfig::qwen3_next_80b_nvfp4().deepseek_main_rope_theta,
        None
    );
}

#[test]
fn internal_field_cannot_replace_missing_checkpoint_declaration() {
    let mut raw = fixture();
    raw.as_object_mut().unwrap().remove("rope_theta");
    raw["deepseek_main_rope_theta"] = json!(10000);
    assert!(parse_config(&raw.to_string()).is_err());
}
