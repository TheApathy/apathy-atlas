// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::config::parse_config;
use serde_json::{Value, json};

fn fixture() -> Value {
    json!({
        "model_type":"deepseek_v4", "hidden_size":4096,
        "num_hidden_layers":43, "num_attention_heads":64,
        "num_key_value_heads":1, "head_dim":512, "vocab_size":129280,
        "max_position_embeddings":1048576, "q_lora_rank":1024,
        "o_lora_rank":1024, "qk_rope_head_dim":64,
        "vision_dim":1024, "vision_inter_dim":2816, "vision_n_layers":32,
        "vision_n_heads":16, "vision_patch_size":14,
        "vision_downsample_ratio":3, "vision_max_n_token":384,
        "vision_max_wh_ratio":8, "vision_min_pixels":147456,
        "vision_rope_theta":10000.0
    })
}

#[test]
fn deepseek_vision_rejects_partial_or_malformed_contract() {
    for key in [
        "vision_dim",
        "vision_n_layers",
        "vision_patch_size",
        "vision_rope_theta",
    ] {
        let mut value = fixture();
        value.as_object_mut().unwrap().remove(key);
        assert!(parse_config(&value.to_string()).is_err(), "missing {key}");
    }
    for bad in [
        json!(null),
        json!(0),
        json!(-1),
        json!(true),
        json!("1024"),
        json!(1024.5),
    ] {
        let mut value = fixture();
        value["vision_dim"] = bad;
        assert!(parse_config(&value.to_string()).is_err());
    }
}

#[test]
fn deepseek_vision_has_a_distinct_typed_contract() {
    let config = parse_config(&fixture().to_string()).unwrap();
    assert!(config.vision.is_none(), "Must not alias Qwen's encoder");
    let vision = config.deepseek_vision.unwrap();
    assert_eq!(vision.hidden_size, 1024);
    assert_eq!(vision.intermediate_size, 2816);
    assert_eq!(vision.num_hidden_layers, 32);
    assert_eq!(vision.num_attention_heads, 16);
    assert_eq!(vision.patch_size, 14);
    assert_eq!(vision.downsample_ratio, 3);
    assert_eq!(vision.max_tokens, 384);
    assert_eq!(vision.max_wh_ratio, Some(8.0));
    assert_eq!(vision.min_pixels, 147456);
    assert_eq!(vision.rope_theta, 10000.0);
}

#[test]
fn deepseek_vision_rejects_unsupported_geometry_and_budget() {
    for (key, value) in [
        ("vision_n_heads", 8),
        ("vision_patch_size", 16),
        ("vision_downsample_ratio", 2),
        ("vision_max_n_token", 385),
        ("vocab_size", 129279),
        ("hidden_size", 2048),
    ] {
        let mut config = fixture();
        config[key] = json!(value);
        assert!(parse_config(&config.to_string()).is_err(), "{key}");
    }
}

#[test]
fn deepseek_text_and_nullable_aspect_ratio_preserve_semantics() {
    let mut value = fixture();
    value["vision_max_wh_ratio"] = Value::Null;
    assert!(
        parse_config(&value.to_string())
            .unwrap()
            .deepseek_vision
            .unwrap()
            .max_wh_ratio
            .is_none()
    );
    value
        .as_object_mut()
        .unwrap()
        .retain(|key, _| !key.starts_with("vision_"));
    let config = parse_config(&value.to_string()).unwrap();
    assert!(config.deepseek_vision.is_none());
    assert!(config.vision.is_none());
}

#[test]
fn deepseek_vision_hf_attention_labels_match_compression_schedule() {
    let mut value = fixture();
    let ratios: Vec<usize> = (0..43)
        .map(|i| {
            if i < 2 {
                0
            } else if i % 2 == 0 {
                4
            } else {
                128
            }
        })
        .collect();
    let labels: Vec<&str> = ratios
        .iter()
        .map(|ratio| match ratio {
            0 => "sliding_attention",
            4 => "compressed_sparse_attention",
            128 => "heavily_compressed_attention",
            _ => unreachable!(),
        })
        .collect();
    value["compress_ratios"] = json!(ratios);
    value["layer_types"] = json!(labels);
    let parsed = parse_config(&value.to_string()).unwrap();
    assert_eq!(parsed.compress_ratios, ratios);
    assert_eq!(
        parsed.layer_types,
        vec![atlas_core::config::LayerType::FullAttention; 43]
    );
    value["layer_types"][2] = json!("heavily_compressed_attention");
    assert!(parse_config(&value.to_string()).is_err());
    value["layer_types"][2] = json!("made_up_attention");
    assert!(parse_config(&value.to_string()).is_err());
}
