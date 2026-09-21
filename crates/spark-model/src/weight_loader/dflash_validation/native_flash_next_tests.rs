// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use serde_json::{Value, json};
use spark_runtime::weights::WeightDtype;

use super::{
    TensorMetadata, TensorMetadataSource, native_flash_next_finiteness_manifest,
    validate_dflash_metadata,
};
use crate::weight_loader::dflash_loader::{DflashConfig, parse_dflash_config};

struct FakeStore(HashMap<String, (Vec<usize>, WeightDtype)>);

impl TensorMetadataSource for FakeStore {
    fn metadata(&self, name: &str) -> Option<TensorMetadata<'_>> {
        self.0.get(name).map(|(shape, dtype)| TensorMetadata {
            shape,
            dtype: *dtype,
        })
    }

    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.0.keys().map(String::as_str))
    }
}

fn native_config_json() -> Value {
    json!({
        "architectures": ["DFlashDraftModel"],
        "block_size": 16,
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 7, 13, 20, 26, 33, 39, 46]
        },
        "head_dim": 128,
        "hidden_size": 2560,
        "intermediate_size": 8704,
        "is_causal": false,
        "model_type": "qwen3",
        "num_attention_heads": 20,
        "num_hidden_layers": 6,
        "num_key_value_heads": 4,
        "num_target_layers": 48,
        "tie_word_embeddings": false,
        "vocab_size": 248320
    })
}

fn native_config() -> DflashConfig {
    parse_dflash_config(&native_config_json().to_string()).expect("exact native config")
}

fn dflash2_config_json() -> Value {
    let mut config = native_config_json();
    config["architectures"] = json!(["DFlash2DraftModel"]);
    config["dflash_config"]["conv_kernel_size"] = json!(2);
    config["dflash_config"]["conv_group_size"] = json!(16);
    config["dflash_config"]["selector_rank"] = json!(256);
    config["dflash_config"]["selector_top_k"] = json!(16);
    config["dflash_config"]["selector_vocab_size"] = json!(248077);
    config["layer_types"] = json!([
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ]);
    config["sliding_window"] = json!(4096);
    config
}

fn dflash2_config() -> DflashConfig {
    parse_dflash_config(&dflash2_config_json().to_string()).expect("exact native DFlash2 config")
}

fn native_store(prefix: &str) -> FakeStore {
    let mut tensors = HashMap::new();
    let mut add = |name: String, shape: &[usize]| {
        tensors.insert(name, (shape.to_vec(), WeightDtype::BF16));
    };
    add(format!("{prefix}fc.weight"), &[2560, 20480]);
    add(format!("{prefix}hidden_norm.weight"), &[2560]);
    add(format!("{prefix}norm.weight"), &[2560]);
    for layer in 0..6 {
        let lp = format!("{prefix}layers.{layer}");
        add(format!("{lp}.input_layernorm.weight"), &[2560]);
        add(format!("{lp}.post_attention_layernorm.weight"), &[2560]);
        add(format!("{lp}.self_attn.q_proj.weight"), &[2560, 2560]);
        add(format!("{lp}.self_attn.k_proj.weight"), &[512, 2560]);
        add(format!("{lp}.self_attn.v_proj.weight"), &[512, 2560]);
        add(format!("{lp}.self_attn.o_proj.weight"), &[2560, 2560]);
        add(format!("{lp}.self_attn.q_norm.weight"), &[128]);
        add(format!("{lp}.self_attn.k_norm.weight"), &[128]);
        add(format!("{lp}.mlp.gate_proj.weight"), &[8704, 2560]);
        add(format!("{lp}.mlp.up_proj.weight"), &[8704, 2560]);
        add(format!("{lp}.mlp.down_proj.weight"), &[2560, 8704]);
    }
    FakeStore(tensors)
}

fn dflash2_store() -> FakeStore {
    let mut store = native_store("");
    let mut add = |name: String, shape: &[usize]| {
        store.0.insert(name, (shape.to_vec(), WeightDtype::BF16));
    };
    for layer in 0..6 {
        for sublayer in ["attention_conv", "mlp_conv"] {
            add(
                format!("layers.{layer}.{sublayer}.base_kernel"),
                &[2, 2, 2560],
            );
            add(
                format!("layers.{layer}.{sublayer}.kernel_projection.weight"),
                &[640, 2560],
            );
        }
    }
    add(
        "candidate_selector.hidden_projection.weight".into(),
        &[256, 2560],
    );
    add(
        "candidate_selector.predecessor_codebook".into(),
        &[248320, 256],
    );
    add(
        "candidate_selector.successor_codebook".into(),
        &[248320, 256],
    );
    store
}

#[test]
fn accepts_exact_native_flash_next_v3_structure() {
    let store = native_store("");
    assert_eq!(store.0.len(), 69);
    assert_eq!(
        validate_dflash_metadata(&store, &native_config()).unwrap(),
        Some("")
    );
}

#[test]
fn exposes_exact_native_finiteness_receipt_manifest() {
    let manifest = native_flash_next_finiteness_manifest(&native_config())
        .unwrap()
        .expect("native receipt manifest");
    let parameters: usize = manifest
        .values()
        .map(|shape| shape.iter().product::<usize>())
        .sum();
    assert_eq!(manifest.len(), 69);
    assert_eq!(parameters, 547_918_336);
    assert_eq!(manifest["fc.weight"], [2560, 20480]);
}

#[test]
fn accepts_exact_native_flash_next_dflash2_structure_and_receipt() {
    let store = dflash2_store();
    assert_eq!(store.0.len(), 96);
    assert_eq!(
        validate_dflash_metadata(&store, &dflash2_config()).unwrap(),
        Some("")
    );

    let manifest = native_flash_next_finiteness_manifest(&dflash2_config())
        .unwrap()
        .expect("native DFlash2 receipt manifest");
    let parameters: usize = manifest
        .values()
        .map(|shape| shape.iter().product::<usize>())
        .sum();
    assert_eq!(manifest.len(), 96);
    assert_eq!(parameters, 695_497_216);
    assert_eq!(
        manifest["layers.5.mlp_conv.kernel_projection.weight"],
        [640, 2560]
    );
    assert_eq!(
        manifest["candidate_selector.predecessor_codebook"],
        [248320, 256]
    );
}

#[test]
fn generic_dflash_has_no_native_finiteness_scan_contract() {
    let mut config = native_config();
    config.num_target_layers = 64;
    config.hidden_size = 2048;
    config.intermediate_size = 11_008;
    config.num_hidden_layers = 5;
    config.dflash_config.as_mut().unwrap().target_layer_ids = vec![1, 10, 18, 27, 35];
    assert!(
        native_flash_next_finiteness_manifest(&config)
            .unwrap()
            .is_none()
    );
}

#[test]
fn rejects_each_native_identity_field_mutation_during_parse() {
    let mutations: &[(&str, fn(&mut Value))] = &[
        ("architectures", |v| v["architectures"] = json!(["Other"])),
        ("model_type", |v| v["model_type"] = json!("qwen4")),
        ("root block", |v| v["block_size"] = json!(32)),
        ("nested block", |v| {
            v["dflash_config"]["block_size"] = json!(32)
        }),
        ("hidden", |v| v["hidden_size"] = json!(2561)),
        ("intermediate", |v| v["intermediate_size"] = json!(8705)),
        ("layers", |v| v["num_hidden_layers"] = json!(7)),
        ("target layers", |v| v["num_target_layers"] = json!(47)),
        ("query heads", |v| v["num_attention_heads"] = json!(21)),
        ("kv heads", |v| v["num_key_value_heads"] = json!(5)),
        ("head dim", |v| v["head_dim"] = json!(64)),
        ("vocab", |v| v["vocab_size"] = json!(248319)),
        ("causal", |v| v["is_causal"] = json!(true)),
        ("mask", |v| {
            v["dflash_config"]["mask_token_id"] = json!(248076)
        }),
        ("taps", |v| {
            v["dflash_config"]["target_layer_ids"][2] = json!(14)
        }),
        ("fc layernorm", |v| {
            v["dflash_config"]["fc_layernorm"] = json!(true)
        }),
        ("projector", |v| {
            v["dflash_config"]["projector_type"] = json!("dflash2")
        }),
        ("conv", |v| {
            v["dflash_config"]["conv_kernel_size"] = json!(2)
        }),
        ("conv group", |v| {
            v["dflash_config"]["conv_group_size"] = json!(16)
        }),
        ("selector", |v| {
            v["dflash_config"]["selector_rank"] = json!(256)
        }),
        ("selector top k", |v| {
            v["dflash_config"]["selector_top_k"] = json!(16)
        }),
        ("selector vocabulary", |v| {
            v["dflash_config"]["selector_vocab_size"] = json!(248077)
        }),
        ("markov", |v| v["markov_rank"] = json!(256)),
        ("confidence", |v| v["enable_confidence_head"] = json!(true)),
        ("nested confidence", |v| {
            v["dflash_config"]["enable_confidence_head"] = json!(true)
        }),
        ("draft vocabulary", |v| {
            v["draft_vocab_size"] = json!(248320)
        }),
        ("tied embeddings", |v| {
            v["tie_word_embeddings"] = json!(true)
        }),
    ];
    for (name, mutate) in mutations {
        let mut config = native_config_json();
        mutate(&mut config);
        let error = match parse_dflash_config(&config.to_string()) {
            Ok(_) => panic!("native mutation {name} was accepted"),
            Err(error) => error,
        };
        assert!(!error.to_string().is_empty(), "{name}");
    }
}

#[test]
fn rejects_each_dflash2_identity_and_layout_mutation() {
    let mutations: &[(&str, fn(&mut Value))] = &[
        ("architecture", |v| {
            v["architectures"] = json!(["DFlashDraftModel"])
        }),
        ("kernel", |v| {
            v["dflash_config"]["conv_kernel_size"] = json!(3)
        }),
        ("group", |v| {
            v["dflash_config"]["conv_group_size"] = json!(32)
        }),
        ("rank", |v| v["dflash_config"]["selector_rank"] = json!(128)),
        ("top k", |v| v["dflash_config"]["selector_top_k"] = json!(8)),
        ("logical vocabulary", |v| {
            v["dflash_config"]["selector_vocab_size"] = json!(248320)
        }),
        ("missing logical vocabulary", |v| {
            v["dflash_config"]
                .as_object_mut()
                .unwrap()
                .remove("selector_vocab_size");
        }),
        ("layer order", |v| {
            v["layer_types"][4] = json!("full_attention")
        }),
        ("missing layer layout", |v| {
            v.as_object_mut().unwrap().remove("layer_types");
        }),
        ("sliding window", |v| v["sliding_window"] = json!(2048)),
        ("missing sliding window", |v| {
            v.as_object_mut().unwrap().remove("sliding_window");
        }),
        ("projector", |v| {
            v["dflash_config"]["projector_type"] = json!("dflash2")
        }),
        ("markov", |v| v["markov_rank"] = json!(256)),
        ("draft vocabulary", |v| {
            v["draft_vocab_size"] = json!(248077)
        }),
        ("unknown nested semantic", |v| {
            v["dflash_config"]["future_selector_mode"] = json!("opaque")
        }),
    ];
    for (name, mutate) in mutations {
        let mut config = dflash2_config_json();
        mutate(&mut config);
        let error = match parse_dflash_config(&config.to_string()) {
            Ok(_) => panic!("native DFlash2 mutation {name} was accepted"),
            Err(error) => error,
        };
        assert!(!error.to_string().is_empty(), "{name}");
    }
}

#[test]
fn dflash2_selector_marker_prevents_multi_axis_native_bypass() {
    let mut config = dflash2_config_json();
    config["num_target_layers"] = json!(47);
    config["hidden_size"] = json!(2561);
    config["intermediate_size"] = json!(8705);
    let error = parse_dflash_config(&config.to_string()).unwrap_err();
    assert!(!error.to_string().is_empty(), "{error:#}");
}

#[test]
fn preserves_plain_v3_unknown_nested_field_behavior() {
    let mut config = native_config_json();
    config["dflash_config"]["future_metadata"] = json!({"opaque": true});
    parse_dflash_config(&config.to_string()).expect("plain V3 remains permissive");
}

#[test]
fn rejects_dflash2_numeric_type_and_bool_coercions() {
    let mutations: &[(&str, fn(&mut Value))] = &[
        ("boolean hidden", |v| v["hidden_size"] = json!(true)),
        ("string block", |v| v["block_size"] = json!("16")),
        ("float group", |v| {
            v["dflash_config"]["conv_group_size"] = json!(16.0)
        }),
        ("boolean selector vocab", |v| {
            v["dflash_config"]["selector_vocab_size"] = json!(true)
        }),
        ("numeric causal", |v| v["is_causal"] = json!(0)),
        ("boolean tap", |v| {
            v["dflash_config"]["target_layer_ids"][0] = json!(true)
        }),
    ];
    for (name, mutate) in mutations {
        let mut config = dflash2_config_json();
        mutate(&mut config);
        let error = parse_dflash_config(&config.to_string()).unwrap_err();
        assert!(!error.to_string().is_empty(), "{name}: {error:#}");
    }
}

#[test]
fn rejects_missing_explicit_native_identity_fields() {
    for path in ["root", "nested", "architectures", "model_type", "is_causal"] {
        let mut config = native_config_json();
        match path {
            "root" => {
                config.as_object_mut().unwrap().remove("block_size");
            }
            "nested" => {
                config["dflash_config"]
                    .as_object_mut()
                    .unwrap()
                    .remove("block_size");
            }
            field => {
                config.as_object_mut().unwrap().remove(field);
            }
        }
        let error = parse_dflash_config(&config.to_string()).unwrap_err();
        assert!(!error.to_string().is_empty(), "{path}: {error:#}");
    }
}

#[test]
fn rejects_native_non_bf16_extra_missing_and_prefixed_manifests() {
    let config = native_config();

    let mut wrong_dtype = native_store("");
    wrong_dtype.0.get_mut("norm.weight").unwrap().1 = WeightDtype::FP32;
    let error = validate_dflash_metadata(&wrong_dtype, &config).unwrap_err();
    assert!(error.to_string().contains("BF16"), "{error:#}");

    let mut extra = native_store("");
    extra
        .0
        .insert("d2t".into(), (vec![248320], WeightDtype::Int64));
    let error = validate_dflash_metadata(&extra, &config).unwrap_err();
    assert!(error.to_string().contains("unexpected"), "{error:#}");

    let mut missing = native_store("");
    missing.0.remove("layers.5.mlp.down_proj.weight");
    let error = validate_dflash_metadata(&missing, &config).unwrap_err();
    assert!(error.to_string().contains("down_proj"), "{error:#}");

    let error = validate_dflash_metadata(&native_store("model."), &config).unwrap_err();
    assert!(error.to_string().contains("bare"), "{error:#}");
}

#[test]
fn rejects_dflash2_partial_extra_wrong_dtype_and_wrong_physical_vocab_manifests() {
    let config = dflash2_config();

    let mut missing = dflash2_store();
    missing
        .0
        .remove("layers.5.mlp_conv.kernel_projection.weight");
    let error = validate_dflash_metadata(&missing, &config).unwrap_err();
    assert!(error.to_string().contains("kernel_projection"), "{error:#}");

    let mut extra = dflash2_store();
    extra.0.insert(
        "candidate_selector.extra".into(),
        (vec![1], WeightDtype::BF16),
    );
    let error = validate_dflash_metadata(&extra, &config).unwrap_err();
    assert!(error.to_string().contains("unexpected"), "{error:#}");

    let mut wrong_dtype = dflash2_store();
    wrong_dtype
        .0
        .get_mut("layers.0.attention_conv.base_kernel")
        .unwrap()
        .1 = WeightDtype::FP32;
    let error = validate_dflash_metadata(&wrong_dtype, &config).unwrap_err();
    assert!(error.to_string().contains("BF16"), "{error:#}");

    let mut cropped_codebook = dflash2_store();
    cropped_codebook
        .0
        .get_mut("candidate_selector.successor_codebook")
        .unwrap()
        .0 = vec![248077, 256];
    let error = validate_dflash_metadata(&cropped_codebook, &config).unwrap_err();
    assert!(error.to_string().contains("248320"), "{error:#}");
}
