// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use spark_runtime::weights::WeightDtype;

use super::{TensorMetadata, TensorMetadataSource, validate_dflash_metadata};
use crate::weight_loader::dflash_loader::{DflashConfig, DflashSubConfig};

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

fn config() -> DflashConfig {
    DflashConfig {
        architectures: vec!["DFlashDraftModel".into()],
        hidden_size: 8,
        num_hidden_layers: 2,
        intermediate_size: 16,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 32,
        draft_vocab_size: None,
        tie_word_embeddings: false,
        block_size: 4,
        dflash_config: Some(DflashSubConfig {
            mask_token_id: 31,
            target_layer_ids: vec![0, 1, 2],
            projector_type: None,
            enable_confidence_head: None,
            confidence_head_with_markov: None,
            fc_layernorm: false,
            block_size: None,
            conv_kernel_size: 0,
            conv_group_size: 0,
            selector_rank: 0,
            selector_top_k: 0,
        }),
        layer_types: None,
        sliding_window: None,
        // `None` = field absent, which keeps the historical
        // `layer_types`-derived causality. Matches the neutral
        // layer_types/sliding_window above; this fixture exercises metadata
        // validation, not the causality derivation.
        is_causal: None,
        rope_theta: 10_000.0,
        rope_scaling: None,
        markov_rank: 0,
        markov_head_type: "vanilla".into(),
        enable_confidence_head: None,
        confidence_head_with_markov: None,
    }
}

fn valid_store(prefix: &str) -> FakeStore {
    let mut tensors = HashMap::new();
    let mut add = |name: String, shape: &[usize]| {
        tensors.insert(name, (shape.to_vec(), WeightDtype::BF16));
    };
    add(format!("{prefix}fc.weight"), &[8, 24]);
    add(format!("{prefix}hidden_norm.weight"), &[8]);
    add(format!("{prefix}norm.weight"), &[8]);
    for layer in 0..2 {
        let lp = format!("{prefix}layers.{layer}");
        add(format!("{lp}.input_layernorm.weight"), &[8]);
        add(format!("{lp}.post_attention_layernorm.weight"), &[8]);
        add(format!("{lp}.self_attn.q_proj.weight"), &[8, 8]);
        add(format!("{lp}.self_attn.k_proj.weight"), &[4, 8]);
        add(format!("{lp}.self_attn.v_proj.weight"), &[4, 8]);
        add(format!("{lp}.self_attn.o_proj.weight"), &[8, 8]);
        add(format!("{lp}.self_attn.q_norm.weight"), &[4]);
        add(format!("{lp}.self_attn.k_norm.weight"), &[4]);
        add(format!("{lp}.mlp.gate_proj.weight"), &[16, 8]);
        add(format!("{lp}.mlp.up_proj.weight"), &[16, 8]);
        add(format!("{lp}.mlp.down_proj.weight"), &[8, 16]);
    }
    FakeStore(tensors)
}

#[test]
fn accepts_complete_bare_and_model_prefixed_schemas() {
    assert_eq!(
        validate_dflash_metadata(&valid_store(""), &config()).unwrap(),
        Some("")
    );
    assert_eq!(
        validate_dflash_metadata(&valid_store("model."), &config()).unwrap(),
        Some("model.")
    );
}

#[test]
fn preserves_none_for_an_unrelated_model_store() {
    let unrelated = FakeStore(HashMap::from([
        (
            "model.embed_tokens.weight".into(),
            (vec![32, 8], WeightDtype::BF16),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight".into(),
            (vec![8, 8], WeightDtype::BF16),
        ),
    ]));
    assert_eq!(
        validate_dflash_metadata(&unrelated, &config()).unwrap(),
        None
    );
}

#[test]
fn rejects_partial_model_prefixed_schema_missing_fc_and_roots() {
    let partial = FakeStore(HashMap::from([(
        "model.layers.0.self_attn.q_proj.weight".into(),
        (vec![8, 8], WeightDtype::BF16),
    )]));
    let error = validate_dflash_metadata(&partial, &config()).unwrap_err();
    assert!(error.to_string().contains("fc.weight"), "{error:#}");
}

#[test]
fn rejects_partial_dflash_schema_missing_fc() {
    let mut store = valid_store("");
    store.0.remove("fc.weight");
    let error = validate_dflash_metadata(&store, &config()).unwrap_err();
    assert!(error.to_string().contains("fc.weight"), "{error:#}");
}

#[test]
fn rejects_each_missing_required_tensor() {
    let names: Vec<_> = valid_store("").0.into_keys().collect();
    for name in names {
        let mut store = valid_store("");
        store.0.remove(&name);
        let error = validate_dflash_metadata(&store, &config()).unwrap_err();
        assert!(error.to_string().contains(&name), "{name}: {error:#}");
    }
}

#[test]
fn rejects_checkpoint_layer_count_mismatch() {
    let mut store = valid_store("");
    store.0.insert(
        "layers.2.input_layernorm.weight".into(),
        (vec![8], WeightDtype::BF16),
    );
    let error = validate_dflash_metadata(&store, &config()).unwrap_err();
    assert!(error.to_string().contains("layer indices"), "{error:#}");
}

#[test]
fn rejects_config_derived_shape_mismatches() {
    for (name, wrong_shape) in [
        ("fc.weight", vec![8, 8]),
        ("layers.0.self_attn.k_proj.weight", vec![8, 8]),
        ("layers.1.mlp.down_proj.weight", vec![16, 8]),
    ] {
        let mut store = valid_store("");
        store.0.get_mut(name).unwrap().0 = wrong_shape;
        let error = validate_dflash_metadata(&store, &config()).unwrap_err();
        assert!(error.to_string().contains(name), "{name}: {error:#}");
        assert!(error.to_string().contains("shape"), "{name}: {error:#}");
    }
}

#[test]
fn rejects_non_bf16_required_tensor() {
    let mut store = valid_store("");
    store
        .0
        .get_mut("layers.0.self_attn.q_proj.weight")
        .unwrap()
        .1 = WeightDtype::FP32;
    let error = validate_dflash_metadata(&store, &config()).unwrap_err();
    assert!(error.to_string().contains("BF16"), "{error:#}");
}

#[test]
fn rejects_fc_layernorm_even_when_config_bypasses_json_parser() {
    let mut config = config();
    config.dflash_config.as_mut().unwrap().fc_layernorm = true;
    let error = validate_dflash_metadata(&valid_store(""), &config).unwrap_err();
    assert!(error.to_string().contains("fc_layernorm"), "{error:#}");
}

#[test]
fn validates_configured_markov_tensor_contract() {
    let mut config = config();
    config.markov_rank = 3;
    let mut store = valid_store("");
    let missing = validate_dflash_metadata(&store, &config).unwrap_err();
    assert!(missing.to_string().contains("markov_w1"), "{missing:#}");

    store.0.insert(
        "markov_head.markov_w1.weight".into(),
        (vec![32, 3], WeightDtype::BF16),
    );
    store.0.insert(
        "markov_head.markov_w2.weight".into(),
        (vec![32, 2], WeightDtype::BF16),
    );
    let wrong_shape = validate_dflash_metadata(&store, &config).unwrap_err();
    assert!(
        wrong_shape.to_string().contains("markov_w2"),
        "{wrong_shape:#}"
    );
}

#[test]
fn rejects_confidence_tensors_even_when_config_claims_the_head_is_disabled() {
    let mut store = valid_store("");
    store.0.insert(
        "confidence_head.proj.weight".into(),
        (vec![1, 11], WeightDtype::BF16),
    );
    store.0.insert(
        "confidence_head.proj.bias".into(),
        (vec![1], WeightDtype::BF16),
    );
    let error = validate_dflash_metadata(&store, &config()).unwrap_err();
    assert!(error.to_string().contains("confidence"), "{error:#}");
    assert!(error.to_string().contains("silently discard"), "{error:#}");
}

fn dspark_confidence_config() -> DflashConfig {
    let mut config = config();
    config.architectures = vec!["DSparkDraftModel".into()];
    config.markov_rank = 3;
    config.enable_confidence_head = Some(true);
    config.confidence_head_with_markov = Some(true);
    let sub = config.dflash_config.as_mut().unwrap();
    sub.projector_type = Some("dspark".into());
    sub.enable_confidence_head = Some(true);
    sub.confidence_head_with_markov = Some(true);
    config
}

fn dspark_confidence_store(prefix: &str) -> FakeStore {
    let mut store = valid_store(prefix);
    for name in ["markov_w1", "markov_w2"] {
        store.0.insert(
            format!("{prefix}markov_head.{name}.weight"),
            (vec![32, 3], WeightDtype::BF16),
        );
    }
    store.0.insert(
        format!("{prefix}confidence_head.proj.weight"),
        (vec![1, 11], WeightDtype::BF16),
    );
    store.0.insert(
        format!("{prefix}confidence_head.proj.bias"),
        (vec![1], WeightDtype::BF16),
    );
    store
}

#[test]
fn accepts_exact_dspark_confidence_tensor_contract_for_both_prefixes() {
    let config = dspark_confidence_config();
    assert_eq!(
        validate_dflash_metadata(&dspark_confidence_store(""), &config).unwrap(),
        Some("")
    );
    assert_eq!(
        validate_dflash_metadata(&dspark_confidence_store("model."), &config).unwrap(),
        Some("model.")
    );
}

#[test]
fn rejects_missing_or_malformed_dspark_confidence_tensors() {
    let config = dspark_confidence_config();

    let mut missing_bias = dspark_confidence_store("");
    missing_bias.0.remove("confidence_head.proj.bias");
    let error = validate_dflash_metadata(&missing_bias, &config).unwrap_err();
    assert!(error.to_string().contains("proj.bias"), "{error:#}");

    let mut wrong_width = dspark_confidence_store("");
    wrong_width
        .0
        .get_mut("confidence_head.proj.weight")
        .unwrap()
        .0 = vec![1, 8];
    let error = validate_dflash_metadata(&wrong_width, &config).unwrap_err();
    assert!(error.to_string().contains("[1, 11]"), "{error:#}");

    let mut wrong_dtype = dspark_confidence_store("");
    wrong_dtype
        .0
        .get_mut("confidence_head.proj.weight")
        .unwrap()
        .1 = WeightDtype::FP32;
    let error = validate_dflash_metadata(&wrong_dtype, &config).unwrap_err();
    assert!(error.to_string().contains("BF16"), "{error:#}");
}
