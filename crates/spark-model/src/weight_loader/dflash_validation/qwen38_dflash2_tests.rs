// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use spark_runtime::weights::WeightDtype;

use super::{TensorMetadata, TensorMetadataSource, validate_dflash_metadata};
use crate::weight_loader::dflash_loader::{DflashConfig, DflashRopeScaling, DflashSubConfig};

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

fn official_config() -> DflashConfig {
    DflashConfig {
        architectures: vec!["DFlash2DraftModel".into()],
        hidden_size: 5120,
        num_hidden_layers: 5,
        intermediate_size: 17408,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: 128,
        vocab_size: 248320,
        draft_vocab_size: None,
        tie_word_embeddings: false,
        block_size: 16,
        dflash_config: Some(DflashSubConfig {
            block_size: Some(8),
            mask_token_id: 248070,
            target_layer_ids: vec![5, 19, 33, 47, 61],
            projector_type: None,
            enable_confidence_head: None,
            confidence_head_with_markov: None,
            fc_layernorm: false,
            conv_kernel_size: 2,
            conv_group_size: 16,
            selector_rank: 256,
            selector_top_k: 16,
            // Both added by theirs' exact-native-DFlash2 admission and absent
            // from HEAD's literal, so the merge produced a struct the merged
            // definition rejects. `None` and an empty map are the permissive
            // values this fixture wants: it is a generic checkpoint, and an
            // empty `unknown_fields` is exactly what exact admission requires.
            selector_vocab_size: None,
            unknown_fields: Default::default(),
        }),
        layer_types: Some(vec!["sliding_attention".into(); 5]),
        sliding_window: Some(2048),
        is_causal: Some(false),
        rms_norm_eps: Some(1e-6),
        hidden_act: Some("silu".into()),
        rope_theta: 10_000_000.0,
        rope_scaling: Some(DflashRopeScaling {
            rope_type: Some("default".into()),
            factor: None,
            beta_fast: None,
            beta_slow: None,
            original_max_position_embeddings: None,
            rope_theta: Some(10_000_000.0),
            attention_factor: None,
        }),
        markov_rank: 0,
        markov_head_type: "vanilla".into(),
        enable_confidence_head: None,
        confidence_head_with_markov: None,
    }
}

fn official_store(prefix: &str) -> FakeStore {
    let mut tensors = HashMap::new();
    let mut add = |name: String, shape: &[usize]| {
        assert!(
            tensors
                .insert(name, (shape.to_vec(), WeightDtype::BF16))
                .is_none()
        );
    };
    add(format!("{prefix}fc.weight"), &[5120, 25600]);
    add(format!("{prefix}hidden_norm.weight"), &[5120]);
    add(format!("{prefix}norm.weight"), &[5120]);
    for layer in 0..5 {
        let lp = format!("{prefix}layers.{layer}");
        add(format!("{lp}.input_layernorm.weight"), &[5120]);
        add(format!("{lp}.post_attention_layernorm.weight"), &[5120]);
        add(format!("{lp}.self_attn.q_proj.weight"), &[4096, 5120]);
        add(format!("{lp}.self_attn.k_proj.weight"), &[1024, 5120]);
        add(format!("{lp}.self_attn.v_proj.weight"), &[1024, 5120]);
        add(format!("{lp}.self_attn.o_proj.weight"), &[5120, 4096]);
        add(format!("{lp}.self_attn.q_norm.weight"), &[128]);
        add(format!("{lp}.self_attn.k_norm.weight"), &[128]);
        add(format!("{lp}.mlp.gate_proj.weight"), &[17408, 5120]);
        add(format!("{lp}.mlp.up_proj.weight"), &[17408, 5120]);
        add(format!("{lp}.mlp.down_proj.weight"), &[5120, 17408]);
        for stem in ["attention_conv", "mlp_conv"] {
            add(format!("{lp}.{stem}.base_kernel"), &[2, 2, 5120]);
            add(
                format!("{lp}.{stem}.kernel_projection.weight"),
                &[1280, 5120],
            );
        }
    }
    add(
        format!("{prefix}candidate_selector.hidden_projection.weight"),
        &[256, 5120],
    );
    add(
        format!("{prefix}candidate_selector.predecessor_codebook"),
        &[248320, 256],
    );
    add(
        format!("{prefix}candidate_selector.successor_codebook"),
        &[248320, 256],
    );
    assert_eq!(tensors.len(), 81);
    FakeStore(tensors)
}

fn legacy_config() -> DflashConfig {
    let mut config = official_config();
    config.architectures = vec!["DFlashDraftModel".into()];
    config.hidden_size = 8;
    config.num_hidden_layers = 1;
    config.intermediate_size = 16;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.head_dim = 4;
    config.vocab_size = 32;
    config.block_size = 4;
    config.layer_types = None;
    config.sliding_window = None;
    config.is_causal = None;
    let sub = config.dflash_config.as_mut().unwrap();
    sub.block_size = None;
    sub.mask_token_id = 31;
    sub.target_layer_ids = vec![0, 1, 2];
    sub.conv_kernel_size = 0;
    sub.conv_group_size = 0;
    sub.selector_rank = 0;
    sub.selector_top_k = 0;
    config
}

fn legacy_store(prefix: &str) -> FakeStore {
    let mut tensors = HashMap::new();
    let mut add = |name: &str, shape: &[usize]| {
        tensors.insert(
            format!("{prefix}{name}"),
            (shape.to_vec(), WeightDtype::BF16),
        );
    };
    add("fc.weight", &[8, 24]);
    add("hidden_norm.weight", &[8]);
    add("norm.weight", &[8]);
    add("layers.0.input_layernorm.weight", &[8]);
    add("layers.0.post_attention_layernorm.weight", &[8]);
    add("layers.0.self_attn.q_proj.weight", &[8, 8]);
    add("layers.0.self_attn.k_proj.weight", &[4, 8]);
    add("layers.0.self_attn.v_proj.weight", &[4, 8]);
    add("layers.0.self_attn.o_proj.weight", &[8, 8]);
    add("layers.0.self_attn.q_norm.weight", &[4]);
    add("layers.0.self_attn.k_norm.weight", &[4]);
    add("layers.0.mlp.gate_proj.weight", &[16, 8]);
    add("layers.0.mlp.up_proj.weight", &[16, 8]);
    add("layers.0.mlp.down_proj.weight", &[8, 16]);
    FakeStore(tensors)
}

#[test]
fn accepts_only_the_exact_81_tensor_official_profile_for_both_prefixes() {
    for prefix in ["", "model."] {
        let store = official_store(prefix);
        assert_eq!(store.0.len(), 81);
        assert_eq!(
            validate_dflash_metadata(&store, &official_config()).unwrap(),
            Some(prefix)
        );
    }
}

#[test]
fn every_official_tensor_name_shape_and_dtype_is_load_bearing() {
    let names = official_store("").0.into_keys().collect::<Vec<_>>();
    assert_eq!(names.len(), 81);
    for name in names {
        let mut missing = official_store("");
        missing.0.remove(&name);
        assert!(validate_dflash_metadata(&missing, &official_config()).is_err());

        let mut wrong_shape = official_store("");
        wrong_shape.0.get_mut(&name).unwrap().0.push(1);
        assert!(validate_dflash_metadata(&wrong_shape, &official_config()).is_err());

        let mut wrong_dtype = official_store("");
        wrong_dtype.0.get_mut(&name).unwrap().1 = WeightDtype::FP32;
        assert!(validate_dflash_metadata(&wrong_dtype, &official_config()).is_err());
    }
}

#[test]
fn rejects_extra_tensor_and_partial_dflash2_declarations() {
    let mut extra = official_store("");
    extra
        .0
        .insert("ignored.weight".into(), (vec![1], WeightDtype::BF16));
    assert!(validate_dflash_metadata(&extra, &official_config()).is_err());

    let mut missing_markers = official_config();
    let sub = missing_markers.dflash_config.as_mut().unwrap();
    sub.conv_kernel_size = 0;
    sub.conv_group_size = 0;
    sub.selector_rank = 0;
    sub.selector_top_k = 0;
    assert!(validate_dflash_metadata(&official_store(""), &missing_markers).is_err());

    for prefix in ["", "model."] {
        let mut disguised = legacy_config();
        disguised.dflash_config.as_mut().unwrap().conv_kernel_size = 2;
        assert!(validate_dflash_metadata(&legacy_store(prefix), &disguised).is_err());

        let mut group_only = legacy_config();
        group_only.dflash_config.as_mut().unwrap().conv_group_size = 16;
        assert!(validate_dflash_metadata(&legacy_store(prefix), &group_only).is_err());

        let mut tensor_only = official_config();
        tensor_only.architectures = vec!["DFlashDraftModel".into()];
        let sub = tensor_only.dflash_config.as_mut().unwrap();
        sub.conv_kernel_size = 0;
        sub.conv_group_size = 0;
        sub.selector_rank = 0;
        sub.selector_top_k = 0;
        assert!(validate_dflash_metadata(&official_store(prefix), &tensor_only).is_err());

        let mut conv_tensor_only = legacy_store(prefix);
        conv_tensor_only.0.insert(
            format!("{prefix}layers.0.attention_conv.base_kernel"),
            (vec![2, 2, 8], WeightDtype::BF16),
        );
        assert!(validate_dflash_metadata(&conv_tensor_only, &legacy_config()).is_err());

        let mut selector_tensor_only = legacy_store(prefix);
        selector_tensor_only.0.insert(
            format!("{prefix}candidate_selector.predecessor_codebook"),
            (vec![32, 1], WeightDtype::BF16),
        );
        assert!(validate_dflash_metadata(&selector_tensor_only, &legacy_config()).is_err());
    }
}

#[test]
fn rejects_every_runtime_critical_official_config_drift() {
    let mutations: &[(&str, fn(&mut DflashConfig))] = &[
        ("architecture", |c| c.architectures.push("Other".into())),
        ("hidden", |c| c.hidden_size = 5119),
        ("intermediate", |c| c.intermediate_size = 17407),
        ("layers", |c| c.num_hidden_layers = 4),
        ("query_heads", |c| c.num_attention_heads = 31),
        ("kv_heads", |c| c.num_key_value_heads = 7),
        ("head_dim", |c| c.head_dim = 64),
        ("vocab", |c| c.vocab_size = 248319),
        ("draft_vocab", |c| c.draft_vocab_size = Some(248319)),
        ("block", |c| {
            c.dflash_config.as_mut().unwrap().block_size = Some(7)
        }),
        ("mask", |c| {
            c.dflash_config.as_mut().unwrap().mask_token_id = 248071
        }),
        ("tap_order", |c| {
            c.dflash_config
                .as_mut()
                .unwrap()
                .target_layer_ids
                .swap(0, 1)
        }),
        ("tap_duplicate", |c| {
            c.dflash_config.as_mut().unwrap().target_layer_ids[1] = 5
        }),
        ("tap_out_of_range", |c| {
            c.dflash_config.as_mut().unwrap().target_layer_ids[4] = 64
        }),
        ("kernel", |c| {
            c.dflash_config.as_mut().unwrap().conv_kernel_size = 3
        }),
        ("group", |c| {
            c.dflash_config.as_mut().unwrap().conv_group_size = 8
        }),
        ("rank", |c| {
            c.dflash_config.as_mut().unwrap().selector_rank = 128
        }),
        ("top_k", |c| {
            c.dflash_config.as_mut().unwrap().selector_top_k = 8
        }),
        ("causal", |c| c.is_causal = Some(true)),
        ("causal_absent", |c| c.is_causal = None),
        ("window", |c| c.sliding_window = Some(4096)),
        ("window_absent", |c| c.sliding_window = None),
        ("layer_mode", |c| {
            c.layer_types.as_mut().unwrap()[2] = "full_attention".into()
        }),
        ("layer_types_absent", |c| c.layer_types = None),
        ("markov", |c| c.markov_rank = 1),
        ("rope_top_theta", |c| c.rope_theta = 10_000.0),
        ("rope_absent", |c| c.rope_scaling = None),
        ("rope_type", |c| {
            c.rope_scaling.as_mut().unwrap().rope_type = Some("yarn".into())
        }),
        ("rope_nested_theta", |c| {
            c.rope_scaling.as_mut().unwrap().rope_theta = Some(10_000.0)
        }),
        ("rope_factor", |c| {
            c.rope_scaling.as_mut().unwrap().factor = Some(2.0)
        }),
        ("rms_norm_eps", |c| c.rms_norm_eps = Some(2e-6)),
        ("rms_norm_eps_absent", |c| c.rms_norm_eps = None),
        ("hidden_act", |c| c.hidden_act = Some("gelu".into())),
        ("hidden_act_absent", |c| c.hidden_act = None),
    ];
    assert_eq!(mutations.len(), 34);
    for (name, mutate) in mutations {
        let mut config = official_config();
        mutate(&mut config);
        let error = validate_dflash_metadata(&official_store(""), &config).unwrap_err();
        assert!(
            !error.to_string().is_empty(),
            "{name} drift was not explained"
        );
    }
}

#[test]
fn preserves_legacy_v2_v3_base_schema_admission() {
    assert_eq!(
        validate_dflash_metadata(&legacy_store(""), &legacy_config()).unwrap(),
        Some("")
    );
}
