// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use atlas_core::config::{ModelConfig, QuantizationConfig};
use spark_runtime::weights::WeightDtype;

use super::{
    MtpMetadata, MtpMetadataSource, Qwen4MtpExpertLayout, classify_qwen4_mtp_metadata,
    packed_bf16_expert_offsets,
};

struct FakeStore(HashMap<String, (Vec<usize>, WeightDtype)>);

impl MtpMetadataSource for FakeStore {
    fn metadata(&self, name: &str) -> Option<MtpMetadata<'_>> {
        self.0.get(name).map(|(shape, dtype)| MtpMetadata {
            shape,
            dtype: *dtype,
        })
    }

    fn names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        Box::new(self.0.keys().map(String::as_str))
    }
}

fn exact_config() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen4_exp".into();
    config.hidden_size = 2_560;
    config.num_hidden_layers = 48;
    config.vocab_size = 248_320;
    config.num_attention_heads = 24;
    config.num_key_value_heads = 2;
    config.head_dim = 256;
    config.num_experts = 512;
    config.num_experts_per_tok = 10;
    config.moe_intermediate_size = 640;
    config.shared_expert_intermediate_size = 640;
    config.hc_count = 4;
    config.hc_lowrank = 320;
    config.mtp_num_hidden_layers = 1;
    config.indexer_n_heads = 4;
    config.indexer_kv_heads = 1;
    config.indexer_head_dim = 128;
    config.quantization_config = Some(QuantizationConfig {
        quant_method: "modelopt".into(),
        quant_algo: "NVFP4".into(),
        format: String::new(),
        ignore_modules: Vec::new(),
        config_groups: Vec::new(),
    });
    config
}

fn insert(store: &mut FakeStore, name: &str, shape: &[usize], dtype: WeightDtype) {
    assert!(
        store
            .0
            .insert(name.into(), (shape.to_vec(), dtype))
            .is_none()
    );
}

fn fixed_store(config: &ModelConfig) -> FakeStore {
    let h = config.hidden_size;
    let r = config.residual_width();
    let rank = config.hc_lowrank;
    let inter = config.moe_intermediate_size;
    let q = config.num_attention_heads * config.head_dim * 2;
    let kv = config.num_key_value_heads * config.head_dim;
    let qsa = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    let mut store = FakeStore(HashMap::new());
    for (name, shape) in [
        ("mtp.fc_embedding.weight", vec![h, h]),
        ("mtp.fc_hidden.weight", vec![h, h]),
        ("mtp.pre_fc_norm_embedding.weight", vec![h]),
        ("mtp.pre_fc_norm_hidden.weight", vec![r]),
        ("mtp.hyper_connection_mixer.hc_norm.weight", vec![r]),
        (
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            vec![rank, r],
        ),
        (
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            vec![r, rank],
        ),
        ("mtp.layers.0.mlp.gate.weight", vec![config.num_experts, h]),
        (
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            vec![inter, h],
        ),
        (
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
            vec![inter, h],
        ),
        (
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
            vec![h, inter],
        ),
        ("mtp.layers.0.mlp.shared_expert_gate.weight", vec![1, h]),
        ("mtp.layers.0.self_attn.q_proj.weight", vec![q, h]),
        ("mtp.layers.0.self_attn.k_proj.weight", vec![kv, h]),
        ("mtp.layers.0.self_attn.v_proj.weight", vec![kv, h]),
        ("mtp.layers.0.self_attn.o_proj.weight", vec![h, q / 2]),
        (
            "mtp.layers.0.self_attn.q_norm.weight",
            vec![config.head_dim],
        ),
        (
            "mtp.layers.0.self_attn.k_norm.weight",
            vec![config.head_dim],
        ),
        (
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
            vec![qsa, h],
        ),
        (
            "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
            vec![config.indexer_head_dim],
        ),
        (
            "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
            vec![config.indexer_head_dim],
        ),
    ] {
        insert(&mut store, name, &shape, WeightDtype::BF16);
    }
    for prefix in [
        "mtp.layers.0.attn_hyper_connection",
        "mtp.layers.0.mlp_hyper_connection",
    ] {
        for (suffix, shape) in [
            ("hc_norm.weight", vec![r]),
            ("input_mix_weight_down.weight", vec![rank, r]),
            ("input_mix_weight_up.weight", vec![r, rank]),
            ("block_inject_weight.weight", vec![config.hc_count, r]),
        ] {
            insert(
                &mut store,
                &format!("{prefix}.{suffix}"),
                &shape,
                WeightDtype::BF16,
            );
        }
    }
    assert_eq!(store.0.len(), 29);
    store
}

fn add_packed(store: &mut FakeStore, config: &ModelConfig) {
    insert(
        store,
        "mtp.layers.0.mlp.experts.gate_up_proj",
        &[
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        ],
        WeightDtype::BF16,
    );
    insert(
        store,
        "mtp.layers.0.mlp.experts.down_proj",
        &[
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        ],
        WeightDtype::BF16,
    );
}

fn add_numbered(store: &mut FakeStore, config: &ModelConfig) {
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    for expert in 0..config.num_experts {
        for projection in ["gate_proj", "up_proj"] {
            let p = format!("mtp.layers.0.mlp.experts.{expert}.{projection}");
            insert(
                store,
                &format!("{p}.weight"),
                &[inter, h / 2],
                WeightDtype::UInt8,
            );
            insert(
                store,
                &format!("{p}.weight_scale"),
                &[inter, h / 16],
                WeightDtype::FP8E4M3,
            );
            insert(
                store,
                &format!("{p}.weight_scale_2"),
                &[],
                WeightDtype::FP32,
            );
            insert(store, &format!("{p}.input_scale"), &[], WeightDtype::FP32);
        }
        let p = format!("mtp.layers.0.mlp.experts.{expert}.down_proj");
        insert(
            store,
            &format!("{p}.weight"),
            &[h, inter / 2],
            WeightDtype::UInt8,
        );
        insert(
            store,
            &format!("{p}.weight_scale"),
            &[h, inter / 16],
            WeightDtype::FP8E4M3,
        );
        insert(
            store,
            &format!("{p}.weight_scale_2"),
            &[],
            WeightDtype::FP32,
        );
        insert(store, &format!("{p}.input_scale"), &[], WeightDtype::FP32);
    }
}

#[test]
fn admits_exact_official_packed_bank_and_logical_vocab() {
    let mut config = exact_config();
    let mut store = fixed_store(&config);
    add_packed(&mut store, &config);
    assert_eq!(store.0.len(), 31);
    assert_eq!(
        classify_qwen4_mtp_metadata(&store, &config).unwrap(),
        Some(Qwen4MtpExpertLayout::PackedBf16)
    );
    config.vocab_size = 248_077;
    assert_eq!(
        classify_qwen4_mtp_metadata(&store, &config).unwrap(),
        Some(Qwen4MtpExpertLayout::PackedBf16)
    );
}

#[test]
fn absent_official_bank_is_inert() {
    let config = exact_config();
    assert_eq!(
        classify_qwen4_mtp_metadata(&FakeStore(HashMap::new()), &config).unwrap(),
        None
    );

    let mut unrelated = FakeStore(HashMap::new());
    insert(
        &mut unrelated,
        "model.embed_tokens.weight",
        &[248_320, 2_560],
        WeightDtype::BF16,
    );
    assert_eq!(
        classify_qwen4_mtp_metadata(&unrelated, &config).unwrap(),
        None
    );
}

#[test]
fn rejects_fixed_only_or_misspelled_expert_bank() {
    let config = exact_config();
    assert!(classify_qwen4_mtp_metadata(&fixed_store(&config), &config).is_err());

    let mut misspelled = fixed_store(&config);
    insert(
        &mut misspelled,
        "mtp.layers.0.mlp.expert.gate_up_proj",
        &[512, 1280, 2560],
        WeightDtype::BF16,
    );
    insert(
        &mut misspelled,
        "mtp.layers.0.mlp.expert.down_proj",
        &[512, 2560, 640],
        WeightDtype::BF16,
    );
    assert!(classify_qwen4_mtp_metadata(&misspelled, &config).is_err());
}

#[test]
fn complete_numbered_sidecar_wins_over_valid_packed_bank() {
    let config = exact_config();
    let mut store = fixed_store(&config);
    add_packed(&mut store, &config);
    add_numbered(&mut store, &config);
    assert_eq!(
        classify_qwen4_mtp_metadata(&store, &config).unwrap(),
        Some(Qwen4MtpExpertLayout::NumberedNvfp4)
    );
}

#[test]
fn rejects_partial_malformed_or_extra_packed_schema() {
    let config = exact_config();
    let mut missing = fixed_store(&config);
    add_packed(&mut missing, &config);
    missing.0.remove("mtp.layers.0.mlp.experts.down_proj");
    assert!(classify_qwen4_mtp_metadata(&missing, &config).is_err());

    let mut shape = fixed_store(&config);
    add_packed(&mut shape, &config);
    shape
        .0
        .get_mut("mtp.layers.0.mlp.experts.gate_up_proj")
        .unwrap()
        .0[1] -= 1;
    assert!(classify_qwen4_mtp_metadata(&shape, &config).is_err());

    let mut dtype = fixed_store(&config);
    add_packed(&mut dtype, &config);
    dtype
        .0
        .get_mut("mtp.layers.0.mlp.experts.down_proj")
        .unwrap()
        .1 = WeightDtype::FP32;
    assert!(classify_qwen4_mtp_metadata(&dtype, &config).is_err());

    let mut wrong_name = fixed_store(&config);
    insert(
        &mut wrong_name,
        "mtp.layers.0.mlp.experts.gate_up_proj.weight",
        &[512, 1280, 2560],
        WeightDtype::BF16,
    );
    assert!(classify_qwen4_mtp_metadata(&wrong_name, &config).is_err());

    let mut extra = fixed_store(&config);
    add_packed(&mut extra, &config);
    insert(&mut extra, "mtp.unexpected", &[1], WeightDtype::BF16);
    assert!(classify_qwen4_mtp_metadata(&extra, &config).is_err());

    let mut fixed = fixed_store(&config);
    add_packed(&mut fixed, &config);
    fixed.0.get_mut("mtp.fc_hidden.weight").unwrap().1 = WeightDtype::FP32;
    assert!(classify_qwen4_mtp_metadata(&fixed, &config).is_err());
}

#[test]
fn rejects_wrong_model_quant_or_incomplete_numbered_mix() {
    let config = exact_config();
    let mut store = fixed_store(&config);
    add_packed(&mut store, &config);

    let mut wrong_model = config.clone();
    wrong_model.model_type = "qwen3_next".into();
    assert!(classify_qwen4_mtp_metadata(&store, &wrong_model).is_err());
    let mut wrong_quant = config.clone();
    wrong_quant.quantization_config.as_mut().unwrap().quant_algo = "FP8".into();
    assert!(classify_qwen4_mtp_metadata(&store, &wrong_quant).is_err());
    let mut wrong_geometry = config.clone();
    wrong_geometry.mtp_num_hidden_layers = 2;
    assert!(classify_qwen4_mtp_metadata(&store, &wrong_geometry).is_err());
    let mut wrong_vocab = config.clone();
    wrong_vocab.vocab_size = 248_319;
    assert!(classify_qwen4_mtp_metadata(&store, &wrong_vocab).is_err());

    insert(
        &mut store,
        "mtp.layers.0.mlp.experts.0.gate_proj.weight",
        &[640, 1280],
        WeightDtype::UInt8,
    );
    assert!(classify_qwen4_mtp_metadata(&store, &config).is_err());
}

#[test]
fn checked_packed_offsets_cover_exact_bank_and_reject_overflow() {
    let first = packed_bf16_expert_offsets(0, 512, 640, 2560).unwrap();
    assert_eq!((first.gate, first.up, first.down), (0, 3_276_800, 0));
    let last = packed_bf16_expert_offsets(511, 512, 640, 2560).unwrap();
    assert_eq!(last.gate, 3_348_889_600);
    assert_eq!(last.up, 3_352_166_400);
    assert_eq!(last.down, 1_674_444_800);
    assert!(packed_bf16_expert_offsets(512, 512, 640, 2560).is_err());
    assert!(packed_bf16_expert_offsets(0, 1, usize::MAX, 2).is_err());
}
