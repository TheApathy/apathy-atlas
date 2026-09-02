// SPDX-License-Identifier: AGPL-3.0-only

use crate::config::{Glm5NextIndexerType, Glm5NextMlpType, LayerType, parse_config};

const GLM53_CONFIG: &str = r#"{
  "model_type": "glm5_next",
  "architectures": ["Glm5NextForConditionalGeneration"],
  "text_config": {
    "model_type": "glm5_next_text",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 45,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "head_dim": 0,
    "qk_head_dim": 256,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "linear_attn_config": {
      "num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
      "kda_layers": [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44],
      "full_attn_layers": [3,7,11,15,19,23,27,31,35,39,43]
    },
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention",
      "linear_attention"
    ],
    "mlp_layer_types": [
      "dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
      "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
      "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
      "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
      "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"
    ],
    "indexer_types": ["full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full"],
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "first_k_dense_replace": 3,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "scoring_func": "sigmoid",
    "norm_topk_prob": true,
    "routed_scaling_factor": 2.5,
    "topk_method": "noaux_tc",
    "hidden_act": "silu",
    "moe_router_dtype": "float32",
    "attention_bias": false,
    "hc_mult": 4,
    "hc_eps": 0.000001,
    "hc_sinkhorn_iters": 20,
    "mhc": true,
    "mla_use_nope": true,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 2048,
    "index_kpool": 4,
    "index_kpool_always_select_tail": true,
    "index_kpool_compress": true,
    "index_share_for_mtp_iteration": true,
    "indexer_rope_interleave": true,
    "swiglu_limit": 10.0,
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 0.00001,
    "vocab_size": 154880,
    "eos_token_id": [154820, 154827, 154829],
    "pad_token_id": 154820,
    "tie_word_embeddings": false
  },
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": [
      "model.layers.0.self_attn.q_proj", "model.embed_tokens", "lm_head"
    ]
  }
}"#;

#[test]
fn parses_minimal_official_glm53_flash_contract() {
    let cfg = parse_config(GLM53_CONFIG).unwrap();
    assert_eq!(cfg.model_type, "glm5_next");
    assert_eq!(cfg.weight_prefix, "model.language_model");
    assert_eq!(cfg.num_hidden_layers, 45);
    assert_eq!(cfg.num_attention_layers(), 11);
    assert_eq!(cfg.num_ssm_layers(), 34);
    assert_eq!(cfg.layer_type(3), LayerType::FullAttention);
    assert_eq!(cfg.num_experts, 288);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.shared_expert_intermediate_size, 2048);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.rotary_dim(), 0);
    assert_eq!(cfg.num_mtp_modules, 1);
    assert_eq!(cfg.eos_token_id, 154820);
    assert_eq!(cfg.stop_token_ids(), [154820, 154827, 154829]);
    assert!(
        cfg.vision.is_none(),
        "initial GLM target is explicitly text-only"
    );
    let quant = cfg.quantization_config.as_ref().unwrap();
    assert_eq!(quant.quant_method, "fp8");
    assert_eq!(quant.format, "e4m3");
    assert!(
        quant
            .ignore_modules
            .contains(&"model.language_model.layers.0.self_attn.q_proj".to_string())
    );
    assert!(
        quant
            .ignore_modules
            .contains(&"model.language_model.embed_tokens".to_string())
    );

    let glm = cfg.glm5_next.unwrap();
    assert_eq!(glm.mlp_layer_types[0], Glm5NextMlpType::Dense);
    assert_eq!(glm.mlp_layer_types[3], Glm5NextMlpType::Sparse);
    assert_eq!(glm.indexer_types[43], Glm5NextIndexerType::Full);
    assert_eq!(glm.eos_token_ids, [154820, 154827, 154829]);
    assert_eq!(glm.index_topk, 2048);
    assert_eq!(glm.index_kpool, 4);
    assert_eq!(glm.linear_lower_bound, -5.0);
}

#[test]
fn rejects_incompatible_kpool_geometry() {
    let json = GLM53_CONFIG.replace("\"index_topk\": 2048", "\"index_topk\": 2047");
    let err = parse_config(&json).unwrap_err().to_string();
    assert!(err.contains("index_topk"), "{err}");
}

#[test]
fn rejects_load_bearing_abi_mutations() {
    for (from, to) in [
        ("\"n_group\": 1", "\"n_group\": 2"),
        ("\"topk_group\": 1", "\"topk_group\": 2"),
        ("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\""),
        ("\"attention_bias\": false", "\"attention_bias\": true"),
        (
            "\"moe_router_dtype\": \"float32\"",
            "\"moe_router_dtype\": \"bfloat16\"",
        ),
        ("\"qk_head_dim\": 256", "\"qk_head_dim\": 128"),
        (
            "\"first_k_dense_replace\": 3",
            "\"first_k_dense_replace\": 2",
        ),
        (
            "\"full_attn_layers\": [3,7,11,15,19,23,27,31,35,39,43]",
            "\"full_attn_layers\": [3,7,11,15,19,23,27,31,35,39,42]",
        ),
    ] {
        let mutated = GLM53_CONFIG.replace(from, to);
        assert_ne!(mutated, GLM53_CONFIG, "mutation pattern must exist: {from}");
        let err = parse_config(&mutated).unwrap_err();
        assert!(format!("{err:#}").contains("unsupported glm5_next geometry"));
    }
}

#[test]
fn rejects_malformed_or_incompatible_fp8_metadata() {
    for (from, to, needle) in [
        ("\"fmt\": \"e4m3\"", "\"fmt\": \"e5m2\"", "fmt"),
        (
            "\"activation_scheme\": \"dynamic\"",
            "\"activation_scheme\": \"static\"",
            "activation_scheme",
        ),
        (
            "\"weight_block_size\": [128, 128]",
            "\"weight_block_size\": [64, 128]",
            "weight_block_size",
        ),
        (
            "\"model.layers.0.self_attn.q_proj\", \"model.embed_tokens\"",
            "17, \"model.embed_tokens\"",
            "modules_to_not_convert",
        ),
    ] {
        let mutated = GLM53_CONFIG.replace(from, to);
        assert_ne!(mutated, GLM53_CONFIG, "mutation pattern must exist: {from}");
        let err = parse_config(&mutated).unwrap_err();
        assert!(format!("{err:#}").contains(needle), "{err:#}");
    }

    let malformed = GLM53_CONFIG.replace(
        "\"quantization_config\": {",
        "\"quantization_config_unused\": {",
    );
    assert!(
        parse_config(&malformed).is_ok(),
        "absence is the explicit BF16 case"
    );
    let malformed = GLM53_CONFIG.replace(
        "\"quantization_config\": {",
        "\"quantization_config\": null, \"unused\": {",
    );
    let err = parse_config(&malformed).unwrap_err();
    assert!(format!("{err:#}").contains("must be an object"));
}
