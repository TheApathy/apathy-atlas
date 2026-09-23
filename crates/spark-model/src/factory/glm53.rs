// SPDX-License-Identifier: AGPL-3.0-only

//! Zero-copy preparation and feature admission for the exact GLM-5.3 target.

#![allow(dead_code)] // Runtime construction is staged behind the typed server store seam.

use anyhow::{Context, Result, anyhow, bail, ensure};
use atlas_core::config::{
    Glm5NextIndexerType, Glm5NextMlpType, LayerType, ModelConfig, VisionConfig,
};
use spark_runtime::weights::gguf::GgufDeviceStore;

use super::{Glm53QuantProfile, ModelSourceKind};
use crate::weight_loader::{
    Glm53Dflash2Weights, Glm53GgufCatalog, Glm53GgufF32, Glm53GgufMatrix, Glm53NextnWeights,
    Glm53TargetLayerWeights,
};

const TARGET_LAYERS: usize = 45;
const EXACT_TENSORS: usize = 1412;

/// All exact target views, with the retained store borrowed for their lifetime.
pub struct PreparedGlm53Target<'a> {
    profile: Glm53QuantProfile,
    store: &'a GgufDeviceStore,
    token_embedding: Glm53GgufMatrix,
    output: Glm53GgufMatrix,
    output_norm: Glm53GgufF32,
    target_layers: [Glm53TargetLayerWeights; TARGET_LAYERS],
    native_nextn: Glm53NextnWeights,
}

impl<'a> PreparedGlm53Target<'a> {
    pub fn new(
        profile: Glm53QuantProfile,
        config: &ModelConfig,
        store: &'a GgufDeviceStore,
    ) -> Result<Self> {
        validate_official_config(config)?;
        validate_store_shape(profile, store.len(), store.total_bytes())?;

        let catalog = Glm53GgufCatalog::new(store)?;
        let target_layers = (0..TARGET_LAYERS)
            .map(|index| catalog.target_layer(index as u32))
            .collect::<Result<Vec<_>>>()?;
        let target_layers: [Glm53TargetLayerWeights; TARGET_LAYERS] =
            target_layers.try_into().map_err(|layers: Vec<_>| {
                anyhow!("expected 45 GLM target layers, got {}", layers.len())
            })?;
        for (index, layer) in target_layers.iter().enumerate() {
            ensure!(
                layer.descriptor.index == index as u32 && !layer.descriptor.is_nextn,
                "GLM target layer ordering drift at index {index}"
            );
        }

        Ok(Self {
            profile,
            store,
            token_embedding: catalog.token_embedding()?,
            output: catalog.output()?,
            output_norm: catalog.output_norm()?,
            target_layers,
            native_nextn: catalog.nextn()?,
        })
    }

    pub fn profile(&self) -> Glm53QuantProfile {
        self.profile
    }

    pub(crate) fn store(&self) -> &'a GgufDeviceStore {
        self.store
    }

    pub fn token_embedding(&self) -> &Glm53GgufMatrix {
        &self.token_embedding
    }

    pub fn output(&self) -> &Glm53GgufMatrix {
        &self.output
    }

    pub fn output_norm(&self) -> &Glm53GgufF32 {
        &self.output_norm
    }

    pub fn target_layers(&self) -> &[Glm53TargetLayerWeights] {
        &self.target_layers
    }

    /// Returns typed dormant layer-45 weights; this does not enable NextN.
    pub fn native_nextn_weights(&self) -> &Glm53NextnWeights {
        &self.native_nextn
    }

    /// Consume the temporary store borrow and move its validated views into
    /// the runtime that will own the store. No pointer-bearing view is copied.
    pub(crate) fn into_runtime_parts(
        self,
    ) -> (
        Glm53QuantProfile,
        Glm53GgufMatrix,
        Glm53GgufMatrix,
        Glm53GgufF32,
        [Glm53TargetLayerWeights; TARGET_LAYERS],
        Glm53NextnWeights,
    ) {
        let Self {
            profile,
            store: _,
            token_embedding,
            output,
            output_norm,
            target_layers,
            native_nextn,
        } = self;
        (
            profile,
            token_embedding,
            output,
            output_norm,
            target_layers,
            native_nextn,
        )
    }
}

/// Caller intent at the final boundary before constructing a GLM runtime.
pub struct Glm53FeatureRequest<'a> {
    pub profile: Glm53QuantProfile,
    pub source_kind: ModelSourceKind,
    pub tp_world_size: usize,
    pub ep_world_size: usize,
    pub lora_requested: bool,
    pub high_speed_swap_requested: bool,
    pub prefix_cache_requested: bool,
    pub legacy_mtp_requested: bool,
    pub enable_dflash2: bool,
    pub dflash2: Option<&'a Glm53Dflash2Weights>,
    pub enable_native_nextn: bool,
    pub native_nextn: Option<&'a Glm53NextnWeights>,
}

/// Feature set that passed the exact GLM source/topology boundary.
pub struct AdmittedGlm53Features<'a> {
    profile: Glm53QuantProfile,
    dflash2: Option<&'a Glm53Dflash2Weights>,
    native_nextn: Option<&'a Glm53NextnWeights>,
}

impl<'a> Glm53FeatureRequest<'a> {
    pub fn admit(self) -> Result<AdmittedGlm53Features<'a>> {
        ensure!(
            self.source_kind.glm53_profile() == Some(self.profile),
            "GLM-5.3 target source profile does not match the requested quant profile"
        );
        ensure!(
            self.tp_world_size == 1 && self.ep_world_size == 1,
            "GLM-5.3 target currently supports one rank only"
        );
        ensure!(!self.lora_requested, "GLM-5.3 LoRA is not implemented");
        ensure!(
            !self.high_speed_swap_requested,
            "GLM-5.3 high-speed swap is not implemented"
        );
        ensure!(
            !self.prefix_cache_requested,
            "GLM-5.3 prefix cache is not implemented"
        );
        ensure!(
            !self.legacy_mtp_requested,
            "GLM-5.3 must not enter the legacy safetensors MTP path"
        );
        ensure!(
            self.enable_dflash2 == self.dflash2.is_some(),
            "GLM-5.3 DFlash2 requires an explicitly prepared typed object"
        );
        ensure!(
            self.enable_native_nextn == self.native_nextn.is_some(),
            "GLM-5.3 native NextN requires an explicitly prepared typed object"
        );
        if let Some(dflash2) = self.dflash2 {
            dflash2.config.validate()?;
            ensure!(
                dflash2.layers.len() == 5,
                "GLM-5.3 DFlash2 prepared layer count drift"
            );
        }
        Ok(AdmittedGlm53Features {
            profile: self.profile,
            dflash2: self.dflash2,
            native_nextn: self.native_nextn,
        })
    }
}

impl<'a> AdmittedGlm53Features<'a> {
    pub fn profile(&self) -> Glm53QuantProfile {
        self.profile
    }

    pub fn dflash2(&self) -> Option<&'a Glm53Dflash2Weights> {
        self.dflash2
    }

    pub fn native_nextn(&self) -> Option<&'a Glm53NextnWeights> {
        self.native_nextn
    }
}

fn validate_store_shape(
    profile: Glm53QuantProfile,
    tensors: usize,
    tensor_bytes: usize,
) -> Result<()> {
    let expected_bytes = usize::try_from(profile.tensor_bytes())
        .context("GLM-5.3 profile tensor bytes exceed host address space")?;
    ensure!(
        tensors == EXACT_TENSORS && tensor_bytes == expected_bytes,
        "GLM-5.3 {profile:?} store must contain exactly {EXACT_TENSORS} tensors and {expected_bytes} bytes"
    );
    Ok(())
}

fn validate_official_config(config: &ModelConfig) -> Result<()> {
    let glm = config
        .glm5_next
        .as_ref()
        .ok_or_else(|| anyhow!("GLM-5.3 target requires parsed glm5_next configuration"))?;
    let exact_core = config.model_type == "glm5_next"
        && config.hidden_size == 4096
        && config.num_hidden_layers == TARGET_LAYERS
        && config.intermediate_size == 12288
        && config.vocab_size == 154880
        && config.num_attention_heads == 64
        && config.num_key_value_heads == 64
        && config.head_dim == 256
        && config.q_lora_rank == 1536
        && config.kv_lora_rank == 512
        && config.qk_nope_head_dim == 256
        && config.qk_rope_head_dim == 0
        && config.v_head_dim == 256
        && config.linear_num_key_heads == 64
        && config.linear_num_value_heads == 64
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && config.linear_conv_kernel_dim == 4
        && config.num_experts == 288
        && config.num_experts_per_tok == 8
        && config.moe_intermediate_size == 2048
        && config.shared_expert_intermediate_size == 2048
        && config.hc_mult == 4
        && config.hc_sinkhorn_iters == 20
        && (config.hc_eps - 1e-6).abs() < f32::EPSILON
        && config.max_position_embeddings == 1_048_576
        && (config.rms_norm_eps - 1e-5).abs() < f64::EPSILON
        && config.scoring_func == "sigmoid"
        && config.norm_topk_prob
        && config.use_routing_bias
        && (config.routed_scaling_factor - 2.5).abs() < f64::EPSILON
        && config.num_mtp_modules == 1
        && config.mtp_transformer_layers == 1
        && config.mtp_num_hidden_layers == 1
        && config.bos_token_id == 0
        && config.eos_token_id == 154820
        && !config.tie_word_embeddings
        && config.partial_rotary_factor == 0.0
        && config.adapter_max_rank == 0
        && config.nested_config
        && config.weight_prefix == "model.language_model"
        && !config.attn_gated
        && config.vision.as_ref().is_none_or(is_exact_glm53_vision)
        && config.tp_world_size == 1
        && config.ep_world_size == 1;
    let schedules = config.layer_types.len() == TARGET_LAYERS
        && config.layer_types.iter().enumerate().all(|(index, kind)| {
            *kind
                == if index % 4 == 3 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
        })
        && glm.mlp_layer_types.len() == TARGET_LAYERS
        && glm.mlp_layer_types.iter().enumerate().all(|(index, kind)| {
            *kind
                == if index < 3 {
                    Glm5NextMlpType::Dense
                } else {
                    Glm5NextMlpType::Sparse
                }
        })
        && glm.indexer_types.len() == TARGET_LAYERS
        && glm
            .indexer_types
            .iter()
            .all(|kind| *kind == Glm5NextIndexerType::Full);
    let exact_glm = glm.eos_token_ids == [154820, 154827, 154829]
        && glm.index_head_dim == 128
        && glm.index_n_heads == 32
        && glm.index_topk == 2048
        && glm.index_kpool == 4
        && glm.index_kpool_always_select_tail
        && glm.index_kpool_compress
        && glm.index_share_for_mtp_iteration
        && glm.indexer_rope_interleave
        && (glm.linear_lower_bound + 5.0).abs() < f32::EPSILON
        && (glm.swiglu_limit - 10.0).abs() < f32::EPSILON
        && glm.mla_use_nope;
    if !exact_core || !schedules || !exact_glm {
        // Name the failing group and the fields that most often differ between
        // a hand-built test config and one parsed from a served checkpoint.
        // A bare "unsupported geometry" costs a full rebuild-and-rerun per
        // guess, which is how `tp/ep_world_size == 0` was originally found.
        let mut why: Vec<String> = Vec::new();
        if !exact_core {
            why.push("core".into());
            // Generated from the exact_core conditions themselves so the
            // list cannot drift out of sync with the check above.
            if config.model_type != "glm5_next" {
                why.push(format!("model_type={:?}", config.model_type));
            }
            if config.hidden_size != 4096 {
                why.push(format!("hidden_size={}", config.hidden_size));
            }
            if config.num_hidden_layers != TARGET_LAYERS {
                why.push(format!("num_hidden_layers={}", config.num_hidden_layers));
            }
            if config.intermediate_size != 12288 {
                why.push(format!("intermediate_size={}", config.intermediate_size));
            }
            if config.vocab_size != 154880 {
                why.push(format!("vocab_size={}", config.vocab_size));
            }
            if config.num_attention_heads != 64 {
                why.push(format!(
                    "num_attention_heads={}",
                    config.num_attention_heads
                ));
            }
            if config.num_key_value_heads != 64 {
                why.push(format!(
                    "num_key_value_heads={}",
                    config.num_key_value_heads
                ));
            }
            if config.head_dim != 256 {
                why.push(format!("head_dim={}", config.head_dim));
            }
            if config.q_lora_rank != 1536 {
                why.push(format!("q_lora_rank={}", config.q_lora_rank));
            }
            if config.kv_lora_rank != 512 {
                why.push(format!("kv_lora_rank={}", config.kv_lora_rank));
            }
            if config.qk_nope_head_dim != 256 {
                why.push(format!("qk_nope_head_dim={}", config.qk_nope_head_dim));
            }
            if config.qk_rope_head_dim != 0 {
                why.push(format!("qk_rope_head_dim={}", config.qk_rope_head_dim));
            }
            if config.v_head_dim != 256 {
                why.push(format!("v_head_dim={}", config.v_head_dim));
            }
            if config.linear_num_key_heads != 64 {
                why.push(format!(
                    "linear_num_key_heads={}",
                    config.linear_num_key_heads
                ));
            }
            if config.linear_num_value_heads != 64 {
                why.push(format!(
                    "linear_num_value_heads={}",
                    config.linear_num_value_heads
                ));
            }
            if config.linear_key_head_dim != 128 {
                why.push(format!(
                    "linear_key_head_dim={}",
                    config.linear_key_head_dim
                ));
            }
            if config.linear_value_head_dim != 128 {
                why.push(format!(
                    "linear_value_head_dim={}",
                    config.linear_value_head_dim
                ));
            }
            if config.linear_conv_kernel_dim != 4 {
                why.push(format!(
                    "linear_conv_kernel_dim={}",
                    config.linear_conv_kernel_dim
                ));
            }
            if config.num_experts != 288 {
                why.push(format!("num_experts={}", config.num_experts));
            }
            if config.num_experts_per_tok != 8 {
                why.push(format!(
                    "num_experts_per_tok={}",
                    config.num_experts_per_tok
                ));
            }
            if config.moe_intermediate_size != 2048 {
                why.push(format!(
                    "moe_intermediate_size={}",
                    config.moe_intermediate_size
                ));
            }
            if config.shared_expert_intermediate_size != 2048 {
                why.push(format!(
                    "shared_expert_intermediate_size={}",
                    config.shared_expert_intermediate_size
                ));
            }
            if config.hc_mult != 4 {
                why.push(format!("hc_mult={}", config.hc_mult));
            }
            if config.hc_sinkhorn_iters != 20 {
                why.push(format!("hc_sinkhorn_iters={}", config.hc_sinkhorn_iters));
            }
            if config.max_position_embeddings != 1_048_576 {
                why.push(format!(
                    "max_position_embeddings={}",
                    config.max_position_embeddings
                ));
            }
            if config.scoring_func != "sigmoid" {
                why.push(format!("scoring_func={:?}", config.scoring_func));
            }
            if config.num_mtp_modules != 1 {
                why.push(format!("num_mtp_modules={}", config.num_mtp_modules));
            }
            if config.mtp_transformer_layers != 1 {
                why.push(format!(
                    "mtp_transformer_layers={}",
                    config.mtp_transformer_layers
                ));
            }
            if config.mtp_num_hidden_layers != 1 {
                why.push(format!(
                    "mtp_num_hidden_layers={}",
                    config.mtp_num_hidden_layers
                ));
            }
            if config.bos_token_id != 0 {
                why.push(format!("bos_token_id={}", config.bos_token_id));
            }
            if config.eos_token_id != 154820 {
                why.push(format!("eos_token_id={}", config.eos_token_id));
            }
            if config.tie_word_embeddings {
                why.push("tie_word_embeddings=true".into());
            }
            if config.partial_rotary_factor != 0.0 {
                why.push(format!(
                    "partial_rotary_factor={}",
                    config.partial_rotary_factor
                ));
            }
            if config.adapter_max_rank != 0 {
                why.push(format!("adapter_max_rank={}", config.adapter_max_rank));
            }
            if config.weight_prefix != "model.language_model" {
                why.push(format!("weight_prefix={:?}", config.weight_prefix));
            }
            if config.attn_gated {
                why.push("attn_gated=true".into());
            }
            if config.tp_world_size != 1 {
                why.push(format!("tp_world_size={}", config.tp_world_size));
            }
            if config.ep_world_size != 1 {
                why.push(format!("ep_world_size={}", config.ep_world_size));
            }
            if config
                .vision
                .as_ref()
                .is_some_and(|vision| !is_exact_glm53_vision(vision))
            {
                why.push("vision=unsupported".into());
            }
        }
        if !schedules {
            why.push(format!(
                "schedules(layer_types={}, mlp={}, indexer={}; want {TARGET_LAYERS})",
                config.layer_types.len(),
                glm.mlp_layer_types.len(),
                glm.indexer_types.len(),
            ));
        }
        if !exact_glm {
            why.push(format!(
                "glm(eos={:?}, index_head_dim={}, index_n_heads={}, index_topk={}, \
                 index_kpool={})",
                glm.eos_token_ids,
                glm.index_head_dim,
                glm.index_n_heads,
                glm.index_topk,
                glm.index_kpool,
            ));
        }
        bail!(
            "unsupported glm5_next geometry; expected exact GLM-5.3-Flash target [{}]",
            why.join(", ")
        );
    }
    Ok(())
}

fn is_exact_glm53_vision(vision: &VisionConfig) -> bool {
    vision.model_type == "glm5_next_vision"
        && vision.depth == 24
        && vision.hidden_size == 1024
        && vision.num_heads == 16
        && vision.patch_size == 14
        && vision.temporal_patch_size == 2
        && vision.spatial_merge_size == 2
        && vision.intermediate_size == 4096
        && vision.out_hidden_size == 4096
        && vision.in_channels == 3
        && vision.image_size == 448
        && vision.projection_intermediate_size == 10240
        && (vision.rms_norm_eps - 1e-5).abs() < f64::EPSILON
        && vision.swiglu_limit == Some(10.0)
        && vision.deepstack_visual_indexes.is_empty()
        && vision.image_pad_token_id == 154854
        && vision.video_pad_token_id == 154855
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::config::Glm5NextConfig;

    fn official_config() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".into();
        config.hidden_size = 4096;
        config.num_hidden_layers = 45;
        config.intermediate_size = 12288;
        config.vocab_size = 154880;
        config.num_attention_heads = 64;
        config.num_key_value_heads = 64;
        config.head_dim = 256;
        config.partial_rotary_factor = 0.0;
        config.q_lora_rank = 1536;
        config.kv_lora_rank = 512;
        config.qk_nope_head_dim = 256;
        config.qk_rope_head_dim = 0;
        config.v_head_dim = 256;
        config.linear_num_key_heads = 64;
        config.linear_num_value_heads = 64;
        config.linear_key_head_dim = 128;
        config.linear_value_head_dim = 128;
        config.linear_conv_kernel_dim = 4;
        config.num_experts = 288;
        config.num_experts_per_tok = 8;
        config.moe_intermediate_size = 2048;
        config.shared_expert_intermediate_size = 2048;
        config.hc_mult = 4;
        config.hc_sinkhorn_iters = 20;
        config.hc_eps = 1e-6;
        config.max_position_embeddings = 1_048_576;
        config.rms_norm_eps = 1e-5;
        config.scoring_func = "sigmoid".into();
        config.norm_topk_prob = true;
        config.use_routing_bias = true;
        config.routed_scaling_factor = 2.5;
        config.num_mtp_modules = 1;
        config.mtp_transformer_layers = 1;
        config.mtp_num_hidden_layers = 1;
        config.bos_token_id = 0;
        config.eos_token_id = 154820;
        config.tie_word_embeddings = false;
        config.adapter_max_rank = 0;
        config.nested_config = true;
        config.weight_prefix = "model.language_model".into();
        config.attn_gated = false;
        config.vision = None;
        config.tp_world_size = 1;
        config.ep_world_size = 1;
        config.layer_types = (0..45)
            .map(|index| {
                if index % 4 == 3 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        config.glm5_next = Some(Glm5NextConfig {
            mlp_layer_types: (0..45)
                .map(|index| {
                    if index < 3 {
                        Glm5NextMlpType::Dense
                    } else {
                        Glm5NextMlpType::Sparse
                    }
                })
                .collect(),
            indexer_types: vec![Glm5NextIndexerType::Full; 45],
            eos_token_ids: vec![154820, 154827, 154829],
            index_head_dim: 128,
            index_n_heads: 32,
            index_topk: 2048,
            index_kpool: 4,
            index_kpool_always_select_tail: true,
            index_kpool_compress: true,
            index_share_for_mtp_iteration: true,
            indexer_rope_interleave: true,
            linear_lower_bound: -5.0,
            swiglu_limit: 10.0,
            mla_use_nope: true,
        });
        config
    }

    fn neutral_features(profile: Glm53QuantProfile) -> Glm53FeatureRequest<'static> {
        Glm53FeatureRequest {
            profile,
            source_kind: ModelSourceKind::Glm53Gguf(profile),
            tp_world_size: 1,
            ep_world_size: 1,
            lora_requested: false,
            high_speed_swap_requested: false,
            prefix_cache_requested: false,
            legacy_mtp_requested: false,
            enable_dflash2: false,
            dflash2: None,
            enable_native_nextn: false,
            native_nextn: None,
        }
    }

    fn official_vision() -> VisionConfig {
        VisionConfig {
            model_type: "glm5_next_vision".into(),
            depth: 24,
            hidden_size: 1024,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            intermediate_size: 4096,
            out_hidden_size: 4096,
            in_channels: 3,
            image_size: 448,
            projection_intermediate_size: 10240,
            rms_norm_eps: 1e-5,
            swiglu_limit: Some(10.0),
            deepstack_visual_indexes: Vec::new(),
            image_pad_token_id: 154854,
            video_pad_token_id: 154855,
        }
    }

    #[test]
    fn official_geometry_is_exact_and_drift_fails() {
        let mut config = official_config();
        validate_official_config(&config).unwrap();
        config.vision = Some(official_vision());
        validate_official_config(&config).unwrap();
        config.vision.as_mut().unwrap().depth = 23;
        assert!(validate_official_config(&config).is_err());
        config.vision = None;
        config.glm5_next.as_mut().unwrap().index_topk = 2047;
        assert!(validate_official_config(&config).is_err());
    }

    #[test]
    fn prepared_store_shape_is_exact_for_each_profile_and_rejects_cross_profile() {
        let q2 = Glm53QuantProfile::UdQ2KXl;
        let iq3 = Glm53QuantProfile::UdIq3Xxs;
        assert_eq!(q2.tensor_bytes(), 108_710_550_904);
        assert_eq!(iq3.tensor_bytes(), 120_358_051_192);
        let q2_bytes = usize::try_from(q2.tensor_bytes()).unwrap();
        let iq3_bytes = usize::try_from(iq3.tensor_bytes()).unwrap();
        validate_store_shape(q2, EXACT_TENSORS, q2_bytes).unwrap();
        validate_store_shape(iq3, EXACT_TENSORS, iq3_bytes).unwrap();
        assert!(validate_store_shape(q2, EXACT_TENSORS, iq3_bytes).is_err());
        assert!(validate_store_shape(iq3, EXACT_TENSORS, q2_bytes).is_err());
        assert!(validate_store_shape(q2, EXACT_TENSORS - 1, q2_bytes).is_err());
    }

    #[test]
    fn neutral_features_pass_but_legacy_and_multirank_fail() {
        for profile in [Glm53QuantProfile::UdQ2KXl, Glm53QuantProfile::UdIq3Xxs] {
            assert_eq!(
                neutral_features(profile).admit().unwrap().profile(),
                profile
            );
        }
        let mut legacy = neutral_features(Glm53QuantProfile::PRIMARY);
        legacy.source_kind = ModelSourceKind::Safetensors;
        assert!(legacy.admit().is_err());
        let mut multirank = neutral_features(Glm53QuantProfile::PRIMARY);
        multirank.tp_world_size = 2;
        assert!(multirank.admit().is_err());
    }

    #[test]
    fn feature_admission_rejects_cross_profile_and_legacy_alias_drift() {
        let mut cross = neutral_features(Glm53QuantProfile::UdQ2KXl);
        cross.source_kind = ModelSourceKind::Glm53Gguf(Glm53QuantProfile::UdIq3Xxs);
        assert!(cross.admit().is_err());
        let mut legacy_alias = neutral_features(Glm53QuantProfile::UdQ2KXl);
        legacy_alias.source_kind = ModelSourceKind::Glm53Iq3Gguf;
        assert!(legacy_alias.admit().is_err());
        let mut iq3_alias = neutral_features(Glm53QuantProfile::UdIq3Xxs);
        iq3_alias.source_kind = ModelSourceKind::Glm53Iq3Gguf;
        assert_eq!(
            iq3_alias.admit().unwrap().profile(),
            Glm53QuantProfile::UdIq3Xxs
        );
    }

    #[test]
    fn unsupported_features_and_unprepared_typed_flags_fail() {
        let mut unsupported = neutral_features(Glm53QuantProfile::PRIMARY);
        unsupported.prefix_cache_requested = true;
        assert!(unsupported.admit().is_err());
        let mut dflash2 = neutral_features(Glm53QuantProfile::PRIMARY);
        dflash2.enable_dflash2 = true;
        assert!(dflash2.admit().is_err());
        let mut nextn = neutral_features(Glm53QuantProfile::PRIMARY);
        nextn.enable_native_nextn = true;
        assert!(nextn.admit().is_err());
    }
}
