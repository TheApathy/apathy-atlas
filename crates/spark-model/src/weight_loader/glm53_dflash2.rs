// SPDX-License-Identifier: AGPL-3.0-only

//! Strict CPU-side admission and typed views for the GLM-5.3 DFlash2 drafter.

use std::collections::BTreeSet;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_map::{DenseWeight, dense};

pub const GLM53_DFLASH2_TENSOR_COUNT: usize = 81;
pub const GLM53_DFLASH2_PAYLOAD_BYTES: usize = 2_342_160_896;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Glm53Dflash2SubConfig {
    pub block_size: usize,
    pub conv_group_size: usize,
    pub conv_kernel_size: usize,
    pub mask_token_id: u32,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub target_layer_ids: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct Glm53Dflash2RopeParameters {
    pub rope_theta: f32,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Glm53Dflash2Config {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub bos_token_id: Option<u32>,
    pub dflash_config: Glm53Dflash2SubConfig,
    pub dtype: String,
    pub eos_token_id: Vec<u32>,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub is_causal: bool,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_target_layers: usize,
    pub pad_token_id: u32,
    pub rms_norm_eps: f32,
    rope_parameters: Glm53Dflash2RopeParameters,
    pub sliding_window: usize,
    pub tie_word_embeddings: bool,
    pub transformers_version: String,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub vocab_size: usize,
}

impl Glm53Dflash2Config {
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters.rope_theta
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.architectures == ["DFlash2DraftModel"],
            "GLM DFlash2 requires architecture DFlash2DraftModel"
        );
        ensure!(
            self.dtype == "bfloat16",
            "GLM DFlash2 requires BF16 weights"
        );
        ensure!(
            (self.hidden_size, self.intermediate_size) == (4096, 12288),
            "GLM DFlash2 hidden/intermediate geometry drift"
        );
        ensure!(
            (self.num_hidden_layers, self.num_target_layers) == (5, 45),
            "GLM DFlash2 layer geometry drift"
        );
        ensure!(
            (
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim
            ) == (32, 8, 128),
            "GLM DFlash2 GQA geometry drift"
        );
        ensure!(self.vocab_size == 154880, "GLM DFlash2 vocabulary drift");
        ensure!(
            self.bos_token_id.is_none()
                && self.pad_token_id == 154820
                && self.eos_token_id == [154820, 154827, 154829],
            "GLM DFlash2 BOS/pad/EOS contract drift"
        );
        ensure!(
            !self.attention_bias
                && self.attention_dropout == 0.0
                && !self.is_causal
                && !self.use_cache
                && !self.tie_word_embeddings,
            "GLM DFlash2 attention/cache/tied-embedding contract drift"
        );
        ensure!(
            self.hidden_act == "silu"
                && self.initializer_range == 0.02
                && self.rms_norm_eps == 0.00001,
            "GLM DFlash2 activation or numeric constant drift"
        );
        ensure!(
            self.layer_types == ["sliding_attention"; 5]
                && self.max_window_layers == 5
                && self.sliding_window == 2048
                && self.use_sliding_window,
            "GLM DFlash2 attention-window contract drift"
        );
        ensure!(
            self.max_position_embeddings == 1_048_576
                && self.model_type == "qwen3"
                && self.rope_parameters.rope_type == "default"
                && self.rope_parameters.rope_theta == 10_000.0
                && self.transformers_version == "5.7.0",
            "GLM DFlash2 model or RoPE identity drift"
        );
        let dflash = &self.dflash_config;
        ensure!(
            (
                dflash.block_size,
                dflash.conv_group_size,
                dflash.conv_kernel_size,
                dflash.mask_token_id,
                dflash.selector_rank,
                dflash.selector_top_k,
            ) == (8, 16, 2, 154856, 256, 16),
            "GLM DFlash2 block/conv/selector contract drift"
        );
        ensure!(
            dflash.target_layer_ids == [5, 14, 24, 33, 42],
            "GLM DFlash2 target capture layers drift"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Dflash2ConvWeights {
    pub base_kernel: DenseWeight,
    pub kernel_projection: DenseWeight,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Dflash2LayerWeights {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub output: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
    pub attention_conv: Glm53Dflash2ConvWeights,
    pub mlp_conv: Glm53Dflash2ConvWeights,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Dflash2CandidateSelectorWeights {
    pub hidden_projection: DenseWeight,
    pub predecessor_codebook: DenseWeight,
    pub successor_codebook: DenseWeight,
}

#[derive(Debug)]
pub struct Glm53Dflash2Weights {
    pub config: Glm53Dflash2Config,
    pub fc: DenseWeight,
    pub hidden_norm: DenseWeight,
    pub norm: DenseWeight,
    pub selector: Glm53Dflash2CandidateSelectorWeights,
    pub layers: Vec<Glm53Dflash2LayerWeights>,
}

pub fn parse_glm53_dflash2_config(json: &str) -> Result<Glm53Dflash2Config> {
    let config: Glm53Dflash2Config =
        serde_json::from_str(json).context("parsing GLM-5.3 DFlash2 config.json")?;
    config.validate()?;
    Ok(config)
}

fn exact_schema() -> Vec<(String, Vec<usize>)> {
    let mut schema = vec![
        (
            "candidate_selector.hidden_projection.weight".into(),
            vec![256, 4096],
        ),
        (
            "candidate_selector.predecessor_codebook".into(),
            vec![154880, 256],
        ),
        (
            "candidate_selector.successor_codebook".into(),
            vec![154880, 256],
        ),
        ("fc.weight".into(), vec![4096, 20480]),
        ("hidden_norm.weight".into(), vec![4096]),
        ("norm.weight".into(), vec![4096]),
    ];
    for layer in 0..5 {
        let base = format!("layers.{layer}");
        for (suffix, shape) in [
            ("attention_conv.base_kernel", vec![2, 2, 4096]),
            ("attention_conv.kernel_projection.weight", vec![1024, 4096]),
            ("input_layernorm.weight", vec![4096]),
            ("mlp.down_proj.weight", vec![4096, 12288]),
            ("mlp.gate_proj.weight", vec![12288, 4096]),
            ("mlp.up_proj.weight", vec![12288, 4096]),
            ("mlp_conv.base_kernel", vec![2, 2, 4096]),
            ("mlp_conv.kernel_projection.weight", vec![1024, 4096]),
            ("post_attention_layernorm.weight", vec![4096]),
            ("self_attn.k_norm.weight", vec![128]),
            ("self_attn.k_proj.weight", vec![1024, 4096]),
            ("self_attn.o_proj.weight", vec![4096, 4096]),
            ("self_attn.q_norm.weight", vec![128]),
            ("self_attn.q_proj.weight", vec![4096, 4096]),
            ("self_attn.v_proj.weight", vec![1024, 4096]),
        ] {
            schema.push((format!("{base}.{suffix}"), shape));
        }
    }
    schema
}

pub fn validate_glm53_dflash2_store(store: &WeightStore) -> Result<()> {
    let schema = exact_schema();
    ensure!(
        schema.len() == GLM53_DFLASH2_TENSOR_COUNT,
        "internal GLM DFlash2 schema count drift"
    );
    let expected_names: BTreeSet<_> = schema.iter().map(|(name, _)| name.as_str()).collect();
    let actual_names: BTreeSet<_> = store.names().collect();
    if expected_names != actual_names {
        let missing: Vec<_> = expected_names.difference(&actual_names).copied().collect();
        let unexpected: Vec<_> = actual_names.difference(&expected_names).copied().collect();
        bail!("GLM DFlash2 tensor names drift: missing={missing:?}, unexpected={unexpected:?}");
    }

    let mut total_bytes = 0usize;
    let mut ranges = Vec::with_capacity(schema.len());
    for (name, shape) in &schema {
        let tensor = store.get(name)?;
        ensure!(
            tensor.dtype == WeightDtype::BF16,
            "GLM DFlash2 {name} is not BF16"
        );
        ensure!(
            tensor.shape.as_slice() == shape.as_slice(),
            "GLM DFlash2 {name} shape drift"
        );
        ensure!(
            tensor.ptr != DevicePtr::NULL,
            "GLM DFlash2 {name} has a null device pointer"
        );
        let elements = shape.iter().try_fold(1usize, |count, &dimension| {
            count
                .checked_mul(dimension)
                .context("GLM DFlash2 tensor element overflow")
        })?;
        let bytes = elements
            .checked_mul(2)
            .context("GLM DFlash2 tensor byte overflow")?;
        total_bytes = total_bytes
            .checked_add(bytes)
            .context("GLM DFlash2 payload byte overflow")?;
        let end = tensor
            .ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .with_context(|| format!("GLM DFlash2 {name} device address overflow"))?;
        ranges.push((name, tensor.ptr.0, end));
    }
    ensure!(
        total_bytes == GLM53_DFLASH2_PAYLOAD_BYTES,
        "GLM DFlash2 payload extent drift"
    );
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].1 < ranges[right].2 && ranges[right].1 < ranges[left].2 {
                bail!(
                    "GLM DFlash2 tensors overlap: {} and {}",
                    ranges[left].0,
                    ranges[right].0
                );
            }
        }
    }
    Ok(())
}

fn conv(store: &WeightStore, base: &str) -> Result<Glm53Dflash2ConvWeights> {
    Ok(Glm53Dflash2ConvWeights {
        base_kernel: dense(store, &format!("{base}.base_kernel"))?,
        kernel_projection: dense(store, &format!("{base}.kernel_projection.weight"))?,
    })
}

pub fn load_glm53_dflash2_weights(
    store: &WeightStore,
    config: &Glm53Dflash2Config,
) -> Result<Glm53Dflash2Weights> {
    config.validate()?;
    validate_glm53_dflash2_store(store)?;
    let mut layers = Vec::with_capacity(5);
    for layer in 0..5 {
        let base = format!("layers.{layer}");
        layers.push(Glm53Dflash2LayerWeights {
            input_layernorm: dense(store, &format!("{base}.input_layernorm.weight"))?,
            post_attention_layernorm: dense(
                store,
                &format!("{base}.post_attention_layernorm.weight"),
            )?,
            q_proj: dense(store, &format!("{base}.self_attn.q_proj.weight"))?,
            k_proj: dense(store, &format!("{base}.self_attn.k_proj.weight"))?,
            v_proj: dense(store, &format!("{base}.self_attn.v_proj.weight"))?,
            output: dense(store, &format!("{base}.self_attn.o_proj.weight"))?,
            q_norm: dense(store, &format!("{base}.self_attn.q_norm.weight"))?,
            k_norm: dense(store, &format!("{base}.self_attn.k_norm.weight"))?,
            gate_proj: dense(store, &format!("{base}.mlp.gate_proj.weight"))?,
            up_proj: dense(store, &format!("{base}.mlp.up_proj.weight"))?,
            down_proj: dense(store, &format!("{base}.mlp.down_proj.weight"))?,
            attention_conv: conv(store, &format!("{base}.attention_conv"))?,
            mlp_conv: conv(store, &format!("{base}.mlp_conv"))?,
        });
    }
    Ok(Glm53Dflash2Weights {
        config: config.clone(),
        fc: dense(store, "fc.weight")?,
        hidden_norm: dense(store, "hidden_norm.weight")?,
        norm: dense(store, "norm.weight")?,
        selector: Glm53Dflash2CandidateSelectorWeights {
            hidden_projection: dense(store, "candidate_selector.hidden_projection.weight")?,
            predecessor_codebook: dense(store, "candidate_selector.predecessor_codebook")?,
            successor_codebook: dense(store, "candidate_selector.successor_codebook")?,
        },
        layers,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spark_runtime::weights::WeightTensor;

    use super::*;

    const CONFIG: &str = r#"{
      "architectures":["DFlash2DraftModel"],
      "attention_bias":false,"attention_dropout":0.0,"bos_token_id":null,
      "dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,
        "mask_token_id":154856,"selector_rank":256,"selector_top_k":16,
        "target_layer_ids":[5,14,24,33,42]},
      "dtype":"bfloat16","eos_token_id":[154820,154827,154829],"head_dim":128,
      "hidden_act":"silu","hidden_size":4096,"initializer_range":0.02,
      "intermediate_size":12288,"is_causal":false,
      "layer_types":["sliding_attention","sliding_attention","sliding_attention",
        "sliding_attention","sliding_attention"],"max_position_embeddings":1048576,
      "max_window_layers":5,"model_type":"qwen3",
      "num_attention_heads":32,"num_hidden_layers":5,"num_key_value_heads":8,
      "num_target_layers":45,"pad_token_id":154820,"rms_norm_eps":0.00001,
      "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
      "sliding_window":2048,"tie_word_embeddings":false,
      "transformers_version":"5.7.0","use_cache":false,
      "use_sliding_window":true,"vocab_size":154880
    }"#;

    fn map_with(mut alter: impl FnMut(&str, &mut WeightTensor)) -> HashMap<String, WeightTensor> {
        let mut map = HashMap::new();
        let mut address = 0x10_0000u64;
        for (name, shape) in exact_schema() {
            let bytes = shape.iter().product::<usize>() * 2;
            let mut tensor = WeightTensor {
                ptr: DevicePtr(address),
                shape,
                dtype: WeightDtype::BF16,
            };
            alter(&name, &mut tensor);
            map.insert(name, tensor);
            address += u64::try_from(bytes).unwrap() + 256;
        }
        map
    }

    #[test]
    fn exact_config_and_typed_store_are_admitted() {
        let config = parse_glm53_dflash2_config(CONFIG).unwrap();
        assert_eq!(config.dflash_config.block_size, 8);
        assert_eq!(config.dflash_config.target_layer_ids, [5, 14, 24, 33, 42]);
        assert_eq!(config.rope_theta(), 10_000.0);
        assert_eq!(config.rms_norm_eps, 1e-5);
        let store = WeightStore::from_map(map_with(|_, _| {}));
        let weights = load_glm53_dflash2_weights(&store, &config).unwrap();
        assert_eq!(weights.layers.len(), 5);
        assert_ne!(weights.selector.successor_codebook.weight, DevicePtr::NULL);
    }

    #[test]
    fn config_geometry_drift_is_rejected() {
        let mut config: Glm53Dflash2Config = serde_json::from_str(CONFIG).unwrap();
        config.dflash_config.block_size = 16;
        assert!(config.validate().is_err());
        config.dflash_config.block_size = 8;
        config.eos_token_id.swap(0, 1);
        assert!(config.validate().is_err());
        config.eos_token_id.swap(0, 1);
        config.rope_parameters.rope_theta = 1_000_000.0;
        assert!(config.validate().is_err());
        config.rope_parameters.rope_theta = 10_000.0;
        config.rms_norm_eps = 1e-6;
        assert!(config.validate().is_err());
        config.rms_norm_eps = 1e-5;
        config.use_sliding_window = false;
        assert!(config.validate().is_err());
    }

    #[test]
    fn shape_dtype_name_and_alias_faults_are_rejected() {
        let wrong_shape = WeightStore::from_map(map_with(|name, tensor| {
            if name == "fc.weight" {
                tensor.shape[1] -= 1;
            }
        }));
        assert!(validate_glm53_dflash2_store(&wrong_shape).is_err());

        let wrong_dtype = WeightStore::from_map(map_with(|name, tensor| {
            if name == "norm.weight" {
                tensor.dtype = WeightDtype::FP32;
            }
        }));
        assert!(validate_glm53_dflash2_store(&wrong_dtype).is_err());

        let mut extra = map_with(|_, _| {});
        extra.insert(
            "unexpected.weight".into(),
            WeightTensor {
                ptr: DevicePtr(0xf000_0000_0000),
                shape: vec![1],
                dtype: WeightDtype::BF16,
            },
        );
        assert!(validate_glm53_dflash2_store(&WeightStore::from_map(extra)).is_err());

        let aliased = WeightStore::from_map(map_with(|name, tensor| {
            if name == "hidden_norm.weight" || name == "norm.weight" {
                tensor.ptr = DevicePtr(0xf000_0000_0000);
            }
        }));
        assert!(validate_glm53_dflash2_store(&aliased).is_err());
    }
}
