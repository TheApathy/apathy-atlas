// SPDX-License-Identifier: AGPL-3.0-only

//! Architecture-first admission for external DFlash drafter checkpoints.

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use spark_runtime::weights::WeightStore;

use super::dflash_loader::{DflashConfig, parse_dflash_config};
use super::glm53_dflash2::{
    Glm53Dflash2Config, parse_glm53_dflash2_config, validate_glm53_dflash2_store,
};

const GLM53_DFLASH2_ARCHITECTURE: &str = "DFlash2DraftModel";
const DFLASH2_ONLY_FIELDS: [&str; 4] = [
    "conv_group_size",
    "conv_kernel_size",
    "selector_rank",
    "selector_top_k",
];

/// Parsed drafter metadata whose architecture choice cannot be erased before
/// weight-store admission or proposer construction.
#[derive(Debug, Clone)]
pub enum DflashDrafterConfig {
    Legacy(DflashConfig),
    Glm53Dflash2(Glm53Dflash2Config),
}

impl DflashDrafterConfig {
    /// Validate architecture-specific store invariants before runtime dispatch.
    pub fn validate_store(&self, store: &WeightStore) -> Result<()> {
        match self {
            Self::Legacy(_) => Ok(()),
            Self::Glm53Dflash2(_) => validate_glm53_dflash2_store(store),
        }
    }

    /// Admit only the legacy variant to the existing block-diffusion head.
    ///
    /// This is deliberately consuming: a caller cannot unwrap a GLM DFlash2
    /// config and accidentally feed it to the permissive legacy path.
    pub fn into_legacy_runtime(self) -> Result<DflashConfig> {
        match self {
            Self::Legacy(config) => Ok(config),
            Self::Glm53Dflash2(_) => bail!(
                "GLM-5.3 DFlash2 passed strict config/store admission, but its typed proposer is \
                 not integrated; refusing the legacy BlockDiffusionDraftHead"
            ),
        }
    }
}

/// Parse a drafter config without giving the permissive legacy parser first
/// refusal over a DFlash2 checkpoint.
pub fn parse_dflash_drafter_config(json: &str) -> Result<DflashDrafterConfig> {
    let value: Value =
        serde_json::from_str(json).context("parsing DFlash architecture envelope")?;
    let object = value
        .as_object()
        .context("DFlash config.json root must be an object")?;

    let architectures = match object.get("architectures") {
        None => Vec::new(),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .context("DFlash architectures entries must be strings")
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => bail!("DFlash architectures must be an array of strings"),
    };
    if architectures.contains(&GLM53_DFLASH2_ARCHITECTURE) {
        return parse_glm53_dflash2_config(json).map(DflashDrafterConfig::Glm53Dflash2);
    }

    let dflash2_fields = object
        .get("dflash_config")
        .and_then(Value::as_object)
        .map(|config| {
            DFLASH2_ONLY_FIELDS
                .iter()
                .filter(|field| config.contains_key(**field))
                .copied()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    ensure!(
        dflash2_fields.is_empty(),
        "DFlash2-only config fields {dflash2_fields:?} require exact architecture \
         DFlash2DraftModel"
    );

    parse_dflash_config(json).map(DflashDrafterConfig::Legacy)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    const LEGACY: &str = r#"{
      "architectures":["Qwen3ForCausalLM"],
      "hidden_size":2048,"num_hidden_layers":8,"intermediate_size":6144,
      "num_attention_heads":32,"num_key_value_heads":4,"head_dim":128,
      "vocab_size":248320,"block_size":16,
      "dflash_config":{"mask_token_id":248070,"target_layer_ids":[1,10,19,28,37]}
    }"#;

    const GLM53_DFLASH2: &str = r#"{
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

    #[test]
    fn legacy_qwen_remains_on_the_legacy_path() {
        let admitted = parse_dflash_drafter_config(LEGACY).unwrap();
        admitted
            .validate_store(&WeightStore::from_map(HashMap::new()))
            .unwrap();
        let config = admitted.into_legacy_runtime().unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.block_size, 16);
    }

    #[test]
    fn exact_glm_config_is_typed_and_cannot_enter_legacy_runtime() {
        let admitted = parse_dflash_drafter_config(GLM53_DFLASH2).unwrap();
        assert!(matches!(&admitted, DflashDrafterConfig::Glm53Dflash2(_)));
        let empty = WeightStore::from_map(HashMap::new());
        assert!(admitted.validate_store(&empty).is_err());
        assert!(admitted.into_legacy_runtime().is_err());
    }

    #[test]
    fn mixed_or_forged_dflash2_identity_is_rejected_not_downgraded() {
        let mixed = GLM53_DFLASH2.replace(
            "[\"DFlash2DraftModel\"]",
            "[\"Qwen3ForCausalLM\",\"DFlash2DraftModel\"]",
        );
        assert!(parse_dflash_drafter_config(&mixed).is_err());

        let forged = GLM53_DFLASH2.replace("DFlash2DraftModel", "DFlashDraftModel");
        let error = parse_dflash_drafter_config(&forged)
            .unwrap_err()
            .to_string();
        assert!(error.contains("DFlash2-only config fields"));
    }

    #[test]
    fn malformed_architecture_envelope_is_rejected_before_legacy_parse() {
        let malformed = LEGACY.replace("[\"Qwen3ForCausalLM\"]", "{\"0\":\"DFlash2DraftModel\"}");
        let error = parse_dflash_drafter_config(&malformed)
            .unwrap_err()
            .to_string();
        assert!(error.contains("array of strings"));
    }
}
