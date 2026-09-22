// SPDX-License-Identifier: AGPL-3.0-only

//! Typed model-source admission.
//!
//! Legacy architectures consume safetensors through [`ModelWeightLoader`].
//! GLM-5.3 consumes a profile-validated GGUF device store through a separate
//! constructor, so its marker deliberately does not implement that trait.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
pub use spark_runtime::weights::gguf::Glm53QuantProfile;

use crate::weight_loader::ModelWeightLoader;

use super::loader_for_config;

const GLM53_MODEL_TYPE: &str = "glm5_next";

/// Physical checkpoint source selected before any store is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSourceKind {
    /// Legacy Hugging Face safetensors loaded into `WeightStore`.
    Safetensors,
    /// Exact four-shard GLM-5.3 GGUF payload with retained profile identity.
    Glm53Gguf(Glm53QuantProfile),
    /// Compatibility spelling for the fallback UD-IQ3_XXS profile.
    Glm53Iq3Gguf,
}

impl ModelSourceKind {
    /// Returns the exact GLM profile, including the legacy IQ3 spelling.
    pub const fn glm53_profile(self) -> Option<Glm53QuantProfile> {
        match self {
            Self::Glm53Gguf(profile) => Some(profile),
            Self::Glm53Iq3Gguf => Some(Glm53QuantProfile::UdIq3Xxs),
            Self::Safetensors => None,
        }
    }
}

/// Marker for the typed GLM-5.3 target constructor.
///
/// This is intentionally not a [`ModelWeightLoader`]: that trait accepts a
/// safetensors `WeightStore`, while the GLM constructor must accept only the
/// validated GGUF store type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53TargetLoader {
    profile: Glm53QuantProfile,
}

impl Glm53TargetLoader {
    const fn new(profile: Glm53QuantProfile) -> Self {
        Self { profile }
    }

    pub const fn profile(self) -> Glm53QuantProfile {
        self.profile
    }
}

/// Loader selected together with its physical checkpoint source.
pub enum ConfiguredModelLoader {
    Safetensors(Box<dyn ModelWeightLoader>),
    Glm53Gguf(Glm53TargetLoader),
}

impl ConfiguredModelLoader {
    pub fn source_kind(&self) -> ModelSourceKind {
        match self {
            Self::Safetensors(_) => ModelSourceKind::Safetensors,
            Self::Glm53Gguf(loader) => ModelSourceKind::Glm53Gguf(loader.profile()),
        }
    }

    pub fn glm53_profile(&self) -> Option<Glm53QuantProfile> {
        match self {
            Self::Safetensors(_) => None,
            Self::Glm53Gguf(loader) => Some(loader.profile()),
        }
    }

    /// GLM starts single-rank. Legacy loaders retain their declared support.
    pub fn supports_tp(&self) -> bool {
        match self {
            Self::Safetensors(loader) => loader.supports_tp(),
            Self::Glm53Gguf(_) => false,
        }
    }
}

/// Select a loader only when the architecture and physical source agree.
///
/// The existing [`loader_for_config`] API remains the safetensors-only legacy
/// registry. Callers that discover checkpoint files must use this typed entry
/// point before opening a store.
pub fn loader_for_source(
    config: &ModelConfig,
    source_kind: ModelSourceKind,
) -> Result<ConfiguredModelLoader> {
    let normalized = normalized_model_type(config);
    if normalized == GLM53_MODEL_TYPE {
        if let Some(profile) = source_kind.glm53_profile() {
            return Ok(ConfiguredModelLoader::Glm53Gguf(Glm53TargetLoader::new(
                profile,
            )));
        }
        bail!(
            "model type '{}' requires an exact profiled GLM-5.3 GGUF source; safetensors are not admitted",
            config.model_type,
        );
    }
    match source_kind {
        ModelSourceKind::Safetensors => {
            loader_for_config(config).map(ConfiguredModelLoader::Safetensors)
        }
        ModelSourceKind::Glm53Gguf(_) | ModelSourceKind::Glm53Iq3Gguf => {
            // Preserve the legacy registry's Unsupported-model diagnostic.
            // A recognized legacy model then fails on the source mismatch.
            let _ = loader_for_config(config)?;
            bail!(
                "model type '{}' requires safetensors; profiled GLM-5.3 GGUF sources are admitted only for glm5_next",
                config.model_type,
            )
        }
    }
}

fn normalized_model_type(config: &ModelConfig) -> String {
    config.model_type.to_lowercase().replace(['-', '.'], "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(model_type: &str) -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = model_type.to_owned();
        config
    }

    #[test]
    fn both_profiles_retain_identity_in_the_typed_loader() {
        for profile in [Glm53QuantProfile::UdQ2KXl, Glm53QuantProfile::UdIq3Xxs] {
            let loader =
                loader_for_source(&config("GLM5.NEXT"), ModelSourceKind::Glm53Gguf(profile))
                    .unwrap();
            assert!(matches!(&loader, ConfiguredModelLoader::Glm53Gguf(_)));
            assert_eq!(loader.glm53_profile(), Some(profile));
            assert_eq!(loader.source_kind(), ModelSourceKind::Glm53Gguf(profile));
            assert!(!loader.supports_tp());
        }
        assert_ne!(
            ModelSourceKind::Glm53Gguf(Glm53QuantProfile::UdQ2KXl),
            ModelSourceKind::Glm53Gguf(Glm53QuantProfile::UdIq3Xxs)
        );
    }

    #[test]
    fn legacy_iq3_spelling_maps_only_to_the_iq3_profile() {
        let loader =
            loader_for_source(&config("glm5_next"), ModelSourceKind::Glm53Iq3Gguf).unwrap();
        assert_eq!(loader.glm53_profile(), Some(Glm53QuantProfile::UdIq3Xxs));
        assert_eq!(
            loader.source_kind(),
            ModelSourceKind::Glm53Gguf(Glm53QuantProfile::UdIq3Xxs)
        );
    }

    #[test]
    fn source_mismatches_fail_closed_in_both_directions() {
        let glm_error = loader_for_source(&config("glm5_next"), ModelSourceKind::Safetensors)
            .err()
            .unwrap();
        assert!(
            glm_error
                .to_string()
                .contains("safetensors are not admitted")
        );

        for profile in [Glm53QuantProfile::UdQ2KXl, Glm53QuantProfile::UdIq3Xxs] {
            let legacy_error =
                loader_for_source(&config("qwen3_next"), ModelSourceKind::Glm53Gguf(profile))
                    .err()
                    .unwrap();
            assert!(legacy_error.to_string().contains("requires safetensors"));
        }
    }

    #[test]
    fn unknown_models_preserve_the_legacy_registry_error() {
        for source_kind in [
            ModelSourceKind::Safetensors,
            ModelSourceKind::Glm53Gguf(Glm53QuantProfile::UdQ2KXl),
            ModelSourceKind::Glm53Gguf(Glm53QuantProfile::UdIq3Xxs),
            ModelSourceKind::Glm53Iq3Gguf,
        ] {
            let error = loader_for_source(&config("unknown.model"), source_kind)
                .err()
                .unwrap();
            assert!(
                error
                    .to_string()
                    .contains("Unsupported model type: 'unknown.model'")
            );
        }
    }

    #[test]
    fn legacy_loader_selection_is_preserved() {
        let config = config("qwen3_next");
        let legacy_supports_tp = loader_for_config(&config).unwrap().supports_tp();
        let configured = loader_for_source(&config, ModelSourceKind::Safetensors).unwrap();
        assert!(matches!(&configured, ConfiguredModelLoader::Safetensors(_)));
        assert_eq!(configured.source_kind(), ModelSourceKind::Safetensors);
        assert_eq!(configured.supports_tp(), legacy_supports_tp);
    }
}
