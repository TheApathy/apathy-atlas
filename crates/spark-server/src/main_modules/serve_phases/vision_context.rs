// SPDX-License-Identifier: AGPL-3.0-only
use atlas_core::config::ModelConfig;

/// Static YaRN keeps image requests text-only, except on a model whose YaRN is
/// native to its attention at every length.
pub(crate) fn text_only_yarn_context(config: &ModelConfig) -> bool {
    yarn_blocks_images(&config.model_type, config.yarn_factor)
}

fn yarn_blocks_images(model_type: &str, yarn_factor: f32) -> bool {
    // DeepSeek-V4.1 applies YaRN inside its own attention at every length and
    // the production server accepts images at any context; its image path is
    // gated by the dsv41 vision config, not by this check.
    yarn_factor > 0.0 && model_type != "deepseek_v41"
}

#[cfg(test)]
mod tests {
    use super::yarn_blocks_images;

    #[test]
    fn deepseek_v41_images_are_not_blocked_by_yarn() {
        assert!(!yarn_blocks_images("deepseek_v41", 16.0));
    }

    #[test]
    fn other_yarn_models_stay_text_only() {
        // Controls: the same YaRN factor on any other model still blocks images.
        assert!(yarn_blocks_images("deepseek_v4", 16.0));
        assert!(yarn_blocks_images("qwen3", 4.0));
        assert!(!yarn_blocks_images("qwen3", 0.0));
    }
}
