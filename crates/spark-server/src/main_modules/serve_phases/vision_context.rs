// SPDX-License-Identifier: AGPL-3.0-only
use atlas_core::config::ModelConfig;

/// Distinguish native compressed-attention YaRN from an unqualified context
/// extension. DeepSeek Vision uses YaRN even inside its original text window.
pub(crate) fn text_only_yarn_context(config: &ModelConfig, max_seq_len: usize) -> bool {
    yarn_blocks_images(
        &config.model_type,
        config.yarn_factor,
        config.deepseek_vision.is_some(),
        config.yarn_original_max_position_embeddings,
        max_seq_len,
    )
}

fn yarn_blocks_images(
    model_type: &str,
    yarn_factor: f32,
    has_deepseek_vision: bool,
    original_max_positions: usize,
    max_seq_len: usize,
) -> bool {
    // DeepSeek-V4.1 applies YaRN inside its own attention at every length and
    // the production server accepts images at any context; its image path is
    // gated by the dsv41 vision config, not by this check.
    if model_type == "deepseek_v41" {
        return false;
    }
    yarn_factor > 0.0
        && !(model_type == "deepseek_v4"
            && has_deepseek_vision
            && original_max_positions > 0
            && max_seq_len > 0
            && max_seq_len <= original_max_positions)
}

#[cfg(test)]
mod tests {
    use super::yarn_blocks_images;

    #[test]
    fn deepseek_v41_images_are_not_blocked_by_yarn() {
        assert!(!yarn_blocks_images("deepseek_v41", 16.0, false, 65536, 1_048_576));
        assert!(!yarn_blocks_images("deepseek_v41", 16.0, false, 65536, 8192));
    }

    #[test]
    fn other_yarn_models_stay_text_only() {
        // Controls: the same YaRN settings on any other model still block images.
        assert!(yarn_blocks_images("deepseek_v4", 16.0, false, 65536, 8192));
        assert!(yarn_blocks_images("deepseek_v4", 16.0, true, 65536, 131072));
        assert!(yarn_blocks_images("qwen3", 4.0, false, 32768, 131072));
        assert!(!yarn_blocks_images("deepseek_v4", 16.0, true, 65536, 65536));
        assert!(!yarn_blocks_images("qwen3", 0.0, false, 0, 131072));
    }
}
