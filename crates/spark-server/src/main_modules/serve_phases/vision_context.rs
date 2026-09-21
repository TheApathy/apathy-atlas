// SPDX-License-Identifier: AGPL-3.0-only
use atlas_core::config::ModelConfig;

/// Distinguish native compressed-attention YaRN from an unqualified context
/// extension. DeepSeek Vision uses YaRN even inside its original text window.
pub(crate) fn text_only_yarn_context(config: &ModelConfig, max_seq_len: usize) -> bool {
    config.yarn_factor > 0.0
        && !(config.model_type == "deepseek_v4"
            && config.deepseek_vision.is_some()
            && config.yarn_original_max_position_embeddings > 0
            && max_seq_len > 0
            && max_seq_len <= config.yarn_original_max_position_embeddings)
}
