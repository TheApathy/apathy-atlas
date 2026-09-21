// SPDX-License-Identifier: AGPL-3.0-only

//! Static YaRN parameters shared by every dense-Qwen RoPE route.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

use super::Qwen3AttentionLayer;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct YarnRopeParams {
    pub factor: f32,
    pub correction_low: f32,
    pub correction_high: f32,
    pub attention_factor: f32,
}

impl YarnRopeParams {
    pub(super) fn from_config(config: &ModelConfig) -> Result<Option<Self>> {
        let factor = config.yarn_factor;
        if factor == 0.0 {
            return Ok(None);
        }
        ensure!(
            factor.is_finite() && factor > 1.0,
            "YaRN factor must be finite and > 1"
        );
        let beta_fast = config.yarn_beta_fast;
        let beta_slow = config.yarn_beta_slow;
        ensure!(
            beta_fast.is_finite() && beta_fast > 0.0,
            "YaRN beta_fast must be positive"
        );
        ensure!(
            beta_slow.is_finite() && beta_slow > 0.0,
            "YaRN beta_slow must be positive"
        );
        ensure!(beta_fast > beta_slow, "YaRN requires beta_fast > beta_slow");
        let original = config.yarn_original_max_position_embeddings;
        ensure!(
            original > 0,
            "YaRN requires original_max_position_embeddings"
        );
        let theta = config.rope_theta as f32;
        ensure!(
            theta.is_finite() && theta > 1.0,
            "YaRN requires a finite rope_theta > 1"
        );
        let rotary_dim = config.rotary_dim();
        ensure!(
            rotary_dim >= 2 && rotary_dim.is_multiple_of(2),
            "YaRN requires a positive even rotary_dim"
        );

        let correction_dim = |rotations: f32| {
            rotary_dim as f32 * (original as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * theta.ln())
        };
        let correction_low = correction_dim(beta_fast).floor().max(0.0);
        let correction_high = correction_dim(beta_slow)
            .ceil()
            .min((rotary_dim - 1) as f32);
        ensure!(
            correction_low.is_finite()
                && correction_high.is_finite()
                && correction_high >= correction_low,
            "invalid YaRN correction range"
        );

        Ok(Some(Self {
            factor,
            correction_low,
            correction_high,
            attention_factor: 1.0 + 0.1 * factor.ln(),
        }))
    }
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_mrope(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        k: DevicePtr,
        pos_t: DevicePtr,
        pos_h: DevicePtr,
        pos_w: DevicePtr,
        seq_len: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        theta: f32,
        stream: u64,
    ) -> Result<()> {
        if let Some(yarn) = self.yarn {
            ensure!(
                self.rope_mrope_interleaved_yarn_k.0 != 0,
                "YaRN requested but rope_forward_mrope_interleaved_yarn is missing"
            );
            ops::rope_mrope_interleaved_yarn(
                gpu,
                self.rope_mrope_interleaved_yarn_k,
                q,
                k,
                pos_t,
                pos_h,
                pos_w,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
                theta,
                yarn.factor,
                yarn.correction_low,
                yarn.correction_high,
                yarn.attention_factor,
                stream,
            )
        } else {
            ensure!(
                self.rope_mrope_interleaved_k.0 != 0,
                "interleaved MRoPE requested but rope_forward_mrope_interleaved is missing"
            );
            ops::rope_mrope_interleaved(
                gpu,
                self.rope_mrope_interleaved_k,
                q,
                k,
                pos_t,
                pos_h,
                pos_w,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
                theta,
                stream,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::YarnRopeParams;
    use atlas_core::config::ModelConfig;

    fn qwen38() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen3_5".into();
        config.hidden_size = 5120;
        config.num_hidden_layers = 64;
        config.num_attention_heads = 24;
        config.num_key_value_heads = 4;
        config.head_dim = 256;
        config.num_experts = 0;
        config.mrope_interleaved = true;
        config.mrope_section = [11, 11, 10];
        config.yarn_factor = 4.0;
        config.yarn_beta_fast = 32.0;
        config.yarn_beta_slow = 1.0;
        config.yarn_original_max_position_embeddings = 262_144;
        config
    }

    #[test]
    fn qwen_1m_parameters_match_static_yarn_defaults() {
        let params = YarnRopeParams::from_config(&qwen38()).unwrap().unwrap();
        assert_eq!(params.factor, 4.0);
        assert!((params.attention_factor - 1.138_629_4).abs() < 1e-6);
        assert_eq!(params.correction_low, 14.0);
        assert_eq!(params.correction_high, 22.0);
    }

    #[test]
    fn invalid_yarn_contracts_fail_closed() {
        let mut config = qwen38();
        config.yarn_factor = 1.0;
        assert!(YarnRopeParams::from_config(&config).is_err());
        config.yarn_factor = 4.0;
        config.yarn_original_max_position_embeddings = 0;
        assert!(YarnRopeParams::from_config(&config).is_err());
    }
}
