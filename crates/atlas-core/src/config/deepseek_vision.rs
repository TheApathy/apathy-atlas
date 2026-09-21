// SPDX-License-Identifier: AGPL-3.0-only

//! The flat Vision-Exp contract from DeepSeek revision
//! 6821d6ad3681a4b137b066b76094fa82ebd0a380. This is deliberately separate
//! from Qwen's LayerNorm/GELU/learned-position vision configuration.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub downsample_ratio: usize,
    pub max_tokens: usize,
    pub max_wh_ratio: Option<f64>,
    pub min_pixels: usize,
    pub rope_theta: f64,
}

impl DeepSeekVisionConfig {
    pub const RMS_NORM_EPS: f32 = 1e-6;
    pub const SPECIAL_TOKEN_COUNT: usize = 5;

    pub(super) fn parse_flat(raw: &Value) -> Result<Option<Self>> {
        let obj = raw
            .as_object()
            .context("DeepSeek config must be an object")?;
        if !obj.keys().any(|key| key.starts_with("vision_")) {
            return Ok(None);
        }
        let integer = |name: &str| -> Result<usize> {
            let value = obj
                .get(name)
                .and_then(Value::as_u64)
                .with_context(|| format!("DeepSeek Vision requires integer {name}"))?;
            let value = usize::try_from(value).context("Vision dimension exceeds host size")?;
            ensure!(value > 0, "DeepSeek Vision {name} must be positive");
            Ok(value)
        };
        let finite = |name: &str| -> Result<f64> {
            let value = obj
                .get(name)
                .and_then(Value::as_f64)
                .with_context(|| format!("DeepSeek Vision requires numeric {name}"))?;
            ensure!(
                value.is_finite() && value > 0.0,
                "DeepSeek Vision {name} must be finite and positive"
            );
            Ok(value)
        };
        let config = Self {
            hidden_size: integer("vision_dim")?,
            intermediate_size: integer("vision_inter_dim")?,
            num_hidden_layers: integer("vision_n_layers")?,
            num_attention_heads: integer("vision_n_heads")?,
            patch_size: integer("vision_patch_size")?,
            downsample_ratio: integer("vision_downsample_ratio")?,
            max_tokens: integer("vision_max_n_token")?,
            max_wh_ratio: match obj.get("vision_max_wh_ratio") {
                Some(Value::Null) => None,
                _ => Some(finite("vision_max_wh_ratio")?),
            },
            min_pixels: integer("vision_min_pixels")?,
            rope_theta: finite("vision_rope_theta")?,
        };
        config.validate()?;
        ensure!(
            integer("hidden_size")? == 4096 && integer("vocab_size")? == 129280,
            "DeepSeek Vision-Exp requires text hidden_size4096 and vocab_size129280"
        );
        Ok(Some(config))
    }

    /// Bound all shape-driven allocations to the implemented Vision-Exp tower.
    /// Additional architectures require explicit implementation and parity.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.patch_size,
                self.downsample_ratio
            ) == (1024, 2816, 32, 16, 14, 3),
            "Unsupported DeepSeek Vision tower geometry"
        );
        ensure!(
            (8..=384).contains(&self.max_tokens),
            "DeepSeek Vision token budget must be8..=384"
        );
        ensure!(
            self.min_pixels > 0 && self.min_pixels <= 147456,
            "Unsupported DeepSeek Vision minimum pixel budget"
        );
        ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "Invalid DeepSeek Vision RoPE theta"
        );
        if let Some(ratio) = self.max_wh_ratio {
            ensure!(
                ratio.is_finite() && (1.0..=8.0).contains(&ratio),
                "Unsupported DeepSeek Vision aspect ratio bound"
            );
        }
        Ok(())
    }
}
