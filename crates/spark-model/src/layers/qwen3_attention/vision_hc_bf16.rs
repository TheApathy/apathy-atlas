// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in Vision HCpost store-boundary experiment, not a quality qualification.
//! Official model.py@6821d6ad Block.hc_post returns y.type_as(x), with BF16 x.
//! Retain native FP32 highway allocations and arithmetic; round final stores.

use std::ffi::OsStr;

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::{HcWeights, Qwen3AttentionLayer};

const FLAG: &str = "ATLAS_VISION_HC_BF16";
const KERNEL: &str = "deepseek_vision_hc_post_bf16";

/// Parsed startup policy. No global cache: invalid environment values cannot
/// be swallowed, and an absent/0 flag performs no kernel lookup or mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionHcBf16(bool);

impl VisionHcBf16 {
    pub fn from_env(config: &ModelConfig) -> Result<Self> {
        Self::parse(std::env::var_os(FLAG).as_deref(), config)
    }

    fn parse(value: Option<&OsStr>, config: &ModelConfig) -> Result<Self> {
        let enabled = match value.map(OsStr::to_str) {
            None | Some(Some("0")) => false,
            Some(Some("1")) => true,
            _ => anyhow::bail!("{FLAG} must be absent, 0 or 1 (valid UTF-8)"),
        };
        if enabled {
            ensure!(
                config.deepseek_vision.is_some()
                    && config.model_type == "deepseek_v4"
                    && config.hidden_size == 4096
                    && config.hc_mult == 4
                    && config.ep_world_size <= 1
                    && config.tp_world_size <= 1,
                "{FLAG}=1 requires typed actual DeepSeek Vision, H4096/HC4 and single GPU"
            );
        }
        Ok(Self(enabled))
    }

    /// Serving validates CLI topology/speculation before initializing GPU or
    /// weights; factory repeats this with the effective model configuration.
    pub fn validate_execution(
        self,
        max_batch_size: usize,
        single_gpu: bool,
        target_only: bool,
    ) -> Result<()> {
        ensure!(
            !self.0 || (max_batch_size == 1 && single_gpu && target_only),
            "{FLAG}=1 requires C1, single GPU, target-only eager Vision execution"
        );
        // Actual typed Vision is unconditionally excluded from decode graph
        // capture by model/trait_impl/decode_a.rs; no debug-env dependency.
        Ok(())
    }

    /// Fail closed before expensive weights load, not on first HC dispatch.
    pub fn validate_kernel(self, gpu: &dyn GpuBackend) -> Result<()> {
        self.resolve(|module, name| gpu.kernel(module, name))?;
        Ok(())
    }

    fn resolve(
        self,
        lookup: impl FnOnce(&str, &str) -> Result<KernelHandle>,
    ) -> Result<Option<KernelHandle>> {
        if !self.0 {
            return Ok(None);
        }
        let handle = lookup(KERNEL, KERNEL)?;
        ensure!(handle.0 != 0, "{FLAG}=1 requires non-null {KERNEL}");
        Ok(Some(handle))
    }

    fn validate_hc(self, hc: Option<&HcWeights>) -> Result<()> {
        if self.0 {
            let hc = hc.ok_or_else(|| anyhow::anyhow!("{FLAG}=1 requires installed HC weights"))?;
            ensure!(hc.hc_mult == 4, "{FLAG}=1 requires installed HC4 state");
            for site in [&hc.attn, &hc.ffn] {
                ensure!(
                    !site.hc_fn.is_null() && !site.hc_base.is_null() && !site.hc_scale.is_null(),
                    "{FLAG}=1 requires non-null attention and FFN HC weights"
                );
            }
        }
        Ok(())
    }
}

impl Qwen3AttentionLayer {
    pub(crate) fn set_vision_hc_bf16(
        &mut self,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let policy = VisionHcBf16::from_env(config)?;
        policy.validate_hc(self.hc.as_ref())?;
        if let Some(handle) = policy.resolve(|module, name| gpu.kernel(module, name))? {
            self.hc_post_k = handle;
            tracing::info!(
                "DeepSeek Vision layer {} HCpost BF16-RNE store boundary ARMED (FP32 storage; prefill+decode)",
                self.attn_layer_idx
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/vision_hc_bf16/unit.rs"]
mod tests;
