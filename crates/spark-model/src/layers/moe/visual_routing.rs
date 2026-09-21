// SPDX-License-Identifier: AGPL-3.0-only

//! Image-aware routing is an explicit per-layer capability, never a fallback.

use super::*;

pub(super) struct DeepSeekVisualRouting {
    bias: DevicePtr,
    vocab_size: u32,
    kernel: KernelHandle,
}

impl MoeLayer {
    pub(crate) fn set_deepseek_visual_routing(
        &mut self,
        bias: DevicePtr,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        anyhow::ensure!(
            config.deepseek_vision.is_some(),
            "visual MoE requires DeepSeek Vision config"
        );
        anyhow::ensure!(
            config.num_experts == 256 && config.num_experts_per_tok == 6,
            "DeepSeek Vision router requires 256 experts and top-6"
        );
        anyhow::ensure!(
            config.scoring_func == "sqrtsoftplus" && config.norm_topk_prob,
            "DeepSeek Vision requires normalized sqrtsoftplus routing"
        );
        anyhow::ensure!(
            config.routed_scaling_factor.is_finite() && config.routed_scaling_factor > 0.0,
            "DeepSeek Vision routing scale must be finite and positive"
        );
        let vocab_size = u32::try_from(config.vocab_size)?;
        anyhow::ensure!(
            vocab_size > 0 && vocab_size.checked_add(5).is_some(),
            "invalid visual vocabulary extent"
        );
        anyhow::ensure!(
            bias != DevicePtr::NULL && self.correction_bias_dev.is_some(),
            "DeepSeek Vision requires both text and visual biases"
        );
        let kernel = gpu.kernel("deepseek_visual_route", "deepseek_visual_route")?;
        anyhow::ensure!(kernel.0 != 0, "DeepSeek Vision routing kernel is missing");
        self.deepseek_visual_routing = Some(DeepSeekVisualRouting {
            bias,
            vocab_size,
            kernel,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn route_deepseek_visual(
        &self,
        ctx: &ForwardContext,
        logits: DevicePtr,
        indices: DevicePtr,
        weights: DevicePtr,
        token_offset: usize,
        rows: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(visual) = &self.deepseek_visual_routing else {
            return Ok(false);
        };
        let ids = ctx.token_ids.ok_or_else(|| {
            anyhow::anyhow!("DeepSeek Vision MoE requires token IDs for every pass")
        })?;
        let offset = token_offset
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("visual token offset overflow"))?;
        ops::deepseek_visual_route(
            ctx.gpu,
            visual.kernel,
            logits,
            self.tid2eid_dev.unwrap_or(DevicePtr::NULL),
            ids.offset(offset),
            self.correction_bias_dev
                .ok_or_else(|| anyhow::anyhow!("missing visual-model text bias"))?,
            visual.bias,
            indices,
            weights,
            visual.vocab_size,
            ctx.config.routed_scaling_factor as f32,
            rows,
            stream,
        )?;
        Ok(true)
    }
}
