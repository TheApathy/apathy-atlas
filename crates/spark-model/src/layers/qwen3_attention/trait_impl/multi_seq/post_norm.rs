// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    pub(super) fn ms_post_norm(
        &self,
        c: &MultiSeqCtx<'_>,
        mixer_out: DevicePtr,
        normed: DevicePtr,
    ) -> Result<()> {
        let serial = c.n > 1 && crate::model::env_diag::DflashSerialControls::current().layer_norms;
        if serial {
            let residual_elem = if c.fwd.config.use_fp32_residual() {
                4
            } else {
                2
            };
            for row in 0..c.n {
                ops::residual_add_rms_norm(
                    c.fwd.gpu,
                    self.residual_add_rms_norm_k,
                    c.hidden.offset(row * c.h * residual_elem),
                    mixer_out.offset(row * c.h * c.bf16),
                    &self.post_attn_norm,
                    normed.offset(row * c.h * c.bf16),
                    c.residual.offset(row * c.h * residual_elem),
                    1,
                    c.h as u32,
                    c.eps,
                    c.stream,
                )?;
            }
            crate::model::control_engagement::engage(
                crate::model::control_engagement::ControlPath::AttnPostNorm,
            )?;
            return Ok(());
        }
        ops::residual_add_rms_norm(
            c.fwd.gpu,
            self.residual_add_rms_norm_k,
            c.hidden,
            mixer_out,
            &self.post_attn_norm,
            normed,
            c.residual,
            c.n as u32,
            c.h as u32,
            c.eps,
            c.stream,
        )
    }
}
