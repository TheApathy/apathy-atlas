// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in prefill: unchanged token-ordered SSM core, followed by one grouped FFN.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_qwen4_moe_batch(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        state: &mut dyn LayerState,
        seq_len_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        crate::layers::qwen4_prefill_moe::validate(ctx, rows, seq_len_start)?;
        crate::layers::qwen4_prefill_moe::validate_ffn(&self.ffn)?;
        let (attn_hyper, mlp_hyper) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("Qwen4 MoE-only SSM prefill requires both hyperconnections"),
        };
        anyhow::ensure!(
            attn_hyper.inject.is_some() && mlp_hyper.inject.is_some(),
            "Qwen4 MoE-only SSM prefill requires both injection weights"
        );
        anyhow::ensure!(
            matches!(&self.ffn, FfnComponent::Moe(_)),
            "Qwen4 MoE-only SSM prefill requires a MoE FFN"
        );
        let width = ctx.config.residual_width();
        anyhow::ensure!(
            attn_hyper.residual_width() == width && mlp_hyper.residual_width() == width,
            "Qwen4 MoE-only SSM hyperconnection width mismatch"
        );
        let row_bytes = width
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Qwen4 SSM row extent overflow"))?;
        let bytes = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("Qwen4 SSM prefill extent overflow"))?;
        anyhow::ensure!(
            !hidden.is_null()
                && !residual.is_null()
                && bytes <= ctx.buffers.sizes().hidden_states
                && bytes <= ctx.buffers.sizes().residual,
            "Qwen4 MoE-only SSM hidden/residual extent is invalid"
        );
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 MoE-only prefill requires SsmLayerState"))?;
        anyhow::ensure!(
            !ssm_state.h_is_f16 && !ssm_state.h_state.is_null() && !ssm_state.conv_state.is_null(),
            "Qwen4 MoE-only SSM prefill requires live FP32 recurrent state"
        );
        anyhow::ensure!(
            self.conv1d_l2norm_f32_k.0 != 0
                && self.gdn_f32_k.0 != 0
                && self.gated_rms_norm_f32_k.0 != 0,
            "Qwen4 MoE-only SSM prefill requires the shipping FP32 core kernels"
        );
        let ranges = [
            (hidden, bytes),
            (residual, bytes),
            (ssm_state.h_state, self.h_state_bytes),
            (ssm_state.conv_state, self.conv_state_bytes),
        ];
        for (i, &(start, extent)) in ranges.iter().enumerate() {
            let end = start
                .0
                .checked_add(extent as u64)
                .ok_or_else(|| anyhow::anyhow!("Qwen4 SSM device extent overflow"))?;
            anyhow::ensure!(extent > 0, "Qwen4 SSM device extent is empty");
            for &(other, other_extent) in &ranges[..i] {
                let other_end = other
                    .0
                    .checked_add(other_extent as u64)
                    .ok_or_else(|| anyhow::anyhow!("Qwen4 SSM device extent overflow"))?;
                anyhow::ensure!(
                    end <= other.0 || other_end <= start.0,
                    "Qwen4 MoE-only SSM input/state buffers overlap"
                );
            }
        }

        let eps = ctx.config.rms_norm_eps as f32;
        let trace = std::env::var("ATLAS_SSM_TRACE").ok().as_deref() == Some("1");
        let exact = qwen4_prefill_exact::selected()?;
        let hyper_exact = crate::layers::qwen4_prefill_moe::hyper_selected()?;
        if hyper_exact {
            attn_hyper.validate_prefill_exact(hidden, residual, rows, ctx.buffers, eps)?;
            mlp_hyper.validate_prefill_exact(hidden, residual, rows, ctx.buffers, eps)?;
        }
        if exact {
            self.prefill_qwen4_ssm_exact(hidden, residual, rows, ssm_state, ctx, stream)?;
        } else {
            if hyper_exact {
                attn_hyper.prepare_prefill_exact(
                    hidden,
                    residual,
                    rows,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?;
            }
            let inject_offset = row_bytes - (width / ctx.config.hidden_size) * 2;
            for row in 0..rows {
                let hidden_row = hidden.offset(row * row_bytes);
                let residual_row = residual.offset(row * row_bytes);
                // Keep the shipping decode core's calls, order, arithmetic and
                // injection lifetime. No same-layer FFN feeds the next core row.
                let (mixed, inject) = if hyper_exact {
                    // norm_output is scratch inside the core; residual retains all
                    // packed input rows and the separately saved injection tail.
                    (residual_row, Some(residual_row.offset(inject_offset)))
                } else {
                    attn_hyper.prepare_decode(
                        hidden_row,
                        residual_row,
                        ctx.buffers,
                        ctx.gpu,
                        eps,
                        stream,
                    )?
                };
                let ssm_out = self.ssm_forward(mixed, ssm_state, ctx, stream, trace)?;
                attn_hyper.inject_decode(
                    hidden_row,
                    ssm_out,
                    inject.expect("injection weights admitted before effects"),
                    ctx.gpu,
                    stream,
                )?;
            }
        }
        crate::layers::qwen4_prefill_moe::finish(
            mlp_hyper, &self.ffn, hidden, residual, rows, ctx, stream,
        )?;
        crate::model::qwen4_prefill_engagement::engage(
            crate::model::qwen4_prefill_engagement::PrefillPath::Ssm,
            rows,
        )?;
        Ok(())
    }
}
