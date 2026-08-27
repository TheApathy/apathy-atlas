// SPDX-License-Identifier: AGPL-3.0-only

//! Exact five-row Qwen4 SSM projection batching with serial recurrence.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ssm_forward_qwen4_k5_exact(
        &self,
        input: DevicePtr,
        state: &mut SsmLayerState,
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        h_intermediate_stride: usize,
        conv_intermediate_stride: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        const ROWS: usize = 5;
        const BF16: usize = 2;
        const FP32: usize = 4;
        let h = ctx.config.hidden_size;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qk_ch = (key_dim * 2) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        anyhow::ensure!(self.sequential_qkvz, "Qwen4 K5 requires sequential QKVZ");
        let qkvz = self
            .qkvz_nvfp4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 K5 requires NVFP4 QKVZ"))?;
        anyhow::ensure!(
            !h_intermediate.is_null() && !conv_intermediate.is_null(),
            "Qwen4 K5 exact SSM requires recurrent intermediates"
        );
        anyhow::ensure!(
            h_intermediate_stride == self.h_state_bytes
                && conv_intermediate_stride == self.conv_state_bytes,
            "Qwen4 K5 recurrent intermediate stride mismatch"
        );
        anyhow::ensure!(
            self.ba_gates_batchn_exact_k.0 != 0
                && self.conv1d_l2norm_f32_k.0 != 0
                && self.gdn_f32_k.0 != 0
                && self.gated_rms_norm_f32_k.0 != 0,
            "Qwen4 K5 exact SSM kernels are incomplete"
        );

        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        self.project_nvfp4_rows_exact_or_k1(
            ctx.gpu,
            input,
            qkvz,
            deinterleaved,
            ROWS as u32,
            qkvz_size as u32,
            h as u32,
            stream,
        )?;

        let gates = ctx.buffers.ssm_gates();
        ops::dense_gemv_ba_gates_batchn(
            ctx.gpu,
            self.ba_gates_batchn_exact_k,
            input,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates,
            gates.offset(nv * FP32),
            ROWS as u32,
            ba_size as u32,
            h as u32,
            vpg as u32,
            (nv * 2) as u32,
            stream,
        )?;

        // The exact M=5 projections amortize both large weight streams. Keep
        // the causal recurrent update in ordinary K=1 operation order: the
        // sequence kernels are not byte-identical for Qwen4 and cannot be used
        // for speculative commit.
        let conv_f32 = ctx.buffers.ssm_conv_out_f32();
        let gdn_f32 = conv_f32.offset(conv_dim * FP32);
        let normed = ctx.buffers.ssm_qkvz();
        for row in 0..ROWS {
            let qkvz_row = deinterleaved.offset(row * qkvz_size * BF16);
            let gates_row = gates.offset(row * nv * 2 * FP32);
            ops::conv1d_update_l2norm(
                ctx.gpu,
                self.conv1d_l2norm_f32_k,
                state.conv_state,
                qkvz_row,
                &self.ssm.conv1d,
                conv_f32,
                conv_dim as u32,
                d_conv as u32,
                1,
                qk_ch,
                kd as u32,
                1e-6,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                state.conv_state,
                conv_intermediate.offset(row * conv_intermediate_stride),
                conv_intermediate_stride,
                stream,
            )?;
            ops::gdn_decode(
                ctx.gpu,
                self.gdn_f32_k,
                state.h_state,
                conv_f32,
                conv_f32.offset(key_dim * FP32),
                conv_f32.offset(key_dim * 2 * FP32),
                gates_row,
                gates_row.offset(nv * FP32),
                gdn_f32,
                1,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                state.h_state,
                h_intermediate.offset(row * h_intermediate_stride),
                h_intermediate_stride,
                stream,
            )?;
            ops::gated_rms_norm(
                ctx.gpu,
                self.gated_rms_norm_f32_k,
                gdn_f32,
                qkvz_row.offset((key_dim * 2 + value_dim) * BF16),
                &self.ssm.norm,
                normed.offset(row * value_dim * BF16),
                nv as u32,
                vd as u32,
                vd as u32,
                eps,
                vd as u32,
                stream,
            )?;
        }

        let output = ctx.buffers.moe_output();
        self.project_nvfp4_rows_exact_or_k1(
            ctx.gpu,
            normed,
            &self.ssm.out_proj,
            output,
            ROWS as u32,
            h as u32,
            value_dim as u32,
            stream,
        )?;
        Ok(output)
    }
}
