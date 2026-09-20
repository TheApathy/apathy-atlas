// SPDX-License-Identifier: AGPL-3.0-only

use super::qwen4_prefill_check::Snapshot;
use super::qwen4_prefill_exact_plan::{Plan, grid32_partition};
use super::qwen4_prefill_gemm::Projection;
use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_qwen4_ssm_exact(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        state: &mut SsmLayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let p = self.preflight_qwen4_prefill_exact(hidden, residual, rows, state, ctx)?;
        let projection_mode = qwen4_prefill_gemm::mode()?;
        let check = if qwen4_prefill_exact::check_selected()? {
            Some(Snapshot::capture(rows, hidden, state, ctx, stream)?)
        } else {
            None
        };
        let attn = self.qwen4_attn_hyper.as_ref().expect("admitted attn HC");
        let eps = ctx.config.rms_norm_eps as f32;
        let input =
            attn.prepare_prefill_exact(hidden, residual, rows, ctx.buffers, ctx.gpu, eps, stream)?;
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        if projection_mode.uses_gemm(Projection::Qkvz) {
            self.project_prefill_gemm(Projection::Qkvz, input, deinterleaved, rows, ctx, stream)?;
        } else {
            self.project_prefill_exact_tiles(
                input,
                self.qkvz_nvfp4.as_ref().expect("admitted NVFP4 QKVZ"),
                deinterleaved,
                &p,
                Plan::QKVZ,
                Plan::H,
                ctx,
                stream,
            )?;
        }
        let gates = ctx.buffers.ssm_gates();
        ops::dense_gemv_ba_gates_batchn(
            ctx.gpu,
            self.ba_gates_batchn_exact_k,
            input,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates,
            gates.offset(48 * 4),
            rows as u32,
            Plan::GATES as u32,
            Plan::H as u32,
            3,
            Plan::GATES as u32,
            stream,
        )?;

        // Each FP32 row holds [Q|K|V|GDN output]. Conv and H state are disjoint;
        // preserve token order inside both kernels, without verify snapshots.
        let conv_rows = ctx.buffers.ssm_conv_out_f32();
        ops::conv1d_update_l2norm_f32_sequence(
            ctx.gpu,
            self.conv1d_l2norm_f32_sequence_k,
            state.conv_state,
            deinterleaved,
            &self.ssm.conv1d,
            conv_rows,
            DevicePtr::NULL,
            rows as u32,
            Plan::CONV_DIM as u32,
            4,
            (Plan::KEY_DIM * 2) as u32,
            128,
            1e-6,
            Plan::QKVZ as u32,
            Plan::QKVZ as u32,
            0,
            stream,
        )?;
        let gdn_rows = conv_rows.offset(Plan::CONV_DIM * 4);
        ops::gdn_decode_f32_sequence(
            ctx.gpu,
            self.gdn_f32_sequence_nosnap_k,
            state.h_state,
            conv_rows,
            conv_rows.offset(Plan::KEY_DIM * 4),
            conv_rows.offset(Plan::KEY_DIM * 2 * 4),
            gates,
            gates.offset(48 * 4),
            gdn_rows,
            DevicePtr::NULL,
            rows as u32,
            16,
            48,
            128,
            128,
            Plan::QKVZ as u32,
            Plan::QKVZ as u32,
            Plan::GATES as u32,
            Plan::QKVZ as u32,
            0,
            stream,
        )?;
        let normed = ctx.buffers.ssm_qkvz();
        ops::gated_rms_norm_f32_multi_seq(
            ctx.gpu,
            self.gated_rms_norm_f32_multi_seq_k,
            gdn_rows,
            deinterleaved.offset(Plan::CONV_DIM * 2),
            &self.ssm.norm,
            normed,
            48,
            rows as u32,
            128,
            eps,
            Plan::QKVZ as u32,
            Plan::QKVZ as u32,
            Plan::VALUE_DIM as u32,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        if projection_mode.uses_gemm(Projection::Output) {
            self.project_prefill_gemm(Projection::Output, normed, output, rows, ctx, stream)?;
        } else {
            self.project_prefill_exact_tiles(
                normed,
                &self.ssm.out_proj,
                output,
                &p,
                Plan::H,
                Plan::VALUE_DIM,
                ctx,
                stream,
            )?;
        }
        attn.inject_saved_batched(hidden, output, residual, rows, ctx.gpu, stream)?;
        if let Some(check) = check {
            check.verify(self, hidden, residual, rows, state, ctx, stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn project_prefill_exact_tiles(
        &self,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        plan: &Plan,
        n: usize,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // GRID32 changes only submission geometry: blockIdx.y selects one
        // disjoint M32 slab while every block preserves the incumbent K1
        // accumulation and BF16 projection output. Incomplete tails retain the
        // exact register tier (or shipping single-row decode dispatcher).
        let grid32 = qwen4_prefill_exact::grid32_selected()?;
        let (full_rows, _) = if grid32 {
            grid32_partition(plan.rows)
        } else {
            (0, plan.rows)
        };
        if full_rows != 0 {
            ops::w4a16_gemv_batch_logits_exact_rt2_m32_grid(
                ctx.gpu,
                self.w4a16_exact_projection_kernels,
                input,
                weight,
                output,
                full_rows as u32,
                n as u32,
                k as u32,
                stream,
            )?;
            static SSM_GRID32_ENGAGED: std::sync::Once = std::sync::Once::new();
            SSM_GRID32_ENGAGED.call_once(|| {
                tracing::info!(
                    "SSM_PREFILL_EXACT_GRID32_ENGAGED selector=ATLAS_QWEN4_PREFILL_SSM_GRID32 rows={full_rows}"
                );
            });
        }
        for (start, count) in plan.tiles_from(full_rows) {
            let a = input.offset(start * k * 2);
            let c = output.offset(start * n * 2);
            if count == 1 {
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    self.w4a16_gemv_sw_k,
                    self.gemv_sw,
                    a,
                    weight,
                    c,
                    n as u32,
                    k as u32,
                    stream,
                )?;
            } else {
                let use_rt2 = ops::w4a16_gemv_rt2_enabled();
                if use_rt2 {
                    static SSM_RT2_ENGAGED: std::sync::Once = std::sync::Once::new();
                    SSM_RT2_ENGAGED.call_once(|| {
                        tracing::info!(
                            "SSM_PREFILL_EXACT_RT2_ENGAGED selector=ATLAS_W4A16_GEMV_RT2 rows={count}"
                        );
                    });
                }
                ops::w4a16_gemv_batch_logits_exact_with(
                    ctx.gpu,
                    self.w4a16_exact_projection_kernels,
                    a,
                    weight,
                    c,
                    count as u32,
                    n as u32,
                    k as u32,
                    stream,
                    // The register-tiled exact family is explicitly selected
                    // by ATLAS_W4A16_GEMV_RT2=1 and preflighted above.
                    use_rt2,
                )?;
            }
        }
        Ok(())
    }
}
