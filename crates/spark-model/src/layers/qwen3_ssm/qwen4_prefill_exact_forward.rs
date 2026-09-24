// SPDX-License-Identifier: AGPL-3.0-only

use super::qwen4_prefill_check::Snapshot;
use super::qwen4_prefill_exact_plan::{Plan, grid32_partition};
use super::qwen4_prefill_gemm::Projection;
use super::*;

const GDN_LAZYFINAL_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_GDN_LAZYFINAL";

const CONV_PARALLEL_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_CONV_PARALLEL";

fn conv_parallel_selected() -> bool {
    static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| std::env::var(CONV_PARALLEL_SELECTOR).ok().as_deref() == Some("1"))
}

fn conv_parallel_kernels(gpu: &dyn GpuBackend) -> Result<(KernelHandle, KernelHandle)> {
    static K: std::sync::OnceLock<Result<(KernelHandle, KernelHandle), String>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let a = gpu
            .kernel("causal_conv1d_prefill_parallel", "causal_conv1d_update_l2norm_f32_prefill_parallel")
            .map_err(|e| e.to_string())?;
        let b = gpu
            .kernel("causal_conv1d_prefill_parallel", "causal_conv1d_prefill_state_commit")
            .map_err(|e| e.to_string())?;
        Ok((a, b))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{CONV_PARALLEL_SELECTOR}: kernel unavailable: {e}"))
}

fn gdn_lazyfinal_selected() -> bool {
    gdn_lazyfinal_kernel_name().is_some()
}

/// 1 = register twin, 2 = its prefetch twin, 3 = H truly in registers (the
/// `_regfinal` H array sits in a local-memory stack frame), 4/5 = 3 plus the
/// token stream staged 15/8 tokens at a time through shared memory. All five
/// share the arithmetic order of the nosnap kernel.
fn gdn_lazyfinal_kernel_name() -> Option<&'static str> {
    static SEL: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| match std::env::var(GDN_LAZYFINAL_SELECTOR).ok().as_deref() {
        Some("1") => Some("gated_delta_rule_prefill_f32_sequence_regfinal"),
        Some("2") => Some("gated_delta_rule_prefill_f32_sequence_regfinal_pf"),
        Some("3") => Some("gated_delta_rule_prefill_f32_sequence_regfinal_full"),
        Some("4") => Some("gated_delta_rule_prefill_f32_sequence_regfinal_chunk15"),
        Some("5") => Some("gated_delta_rule_prefill_f32_sequence_regfinal_chunk8"),
        _ => None,
    })
}

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
        let fast = crate::layers::qwen4_fast_proj::selection()?;
        if fast.ssm {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!(
                    "SSM_PREFILL_FAST_ENGAGED selector={} rows={rows} projection=dequant_cublaslt_bf16_non_bit_exact",
                    crate::layers::qwen4_fast_proj::SELECTOR
                );
            });
            crate::layers::qwen4_fast_proj::dequant_gemm(
                ctx.gpu,
                input,
                self.qkvz_nvfp4.as_ref().expect("admitted NVFP4 QKVZ"),
                deinterleaved,
                rows,
                Plan::QKVZ,
                Plan::H,
                stream,
            )?;
        } else if projection_mode.uses_gemm(Projection::Qkvz) {
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
        if conv_parallel_selected() {
            self.conv_prefill_parallel(state.conv_state, deinterleaved, conv_rows, rows, ctx, stream)?;
        } else {
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
        }
        let gdn_rows = conv_rows.offset(Plan::CONV_DIM * 4);
        // Default-off: the register-resident lazy-commit twin of the nosnap
        // kernel (identical arithmetic order, no per-token snapshot writes,
        // H kept in registers instead of 2 global passes per token).
        let gdn_kernel = if gdn_lazyfinal_selected() {
            static K: std::sync::OnceLock<Result<KernelHandle, String>> = std::sync::OnceLock::new();
            let k = K
                .get_or_init(|| {
                    ctx.gpu
                        .kernel(
                            "gated_delta_rule_prefill_regfinal",
                            gdn_lazyfinal_kernel_name().expect("selected"),
                        )
                        .map_err(|e| e.to_string())
                })
                .clone()
                .map_err(|e| anyhow::anyhow!("{GDN_LAZYFINAL_SELECTOR}: kernel unavailable: {e}"))?;
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!(
                    "SSM_PREFILL_GDN_REGFINAL_ENGAGED selector={GDN_LAZYFINAL_SELECTOR} kernel={} rows={rows}",
                    gdn_lazyfinal_kernel_name().unwrap_or("?")
                );
            });
            k
        } else {
            self.gdn_f32_sequence_nosnap_k
        };
        ops::gdn_decode_f32_sequence(
            ctx.gpu,
            gdn_kernel,
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
        if fast.ssm {
            crate::layers::qwen4_fast_proj::dequant_gemm(
                ctx.gpu,
                normed,
                &self.ssm.out_proj,
                output,
                rows,
                Plan::H,
                Plan::VALUE_DIM,
                stream,
            )?;
        } else if projection_mode.uses_gemm(Projection::Output) {
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

    /// Token-parallel exact conv1d (+ per-head L2) and shift-register commit.
    fn conv_prefill_parallel(
        &self,
        conv_state: DevicePtr,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
        let (conv_k, commit_k) = conv_parallel_kernels(ctx.gpu)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!("SSM_PREFILL_CONV_PARALLEL_ENGAGED selector={CONV_PARALLEL_SELECTOR} rows={rows}");
        });
        KernelLaunch::new(ctx.gpu, conv_k)
            .grid([div_ceil(Plan::CONV_DIM as u32, 256), rows as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(conv_state)
            .arg_ptr(input)
            .arg_ptr(self.ssm.conv1d.weight)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(output)
            .arg_u32(rows as u32)
            .arg_u32(Plan::CONV_DIM as u32)
            .arg_u32(4)
            .arg_u32((Plan::KEY_DIM * 2) as u32)
            .arg_u32(128)
            .arg_f32(1e-6)
            .arg_u32(Plan::QKVZ as u32)
            .arg_u32(Plan::QKVZ as u32)
            .launch(stream)?;
        KernelLaunch::new(ctx.gpu, commit_k)
            .grid([div_ceil(Plan::CONV_DIM as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(conv_state)
            .arg_ptr(input)
            .arg_u32(rows as u32)
            .arg_u32(Plan::CONV_DIM as u32)
            .arg_u32(4)
            .arg_u32(Plan::QKVZ as u32)
            .launch(stream)
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
