// SPDX-License-Identifier: AGPL-3.0-only

//! Exact small-row Qwen4 SSM projection batching with serial recurrence.

use super::*;

pub(super) const QWEN4_K16_BATCHED_VERIFY_ENV: &str = "ATLAS_QWEN4_K16_BATCHED_VERIFY";
pub(super) const QWEN4_K16_EXACT_ENV: &str = "ATLAS_QWEN4_K16_EXACT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen4ExactSsmRowsRoute {
    LegacyK5OrK9,
    NativeK16,
}

impl Qwen4ExactSsmRowsRoute {
    fn admits(self, rows: usize) -> bool {
        match self {
            Self::LegacyK5OrK9 => matches!(rows, 4 | 5 | 9),
            Self::NativeK16 => rows == 16,
        }
    }

    fn force_sequence_kernels(self) -> bool {
        self == Self::NativeK16
    }
}

fn parse_exact_bool_env_value(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => {
            anyhow::bail!("{name} must be exactly 0 or 1, got {other:?}")
        }
    }
}

fn exact_bool_env(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => parse_exact_bool_env_value(name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_exact_bool_env_value(name, None),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{name} must be valid UTF-8 and exactly 0 or 1")
        }
    }
}

#[cfg(test)]
pub(super) fn parse_qwen4_k16_batched_verify(value: Option<&str>) -> Result<bool> {
    parse_exact_bool_env_value(QWEN4_K16_BATCHED_VERIFY_ENV, value)
}

#[cfg(test)]
pub(super) fn parse_qwen4_k16_exact(value: Option<&str>) -> Result<bool> {
    parse_exact_bool_env_value(QWEN4_K16_EXACT_ENV, value)
}

pub(super) fn qwen4_k16_batched_verify_requested() -> Result<bool> {
    exact_bool_env(QWEN4_K16_BATCHED_VERIFY_ENV)
}

pub(super) fn qwen4_k16_exact_requested() -> Result<bool> {
    exact_bool_env(QWEN4_K16_EXACT_ENV)
}

pub(super) fn qwen4_exact_ssm_rows_route(
    rows: usize,
    legacy_hybrid: bool,
    legacy_batch_ssm: bool,
    k16_requested: bool,
    k16_exact_requested: bool,
    has_recurrent_intermediates: bool,
) -> Option<Qwen4ExactSsmRowsRoute> {
    if rows == 16 && k16_requested && k16_exact_requested && has_recurrent_intermediates {
        Some(Qwen4ExactSsmRowsRoute::NativeK16)
    } else if matches!(rows, 4 | 5 | 9) && legacy_hybrid && legacy_batch_ssm {
        Some(Qwen4ExactSsmRowsRoute::LegacyK5OrK9)
    } else {
        None
    }
}

fn checked_row_bytes(rows: usize, width: usize, element_bytes: usize, name: &str) -> Result<usize> {
    rows.checked_mul(width)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(|| anyhow::anyhow!("Qwen4 exact SSM {name} extent overflow"))
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn preflight_qwen4_exact_ssm_rows(
        &self,
        route: Qwen4ExactSsmRowsRoute,
        rows: usize,
        state: &SsmLayerState,
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        h_intermediate_stride: usize,
        conv_intermediate_stride: usize,
        ctx: &ForwardContext,
    ) -> Result<()> {
        const BF16: usize = 2;
        const FP32: usize = 4;
        anyhow::ensure!(
            route.admits(rows),
            "Qwen4 exact SSM route does not admit {rows} rows"
        );

        let h = ctx.config.hidden_size;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ba_size = ctx.config.ssm_ba_size();
        let d_conv = ctx.config.linear_conv_kernel_dim;
        if route == Qwen4ExactSsmRowsRoute::NativeK16 {
            anyhow::ensure!(
                ctx.config.is_qwen4_exp()
                    && h == 2_560
                    && ctx.config.residual_width() == 10_240
                    && nk == 16
                    && kd == 128
                    && nv == 48
                    && vd == 128
                    && d_conv == 4
                    && qkvz_size == 16_384
                    && ba_size == 96,
                "Qwen4 native K16 SSM requires exact Flash-Next recurrent geometry"
            );
        }
        anyhow::ensure!(
            nk > 0 && nv.is_multiple_of(nk),
            "Qwen4 exact SSM requires integral value-head grouping"
        );
        anyhow::ensure!(
            self.sequential_qkvz,
            "Qwen4 exact SSM requires sequential QKVZ"
        );
        anyhow::ensure!(
            self.qkvz_nvfp4.is_some(),
            "Qwen4 exact SSM requires NVFP4 QKVZ"
        );
        anyhow::ensure!(
            !state.h_is_f16 && !state.h_state.is_null() && !state.conv_state.is_null(),
            "Qwen4 exact SSM requires FP32 live recurrent state"
        );
        anyhow::ensure!(
            !h_intermediate.is_null() && !conv_intermediate.is_null(),
            "Qwen4 exact SSM requires recurrent intermediates"
        );
        anyhow::ensure!(
            h_intermediate != state.h_state && conv_intermediate != state.conv_state,
            "Qwen4 exact SSM live and checkpoint state must not alias"
        );
        anyhow::ensure!(
            h_intermediate_stride == self.h_state_bytes
                && conv_intermediate_stride == self.conv_state_bytes
                && h_intermediate_stride.is_multiple_of(FP32)
                && conv_intermediate_stride.is_multiple_of(FP32),
            "Qwen4 exact SSM recurrent intermediate stride mismatch"
        );

        let batch_conv = route.force_sequence_kernels()
            || std::env::var("ATLAS_QWEN4_K5_BATCH_CONV").ok().as_deref() == Some("1");
        let batch_gdn = route.force_sequence_kernels()
            || std::env::var("ATLAS_QWEN4_K5_BATCH_GDN").ok().as_deref() == Some("1");
        anyhow::ensure!(
            self.ba_gates_batchn_exact_k.0 != 0
                && self.conv1d_l2norm_f32_k.0 != 0
                && self.gdn_f32_k.0 != 0
                && self.gated_rms_norm_f32_k.0 != 0
                && (!batch_conv || self.conv1d_l2norm_f32_sequence_k.0 != 0)
                && (!batch_gdn || self.gdn_f32_sequence_k.0 != 0),
            "Qwen4 exact SSM kernels are incomplete"
        );

        let sizes = ctx.buffers.sizes();
        anyhow::ensure!(
            checked_row_bytes(rows, qkvz_size, BF16, "deinterleaved")? <= sizes.ssm_deinterleaved,
            "Qwen4 exact SSM deinterleaved staging exceeds its arena"
        );
        anyhow::ensure!(
            checked_row_bytes(rows, nv * 2, FP32, "gate")? <= sizes.ssm_gates,
            "Qwen4 exact SSM gate staging exceeds its arena"
        );
        anyhow::ensure!(
            checked_row_bytes(rows, qkvz_size, FP32, "conv/GDN")? <= sizes.ssm_conv_out_f32,
            "Qwen4 exact SSM conv/GDN staging exceeds its arena"
        );
        anyhow::ensure!(
            checked_row_bytes(rows, nv * vd, BF16, "normalization")? <= sizes.ssm_qkvz,
            "Qwen4 exact SSM normalization staging exceeds its arena"
        );
        anyhow::ensure!(
            checked_row_bytes(rows, h, BF16, "output")? <= sizes.moe_output,
            "Qwen4 exact SSM output staging exceeds its arena"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ssm_forward_qwen4_exact_rows(
        &self,
        route: Qwen4ExactSsmRowsRoute,
        input: DevicePtr,
        rows: usize,
        state: &mut SsmLayerState,
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        h_intermediate_stride: usize,
        conv_intermediate_stride: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
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
        let batch_conv = route.force_sequence_kernels()
            || std::env::var("ATLAS_QWEN4_K5_BATCH_CONV").ok().as_deref() == Some("1");
        let batch_gdn = route.force_sequence_kernels()
            || std::env::var("ATLAS_QWEN4_K5_BATCH_GDN").ok().as_deref() == Some("1");
        anyhow::ensure!(
            route.admits(rows),
            "Qwen4 exact SSM route does not admit {rows} rows"
        );
        anyhow::ensure!(
            self.sequential_qkvz,
            "Qwen4 exact SSM requires sequential QKVZ"
        );
        let qkvz = self
            .qkvz_nvfp4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 exact SSM requires NVFP4 QKVZ"))?;
        anyhow::ensure!(
            !h_intermediate.is_null() && !conv_intermediate.is_null(),
            "Qwen4 exact SSM requires recurrent intermediates"
        );
        anyhow::ensure!(
            h_intermediate_stride == self.h_state_bytes
                && conv_intermediate_stride == self.conv_state_bytes,
            "Qwen4 exact SSM recurrent intermediate stride mismatch"
        );
        anyhow::ensure!(
            self.ba_gates_batchn_exact_k.0 != 0
                && self.conv1d_l2norm_f32_k.0 != 0
                && self.gdn_f32_k.0 != 0
                && self.gated_rms_norm_f32_k.0 != 0
                && (!batch_conv || self.conv1d_l2norm_f32_sequence_k.0 != 0)
                && (!batch_gdn || self.gdn_f32_sequence_k.0 != 0),
            "Qwen4 exact SSM kernels are incomplete"
        );

        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        self.project_nvfp4_rows_exact_or_k1(
            ctx.gpu,
            input,
            qkvz,
            deinterleaved,
            rows as u32,
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
            rows as u32,
            ba_size as u32,
            h as u32,
            vpg as u32,
            (nv * 2) as u32,
            stream,
        )?;

        // Stage every FP32 conv row so conv batching can be qualified
        // independently from GDN. Delaying GDN is safe: the two recurrent
        // states are disjoint and GDN consumes only its corresponding conv row.
        let conv_rows = ctx.buffers.ssm_conv_out_f32();
        if batch_conv {
            ops::conv1d_update_l2norm_f32_sequence(
                ctx.gpu,
                self.conv1d_l2norm_f32_sequence_k,
                state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                conv_rows,
                conv_intermediate,
                rows as u32,
                conv_dim as u32,
                d_conv as u32,
                qk_ch,
                kd as u32,
                1e-6,
                qkvz_size as u32,
                qkvz_size as u32,
                (conv_intermediate_stride / FP32) as u32,
                stream,
            )?;
        } else {
            for row in 0..rows {
                let qkvz_row = deinterleaved.offset(row * qkvz_size * BF16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_k,
                    state.conv_state,
                    qkvz_row,
                    &self.ssm.conv1d,
                    conv_rows.offset(row * qkvz_size * FP32),
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
            }
        }

        let gdn_rows = conv_rows.offset(conv_dim * FP32);
        // ATLAS_SSM_GDN_LAZY: the async commit decides replay-vs-copy from
        // `gdn_seq_lazy_engaged(rows)`, so this dispatch must take the lazy
        // kernel exactly when that predicate holds (and fail closed where it
        // cannot), or the commit would replay a retain buffer nobody wrote.
        let lazy = self.gdn_seq_lazy_engaged_inner(rows);
        anyhow::ensure!(
            !lazy || (batch_gdn && ctx.ddtree_parent_ids_dev.is_none()),
            "ATLAS_SSM_GDN_LAZY on the Qwen4 exact route requires the batched GDN sequence \
             (ATLAS_QWEN4_K5_BATCH_GDN=1) and no tree payload"
        );
        if lazy {
            // Retain this step's GDN inputs for the commit replay, in the
            // layout the generic lazy path uses: rows x qkvz FP32, then the
            // per-row gate/beta pairs at max_k x qkvz.
            let max_k = crate::layers::qwen3_ssm::GDN_LAZY_MAX_K;
            let retain = match self.gdn_lazy_retain.get() {
                Some(pp) => *pp,
                None => {
                    let bytes = max_k * qkvz_size * FP32 + max_k * nv * 2 * FP32;
                    let pp = ctx.gpu.alloc(bytes)?;
                    let _ = self.gdn_lazy_retain.set(pp);
                    pp
                }
            };
            ctx.gpu
                .copy_d2d_async(conv_rows, retain, rows * qkvz_size * FP32, stream)?;
            ctx.gpu.copy_d2d_async(
                gates,
                retain.offset(max_k * qkvz_size * FP32),
                rows * nv * 2 * FP32,
                stream,
            )?;
            // Final H only, into inter[rows - 1]; h_state stays step-initial.
            ops::gdn_decode_f32_sequence_persistent(
                ctx.gpu,
                self.gdn_f32_sequence_lazyfinal_k,
                state.h_state,
                conv_rows,
                conv_rows.offset(key_dim * FP32),
                conv_rows.offset(key_dim * 2 * FP32),
                gates,
                gates.offset(nv * FP32),
                gdn_rows,
                h_intermediate,
                rows as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                (h_intermediate_stride / FP32) as u32,
                stream,
            )?;
        } else if batch_gdn {
            ops::gdn_decode_f32_sequence(
                ctx.gpu,
                self.gdn_f32_sequence_k,
                state.h_state,
                conv_rows,
                conv_rows.offset(key_dim * FP32),
                conv_rows.offset(key_dim * 2 * FP32),
                gates,
                gates.offset(nv * FP32),
                gdn_rows,
                h_intermediate,
                rows as u32,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                qkvz_size as u32,
                qkvz_size as u32,
                (nv * 2) as u32,
                qkvz_size as u32,
                (h_intermediate_stride / FP32) as u32,
                stream,
            )?;
        } else {
            for row in 0..rows {
                let gates_row = gates.offset(row * nv * 2 * FP32);
                let conv_row = conv_rows.offset(row * qkvz_size * FP32);
                let gdn_row = gdn_rows.offset(row * qkvz_size * FP32);
                ops::gdn_decode(
                    ctx.gpu,
                    self.gdn_f32_k,
                    state.h_state,
                    conv_row,
                    conv_row.offset(key_dim * FP32),
                    conv_row.offset(key_dim * 2 * FP32),
                    gates_row,
                    gates_row.offset(nv * FP32),
                    gdn_row,
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
            }
        }

        let normed = ctx.buffers.ssm_qkvz();
        for row in 0..rows {
            let qkvz_row = deinterleaved.offset(row * qkvz_size * BF16);
            let gdn_row = gdn_rows.offset(row * qkvz_size * FP32);
            ops::gated_rms_norm(
                ctx.gpu,
                self.gated_rms_norm_f32_k,
                gdn_row,
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
            rows as u32,
            h as u32,
            value_dim as u32,
            stream,
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod qwen4_k16_ssm_tests {
    include!("qwen4_k16_ssm_tests.rs");
}
