// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off exact-family projections with ordered FP32 sequence recurrence.

use super::qwen4_prefill_check_raw::validate_check;
use super::qwen4_prefill_exact_plan::{Plan, parse_selector, validate_regions, validate_request};
use super::*;
use atlas_core::config::ModelConfig;

const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_SSM_EXACT";
const CHECK_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_SSM_CHECK";
const GRID32_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_SSM_GRID32";

pub(crate) fn check_selected() -> Result<bool> {
    if crate::layers::qwen4_prefill_moe::family_suppressed() {
        return Ok(false);
    }
    parse_selector(std::env::var_os(CHECK_SELECTOR).as_deref())
        .map_err(|error| anyhow::anyhow!("{CHECK_SELECTOR}: {error}"))
}

pub(crate) fn selected() -> Result<bool> {
    if crate::layers::qwen4_prefill_moe::family_suppressed() {
        return Ok(false);
    }
    parse_selector(std::env::var_os(SELECTOR).as_deref())
        .map_err(|error| anyhow::anyhow!("{SELECTOR}: {error}"))
}

pub(crate) fn grid32_selected() -> Result<bool> {
    if crate::layers::qwen4_prefill_moe::family_suppressed() {
        return Ok(false);
    }
    parse_selector(std::env::var_os(GRID32_SELECTOR).as_deref())
        .map_err(|error| anyhow::anyhow!("{GRID32_SELECTOR}: {error}"))
}

/// Global request admission, also evaluated when the parent F8 flag is absent.
pub(crate) fn admit_request(config: &ModelConfig, rows: usize, start: usize) -> Result<()> {
    let exact = selected()?;
    let check = check_selected()?;
    let grid32 = grid32_selected()?;
    validate_check(exact, check).map_err(anyhow::Error::msg)?;
    qwen4_prefill_gemm::admit(exact, check)?;
    anyhow::ensure!(!grid32 || exact, "{GRID32_SELECTOR} requires {SELECTOR}=1");
    anyhow::ensure!(
        !grid32 || ops::w4a16_gemv_rt2_enabled(),
        "{GRID32_SELECTOR} requires ATLAS_W4A16_GEMV_RT2=1"
    );
    if !exact {
        return Ok(());
    }
    anyhow::ensure!(
        crate::layers::qwen4_prefill_moe::selected()?
            && crate::layers::qwen4_prefill_moe::hyper_selected()?,
        "{SELECTOR} requires ATLAS_QWEN4_PREFILL_MOE_BATCH=1 and ATLAS_QWEN4_PREFILL_HC_EXACT=1"
    );
    anyhow::ensure!(
        config.is_qwen4_exp()
            && config.hidden_size == Plan::H
            && config.residual_width() == 10_240
            && config.linear_num_key_heads == 16
            && config.linear_key_head_dim == 128
            && config.linear_num_value_heads == 48
            && config.linear_value_head_dim == 128
            && config.linear_conv_kernel_dim == 4
            && config.output_gate_type == "sigmoid"
            && !config.use_fp32_residual(),
        "{SELECTOR} requires canonical Flash-Next FP32 recurrence and BF16 sigmoid output"
    );
    validate_request(rows, start).map_err(anyhow::Error::msg)
}

impl Qwen3SsmLayer {
    /// All kernel, arena and state admission completes before the first effect.
    pub(super) fn preflight_qwen4_prefill_exact(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        state: &SsmLayerState,
        ctx: &ForwardContext,
    ) -> Result<Plan> {
        anyhow::ensure!(selected()?, "exact SSM prefill was not selected");
        admit_request(ctx.config, rows, 0)?;
        qwen4_prefill_gemm::validate_handle(self.w4a16_gemm_k.0)?;
        let sizes = ctx.buffers.sizes();
        let plan = Plan::new(
            rows,
            ctx.buffers.max_batch_tokens(),
            [
                sizes.norm_output,
                sizes.ssm_deinterleaved,
                sizes.ssm_conv_out_f32,
                sizes.ssm_gates,
                sizes.ssm_qkvz,
                sizes.moe_output,
            ],
        )
        .map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            plan.residual_bytes <= sizes.hidden_states && plan.residual_bytes <= sizes.residual,
            "exact SSM residual slabs exceed their arenas"
        );
        let (attn, mlp) = match (&self.qwen4_attn_hyper, &self.qwen4_mlp_hyper) {
            (Some(attn), Some(mlp)) => (attn, mlp),
            _ => anyhow::bail!("exact SSM requires both hyperconnections"),
        };
        anyhow::ensure!(
            attn.inject.is_some()
                && mlp.inject.is_some()
                && attn.residual_width() == 10_240
                && mlp.residual_width() == 10_240,
            "exact SSM requires canonical injecting hyperconnections"
        );
        let eps = ctx.config.rms_norm_eps as f32;
        attn.validate_prefill_exact(hidden, residual, rows, ctx.buffers, eps)?;
        mlp.validate_prefill_exact(hidden, residual, rows, ctx.buffers, eps)?;
        anyhow::ensure!(
            self.sequential_qkvz
                && self.qkvz_fp8w.is_none()
                && self.out_proj_fp8w.is_none()
                && self.out_proj_dense.is_none(),
            "exact SSM requires original-layout NVFP4 projections without alternative weights"
        );
        let qkvz = self
            .qkvz_nvfp4
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("exact SSM requires NVFP4 QKVZ"))?;
        for weight in [qkvz, &self.ssm.out_proj] {
            anyhow::ensure!(
                !weight.weight.is_null()
                    && !weight.weight_scale.is_null()
                    && weight.weight_scale_2.is_finite()
                    && weight.weight_scale_2 > 0.0,
                "invalid exact SSM projection weight"
            );
        }
        // Weight dimensions/dtypes are fixed by the admitted loaded model;
        // these weight wrappers carry pointers, not independent shape metadata.
        for weight in [
            &self.ssm.in_proj_ba,
            &self.ssm.a_log,
            &self.ssm.dt_bias,
            &self.ssm.conv1d,
            &self.ssm.norm,
        ] {
            anyhow::ensure!(!weight.weight.is_null(), "missing exact SSM dense weight");
        }
        anyhow::ensure!(
            !state.h_is_f16
                && self.h_state_bytes == Plan::H_STATE_BYTES
                && self.conv_state_bytes == Plan::CONV_STATE_BYTES,
            "exact SSM requires canonical FP32 recurrent state extents"
        );
        for kernel in [
            self.ba_gates_batchn_exact_k,
            self.conv1d_l2norm_f32_sequence_k,
            self.gdn_f32_sequence_nosnap_k,
            self.gated_rms_norm_f32_multi_seq_k,
            self.w4a16_gemv_k,
        ] {
            anyhow::ensure!(
                kernel.0 != 0,
                "missing exact SSM sequence/projection kernel"
            );
        }
        let use_rt2 = ops::w4a16_gemv_rt2_enabled();
        if grid32_selected()? && plan.rows >= 32 {
            anyhow::ensure!(
                self.w4a16_exact_projection_kernels.rt2_m32_grid().0 != 0,
                "selected exact SSM GRID32 projection kernel is unavailable"
            );
        }
        for (_, count) in plan.tiles() {
            let tier = ops::exact_lm_head_tier_for_rows(count as u32);
            anyhow::ensure!(
                count == 1
                    || matches!(
                        self.w4a16_exact_projection_kernels
                            .route_for_rows(count as u32),
                        Some(ops::ExactLmHeadRoute::Exact(_))
                    ),
                "missing exact SSM register projection tier"
            );
            anyhow::ensure!(
                count == 1
                    || !use_rt2
                    || self
                        .w4a16_exact_projection_kernels
                        .rt2_for_tier(tier.expect("multi-row exact tier"))
                        .0
                        != 0,
                "selected exact SSM RT2 projection tier is unavailable"
            );
        }
        let b = ctx.buffers;
        validate_regions(&[
            (hidden.0, plan.residual_bytes, 2),
            (residual.0, plan.residual_bytes, 2),
            (state.h_state.0, self.h_state_bytes, 4),
            (state.conv_state.0, self.conv_state_bytes, 4),
            (b.norm_output().0, plan.bytes[0], 2),
            (b.ssm_deinterleaved().0, plan.bytes[1], 2),
            (b.ssm_conv_out_f32().0, plan.bytes[2], 4),
            (b.ssm_gates().0, plan.bytes[3], 4),
            (b.ssm_qkvz().0, plan.bytes[4], 2),
            (b.moe_output().0, plan.bytes[5], 2),
            (b.ssm_ba().0, sizes.ssm_ba, 2),
            (b.qkv_output().0, sizes.qkv_output, 2),
        ])
        .map_err(anyhow::Error::msg)?;
        Ok(plan)
    }
}
