// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in numerical candidate using already-resident original-layout weights.
//! BF16-operand MMA with FP32 accumulation differs from exact GEMV arithmetic.

use super::qwen4_prefill_exact_plan::Plan;
use super::*;

mod plan;
pub(crate) use plan::{Mode, Projection};

const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_SSM_GEMM";

pub(crate) fn mode() -> Result<Mode> {
    plan::parse_mode(std::env::var_os(SELECTOR).as_deref())
        .map_err(|error| anyhow::anyhow!("{SELECTOR}: {error}"))
}

pub(super) fn admit(exact: bool, check: bool) -> Result<()> {
    mode()?
        .admit(exact, check)
        .map_err(|error| anyhow::anyhow!("{SELECTOR}: {error}"))
}

pub(super) fn validate_handle(handle: u64) -> Result<()> {
    mode()?
        .validate_handle(handle)
        .map_err(|error| anyhow::anyhow!("{SELECTOR}: {error}"))
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project_prefill_gemm(
        &self,
        projection: Projection,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Admission has validated both projection weights, row extents, and
        // every live arena before hyperconnection or recurrent-state effects.
        let (weight, n, k) = match projection {
            Projection::Qkvz => (
                self.qkvz_nvfp4.as_ref().expect("admitted NVFP4 QKVZ"),
                Plan::QKVZ,
                Plan::H,
            ),
            Projection::Output => (&self.ssm.out_proj, Plan::H, Plan::VALUE_DIM),
        };
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm_k,
            input,
            weight,
            output,
            rows as u32,
            n as u32,
            k as u32,
            stream,
        )
    }
}
