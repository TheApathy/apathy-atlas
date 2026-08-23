// SPDX-License-Identifier: AGPL-3.0-only

//! Launcher and fail-closed routing policy for the exact multi-row NVFP4
//! LM-head GEMV family.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

#[path = "gemv_exact_lm_head/route.rs"]
mod route;
pub use route::*;

pub const W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK: u32 = 4;

/// Output rows per lane group in the register-tiled family. The grid shrinks
/// by this factor because each lane group now covers T adjacent outputs.
pub const W4A16_EXACT_LM_HEAD_RT2_TILE: u32 = 2;

/// `ATLAS_W4A16_GEMV_RT2=1` opts into the register-tiled exact GEMV family.
///
/// Default OFF. Unset, or any value other than `1`, leaves every launch on the
/// kernel it uses today — the register-tiled twin is numerics-preserving but
/// this stays opt-in so a benchmark campaign cannot pick it up implicitly.
pub fn w4a16_gemv_rt2_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_W4A16_GEMV_RT2").as_deref() == Ok("1"))
}

/// Resolved handles for the four register-sized exact kernels.
///
/// A missing handle selects the separately qualified serial-K1 fallback in
/// production rather than a numerically different batch/MMA kernel.
#[derive(Debug, Clone, Copy)]
pub struct W4a16ExactLmHeadKernels {
    m4: KernelHandle,
    m8: KernelHandle,
    m17: KernelHandle,
    m32: KernelHandle,
    rt2_m4: KernelHandle,
    rt2_m8: KernelHandle,
    rt2_m17: KernelHandle,
    rt2_m32: KernelHandle,
}

impl W4a16ExactLmHeadKernels {
    pub const fn new(
        m4: KernelHandle,
        m8: KernelHandle,
        m17: KernelHandle,
        m32: KernelHandle,
    ) -> Self {
        Self {
            m4,
            m8,
            m17,
            m32,
            rt2_m4: KernelHandle(0),
            rt2_m8: KernelHandle(0),
            rt2_m17: KernelHandle(0),
            rt2_m32: KernelHandle(0),
        }
    }

    /// Attach the register-tiled twins. Callers that never do this keep the
    /// zero handles, which the launcher treats as "rt2 unavailable" and falls
    /// through to the shipping kernel regardless of the env gate.
    pub const fn with_rt2(
        mut self,
        m4: KernelHandle,
        m8: KernelHandle,
        m17: KernelHandle,
        m32: KernelHandle,
    ) -> Self {
        self.rt2_m4 = m4;
        self.rt2_m8 = m8;
        self.rt2_m17 = m17;
        self.rt2_m32 = m32;
        self
    }

    pub const fn rt2_for_tier(self, tier: ExactLmHeadTier) -> KernelHandle {
        match tier {
            ExactLmHeadTier::M4 => self.rt2_m4,
            ExactLmHeadTier::M8 => self.rt2_m8,
            ExactLmHeadTier::M17 => self.rt2_m17,
            ExactLmHeadTier::M32 => self.rt2_m32,
        }
    }

    pub const fn for_tier(self, tier: ExactLmHeadTier) -> KernelHandle {
        match tier {
            ExactLmHeadTier::M4 => self.m4,
            ExactLmHeadTier::M8 => self.m8,
            ExactLmHeadTier::M17 => self.m17,
            ExactLmHeadTier::M32 => self.m32,
        }
    }

    pub const fn is_present(self, tier: ExactLmHeadTier) -> bool {
        self.for_tier(tier).0 != 0
    }

    pub const fn route_for_rows(self, rows: u32) -> Option<ExactLmHeadRoute> {
        let Some(tier) = exact_lm_head_tier_for_rows(rows) else {
            return None;
        };
        exact_lm_head_route_for_rows(rows, self.is_present(tier))
    }
}

/// Launch the exact row-major NVFP4 LM-head GEMV selected by `rows`.
///
/// ABI: `(A, B_packed, B_scale, scale2, C, M, N, K)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batch_logits_exact(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactLmHeadKernels,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    w4a16_gemv_batch_logits_exact_with(
        gpu,
        kernels,
        input,
        weight,
        output,
        rows,
        n,
        k,
        stream,
        w4a16_gemv_rt2_enabled(),
    )
}

/// Same launch with the register-tiled choice passed in rather than read from
/// the environment, so a parity harness can drive both families in one process.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batch_logits_exact_with(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactLmHeadKernels,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
    use_rt2: bool,
) -> Result<()> {
    let tier = exact_lm_head_tier_for_rows(rows).ok_or_else(|| {
        anyhow::anyhow!(
            "exact NVFP4 LM-head rows must be in 2..=32 (M=1 uses ordinary GEMV), got {rows}"
        )
    })?;
    let kernel = kernels.for_tier(tier);

    ensure!(
        kernel.0 != 0,
        "missing exact LM-head kernel {}",
        tier.symbol()
    );

    // Register-tiled substitution. Only the kernel handle and the grid divisor
    // change; every argument, the block shape, and the per-output arithmetic
    // are the same, so this is a bandwidth swap rather than a routing decision.
    let rt2 = kernels.rt2_for_tier(tier);
    let (kernel, outs_per_block) = if use_rt2 && rt2.0 != 0 {
        (
            rt2,
            W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK * W4A16_EXACT_LM_HEAD_RT2_TILE,
        )
    } else {
        (kernel, W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK)
    };
    ensure!(n > 0, "exact NVFP4 LM-head vocab must be non-zero");
    ensure!(k > 0, "exact NVFP4 LM-head hidden width must be non-zero");
    ensure!(
        k.is_multiple_of(16),
        "exact NVFP4 LM-head hidden width must be divisible by 16, got {k}"
    );

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, outs_per_block), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
