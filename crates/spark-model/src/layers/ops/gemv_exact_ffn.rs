// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers and fail-closed routing for exact multi-row dense NVFP4 FFN.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

#[path = "gemv_exact_ffn/route.rs"]
mod route;
pub use route::*;

#[path = "gemv_exact_ffn/materialized_m8.rs"]
mod materialized_m8;
pub use materialized_m8::*;

#[cfg(test)]
#[path = "gemv_exact_ffn/static_tests.rs"]
mod static_tests;

pub const W4A16_EXACT_FFN_OUTS_PER_BLOCK: u32 = 4;

/// Exact gate/up and SiLU-input handles for every register-sized row tier.
#[derive(Debug, Clone, Copy)]
pub struct W4a16ExactFfnKernels {
    dual_m4: KernelHandle,
    dual_m8: KernelHandle,
    dual_m17: KernelHandle,
    dual_m32: KernelHandle,
    silu_m4: KernelHandle,
    silu_m8: KernelHandle,
    silu_m17: KernelHandle,
    silu_m32: KernelHandle,
    dual_silu_f32_m8: KernelHandle,
    f32_input_m8: KernelHandle,
    dual_silu_f32_m17: KernelHandle,
    f32_input_m17: KernelHandle,
    dual_materialize_f32_m17: KernelHandle,
    pub(crate) rt2_dual_materialize_f32_m17: KernelHandle,
    pub(crate) rt2_f32_input_m8: KernelHandle,
    pub(crate) rt2_f32_input_m17: KernelHandle,
}

impl W4a16ExactFfnKernels {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        dual_m4: KernelHandle,
        dual_m8: KernelHandle,
        dual_m17: KernelHandle,
        dual_m32: KernelHandle,
        silu_m4: KernelHandle,
        silu_m8: KernelHandle,
        silu_m17: KernelHandle,
        silu_m32: KernelHandle,
    ) -> Self {
        Self {
            rt2_dual_materialize_f32_m17: KernelHandle(0),
            rt2_f32_input_m8: KernelHandle(0),
            rt2_f32_input_m17: KernelHandle(0),
            dual_m4,
            dual_m8,
            dual_m17,
            dual_m32,
            silu_m4,
            silu_m8,
            silu_m17,
            silu_m32,
            dual_silu_f32_m8: KernelHandle(0),
            f32_input_m8: KernelHandle(0),
            dual_silu_f32_m17: KernelHandle(0),
            f32_input_m17: KernelHandle(0),
            dual_materialize_f32_m17: KernelHandle(0),
        }
    }

    pub const fn with_materialized_m8(
        mut self,
        dual_silu_f32_m8: KernelHandle,
        f32_input_m8: KernelHandle,
    ) -> Self {
        self.dual_silu_f32_m8 = dual_silu_f32_m8;
        self.f32_input_m8 = f32_input_m8;
        self
    }

    pub const fn dual_silu_f32_m8_is_present(self) -> bool {
        self.dual_silu_f32_m8.0 != 0
    }

    pub const fn f32_input_m8_is_present(self) -> bool {
        self.f32_input_m8.0 != 0
    }

    pub const fn materialized_m8_is_complete(self) -> bool {
        self.dual_silu_f32_m8_is_present() && self.f32_input_m8_is_present()
    }

    pub const fn with_materialized_m17(
        mut self,
        dual_silu_f32_m17: KernelHandle,
        f32_input_m17: KernelHandle,
    ) -> Self {
        self.dual_silu_f32_m17 = dual_silu_f32_m17;
        self.f32_input_m17 = f32_input_m17;
        self
    }

    pub const fn dual_silu_f32_m17_is_present(self) -> bool {
        self.dual_silu_f32_m17.0 != 0
    }

    pub const fn f32_input_m17_is_present(self) -> bool {
        self.f32_input_m17.0 != 0
    }

    pub const fn materialized_m17_is_complete(self) -> bool {
        self.dual_silu_f32_m17_is_present() && self.f32_input_m17_is_present()
    }

    pub const fn with_fused_materialized_m17(mut self, kernel: KernelHandle) -> Self {
        self.dual_materialize_f32_m17 = kernel;
        self
    }

    pub const fn fused_materialized_m17_is_present(self) -> bool {
        self.dual_materialize_f32_m17.0 != 0
    }

    /// Attach the register-tiled twins of the two kernels that carry the
    /// default M=17 verify FFN. Callers that skip this keep zero handles, and
    /// the launchers then use the shipping kernels regardless of the env gate.
    pub const fn with_rt2(
        mut self,
        dual_materialize_f32_m17: KernelHandle,
        f32_input_m8: KernelHandle,
        f32_input_m17: KernelHandle,
    ) -> Self {
        self.rt2_dual_materialize_f32_m17 = dual_materialize_f32_m17;
        self.rt2_f32_input_m8 = f32_input_m8;
        self.rt2_f32_input_m17 = f32_input_m17;
        self
    }

    pub const fn rt2_f32_input_for_tier(self, tier: ExactFfnTier) -> KernelHandle {
        match tier {
            ExactFfnTier::M8 => self.rt2_f32_input_m8,
            ExactFfnTier::M17 => self.rt2_f32_input_m17,
            _ => KernelHandle(0),
        }
    }

    pub const fn dual_for_tier(self, tier: ExactFfnTier) -> KernelHandle {
        match tier {
            ExactFfnTier::M4 => self.dual_m4,
            ExactFfnTier::M8 => self.dual_m8,
            ExactFfnTier::M17 => self.dual_m17,
            ExactFfnTier::M32 => self.dual_m32,
        }
    }

    pub const fn silu_input_for_tier(self, tier: ExactFfnTier) -> KernelHandle {
        match tier {
            ExactFfnTier::M4 => self.silu_m4,
            ExactFfnTier::M8 => self.silu_m8,
            ExactFfnTier::M17 => self.silu_m17,
            ExactFfnTier::M32 => self.silu_m32,
        }
    }

    pub const fn route_for_rows(self, rows: u32) -> Option<ExactFfnRoute> {
        let Some(tier) = exact_ffn_tier_for_rows(rows) else {
            return None;
        };
        exact_ffn_route_for_rows(
            rows,
            self.dual_for_tier(tier).0 != 0,
            self.silu_input_for_tier(tier).0 != 0,
        )
    }

    pub const fn tier_is_complete(self, tier: ExactFfnTier) -> bool {
        self.dual_for_tier(tier).0 != 0 && self.silu_input_for_tier(tier).0 != 0
    }
}

fn validate_exact_ffn_shape(rows: u32, n: u32, k: u32) -> Result<ExactFfnTier> {
    let tier = exact_ffn_tier_for_rows(rows)
        .ok_or_else(|| anyhow::anyhow!("exact dense FFN rows must be in 2..=32, got {rows}"))?;
    ensure!(n > 0, "exact dense FFN output width must be non-zero");
    ensure!(k > 0, "exact dense FFN input width must be non-zero");
    ensure!(
        k.is_multiple_of(16),
        "exact dense FFN input width must be divisible by 16, got {k}"
    );
    Ok(tier)
}

/// Exact row-major dual gate/up projection. CUDA ABI ends `(M, N, K)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_exact(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnKernels,
    input: DevicePtr,
    gate_weight: &QuantizedWeight,
    gate_output: DevicePtr,
    up_weight: &QuantizedWeight,
    up_output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let tier = validate_exact_ffn_shape(rows, n, k)?;
    let kernel = kernels.dual_for_tier(tier);
    ensure!(
        kernels.tier_is_complete(tier),
        "incomplete exact FFN tier: {} and {} must both resolve",
        tier.dual_symbol(),
        tier.silu_input_symbol()
    );
    ensure!(
        kernel.0 != 0,
        "missing exact FFN kernel {}",
        tier.dual_symbol()
    );

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, W4A16_EXACT_FFN_OUTS_PER_BLOCK), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_weight.weight)
        .arg_ptr(gate_weight.weight_scale)
        .arg_f32(gate_weight.weight_scale_2)
        .arg_ptr(gate_output)
        .arg_ptr(up_weight.weight)
        .arg_ptr(up_weight.weight_scale)
        .arg_f32(up_weight.weight_scale_2)
        .arg_ptr(up_output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Exact row-major SiLU(gate)*up down projection. CUDA ABI ends `(M, N, K)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_silu_input_exact(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnKernels,
    gate_output: DevicePtr,
    up_output: DevicePtr,
    down_weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let tier = validate_exact_ffn_shape(rows, n, k)?;
    let kernel = kernels.silu_input_for_tier(tier);
    ensure!(
        kernels.tier_is_complete(tier),
        "incomplete exact FFN tier: {} and {} must both resolve",
        tier.dual_symbol(),
        tier.silu_input_symbol()
    );
    ensure!(
        kernel.0 != 0,
        "missing exact FFN kernel {}",
        tier.silu_input_symbol()
    );

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, W4A16_EXACT_FFN_OUTS_PER_BLOCK), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_output)
        .arg_ptr(up_output)
        .arg_ptr(down_weight.weight)
        .arg_ptr(down_weight.weight_scale)
        .arg_f32(down_weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
