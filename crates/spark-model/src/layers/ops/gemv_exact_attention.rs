// SPDX-License-Identifier: AGPL-3.0-only

//! Exact K1-order multi-row NVFP4 attention projection kernels.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

const OUTS_PER_BLOCK: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactAttentionKernelArg {
    Buffer(DevicePtr),
    F32Bits(u32),
    U32(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExactAttentionLaunchPlan<const ARGS: usize> {
    grid: [u32; 3],
    block: [u32; 3],
    args: [ExactAttentionKernelArg; ARGS],
}

impl<const ARGS: usize> ExactAttentionLaunchPlan<ARGS> {
    fn launch(self, gpu: &dyn GpuBackend, kernel: KernelHandle, stream: u64) -> Result<()> {
        let mut launch = KernelLaunch::new(gpu, kernel)
            .grid(self.grid)
            .block(self.block);
        for arg in self.args {
            launch = match arg {
                ExactAttentionKernelArg::Buffer(ptr) => launch.arg_ptr(ptr),
                ExactAttentionKernelArg::F32Bits(bits) => launch.arg_f32(f32::from_bits(bits)),
                ExactAttentionKernelArg::U32(value) => launch.arg_u32(value),
            };
        }
        launch.launch(stream)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactAttentionQkvRoute {
    ExactM4,
    SerialK1M4,
    ExactM17,
    SerialK1M17,
    ExactM32,
    SerialK1M32,
}

/// The gated-Q and dual-KV kernels form one atomic arithmetic route at each
/// M4/M17/M32 tier. If either selected handle is absent, production must use
/// the complete ordinary K1 path rather than mix exact multi-row and serial
/// projections.
#[derive(Debug, Clone, Copy)]
pub struct W4a16ExactAttentionKernels {
    qg_m4: KernelHandle,
    dual_kv_m4: KernelHandle,
    qg_m17: KernelHandle,
    dual_kv_m17: KernelHandle,
    qg_m32: KernelHandle,
    dual_kv_m32: KernelHandle,
}

impl W4a16ExactAttentionKernels {
    pub const fn new(qg_m17: KernelHandle, dual_kv_m17: KernelHandle) -> Self {
        Self {
            qg_m4: KernelHandle(0),
            dual_kv_m4: KernelHandle(0),
            qg_m17,
            dual_kv_m17,
            qg_m32: KernelHandle(0),
            dual_kv_m32: KernelHandle(0),
        }
    }

    pub const fn with_m4(mut self, qg_m4: KernelHandle, dual_kv_m4: KernelHandle) -> Self {
        self.qg_m4 = qg_m4;
        self.dual_kv_m4 = dual_kv_m4;
        self
    }

    pub const fn with_m32(mut self, qg_m32: KernelHandle, dual_kv_m32: KernelHandle) -> Self {
        self.qg_m32 = qg_m32;
        self.dual_kv_m32 = dual_kv_m32;
        self
    }

    pub const fn qg_for_rows(self, rows: usize) -> KernelHandle {
        match rows {
            4 => self.qg_m4,
            // The M17 kernel accumulates MAX_M=17 rows with the identical
            // K1 K8 lane assignment / shuffle tree / cross-warp add as the
            // M9..=17 tier; running it at M=5..=8 is bit-exact and only
            // wastes unused accumulator registers (no idle row writes).
            // This closes the serial-per-token gap for γ=4..=7 (n=5..=8).
            5..=17 => self.qg_m17,
            18..=32 => self.qg_m32,
            _ => KernelHandle(0),
        }
    }

    pub const fn dual_kv_for_rows(self, rows: usize) -> KernelHandle {
        match rows {
            4 => self.dual_kv_m4,
            5..=17 => self.dual_kv_m17,
            18..=32 => self.dual_kv_m32,
            _ => KernelHandle(0),
        }
    }

    const fn complete_for_rows(self, rows: usize) -> bool {
        self.qg_for_rows(rows).0 != 0 && self.dual_kv_for_rows(rows).0 != 0
    }
}

/// ABI-separate M17 activation-staging experiment. Keeping this pair outside
/// [`W4a16ExactAttentionKernels`] ensures the proven M4/M17 route and its
/// fallback decisions remain byte-for-byte independent of the experiment.
#[derive(Debug, Clone, Copy)]
pub struct W4a16ExactAttentionM17AStageKernels {
    qg: KernelHandle,
    dual_kv: KernelHandle,
}

impl W4a16ExactAttentionM17AStageKernels {
    pub const fn new(qg: KernelHandle, dual_kv: KernelHandle) -> Self {
        Self { qg, dual_kv }
    }

    pub const fn qg(self) -> KernelHandle {
        self.qg
    }

    pub const fn dual_kv(self) -> KernelHandle {
        self.dual_kv
    }

    pub const fn complete(self) -> bool {
        self.qg.0 != 0 && self.dual_kv.0 != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactAttentionM17AStageRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

/// Pure fail-closed selector for the ABI-separate staged pair. `Missing` is a
/// distinct state so an explicit opt-in can never become a silent baseline or
/// a mixed QG/dual-KV route when the compiled kernel bundle is stale.
pub const fn exact_attention_m17_astage_route(
    requested: bool,
    rows: usize,
    gated: bool,
    ordinary_nvfp4: bool,
    kernels: W4a16ExactAttentionM17AStageKernels,
) -> ExactAttentionM17AStageRoute {
    if !requested {
        return ExactAttentionM17AStageRoute::Disabled;
    }
    if !gated || !ordinary_nvfp4 || rows < 5 || rows > 17 {
        return ExactAttentionM17AStageRoute::Ineligible;
    }
    if kernels.complete() {
        ExactAttentionM17AStageRoute::Complete
    } else {
        ExactAttentionM17AStageRoute::Missing
    }
}

/// Select the exact path qualified for gated ordinary-NVFP4 attention.
pub const fn exact_attention_qkv_route(
    rows: usize,
    gated: bool,
    ordinary_nvfp4: bool,
    kernels: W4a16ExactAttentionKernels,
) -> Option<ExactAttentionQkvRoute> {
    if !gated || !ordinary_nvfp4 {
        return None;
    }
    match rows {
        4 => Some(if kernels.complete_for_rows(rows) {
            ExactAttentionQkvRoute::ExactM4
        } else {
            ExactAttentionQkvRoute::SerialK1M4
        }),
        5..=17 => Some(if kernels.complete_for_rows(rows) {
            ExactAttentionQkvRoute::ExactM17
        } else {
            ExactAttentionQkvRoute::SerialK1M17
        }),
        18..=32 => Some(if kernels.complete_for_rows(rows) {
            ExactAttentionQkvRoute::ExactM32
        } else {
            ExactAttentionQkvRoute::SerialK1M32
        }),
        _ => None,
    }
}

const fn exact_attention_rows_supported(rows: u32) -> bool {
    rows == 4 || (rows >= 5 && rows <= 32)
}

#[allow(clippy::too_many_arguments)]
fn qg_launch_plan(
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    out_stride: u32,
) -> Result<ExactAttentionLaunchPlan<11>> {
    ensure!(
        exact_attention_rows_supported(rows),
        "exact attention QG rows must be 4 or in 5..=32"
    );
    ensure!(
        n == 2 * num_heads * head_dim,
        "gated-Q output width mismatch"
    );
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "gated-Q K must be a positive multiple of 16"
    );
    Ok(ExactAttentionLaunchPlan {
        grid: [div_ceil(n, OUTS_PER_BLOCK), 1, 1],
        block: [256, 1, 1],
        args: [
            ExactAttentionKernelArg::Buffer(input),
            ExactAttentionKernelArg::Buffer(weight.weight),
            ExactAttentionKernelArg::Buffer(weight.weight_scale),
            ExactAttentionKernelArg::F32Bits(weight.weight_scale_2.to_bits()),
            ExactAttentionKernelArg::Buffer(output),
            ExactAttentionKernelArg::U32(rows),
            ExactAttentionKernelArg::U32(n),
            ExactAttentionKernelArg::U32(k),
            ExactAttentionKernelArg::U32(num_heads),
            ExactAttentionKernelArg::U32(head_dim),
            ExactAttentionKernelArg::U32(out_stride),
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn dual_kv_launch_plan(
    input: DevicePtr,
    k_weight: &QuantizedWeight,
    k_output: DevicePtr,
    v_weight: &QuantizedWeight,
    v_output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    out_stride: u32,
) -> Result<ExactAttentionLaunchPlan<13>> {
    ensure!(
        exact_attention_rows_supported(rows),
        "exact attention dual-KV rows must be 4 or in 5..=32"
    );
    ensure!(n > 0, "exact attention dual-KV width must be non-zero");
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "dual-KV K must be a positive multiple of 16"
    );
    Ok(ExactAttentionLaunchPlan {
        grid: [div_ceil(n, OUTS_PER_BLOCK), 1, 2],
        block: [256, 1, 1],
        args: [
            ExactAttentionKernelArg::Buffer(input),
            ExactAttentionKernelArg::Buffer(k_weight.weight),
            ExactAttentionKernelArg::Buffer(k_weight.weight_scale),
            ExactAttentionKernelArg::F32Bits(k_weight.weight_scale_2.to_bits()),
            ExactAttentionKernelArg::Buffer(k_output),
            ExactAttentionKernelArg::Buffer(v_weight.weight),
            ExactAttentionKernelArg::Buffer(v_weight.weight_scale),
            ExactAttentionKernelArg::F32Bits(v_weight.weight_scale_2.to_bits()),
            ExactAttentionKernelArg::Buffer(v_output),
            ExactAttentionKernelArg::U32(rows),
            ExactAttentionKernelArg::U32(n),
            ExactAttentionKernelArg::U32(k),
            ExactAttentionKernelArg::U32(out_stride),
        ],
    })
}

/// Exact gated-Q projection.
///
/// ABI: `(A, B_packed, B_scale, scale2, C, M, N, K, num_heads, head_dim,
/// out_stride)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_exact(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "missing exact attention QG kernel");
    qg_launch_plan(
        input, weight, output, rows, n, k, num_heads, head_dim, out_stride,
    )?
    .launch(gpu, kernel, stream)
}

/// Exact dual K/V projection.
///
/// ABI: `(A, K_packed, K_scale, K_scale2, K_out, V_packed, V_scale,
/// V_scale2, V_out, M, N, K, out_stride)` with `grid.z=2` selecting K
/// versus V.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_kv_exact(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    k_weight: &QuantizedWeight,
    k_output: DevicePtr,
    v_weight: &QuantizedWeight,
    v_output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(kernel.0 != 0, "missing exact attention dual-KV kernel");
    dual_kv_launch_plan(
        input, k_weight, k_output, v_weight, v_output, rows, n, k, out_stride,
    )?
    .launch(gpu, kernel, stream)
}

#[cfg(test)]
#[path = "gemv_exact_attention/static_tests.rs"]
mod static_tests;
