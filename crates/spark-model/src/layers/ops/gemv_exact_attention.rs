// SPDX-License-Identifier: AGPL-3.0-only

//! Exact K1-order multi-row NVFP4 attention projection kernels.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

const OUTS_PER_BLOCK: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactAttentionQkvRoute {
    ExactM4,
    SerialK1M4,
    ExactM17,
    SerialK1M17,
}

/// The gated-Q and dual-KV kernels form one atomic arithmetic route. If either
/// selected handle is absent, production must use the complete ordinary K1
/// path rather than mix exact multi-row and serial projections.
#[derive(Debug, Clone, Copy)]
pub struct W4a16ExactAttentionKernels {
    qg_m4: KernelHandle,
    dual_kv_m4: KernelHandle,
    qg_m17: KernelHandle,
    dual_kv_m17: KernelHandle,
}

impl W4a16ExactAttentionKernels {
    pub const fn new(qg_m17: KernelHandle, dual_kv_m17: KernelHandle) -> Self {
        Self {
            qg_m4: KernelHandle(0),
            dual_kv_m4: KernelHandle(0),
            qg_m17,
            dual_kv_m17,
        }
    }

    pub const fn with_m4(mut self, qg_m4: KernelHandle, dual_kv_m4: KernelHandle) -> Self {
        self.qg_m4 = qg_m4;
        self.dual_kv_m4 = dual_kv_m4;
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
            _ => KernelHandle(0),
        }
    }

    pub const fn dual_kv_for_rows(self, rows: usize) -> KernelHandle {
        match rows {
            4 => self.dual_kv_m4,
            5..=17 => self.dual_kv_m17,
            _ => KernelHandle(0),
        }
    }

    const fn complete_for_rows(self, rows: usize) -> bool {
        self.qg_for_rows(rows).0 != 0 && self.dual_kv_for_rows(rows).0 != 0
    }
}

/// Select the narrow exact path qualified for gated ordinary-NVFP4 attention.
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
        _ => None,
    }
}

const fn exact_attention_rows_supported(rows: u32) -> bool {
    rows == 4 || (rows >= 5 && rows <= 17)
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
    ensure!(
        exact_attention_rows_supported(rows),
        "exact attention QG rows must be 4 or in 9..=17"
    );
    ensure!(
        n == 2 * num_heads * head_dim,
        "gated-Q output width mismatch"
    );
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "gated-Q K must be a positive multiple of 16"
    );

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, OUTS_PER_BLOCK), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(out_stride)
        .launch(stream)
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
    ensure!(
        exact_attention_rows_supported(rows),
        "exact attention dual-KV rows must be 4 or in 9..=17"
    );
    ensure!(n > 0, "exact attention dual-KV width must be non-zero");
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "dual-KV K must be a positive multiple of 16"
    );

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, OUTS_PER_BLOCK), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(k_weight.weight)
        .arg_ptr(k_weight.weight_scale)
        .arg_f32(k_weight.weight_scale_2)
        .arg_ptr(k_output)
        .arg_ptr(v_weight.weight)
        .arg_ptr(v_weight.weight_scale)
        .arg_f32(v_weight.weight_scale_2)
        .arg_ptr(v_output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

#[cfg(test)]
#[path = "gemv_exact_attention/static_tests.rs"]
mod static_tests;
