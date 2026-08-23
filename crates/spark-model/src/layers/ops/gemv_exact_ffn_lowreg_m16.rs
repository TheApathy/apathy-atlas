// SPDX-License-Identifier: AGPL-3.0-only

//! Low-register exact M16 gate/up projection route.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

const OUTS_PER_BLOCK: u32 = 4;
const POINTWISE_THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactFfnLowregM16Route {
    Lowreg,
    ExistingSplitM8,
}

#[derive(Debug, Clone, Copy)]
pub struct W4a16ExactFfnLowregM16Kernels {
    gate: KernelHandle,
    up: KernelHandle,
    materialize: KernelHandle,
}

impl W4a16ExactFfnLowregM16Kernels {
    pub const fn new(gate: KernelHandle, up: KernelHandle, materialize: KernelHandle) -> Self {
        Self {
            gate,
            up,
            materialize,
        }
    }

    pub const fn complete(self) -> bool {
        self.gate.0 != 0 && self.up.0 != 0 && self.materialize.0 != 0
    }
}

pub const fn exact_ffn_lowreg_m16_route(
    rows: u32,
    enabled: bool,
    kernels: W4a16ExactFfnLowregM16Kernels,
) -> Option<ExactFfnLowregM16Route> {
    if rows != 16 || !enabled {
        return None;
    }
    Some(if kernels.complete() {
        ExactFfnLowregM16Route::Lowreg
    } else {
        ExactFfnLowregM16Route::ExistingSplitM8
    })
}

#[allow(clippy::too_many_arguments)]
fn launch_projection(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "missing exact FFN low-register M16 projection"
    );
    ensure!(
        rows == 16,
        "low-register exact FFN requires exactly 16 rows"
    );
    ensure!(
        n > 0,
        "low-register exact FFN output width must be non-zero"
    );
    ensure!(
        k > 0 && k.is_multiple_of(16),
        "low-register exact FFN K must be a positive multiple of 16"
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
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_gate_exact_m16_lowreg(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnLowregM16Kernels,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    launch_projection(gpu, kernels.gate, input, weight, output, rows, n, k, stream)
}

#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_up_exact_m16_lowreg(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnLowregM16Kernels,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    launch_projection(gpu, kernels.up, input, weight, output, rows, n, k, stream)
}

pub fn w4a16_gate_up_materialize_f32_m16(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactFfnLowregM16Kernels,
    gate: DevicePtr,
    up: DevicePtr,
    activation: DevicePtr,
    rows: u32,
    n: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernels.materialize.0 != 0,
        "missing exact FFN M16 materializer"
    );
    ensure!(rows == 16, "exact FFN M16 materializer requires 16 rows");
    ensure!(n > 0, "exact FFN M16 materializer width must be non-zero");
    let elements = rows
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("exact FFN M16 materializer shape overflow"))?;
    KernelLaunch::new(gpu, kernels.materialize)
        .grid([div_ceil(elements, POINTWISE_THREADS), 1, 1])
        .block([POINTWISE_THREADS, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(activation)
        .arg_u32(rows)
        .arg_u32(n)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::KernelHandle;

    use super::{
        ExactFfnLowregM16Route, W4a16ExactFfnLowregM16Kernels, exact_ffn_lowreg_m16_route,
    };

    fn kernels(gate: u64, up: u64, materialize: u64) -> W4a16ExactFfnLowregM16Kernels {
        W4a16ExactFfnLowregM16Kernels::new(
            KernelHandle(gate),
            KernelHandle(up),
            KernelHandle(materialize),
        )
    }

    #[test]
    fn selected_m16_route_requires_every_handle() {
        assert_eq!(
            exact_ffn_lowreg_m16_route(16, true, kernels(1, 2, 3)),
            Some(ExactFfnLowregM16Route::Lowreg)
        );
        for incomplete in [kernels(0, 2, 3), kernels(1, 0, 3), kernels(1, 2, 0)] {
            assert_eq!(
                exact_ffn_lowreg_m16_route(16, true, incomplete),
                Some(ExactFfnLowregM16Route::ExistingSplitM8)
            );
        }
    }

    #[test]
    fn route_is_narrow_to_enabled_physical_m16() {
        let complete = kernels(1, 2, 3);
        assert_eq!(exact_ffn_lowreg_m16_route(16, false, complete), None);
        for rows in [8, 15, 17, 32] {
            assert_eq!(exact_ffn_lowreg_m16_route(rows, true, complete), None);
        }
    }

    #[test]
    fn cuda_source_pins_k8_order_bf16_boundary_and_symbols() {
        let source =
            include_str!("../../../../../kernels/gb10/common/w4a16_gemv_exact_ffn_lowreg_m16.cu");
        assert!(source.contains("w4a16_gemv_gate_exact_m16_lowreg"));
        assert!(source.contains("w4a16_gemv_up_exact_m16_lowreg"));
        assert!(source.contains("w4a16_gate_up_materialize_f32_m16"));
        assert!(source.contains("__launch_bounds__(256, 4)"));
        assert!(source.contains("k8 = lane; k8 < K8; k8 += 64u"));
        assert!(source.contains("acc[row] += __bfloat162float(a_lo) * w_lo[b];"));
        assert!(source.contains("acc[row] += __bfloat162float(a_hi) * w_hi[b];"));
        assert!(source.contains("__float2bfloat16(smem[base] + smem[base + 1])"));
        assert!(source.contains("const __nv_bfloat16 gate_bf16 = gate[idx]"));
        assert!(source.contains("const __nv_bfloat16 up_bf16 = up[idx]"));
        assert!(source.contains("(gate_f32 / (1.0f + __expf(-gate_f32))) * up_f32"));
    }

    #[test]
    fn dense_ffn_integration_is_env_gated_and_falls_through_existing_split_m8() {
        let source = include_str!("../dense_ffn.rs");
        assert!(source.contains("ATLAS_EXACT_FFN_LOWREG_GATE_UP_M16"));
        assert!(source.contains("exact_ffn_lowreg_m16_route("));
        assert!(source.contains("w4a16_gemv_gate_exact_m16_lowreg("));
        assert!(source.contains("w4a16_gemv_up_exact_m16_lowreg("));
        assert!(source.contains("w4a16_gate_up_materialize_f32_m16("));
        assert!(source.contains("forward_kgamma_exact_split_m8_down("));
    }
}
