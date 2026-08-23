// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_model::layers::ops::W4a16ExactFfnKernels;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) serial_dual: KernelHandle,
    pub(crate) serial_silu: KernelHandle,
    pub(crate) exact: W4a16ExactFfnKernels,
}

pub(crate) fn load_kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    let module = "w4a16_gemv_fused";
    let rt = "w4a16_gemv_fused_rt";
    let exact = W4a16ExactFfnKernels::new(
        gpu.kernel(module, "w4a16_gemv_dual_exact_m4")?,
        gpu.kernel(module, "w4a16_gemv_dual_exact_m8")?,
        gpu.kernel(module, "w4a16_gemv_dual_exact_m17")?,
        gpu.kernel(module, "w4a16_gemv_dual_exact_m32")?,
        gpu.kernel(module, "w4a16_gemv_silu_input_exact_m4")?,
        gpu.kernel(module, "w4a16_gemv_silu_input_exact_m8")?,
        gpu.kernel(module, "w4a16_gemv_silu_input_exact_m17")?,
        gpu.kernel(module, "w4a16_gemv_silu_input_exact_m32")?,
    )
    .with_materialized_m8(
        gpu.kernel(module, "w4a16_gemv_dual_silu_f32_exact_m8")?,
        gpu.kernel(module, "w4a16_gemv_f32_input_exact_m8")?,
    )
    .with_materialized_m17(
        gpu.kernel(module, "w4a16_gemv_dual_silu_f32_exact_m17")?,
        gpu.kernel(module, "w4a16_gemv_f32_input_exact_m17")?,
    )
    .with_fused_materialized_m17(gpu.kernel(module, "w4a16_gemv_dual_exact_materialize_f32_m17")?)
    .with_rt2(
        gpu.kernel(rt, "w4a16_gemv_dual_exact_materialize_f32_rt2_m17")?,
        gpu.kernel(rt, "w4a16_gemv_f32_input_exact_rt2_m8")?,
        gpu.kernel(rt, "w4a16_gemv_f32_input_exact_rt2_m17")?,
    );
    Ok(Kernels {
        serial_dual: gpu.kernel(module, "w4a16_gemv_dual")?,
        serial_silu: gpu.kernel(module, "w4a16_gemv_silu_input")?,
        exact,
    })
}
