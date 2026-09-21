// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_model::layers::ops;
use spark_model::layers::ops::nvfp4_dynamic_scale::{
    Nvfp4DynamicScaleBuffers, Nvfp4DynamicScaleKernels, Nvfp4DynamicScalePlan,
};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_static_reference(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    packed: DevicePtr,
    scales: DevicePtr,
    scale2: f32,
    plan: Nvfp4DynamicScalePlan,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([plan.padded_rows.min(96), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(packed)
        .arg_ptr(scales)
        .arg_f32(scale2)
        .arg_u32(plan.rows)
        .arg_u32(plan.cols)
        .launch(stream)
}

pub(super) fn launch_quantizer_without_alpha_guard(
    gpu: &dyn GpuBackend,
    kernels: Nvfp4DynamicScaleKernels,
    buffers: Nvfp4DynamicScaleBuffers,
    plan: Nvfp4DynamicScalePlan,
    stream: u64,
) -> Result<()> {
    gpu.memset_async(buffers.global_max_f32, 0, 4, stream)?;
    gpu.memset_async(buffers.status_u32, 0, 4, stream)?;
    ops::nvfp4_global_absmax(
        gpu,
        kernels.absmax,
        buffers.input_bf16,
        buffers.global_max_f32,
        plan.total_elements,
        stream,
    )?;
    KernelLaunch::new(gpu, kernels.quantize_from_absmax)
        .grid([plan.padded_rows.min(96), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(buffers.input_bf16)
        .arg_ptr(buffers.packed_e2m1)
        .arg_ptr(buffers.scales_e4m3_128x4)
        .arg_ptr(buffers.global_max_f32)
        .arg_ptr(buffers.scale2_a_f32)
        .arg_ptr(buffers.status_u32)
        .arg_u32(plan.rows)
        .arg_u32(plan.cols)
        .launch(stream)
}
