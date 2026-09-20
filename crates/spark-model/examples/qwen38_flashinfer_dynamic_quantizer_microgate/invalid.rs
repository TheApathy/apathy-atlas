// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail, ensure};
use spark_model::layers::ops::nvfp4_dynamic_scale::{
    Nvfp4DynamicScaleBuffers, Nvfp4DynamicScaleKernels, Nvfp4DynamicScalePlan,
    enqueue_qwen38_ssm_projection_dynamic_scale,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

use super::contract::WEIGHT_SCALE_2;
use super::fixtures::{invalid_input, u32_payload};
use super::guarded::Guarded;
use super::launch::launch_quantizer_without_alpha_guard;
use super::provenance::exact_bundle;

pub(super) fn invalid_status_case(
    gpu: &dyn GpuBackend,
    kernels: Nvfp4DynamicScaleKernels,
    kind: &str,
    cols: u32,
    stream: u64,
) -> Result<()> {
    let plan = Nvfp4DynamicScalePlan::qwen38_ssm_projection(2_079, cols)?;
    let input = Guarded::input(gpu, invalid_input(2_079, cols, kind)?, 0x41)?;
    let packed = Guarded::output(gpu, plan.packed_bytes, 0x42)?;
    let scales = Guarded::output(gpu, plan.physical_scale_bytes, 0x43)?;
    let maximum = Guarded::output(gpu, 4, 0x44)?;
    let scale2 = Guarded::output(gpu, 4, 0x45)?;
    let status = Guarded::output(gpu, 4, 0x46)?;
    let alpha = Guarded::output(gpu, 4, 0x47)?;
    let buffers = Nvfp4DynamicScaleBuffers {
        input_bf16: input.ptr(),
        packed_e2m1: packed.ptr(),
        scales_e4m3_128x4: scales.ptr(),
        global_max_f32: maximum.ptr(),
        scale2_a_f32: scale2.ptr(),
        status_u32: status.ptr(),
        combined_alpha_f32: alpha.ptr(),
    };
    launch_quantizer_without_alpha_guard(gpu, kernels, buffers, plan, stream)?;
    gpu.synchronize(stream)?;
    let status_value = u32_payload(&status, gpu, "invalid-status")?;
    ensure!(
        status_value != 0,
        "{kind}: nonfinite input was not recorded"
    );
    ensure!(
        packed.payload(gpu, "invalid-packed")?[..8]
            .iter()
            .all(|byte| *byte == 0),
        "{kind}: invalid first group packed bytes are not deterministic zero"
    );
    let _ = scales.payload(gpu, "invalid-scales")?;
    let _ = maximum.payload(gpu, "invalid-maximum")?;
    let _ = scale2.payload(gpu, "invalid-scale2")?;
    let _ = alpha.payload(gpu, "invalid-alpha")?;
    input.check_immutable(gpu, "invalid BF16 input")?;
    println!(
        "INVALID_STATUS kind={kind} K={cols} status=0x{status_value:08x} invalid_group_zero=PASS redzones=PASS immutable=PASS"
    );
    for buffer in [&input, &packed, &scales, &maximum, &scale2, &status, &alpha] {
        buffer.free(gpu)?;
    }
    Ok(())
}

pub(super) fn invalid_trap_child(kind: &str, cols: u32) -> Result<()> {
    let modules = exact_bundle()?;
    let backend = AtlasCudaBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let kernels = Nvfp4DynamicScaleKernels::load(gpu)?;
    let plan = Nvfp4DynamicScalePlan::qwen38_ssm_projection(2_079, cols)?;
    let input = Guarded::input(gpu, invalid_input(2_079, cols, kind)?, 0x51)?;
    let packed = Guarded::output(gpu, plan.packed_bytes, 0x52)?;
    let scales = Guarded::output(gpu, plan.physical_scale_bytes, 0x53)?;
    let maximum = Guarded::output(gpu, 4, 0x54)?;
    let scale2 = Guarded::output(gpu, 4, 0x55)?;
    let status = Guarded::output(gpu, 4, 0x56)?;
    let alpha = Guarded::output(gpu, 4, 0x57)?;
    let result = enqueue_qwen38_ssm_projection_dynamic_scale(
        gpu,
        kernels,
        Nvfp4DynamicScaleBuffers {
            input_bf16: input.ptr(),
            packed_e2m1: packed.ptr(),
            scales_e4m3_128x4: scales.ptr(),
            global_max_f32: maximum.ptr(),
            scale2_a_f32: scale2.ptr(),
            status_u32: status.ptr(),
            combined_alpha_f32: alpha.ptr(),
        },
        2_079,
        cols,
        WEIGHT_SCALE_2,
        stream,
    );
    match result.and_then(|_| gpu.synchronize(stream)) {
        Err(error) => {
            println!("INVALID_TRAP kind={kind} K={cols} observed=PASS error={error:#}");
            Ok(())
        }
        Ok(()) => bail!("{kind}: invalid device data reached synchronization without error"),
    }
}
