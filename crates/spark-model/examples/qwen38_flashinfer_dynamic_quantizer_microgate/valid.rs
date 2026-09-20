// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use spark_model::layers::ops;
use spark_model::layers::ops::nvfp4_dynamic_scale::{
    Nvfp4DynamicScaleBuffers, Nvfp4DynamicScaleKernels, Nvfp4DynamicScalePlan,
    enqueue_qwen38_ssm_projection_dynamic_scale,
};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::contract::{Fixture, WEIGHT_SCALE_2};
use super::fixtures::{f32_payload, fnv1a64, input_bytes, require_exact, u32_payload};
use super::guarded::Guarded;
use super::launch::launch_static_reference;

pub(super) fn valid_case(
    gpu: &dyn GpuBackend,
    kernels: Nvfp4DynamicScaleKernels,
    static_quantizer: KernelHandle,
    rows: u32,
    cols: u32,
    fixture: Fixture,
    stream: u64,
) -> Result<()> {
    let plan = Nvfp4DynamicScalePlan::qwen38_ssm_projection(rows, cols)?;
    let input = Guarded::input(gpu, input_bytes(rows, cols, fixture), 0x11)?;
    let reference_max = Guarded::output(gpu, 4, 0x12)?;
    let reference_packed = Guarded::output(gpu, plan.packed_bytes, 0x21)?;
    let reference_scales = Guarded::output(gpu, plan.physical_scale_bytes, 0x22)?;
    let candidate_packed = Guarded::output(gpu, plan.packed_bytes, 0x31)?;
    let candidate_scales = Guarded::output(gpu, plan.physical_scale_bytes, 0x32)?;
    let candidate_max = Guarded::output(gpu, 4, 0x33)?;
    let candidate_scale2 = Guarded::output(gpu, 4, 0x34)?;
    let candidate_status = Guarded::output(gpu, 4, 0x35)?;
    let candidate_alpha = Guarded::output(gpu, 4, 0x36)?;

    gpu.memset_async(reference_max.ptr(), 0, 4, stream)?;
    ops::nvfp4_global_absmax(
        gpu,
        kernels.absmax,
        input.ptr(),
        reference_max.ptr(),
        plan.total_elements,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let maximum = f32_payload(&reference_max, gpu, "reference-max")?;
    ensure!(
        maximum.is_finite() && maximum >= 0.0,
        "invalid reference maximum"
    );
    let host_scale2 = if maximum > 0.0 {
        maximum / (6.0 * 448.0)
    } else {
        1.0
    };
    launch_static_reference(
        gpu,
        static_quantizer,
        input.ptr(),
        reference_packed.ptr(),
        reference_scales.ptr(),
        host_scale2,
        plan,
        stream,
    )?;

    let buffers = Nvfp4DynamicScaleBuffers {
        input_bf16: input.ptr(),
        packed_e2m1: candidate_packed.ptr(),
        scales_e4m3_128x4: candidate_scales.ptr(),
        global_max_f32: candidate_max.ptr(),
        scale2_a_f32: candidate_scale2.ptr(),
        status_u32: candidate_status.ptr(),
        combined_alpha_f32: candidate_alpha.ptr(),
    };
    let output = enqueue_qwen38_ssm_projection_dynamic_scale(
        gpu,
        kernels,
        buffers,
        rows,
        cols,
        WEIGHT_SCALE_2,
        stream,
    )?;
    ensure!(
        output.for_flashinfer(stream + 1).is_err(),
        "cross-stream output accepted"
    );
    let flashinfer = output.for_flashinfer(stream)?;
    ensure!(
        flashinfer.activation_fp4 == candidate_packed.ptr(),
        "packed pointer changed"
    );
    ensure!(
        flashinfer.activation_scales == candidate_scales.ptr(),
        "scale pointer changed"
    );
    ensure!(
        flashinfer.global_scale_f32 == candidate_alpha.ptr(),
        "alpha pointer changed"
    );
    ensure!(
        output.scale2_a_f32() == candidate_scale2.ptr(),
        "scale2 pointer changed"
    );
    ensure!(
        output.status_u32() == candidate_status.ptr(),
        "status pointer changed"
    );
    gpu.synchronize(stream)?;

    let reference_packed_bytes = reference_packed.payload(gpu, "reference-packed")?;
    let reference_scale_bytes = reference_scales.payload(gpu, "reference-scales")?;
    let first_packed = candidate_packed.payload(gpu, "candidate-packed")?;
    let first_scales = candidate_scales.payload(gpu, "candidate-scales")?;
    require_exact("packed E2M1", &reference_packed_bytes, &first_packed)?;
    require_exact("physical 128x4 E4M3", &reference_scale_bytes, &first_scales)?;
    ensure!(
        f32_payload(&candidate_max, gpu, "candidate-max")?.to_bits() == maximum.to_bits(),
        "device absmax bits differ"
    );
    ensure!(
        f32_payload(&candidate_scale2, gpu, "candidate-scale2")?.to_bits() == host_scale2.to_bits(),
        "device scale2 bits differ: host={:08x} device={:08x}",
        host_scale2.to_bits(),
        f32_payload(&candidate_scale2, gpu, "candidate-scale2-repeat")?.to_bits()
    );
    let host_alpha = host_scale2 * WEIGHT_SCALE_2;
    ensure!(
        f32_payload(&candidate_alpha, gpu, "candidate-alpha")?.to_bits() == host_alpha.to_bits(),
        "device combined alpha bits differ"
    );
    ensure!(
        u32_payload(&candidate_status, gpu, "candidate-status")? == 0,
        "valid status set"
    );
    ensure!(
        first_scales.iter().all(|byte| byte & 0x7f != 0x7f),
        "candidate emitted E4M3 NaN"
    );

    let _ = enqueue_qwen38_ssm_projection_dynamic_scale(
        gpu,
        kernels,
        buffers,
        rows,
        cols,
        WEIGHT_SCALE_2,
        stream,
    )?;
    gpu.synchronize(stream)?;
    require_exact(
        "deterministic packed",
        &first_packed,
        &candidate_packed.payload(gpu, "candidate-packed-second")?,
    )?;
    require_exact(
        "deterministic scales",
        &first_scales,
        &candidate_scales.payload(gpu, "candidate-scales-second")?,
    )?;
    input.check_immutable(gpu, "BF16 input")?;

    println!(
        "PARITY fixture={fixture:?} M={rows} K={cols} max_bits={:08x} scale2_bits={:08x} alpha_bits={:08x} packed_hash=fnv1a64:{:016x} scale_hash=fnv1a64:{:016x} packed=EXACT physical_scale=EXACT padding=EXACT redzones=PASS immutable=PASS deterministic=PASS status=0",
        maximum.to_bits(),
        host_scale2.to_bits(),
        host_alpha.to_bits(),
        fnv1a64(&first_packed),
        fnv1a64(&first_scales),
    );

    for buffer in [
        &input,
        &reference_max,
        &reference_packed,
        &reference_scales,
        &candidate_packed,
        &candidate_scales,
        &candidate_max,
        &candidate_scale2,
        &candidate_status,
        &candidate_alpha,
    ] {
        buffer.free(gpu)?;
    }
    Ok(())
}
