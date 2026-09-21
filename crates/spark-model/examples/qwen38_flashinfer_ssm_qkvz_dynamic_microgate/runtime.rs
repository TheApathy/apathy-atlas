// SPDX-License-Identifier: AGPL-3.0-only

use super::buffers::*;
use super::evidence::{K, N};
use super::*;
use serde_json::{Value, json};
use spark_model::layers::ops::flashinfer_sm121::{
    BorrowedZeroWorkspaceFlashInferSm121, FlashInferSm121, FlashInferSm121Buffers,
    FlashInferSm121Shape,
};
use spark_model::layers::ops::nvfp4_dynamic_scale::{
    Nvfp4DynamicScaleBuffers, Nvfp4DynamicScaleKernels, Nvfp4DynamicScalePlan,
    enqueue_qwen38_ssm_qkvz_dynamic_scale, validate_qwen38_ssm_projection_dynamic_scale,
};
use spark_model::weight_map::QuantizedWeight;
use spark_model::weight_map::cutlass_scale_layout::{
    NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4,
};

#[rustfmt::skip]
fn dynamic_buffers(buffers: &CaseBuffers, replay: bool) -> Nvfp4DynamicScaleBuffers {
    let (packed, scales, maximum, scale2, status, alpha) = if replay {
        (&buffers.replay_packed, &buffers.replay_scales, &buffers.replay_maximum,
         &buffers.replay_scale2, &buffers.replay_status, &buffers.replay_alpha)
    } else {
        (&buffers.packed, &buffers.scales, &buffers.maximum,
         &buffers.scale2, &buffers.status, &buffers.alpha)
    };
    Nvfp4DynamicScaleBuffers {
        input_bf16: buffers.input.ptr(), packed_e2m1: packed.ptr(),
        scales_e4m3_128x4: scales.ptr(), global_max_f32: maximum.ptr(),
        scale2_a_f32: scale2.ptr(), status_u32: status.ptr(), combined_alpha_f32: alpha.ptr(),
    }
}

#[allow(clippy::too_many_arguments)]
#[rustfmt::skip]
fn candidate(
    gpu: &dyn GpuBackend,
    kernels: Nvfp4DynamicScaleKernels,
    prepared: &mut BorrowedZeroWorkspaceFlashInferSm121<'_>,
    buffers: &CaseBuffers,
    weight_scale2: f32,
    rows: usize,
    stream: u64,
    replay: bool,
) -> Result<()> {
    let dynamic = dynamic_buffers(buffers, replay);
    let output = enqueue_qwen38_ssm_qkvz_dynamic_scale(
        gpu, kernels, dynamic, rows as u32, K as u32, weight_scale2, stream)?;
    ensure!(output.for_flashinfer(stream + 1).is_err(), "dynamic result accepted wrong stream");
    let input = output.for_flashinfer(stream)?;
    let output_bf16 = if replay { buffers.replay.ptr() } else { buffers.candidate.ptr() };
    prepared.launch_eager(FlashInferSm121Buffers {
        output_bf16, activation_fp4: input.activation_fp4,
        weight_fp4: buffers.weight.ptr(), activation_scales: input.activation_scales,
        weight_scales: buffers.weight_scales.ptr(), global_scale_f32: input.global_scale_f32,
    }, stream)
}

#[rustfmt::skip]
fn parent(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    buffers: &CaseBuffers,
    weight_scale2: f32,
    rows: usize,
    stream: u64,
) -> Result<()> {
    let weight = QuantizedWeight { weight: buffers.weight_t.ptr(),
        weight_scale: buffers.weight_scales_t.ptr(), weight_scale_2: weight_scale2,
        input_scale: DevicePtr::NULL };
    ops::w4a16_gemm_n128_m128(
        gpu, kernel, buffers.input.ptr(), &weight, buffers.parent.ptr(),
        rows as u32, N as u32, K as u32, stream,
    )
}

#[rustfmt::skip]
fn timed<F: FnMut() -> Result<()>>(gpu: &dyn GpuBackend, stream: u64, mut launch: F) -> Result<f64> {
    gpu.synchronize(stream)?;
    let start = Instant::now();
    launch()?;
    gpu.synchronize(stream)?;
    Ok(start.elapsed().as_secs_f64() * 1_000.0)
}

#[rustfmt::skip]
pub(super) fn run_shape(
    gpu: &dyn GpuBackend,
    library: &FlashInferSm121,
    data: &ProjectionData,
    physical_weight_scales: &[u8],
    rows: usize,
    reps: usize,
    stream: u64,
) -> Result<Value> {
    let plan = Nvfp4DynamicScalePlan::qwen38_ssm_qkvz(rows as u32, K as u32)?;
    ensure!(reps >= 21 && reps.is_multiple_of(2), "timing reps must be balanced and >=21");
    let tactic = if rows == 2_079 { 4 } else { 2 };
    let shape = FlashInferSm121Shape::new(tactic, rows, N, K, 1)?;
    let mut prepared = library.prepare_borrowed_zero_workspace(gpu, shape, stream)?;
    ensure!(prepared.stream() == stream && prepared.shape() == shape, "prepared route drift");
    let buffers = CaseBuffers::new(gpu, stream, data, physical_weight_scales, plan)?;
    let kernels = Nvfp4DynamicScaleKernels::load(gpu)?;
    let parent_kernel = gpu.kernel("w4a16", "w4a16_gemm_t_m128")?;
    let static_quantizer = gpu.kernel("quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4")?;
    ensure!(parent_kernel.0 != 0 && static_quantizer.0 != 0, "raw gate kernel handle missing");
    let admitted = validate_qwen38_ssm_projection_dynamic_scale(
        kernels, dynamic_buffers(&buffers, false), rows as u32, K as u32, data.weight_scale2)?;
    ensure!(admitted == plan, "dynamic preflight plan drift");

    parent(gpu, parent_kernel, &buffers, data.weight_scale2, rows, stream)?;
    candidate(gpu, kernels, &mut prepared, &buffers, data.weight_scale2, rows, stream, false)?;
    candidate(gpu, kernels, &mut prepared, &buffers, data.weight_scale2, rows, stream, true)?;
    gpu.synchronize(stream)?;
    let maximum = scalar_f32(&buffers.maximum, gpu, "maximum")?;
    ensure!(maximum.is_finite() && maximum > 0.0, "invalid observed maximum");
    let host_scale2 = maximum / 2_688.0;
    ops::quantize_bf16_to_nvfp4_atlas_128x4(gpu, static_quantizer,
        buffers.input.ptr(), buffers.reference_packed.ptr(), buffers.reference_scales.ptr(),
        host_scale2, rows as u32, K as u32, stream)?;
    gpu.synchronize(stream)?;

    let packed = buffers.packed.payload(gpu, "packed")?;
    let scales = buffers.scales.payload(gpu, "scales")?;
    ensure!(packed == buffers.replay_packed.payload(gpu, "replay-packed")?, "packed replay differs");
    ensure!(scales == buffers.replay_scales.payload(gpu, "replay-scales")?, "scale replay differs");
    ensure!(packed == buffers.reference_packed.payload(gpu, "reference-packed")?, "packed reference differs");
    ensure!(scales == buffers.reference_scales.payload(gpu, "reference-scales")?, "scale reference differs");
    for (lhs, rhs, label) in [
        (&buffers.maximum, &buffers.replay_maximum, "maximum"),
        (&buffers.scale2, &buffers.replay_scale2, "scale2"),
        (&buffers.status, &buffers.replay_status, "status"),
        (&buffers.alpha, &buffers.replay_alpha, "alpha"),
    ] {
        ensure!(lhs.payload(gpu, label)? == rhs.payload(gpu, label)?, "{label} replay differs");
    }
    ensure!(scalar_u32(&buffers.status, gpu, "status")? == 0, "dynamic status was nonzero");
    ensure!(scalar_f32(&buffers.scale2, gpu, "scale2")?.to_bits() == host_scale2.to_bits(), "scale2 bits differ");
    let host_alpha = host_scale2 * data.weight_scale2;
    ensure!(scalar_f32(&buffers.alpha, gpu, "alpha")?.to_bits() == host_alpha.to_bits(), "alpha bits differ");
    let logical = deinterleave_nvfp4_scales_128x4(&scales,
        &[plan.padded_rows as usize, K / NVFP4_GROUP_SIZE], NVFP4_GROUP_SIZE)?;
    let tail_offset = rows * K / NVFP4_GROUP_SIZE;
    ensure!(logical[tail_offset..].iter().all(|byte| *byte == 0), "physical scale tail is not zero");
    let parent_bytes = buffers.parent.payload(gpu, "parent")?;
    let candidate_bytes = buffers.candidate.payload(gpu, "candidate")?;
    ensure!(candidate_bytes == buffers.replay.payload(gpu, "replay-output")?, "output replay differs");
    let quality = metrics(&parent_bytes, &candidate_bytes)?;
    buffers.check_inputs(gpu)?;

    let _ = timed(gpu, stream, || parent(gpu, parent_kernel, &buffers, data.weight_scale2, rows, stream))?;
    let _ = timed(gpu, stream, || candidate(gpu, kernels, &mut prepared, &buffers, data.weight_scale2, rows, stream, false))?;
    let (mut parent_ms, mut candidate_ms, mut deltas) = (Vec::new(), Vec::new(), Vec::new());
    for index in 0..reps {
        let (p, c) = if index.is_multiple_of(2) {
            let p = timed(gpu, stream, || parent(gpu, parent_kernel, &buffers, data.weight_scale2, rows, stream))?;
            let c = timed(gpu, stream, || candidate(gpu, kernels, &mut prepared, &buffers, data.weight_scale2, rows, stream, false))?;
            (p, c)
        } else {
            let c = timed(gpu, stream, || candidate(gpu, kernels, &mut prepared, &buffers, data.weight_scale2, rows, stream, false))?;
            let p = timed(gpu, stream, || parent(gpu, parent_kernel, &buffers, data.weight_scale2, rows, stream))?;
            (p, c)
        };
        parent_ms.push(p);
        candidate_ms.push(c);
        deltas.push(p - c);
    }
    let (pm, pp, cm, cp, dm) = (median(&parent_ms), p90(&parent_ms), median(&candidate_ms), p90(&candidate_ms), median(&deltas));
    ensure!(cm < pm && cp < pp && dm > 0.0, "candidate failed median/p90/paired improvement");
    let post_parent = buffers.parent.payload(gpu, "post-parent")?;
    let post_candidate = buffers.candidate.payload(gpu, "post-candidate")?;
    ensure!(buffers.packed.payload(gpu, "post-packed")? == packed, "timed packed drift");
    ensure!(buffers.scales.payload(gpu, "post-scales")? == scales, "timed scales drift");
    ensure!(scalar_u32(&buffers.status, gpu, "post-status")? == 0, "timed status drift");
    ensure!(scalar_f32(&buffers.scale2, gpu, "post-scale2")?.to_bits() == host_scale2.to_bits(), "timed scale2 drift");
    ensure!(scalar_f32(&buffers.alpha, gpu, "post-alpha")?.to_bits() == host_alpha.to_bits(), "timed alpha drift");
    let post_quality = metrics(&post_parent, &post_candidate)?;
    buffers.check_inputs(gpu)?;
    let receipt = json!({
        "m":rows,"n":N,"k":K,"tactic":tactic,"reps_per_arm":reps,"workspace_bytes":0,
        "input":{"layout":"row-major-bf16","row_stride_elements":K,"immutable":true},
        "weight":{"contract":"checkpoint-merged-qkv-plus-z-generated-requantized","immutable":true},
        "extents":{"packed_bytes":plan.packed_bytes,"physical_scale_bytes":plan.physical_scale_bytes,
            "padded_rows":plan.padded_rows,"zero_tail_bytes":logical.len()-tail_offset},
        "exact":{"packed_sha256":sha256_bytes(&packed)?,"physical_scales_sha256":sha256_bytes(&scales)?,
            "scale2_bits":host_scale2.to_bits(),"alpha_bits":host_alpha.to_bits(),"status":0,"replay":true},
        "output":{"parent_sha256":sha256_bytes(&post_parent)?,"candidate_sha256":sha256_bytes(&post_candidate)?,
            "cosine":post_quality.cosine,"relative_rms":post_quality.relative_rms,"differing_bf16":post_quality.differing,
            "initial_cosine":quality.cosine,"initial_relative_rms":quality.relative_rms},
        "timing_ms":{"parent_median":pm,"parent_p90":pp,"candidate_median":cm,"candidate_p90":cp,"paired_delta_median":dm},
        "guards":"PASS","input_immutability":"PASS","same_nondefault_stream":stream != 0,
    });
    buffers.free(gpu)?;
    Ok(receipt)
}
