// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use serde_json::{Value, json};
use spark_model::layers::ops;
use spark_model::weight_map::cutlass_scale_layout::{
    NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4,
};
use spark_runtime::gpu::GpuBackend;

use super::guarded::{Guarded, input};
use super::launch::{equal, hash, launch};
use super::timing::measure_balanced;
use super::{BinaryIdentity, BundleIdentity, DOWN_N, K, Kernels, Plan, SCALE2, SCHEMA};

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    plan: Plan,
    weight: &Guarded,
    weight_scales: &Guarded,
    nonce: &str,
    binary: &BinaryIdentity,
    bundle: &BundleIdentity,
    sources: &[(&str, &str)],
) -> Result<Value> {
    let gate = Guarded::input(gpu, input(plan.elements, false), 1)?;
    let up = Guarded::input(gpu, input(plan.elements, true), 2)?;
    ensure!(gate.ptr().0 != up.ptr().0, "gate/up allocations alias");
    let temporary = Guarded::output(gpu, plan.elements * 2, 3)?;
    let parent_packed = Guarded::output(gpu, plan.packed, 4)?;
    let parent_scales = Guarded::output(gpu, plan.scales, 5)?;
    let candidate_packed = Guarded::output(gpu, plan.packed, 6)?;
    let candidate_scales = Guarded::output(gpu, plan.scales, 7)?;
    let live = [
        &gate,
        &up,
        &temporary,
        &parent_packed,
        &parent_scales,
        &candidate_packed,
        &candidate_scales,
        weight,
        weight_scales,
    ];
    for (index, left) in live.iter().enumerate() {
        let left = left.span()?;
        for right in &live[index + 1..] {
            let right = right.span()?;
            ensure!(
                left.1 <= right.0 || right.1 <= left.0,
                "live device allocations overlap"
            );
        }
    }
    launch(
        false,
        gpu,
        stream,
        kernels,
        plan,
        gate.ptr(),
        up.ptr(),
        temporary.ptr(),
        parent_packed.ptr(),
        parent_scales.ptr(),
    )?;
    launch(
        true,
        gpu,
        stream,
        kernels,
        plan,
        gate.ptr(),
        up.ptr(),
        temporary.ptr(),
        candidate_packed.ptr(),
        candidate_scales.ptr(),
    )?;
    gpu.synchronize(stream)?;
    let parent_packed_bytes = parent_packed.read(gpu, "parent-packed")?;
    let parent_scale_bytes = parent_scales.read(gpu, "parent-scales")?;
    let candidate_packed_bytes = candidate_packed.read(gpu, "candidate-packed-1")?;
    let candidate_scale_bytes = candidate_scales.read(gpu, "candidate-scales-1")?;
    equal("packed", &parent_packed_bytes, &candidate_packed_bytes)?;
    equal(
        "full physical scales",
        &parent_scale_bytes,
        &candidate_scale_bytes,
    )?;
    launch(
        true,
        gpu,
        stream,
        kernels,
        plan,
        gate.ptr(),
        up.ptr(),
        temporary.ptr(),
        candidate_packed.ptr(),
        candidate_scales.ptr(),
    )?;
    gpu.synchronize(stream)?;
    equal(
        "candidate packed determinism",
        &candidate_packed_bytes,
        &candidate_packed.read(gpu, "candidate-packed-2")?,
    )?;
    equal(
        "candidate scale determinism",
        &candidate_scale_bytes,
        &candidate_scales.read(gpu, "candidate-scales-2")?,
    )?;
    let parent_logical = deinterleave_nvfp4_scales_128x4(
        &parent_scale_bytes,
        &[plan.padded as usize, (K / 16) as usize],
        NVFP4_GROUP_SIZE,
    )?;
    let candidate_logical = deinterleave_nvfp4_scales_128x4(
        &candidate_scale_bytes,
        &[plan.padded as usize, (K / 16) as usize],
        NVFP4_GROUP_SIZE,
    )?;
    ensure!(
        parent_logical[plan.m as usize * (K / 16) as usize..].len() == plan.tail
            && parent_logical[plan.m as usize * (K / 16) as usize..]
                .iter()
                .all(|byte| *byte == 0),
        "physical padded tail is not exactly zero"
    );
    let logical = plan.m as usize * (K / 16) as usize;
    let parent_logical_scales = Guarded::input(gpu, parent_logical[..logical].to_vec(), 8)?;
    let candidate_logical_scales = Guarded::input(gpu, candidate_logical[..logical].to_vec(), 9)?;
    let parent_output = Guarded::output(gpu, plan.m as usize * DOWN_N as usize * 2, 10)?;
    let candidate_output = Guarded::output(gpu, plan.m as usize * DOWN_N as usize * 2, 11)?;
    for (scales, output) in [
        (&parent_logical_scales, &parent_output),
        (&candidate_logical_scales, &candidate_output),
    ] {
        ops::nvfp4_nvfp4_gemm_kmajor_m256(
            gpu,
            kernels.down,
            if std::ptr::eq(scales, &parent_logical_scales) {
                parent_packed.ptr()
            } else {
                candidate_packed.ptr()
            },
            scales.ptr(),
            weight.ptr(),
            weight_scales.ptr(),
            SCALE2,
            output.ptr(),
            plan.m,
            DOWN_N,
            K,
            stream,
        )?;
    }
    gpu.synchronize(stream)?;
    let down = parent_output.read(gpu, "parent-down")?;
    equal(
        "synthetic down BF16",
        &down,
        &candidate_output.read(gpu, "candidate-down")?,
    )?;
    let timing = measure_balanced(
        gpu,
        stream,
        kernels,
        plan,
        gate.ptr(),
        up.ptr(),
        temporary.ptr(),
        parent_packed.ptr(),
        parent_scales.ptr(),
        candidate_packed.ptr(),
        candidate_scales.ptr(),
    )?;
    equal(
        "post-timing parent packed",
        &parent_packed_bytes,
        &parent_packed.read(gpu, "timed-parent-packed")?,
    )?;
    equal(
        "post-timing parent scales",
        &parent_scale_bytes,
        &parent_scales.read(gpu, "timed-parent-scales")?,
    )?;
    equal(
        "post-timing candidate packed",
        &candidate_packed_bytes,
        &candidate_packed.read(gpu, "timed-candidate-packed")?,
    )?;
    equal(
        "post-timing candidate scales",
        &candidate_scale_bytes,
        &candidate_scales.read(gpu, "timed-candidate-scales")?,
    )?;
    let _ = temporary.read(gpu, "timed-parent-silu")?;
    gate.immutable(gpu, "gate")?;
    up.immutable(gpu, "up")?;
    parent_logical_scales.immutable(gpu, "parent-logical-scales")?;
    candidate_logical_scales.immutable(gpu, "candidate-logical-scales")?;
    weight.immutable(gpu, "down-weight")?;
    weight_scales.immutable(gpu, "down-weight-scales")?;
    let receipt = json!({"schema":SCHEMA,"status":"qualified","nonce":nonce,"m":plan.m,"k":K,"padded_m":plan.padded,"reps":plan.reps,"stream":stream,"scale2_bits":SCALE2.to_bits(),"gate_input_hash":format!("fnv1a64:{:016x}",hash(gate.original.as_ref().unwrap())),"up_input_hash":format!("fnv1a64:{:016x}",hash(up.original.as_ref().unwrap())),"down_weight_hash":format!("fnv1a64:{:016x}",hash(weight.original.as_ref().unwrap())),"down_weight_scale_hash":format!("fnv1a64:{:016x}",hash(weight_scales.original.as_ref().unwrap())),"packed_hash":format!("fnv1a64:{:016x}",hash(&candidate_packed_bytes)),"physical_scale_hash":format!("fnv1a64:{:016x}",hash(&candidate_scale_bytes)),"down_bf16_hash":format!("fnv1a64:{:016x}",hash(&down)),"tail_zero_bytes":plan.tail,"parent_median_ms":timing.parent_median,"candidate_median_ms":timing.candidate_median,"parent_p90_ms":timing.parent_p90,"candidate_p90_ms":timing.candidate_p90,"paired_median_ms":timing.paired_median,"source_sha256":sources,"running_executable":{"path":binary.path,"sha256":binary.sha256,"profile":binary.profile},"embedded_bundle":{"target":bundle.target,"module_count":bundle.module_count,"sha256":bundle.sha256,"direct_modules":bundle.direct_modules},"exact":true,"redzones":true,"immutable":true,"deterministic":true,"same_nondefault_stream":true});
    for buffer in [
        &gate,
        &up,
        &temporary,
        &parent_packed,
        &parent_scales,
        &candidate_packed,
        &candidate_scales,
        &parent_logical_scales,
        &candidate_logical_scales,
        &parent_output,
        &candidate_output,
    ] {
        buffer.free(gpu)?;
    }
    Ok(receipt)
}
