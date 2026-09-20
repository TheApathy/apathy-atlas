// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use serde_json::{Value, json};
use spark_model::layers::ops;
use spark_model::weight_map::cutlass_scale_layout::{
    NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4,
};
use spark_runtime::gpu::GpuBackend;

use super::guarded::{Guarded, disjoint_sentinel, merged_input};
use super::launch::{equal, exact_split, hash, launch, split_payloads};
use super::owner::{AllocationOwner, RetainedGpuFailure};
use super::timing::measure_balanced;
use super::{BinaryIdentity, BundleIdentity, DOWN_N, K, Kernels, Plan, SCALE2, SCHEMA};

pub(super) fn run(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    plan: Plan,
    session_id: &str,
    binary: &BinaryIdentity,
    bundle: &BundleIdentity,
    sources: &[(&str, &str)],
) -> std::result::Result<Value, RetainedGpuFailure> {
    let mut owner = AllocationOwner::new();
    let outcome = run_owned(
        &mut owner, gpu, stream, kernels, plan, session_id, binary, bundle, sources,
    );
    owner.finish(gpu, outcome)
}

#[allow(clippy::too_many_arguments)]
fn run_owned(
    owner: &mut AllocationOwner,
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    plan: Plan,
    session_id: &str,
    binary: &BinaryIdentity,
    bundle: &BundleIdentity,
    sources: &[(&str, &str)],
) -> Result<Value> {
    let merged = Guarded::input(owner, gpu, merged_input(plan.m, K), 0x11)?;
    ensure!(merged.len == plan.merged_bytes, "merged extent changed");
    let (expected_gate, expected_up) = split_payloads(merged.original.as_ref().unwrap(), plan)?;
    let mut gate = Guarded::output(owner, gpu, plan.elements * 2, 0x21)?;
    let mut up = Guarded::output(owner, gpu, plan.elements * 2, 0x22)?;
    gate.replace_payload(gpu, disjoint_sentinel(&expected_gate, 0x21), false)?;
    up.replace_payload(gpu, disjoint_sentinel(&expected_up, 0x22), false)?;
    let mut parent_packed = Guarded::output(owner, gpu, plan.packed, 0x31)?;
    let mut parent_scales = Guarded::output(owner, gpu, plan.scales, 0x41)?;
    let mut candidate_packed = Guarded::output(owner, gpu, plan.packed, 0xc7)?;
    let mut candidate_scales = Guarded::output(owner, gpu, plan.scales, 0xd3)?;
    let weight = Guarded::input(owner, gpu, vec![0x11; (K / 2 * DOWN_N) as usize], 0x51)?;
    let weight_scales = Guarded::input(owner, gpu, vec![0x38; (K / 16 * DOWN_N) as usize], 0x52)?;
    let logical = plan.m as usize * (K / 16) as usize;
    let mut parent_logical_scales = Guarded::output(owner, gpu, logical, 0x61)?;
    let mut candidate_logical_scales = Guarded::output(owner, gpu, logical, 0x62)?;
    let mut parent_output =
        Guarded::output(owner, gpu, plan.m as usize * DOWN_N as usize * 2, 0x71)?;
    let mut candidate_output =
        Guarded::output(owner, gpu, plan.m as usize * DOWN_N as usize * 2, 0x72)?;
    complete_nonalias(&[
        &merged,
        &gate,
        &up,
        &parent_packed,
        &parent_scales,
        &candidate_packed,
        &candidate_scales,
        &weight,
        &weight_scales,
        &parent_logical_scales,
        &candidate_logical_scales,
        &parent_output,
        &candidate_output,
    ])?;

    launch(
        false,
        gpu,
        stream,
        kernels,
        plan,
        merged.ptr(),
        gate.ptr(),
        up.ptr(),
        parent_packed.ptr(),
        parent_scales.ptr(),
    )?;
    launch(
        true,
        gpu,
        stream,
        kernels,
        plan,
        merged.ptr(),
        gate.ptr(),
        up.ptr(),
        candidate_packed.ptr(),
        candidate_scales.ptr(),
    )?;
    gpu.synchronize(stream)?;
    exact_split(
        merged.original.as_ref().unwrap(),
        &gate.read(gpu, "split-gate")?,
        &up.read(gpu, "split-up")?,
        plan,
    )?;
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

    let down_bytes = replay_and_down(
        gpu,
        stream,
        kernels,
        plan,
        logical,
        &merged,
        &gate,
        &up,
        &mut candidate_packed,
        &mut candidate_scales,
        &mut parent_packed,
        &mut parent_scales,
        &parent_packed_bytes,
        &parent_scale_bytes,
        &candidate_packed_bytes,
        &candidate_scale_bytes,
        &mut parent_logical_scales,
        &mut candidate_logical_scales,
        &weight,
        &weight_scales,
        &parent_output,
        &mut candidate_output,
    )?;
    parent_output.replace_payload(gpu, disjoint_sentinel(&down_bytes, 0x7e), false)?;
    down(
        gpu,
        stream,
        kernels,
        plan,
        &parent_packed,
        &parent_logical_scales,
        &weight,
        &weight_scales,
        &parent_output,
    )?;
    gpu.synchronize(stream)?;
    equal(
        "parent down complete overwrite",
        &down_bytes,
        &parent_output.read(gpu, "parent-down-2")?,
    )?;

    let timing = measure_balanced(
        gpu,
        stream,
        kernels,
        plan,
        merged.ptr(),
        gate.ptr(),
        up.ptr(),
        parent_packed.ptr(),
        parent_scales.ptr(),
        candidate_packed.ptr(),
        candidate_scales.ptr(),
    )?;
    postchecks(
        gpu,
        plan,
        &merged,
        &gate,
        &up,
        &parent_packed,
        &parent_scales,
        &candidate_packed,
        &candidate_scales,
        &parent_packed_bytes,
        &parent_scale_bytes,
        &candidate_packed_bytes,
        &candidate_scale_bytes,
        &parent_logical_scales,
        &candidate_logical_scales,
        &weight,
        &weight_scales,
    )?;
    Ok(
        json!({"schema":SCHEMA,"status":"qualified","performance_claim":true,"scheduler_session_id":session_id,
        "m":plan.m,"k":K,"padded_m":plan.padded,"reps":plan.reps,"timing_order":"C/I/I/C",
        "stream":stream,"scale2_bits":SCALE2.to_bits(),
        "merged_input_hash":format!("fnv1a64:{:016x}",hash(merged.original.as_ref().unwrap())),
        "packed_hash":format!("fnv1a64:{:016x}",hash(&candidate_packed_bytes)),
        "physical_scale_hash":format!("fnv1a64:{:016x}",hash(&candidate_scale_bytes)),
        "down_bf16_hash":format!("fnv1a64:{:016x}",hash(&down_bytes)),"tail_zero_bytes":plan.tail,
        "timing":timing,"source_sha256":sources,
        "running_executable":{"path":binary.path,"sha256":binary.sha256,"profile":binary.profile},
        "embedded_bundle":{"target":bundle.target,"module_count":bundle.module_count,"sha256":bundle.sha256,"direct_modules":bundle.direct_modules},
        "exact":true,"redzones":true,"immutable":true,"deterministic":true,
        "complete_nonalias_count":13,"same_nondefault_stream":true}),
    )
}

include!("runtime_helpers.inc.rs");
