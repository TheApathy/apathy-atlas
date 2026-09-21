// SPDX-License-Identifier: AGPL-3.0-only
fn logical_scales(bytes: &[u8], plan: Plan) -> Result<Vec<u8>> {
    Ok(deinterleave_nvfp4_scales_128x4(
        bytes,
        &[plan.padded as usize, (K / 16) as usize],
        NVFP4_GROUP_SIZE,
    )?)
}

fn complete_nonalias(live: &[&Guarded]) -> Result<()> {
    ensure!(live.len() == 13, "complete live allocation census changed");
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
    Ok(())
}

fn down(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    plan: Plan,
    packed: &Guarded,
    scales: &Guarded,
    weight: &Guarded,
    weight_scales: &Guarded,
    output: &Guarded,
) -> Result<()> {
    ops::nvfp4_nvfp4_gemm_kmajor_m256(
        gpu,
        kernels.down,
        packed.ptr(),
        scales.ptr(),
        weight.ptr(),
        weight_scales.ptr(),
        SCALE2,
        output.ptr(),
        plan.m,
        DOWN_N,
        K,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn replay_and_down(
    gpu: &dyn GpuBackend,
    stream: u64,
    kernels: Kernels,
    plan: Plan,
    logical: usize,
    merged: &Guarded,
    gate: &Guarded,
    up: &Guarded,
    candidate_packed: &mut Guarded,
    candidate_scales: &mut Guarded,
    parent_packed: &mut Guarded,
    parent_scales: &mut Guarded,
    parent_packed_bytes: &[u8],
    parent_scale_bytes: &[u8],
    candidate_packed_bytes: &[u8],
    candidate_scale_bytes: &[u8],
    parent_logical_scales: &mut Guarded,
    candidate_logical_scales: &mut Guarded,
    weight: &Guarded,
    weight_scales: &Guarded,
    parent_output: &Guarded,
    candidate_output: &mut Guarded,
) -> Result<Vec<u8>> {
    parent_packed.replace_payload(gpu, disjoint_sentinel(parent_packed_bytes, 0x92), false)?;
    parent_scales.replace_payload(gpu, disjoint_sentinel(parent_scale_bytes, 0xa4), false)?;
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
    gpu.synchronize(stream)?;
    equal(
        "parent packed complete overwrite",
        parent_packed_bytes,
        &parent_packed.read(gpu, "parent-packed-2")?,
    )?;
    equal(
        "parent scale complete overwrite",
        parent_scale_bytes,
        &parent_scales.read(gpu, "parent-scales-2")?,
    )?;
    candidate_packed.replace_payload(
        gpu,
        disjoint_sentinel(candidate_packed_bytes, 0xe3),
        false,
    )?;
    candidate_scales.replace_payload(gpu, disjoint_sentinel(candidate_scale_bytes, 0x19), false)?;
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
    equal(
        "candidate packed determinism",
        candidate_packed_bytes,
        &candidate_packed.read(gpu, "candidate-packed-2")?,
    )?;
    equal(
        "candidate scale determinism",
        candidate_scale_bytes,
        &candidate_scales.read(gpu, "candidate-scales-2")?,
    )?;
    let parent_logical = logical_scales(parent_scale_bytes, plan)?;
    let candidate_logical = logical_scales(candidate_scale_bytes, plan)?;
    for (label, bytes) in [
        ("parent", &parent_logical),
        ("candidate", &candidate_logical),
    ] {
        ensure!(
            bytes[logical..].len() == plan.tail && bytes[logical..].iter().all(|byte| *byte == 0),
            "{label} physical padded tail is not exactly zero"
        );
    }
    parent_logical_scales.replace_payload(gpu, parent_logical[..logical].to_vec(), true)?;
    candidate_logical_scales.replace_payload(gpu, candidate_logical[..logical].to_vec(), true)?;
    down(
        gpu,
        stream,
        kernels,
        plan,
        parent_packed,
        parent_logical_scales,
        weight,
        weight_scales,
        parent_output,
    )?;
    down(
        gpu,
        stream,
        kernels,
        plan,
        candidate_packed,
        candidate_logical_scales,
        weight,
        weight_scales,
        candidate_output,
    )?;
    gpu.synchronize(stream)?;
    let down_bytes = parent_output.read(gpu, "parent-down")?;
    equal(
        "synthetic down BF16",
        &down_bytes,
        &candidate_output.read(gpu, "candidate-down-1")?,
    )?;
    candidate_output.replace_payload(gpu, disjoint_sentinel(&down_bytes, 0x8d), false)?;
    down(
        gpu,
        stream,
        kernels,
        plan,
        candidate_packed,
        candidate_logical_scales,
        weight,
        weight_scales,
        candidate_output,
    )?;
    gpu.synchronize(stream)?;
    equal(
        "candidate down determinism",
        &down_bytes,
        &candidate_output.read(gpu, "candidate-down-2")?,
    )?;
    Ok(down_bytes)
}

#[allow(clippy::too_many_arguments)]
fn postchecks(
    gpu: &dyn GpuBackend,
    plan: Plan,
    merged: &Guarded,
    gate: &Guarded,
    up: &Guarded,
    parent_packed: &Guarded,
    parent_scales: &Guarded,
    candidate_packed: &Guarded,
    candidate_scales: &Guarded,
    parent_packed_bytes: &[u8],
    parent_scale_bytes: &[u8],
    candidate_packed_bytes: &[u8],
    candidate_scale_bytes: &[u8],
    parent_logical_scales: &Guarded,
    candidate_logical_scales: &Guarded,
    weight: &Guarded,
    weight_scales: &Guarded,
) -> Result<()> {
    for (label, expected, buffer) in [
        (
            "post-timing parent packed",
            parent_packed_bytes,
            parent_packed,
        ),
        (
            "post-timing parent scales",
            parent_scale_bytes,
            parent_scales,
        ),
        (
            "post-timing candidate packed",
            candidate_packed_bytes,
            candidate_packed,
        ),
        (
            "post-timing candidate scales",
            candidate_scale_bytes,
            candidate_scales,
        ),
    ] {
        equal(label, expected, &buffer.read(gpu, label)?)?;
    }
    exact_split(
        merged.original.as_ref().unwrap(),
        &gate.read(gpu, "timed-gate")?,
        &up.read(gpu, "timed-up")?,
        plan,
    )?;
    merged.immutable(gpu, "merged")?;
    parent_logical_scales.immutable(gpu, "parent-logical-scales")?;
    candidate_logical_scales.immutable(gpu, "candidate-logical-scales")?;
    weight.immutable(gpu, "down-weight")?;
    weight_scales.immutable(gpu, "down-weight-scales")?;
    Ok(())
}
