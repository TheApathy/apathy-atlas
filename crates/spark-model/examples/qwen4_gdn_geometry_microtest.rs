// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;

fn main() -> Result<()> {
    const NK: usize = 16;
    const NV: usize = 48;
    const D: usize = 128;
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();

    let h = gpu.alloc(NV * D * D * 4)?;
    let q = gpu.alloc(NK * D * 4)?;
    let k = gpu.alloc(NK * D * 4)?;
    let v = gpu.alloc(NV * D * 4)?;
    let gate = gpu.alloc(NV * 4)?;
    let beta = gpu.alloc(NV * 4)?;
    let gdn = gpu.alloc(NV * D * 4)?;
    let z = gpu.alloc(NV * D * 2)?;
    let weight = gpu.alloc(D * 2)?;
    let output = gpu.alloc(NV * D * 2)?;
    for (ptr, bytes) in [
        (h, NV * D * D * 4),
        (q, NK * D * 4),
        (k, NK * D * 4),
        (v, NV * D * 4),
        (gate, NV * 4),
        (beta, NV * 4),
        (gdn, NV * D * 4),
        (z, NV * D * 2),
        (weight, D * 2),
        (output, NV * D * 2),
    ] {
        gpu.memset(ptr, 0, bytes)?;
    }

    KernelLaunch::new(
        &gpu,
        gpu.kernel("gated_delta_rule", "gated_delta_rule_decode_f32")?,
    )
    .grid([NV as u32, 1, 1])
    .block([D as u32, 1, 1])
    .arg_ptr(h)
    .arg_ptr(q)
    .arg_ptr(k)
    .arg_ptr(v)
    .arg_ptr(gate)
    .arg_ptr(beta)
    .arg_ptr(gdn)
    .arg_u32(1)
    .arg_u32(NK as u32)
    .arg_u32(NV as u32)
    .arg_u32(D as u32)
    .arg_u32(D as u32)
    .launch(stream)?;
    gpu.synchronize(stream)?;
    println!("48-head GDN: PASS");

    KernelLaunch::new(
        &gpu,
        gpu.kernel("qwen4_hyper", "qwen4_gated_rms_norm_sigmoid_f32")?,
    )
    .grid([NV as u32, 1, 1])
    .block([D as u32, 1, 1])
    .arg_ptr(gdn)
    .arg_ptr(z)
    .arg_ptr(weight)
    .arg_ptr(output)
    .arg_u32(D as u32)
    .arg_f32(1e-6)
    .arg_u32(D as u32)
    .arg_u32(D as u32)
    .launch(stream)?;
    gpu.synchronize(stream)?;
    println!("48-head Qwen4 gated RMS norm: PASS");
    Ok(())
}
