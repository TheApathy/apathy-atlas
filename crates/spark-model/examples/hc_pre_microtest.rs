// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone timing for `hc_pre` at prefill width (T=2410, hc=4, H=4096) —
//! decides the optimization route for map item 2 (hc glue −0.4 s):
//! register-x rewrite (if the 24x stream re-reads dominate) vs
//! GEMM-ification with a bf16 fn mirror (if the 1.5 MB/token fn traffic
//! dominates). Also times hc_post for the same call width.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const T: usize = 2410;
const HC: usize = 4;
const H: usize = 4096;
const HC_DIM: usize = HC * H;
const MIX: usize = (2 + HC) * HC; // 24
const ITERS: u32 = 50;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

fn upload_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let pre_h = gpu.kernel("hyper_connection", "hc_pre")?;
    let post_h = gpu.kernel("hyper_connection", "hc_post")?;

    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let streams: Vec<f32> = (0..T * HC_DIM).map(|_| rnd()).collect();
    let fn_w: Vec<f32> = (0..MIX * HC_DIM).map(|_| rnd() * 0.01).collect();
    let scale: Vec<f32> = vec![1.0, 1.0, 1.0];
    let base: Vec<f32> = (0..MIX).map(|_| rnd() * 0.1).collect();

    let streams_d = upload_f32(&gpu, &streams)?;
    let fn_d = upload_f32(&gpu, &fn_w)?;
    let scale_d = upload_f32(&gpu, &scale)?;
    let base_d = upload_f32(&gpu, &base)?;
    let y_d = gpu.alloc(T * H * 2)?;
    let post_d = gpu.alloc(T * HC * 4)?;
    let comb_d = gpu.alloc(T * HC * HC * 4)?;
    let attn_d = gpu.alloc(T * H * 2)?; // bf16 sublayer output for hc_post

    let launch_pre = || -> Result<()> {
        KernelLaunch::new(&gpu, pre_h)
            .grid([T as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(streams_d)
            .arg_ptr(fn_d)
            .arg_ptr(scale_d)
            .arg_ptr(base_d)
            .arg_ptr(y_d)
            .arg_ptr(post_d)
            .arg_ptr(comb_d)
            .arg_u32(H as u32)
            .arg_u32(HC as u32)
            .arg_u32(3)
            .arg_f32(1e-6)
            .arg_f32(1e-6)
            .launch(stream)
    };
    let launch_post = || -> Result<()> {
        KernelLaunch::new(&gpu, post_h)
            .grid([T as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(attn_d)
            .arg_ptr(streams_d)
            .arg_ptr(post_d)
            .arg_ptr(comb_d)
            .arg_ptr(streams_d)
            .arg_u32(H as u32)
            .arg_u32(HC as u32)
            .launch(stream)
    };

    let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        for _ in 0..5 {
            f()?;
        }
        gpu.synchronize(stream)?;
        let (mut e0, mut e1) = (0u64, 0u64);
        unsafe {
            cuEventCreate(&mut e0, 0);
            cuEventCreate(&mut e1, 0);
            cuEventRecord(e0, stream);
        }
        for _ in 0..ITERS {
            f()?;
        }
        unsafe { cuEventRecord(e1, stream) };
        gpu.synchronize(stream)?;
        let mut ms = 0f32;
        unsafe {
            cuEventSynchronize(e1);
            cuEventElapsedTime(&mut ms, e0, e1);
            cuEventDestroy_v2(e0);
            cuEventDestroy_v2(e1);
        }
        Ok(ms as f64 / ITERS as f64)
    };

    // v2: bit-exactness gate + timing
    let pre2_h = gpu.kernel("hyper_connection", "hc_pre_v2")?;
    let y2_d = gpu.alloc(T * H * 2)?;
    let post2_d = gpu.alloc(T * HC * 4)?;
    let comb2_d = gpu.alloc(T * HC * HC * 4)?;
    let launch_pre2 = || -> Result<()> {
        KernelLaunch::new(&gpu, pre2_h)
            .grid([T as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(streams_d)
            .arg_ptr(fn_d)
            .arg_ptr(scale_d)
            .arg_ptr(base_d)
            .arg_ptr(y2_d)
            .arg_ptr(post2_d)
            .arg_ptr(comb2_d)
            .arg_u32(H as u32)
            .arg_u32(HC as u32)
            .arg_u32(3)
            .arg_f32(1e-6)
            .arg_f32(1e-6)
            .launch(stream)
    };
    launch_pre()?;
    launch_pre2()?;
    gpu.synchronize(stream)?;
    for (a, b, len, name) in [
        (y_d, y2_d, T * H * 2, "y"),
        (post_d, post2_d, T * HC * 4, "post"),
        (comb_d, comb2_d, T * HC * HC * 4, "comb"),
    ] {
        let mut ra = vec![0u8; len];
        let mut rb = vec![0u8; len];
        gpu.copy_d2h(a, &mut ra)?;
        gpu.copy_d2h(b, &mut rb)?;
        if ra != rb {
            anyhow::bail!("hc_pre_v2 {name} NOT bit-identical");
        }
    }
    println!("  GATE PASS: hc_pre_v2 == hc_pre bit-identical (y/post/comb)");
    let t_pre2 = time(&launch_pre2)?;
    println!("hc_pre_v2 T={T}: {t_pre2:.3} ms/call");

    let t_pre = time(&launch_pre)?;
    let t_post = time(&launch_post)?;
    println!("hc_pre  T={T}: {t_pre:.3} ms/call  (x2 calls/layer x43 = {:.0} ms/pass)", t_pre * 86.0);
    println!("hc_post T={T}: {t_post:.3} ms/call (x2 calls/layer x43 = {:.0} ms/pass)", t_post * 86.0);
    Ok(())
}
