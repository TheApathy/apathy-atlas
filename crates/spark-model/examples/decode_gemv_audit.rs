// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-path GEMV bandwidth audit for DeepSeek-V4-Flash on GB10.
//!
//! Times every (kernel, shape) pair the M=1 plain-decode path actually
//! dispatches — the list comes from a shape-logged serve run
//! (ATLAS_GEMM_SHAPE_LOG=1, prose decode) — with an L2-defeating weight
//! rotation so each iteration reads cold weight bytes, matching the serve
//! reality where all 43 layers stream their weights every token.
//!
//! Purpose: find the decode-side version of the prefill disease (fast
//! kernel in-tree, slow one wired). Prefill audit found five such sites
//! worth 18-30x each; plain decode sits at 154 GB/s achieved vs the 229
//! GB/s ceiling with the MLA chain at ~109 GB/s, so the suspects are the
//! per-projection GEMVs below, all sharing grid=(N/4) / block=256.
//!
//! Reports GB/s of WEIGHT bytes (the dominant traffic at M=1). No cosine
//! gate: incumbents only — this is a bandwidth census, not a swap gate.
//!
//! Usage: ATLAS_TARGET_MODEL=deepseek-v4-flash cargo run --release \
//!          -p spark-model --example decode_gemv_audit --features cuda,gpu-examples

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const AGG_BYTES: usize = 512 << 20; // rotation footprint per shape
const ITERS: u32 = 200;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

/// Which launch/argument convention the kernel uses. All are grid=(N/4),
/// block=256 — they differ only in the weight/scale argument list.
#[derive(Clone, Copy)]
enum Conv {
    DenseBf16,  // (A, B_bf16, C, N, K)                weight bytes = N*K*2
    Fp8Row,     // (A, B_fp8, row_scale_f32, C, N, K)  weight bytes = N*K
    W4a16,      // (A, B_pack, B_scale, scale2, C, N, K) bytes = N*K/2 + N*K/16
}

#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &dyn GpuBackend,
    conv: Conv,
    h: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    s: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let l = KernelLaunch::new(gpu, h)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b);
    match conv {
        Conv::DenseBf16 => l.arg_ptr(c).arg_u32(n).arg_u32(k).launch(stream),
        Conv::Fp8Row => l
            .arg_ptr(s)
            .arg_ptr(c)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream),
        Conv::W4a16 => l
            .arg_ptr(s)
            .arg_f32(1.0)
            .arg_ptr(c)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream),
    }
}

fn weight_bytes(conv: Conv, n: usize, k: usize) -> usize {
    match conv {
        Conv::DenseBf16 => n * k * 2,
        Conv::Fp8Row => n * k,
        Conv::W4a16 => n * k / 2 + n * k / 16, // e2m1 pairs + e4m3 per-16 scales
    }
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let h_dense = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let h_fp8 = gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?;
    let h_w8 = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let h_w4 = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;

    // Every M=1 dispatch from the shape-logged prose-decode run, tagged with
    // its serve-side role. 43x = per-layer (MLA chain), 1x = per-step.
    let sites: &[(&str, Conv, KernelHandle, usize, usize)] = &[
        ("wq_a.fp8      43x", Conv::Fp8Row, h_w8, 1024, 4096),
        ("kv_proj.fp8   43x", Conv::Fp8Row, h_w8, 512, 4096),
        ("wq_b.nvfp4    43x", Conv::W4a16, h_w4, 32768, 1024),
        ("wo_b.nvfp4    43x", Conv::W4a16, h_w4, 4096, 8192),
        ("bf16.n512     43x", Conv::DenseBf16, h_dense, 512, 4096),
        ("bf16.n144     43x", Conv::DenseBf16, h_dense, 144, 4096),
        ("drafter.down   1x", Conv::DenseBf16, h_dense, 4096, 12288),
        ("drafter.lmh    1x", Conv::DenseBf16, h_dense, 129280, 256),
        ("lm_head.fp8    1x", Conv::Fp8Row, h_fp8, 129280, 4096),
    ];

    println!(
        "{:<18} {:>7} {:>7} {:>9} {:>10} {:>9}",
        "site", "N", "K", "MB/call", "us/call", "GB/s"
    );
    let mut census: Vec<(String, f64, f64)> = Vec::new();
    for &(tag, conv, h, n, k) in sites {
        let mut rng = Rng(0xDECA0DE ^ ((n * 31 + k) as u64));
        let wb = weight_bytes(conv, n, k);
        // Inputs: bounded-exponent bytes; values don't affect timing.
        let a: Vec<u8> = (0..k * 2).map(|_| (rng.next_u64() & 0x3F) as u8).collect();
        let b: Vec<u8> = (0..wb).map(|_| (rng.next_u64() & 0x77) as u8).collect();
        let s: Vec<u8> = (0..(n.max(k * n / 16)) * 4)
            .map(|_| (rng.next_u64() & 0x3D) as u8)
            .collect();

        let a_p = upload(gpu, &a)?;
        let s_p = upload(gpu, &s)?;
        let c_p = gpu.alloc(n * 2)?;
        let rot = (AGG_BYTES / wb).clamp(2, 64);
        let b_ps: Vec<DevicePtr> = (0..rot)
            .map(|_| upload(gpu, &b))
            .collect::<Result<_>>()?;

        for i in 0..(2 * rot as u32) {
            launch(
                gpu, conv, h, a_p, b_ps[i as usize % rot], s_p, c_p, n as u32, k as u32, stream,
            )?;
        }
        gpu.synchronize(stream)?;
        let t0 = std::time::Instant::now();
        for i in 0..ITERS {
            launch(
                gpu, conv, h, a_p, b_ps[i as usize % rot], s_p, c_p, n as u32, k as u32, stream,
            )?;
        }
        gpu.synchronize(stream)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        let gbs = wb as f64 / (us * 1e-6) / 1e9;
        println!(
            "{tag:<18} {n:>7} {k:>7} {:>9.2} {us:>10.2} {gbs:>9.1}",
            wb as f64 / 1e6
        );
        census.push((tag.to_string(), wb as f64, us));
        for p in b_ps {
            gpu.free(p).ok();
        }
        for p in [a_p, s_p, c_p] {
            gpu.free(p).ok();
        }
    }

    // Serve-projected MLA-chain cost: per-layer sites x 43.
    let (mut mla_us, mut mla_mb) = (0f64, 0f64);
    for (tag, wb, us) in &census {
        if tag.ends_with("43x") {
            mla_us += us * 43.0;
            mla_mb += wb * 43.0 / 1e6;
        }
    }
    println!(
        "\nMLA-chain projection: {mla_mb:.0} MB/token in {:.2} ms -> {:.1} GB/s aggregate (ceiling 229)",
        mla_us / 1e3,
        mla_mb / 1e3 / (mla_us / 1e6)
    );
    Ok(())
}
