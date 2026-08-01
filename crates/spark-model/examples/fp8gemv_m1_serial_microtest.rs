// SPDX-License-Identifier: AGPL-3.0-only

//! Cold M=1 bench: `dense_gemv_fp8w` (the serial-decode FP8-mirror GEMV)
//! vs `fp8_gemm_t_row_scaled_mtile8` / `_n32` (the proven M≤8 verify GEMMs)
//! at the SERIAL decode shapes of Laguna-S-2.1 (h=3072, 72 q-heads, 8
//! kv-heads, hd=128, vocab=100352):
//!
//!   q     [1,3072] @ [9216,3072]^T   (ungated q_proj mirror)
//!   kv    [1,3072] @ [1024,3072]^T   (k_proj / v_proj mirror)
//!   o     [1,9216] @ [3072,9216]^T   (o_proj mirror)
//!   lmhead[1,3072] @ [100352,3072]^T (FP8 lm_head, serial K=1 tier)
//!
//! Plus the 48-head sibling shapes from the M=7 verify bench for coverage.
//!
//! Every iteration reads a DIFFERENT copy of the FP8 weight (round-robin,
//! ≥256MB aggregate) so weight bytes are never L2-resident — matching the
//! serve-side reality where all 12 attention layers' mirrors are cold every
//! serial step. Also cross-checks GEMV-vs-GEMM output cosine per shape
//! (the two kernels use different accumulation orders; gate 0.999).
//!
//! Usage: cargo run --release -p spark-model --example \
//!          fp8gemv_m1_serial_microtest --features cuda,gpu-examples
//! Exit 0 = PASS (all cosines >= gate), 1 = FAIL — scriptable.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const COSINE_GATE: f64 = 0.999;
const AGG_BYTES: usize = 256 << 20; // rotation footprint per shape
const ITERS: u32 = 200;

// splitmix64 — reproducible inputs, no rand dependency.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (round >> 16) as u16
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

#[derive(Clone, Copy)]
enum Variant {
    Gemv,      // dense_gemv_fp8w: grid ceil(N/4), block 256
    Mtile8,    // fp8_gemm_t_row_scaled_mtile8: grid ceil(N/64), block 128
    Mtile8N32, // _n32: grid ceil(N/32), block 128
}

#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &dyn GpuBackend,
    v: Variant,
    h: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    s: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    match v {
        Variant::Gemv => KernelLaunch::new(gpu, h)
            .grid([div_ceil(n, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b)
            .arg_ptr(s)
            .arg_ptr(c)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream),
        Variant::Mtile8 | Variant::Mtile8N32 => {
            let n_tile = if matches!(v, Variant::Mtile8) { 64 } else { 32 };
            KernelLaunch::new(gpu, h)
                .grid([div_ceil(n, n_tile), 1, 1])
                .block([128, 1, 1])
                .arg_ptr(a)
                .arg_ptr(b)
                .arg_ptr(s)
                .arg_ptr(c)
                .arg_u32(1) // M = 1
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
        }
    }
}

fn read_row_f64(gpu: &dyn GpuBackend, c: DevicePtr, n: usize) -> Result<Vec<f64>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(c, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| bf16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64)
        .collect())
}

fn cosine(x: &[f64], y: &[f64]) -> f64 {
    let (mut dot, mut nx, mut ny) = (0f64, 0f64, 0f64);
    for (a, b) in x.iter().zip(y) {
        dot += a * b;
        nx += a * a;
        ny += b * b;
    }
    dot / (nx.sqrt() * ny.sqrt() + 1e-30)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let h_gemv = gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?;
    let h8 = gpu.kernel("w4a16", "fp8_gemm_t_row_scaled_mtile8")?;
    let h8n32 = gpu.kernel("w4a16", "fp8_gemm_t_row_scaled_mtile8_n32")?;

    // Serial M=1 shapes for Laguna-S-2.1 + the 48-head verify siblings.
    let shapes: &[(&str, usize, usize)] = &[
        ("q72", 9216, 3072),
        ("q48", 6144, 3072),
        ("kv", 1024, 3072),
        ("o72", 3072, 9216),
        ("o48", 3072, 6144),
        ("lmhead", 100352, 3072),
    ];

    let mut all_pass = true;
    println!(
        "{:<8} {:<10} {:>10} {:>12} {:>10}",
        "shape", "kernel", "us/iter", "GB/s(cold)", "cosine"
    );
    for &(tag, n, k) in shapes {
        let mut rng = Rng(0x5E41A1 ^ ((n * 31 + k) as u64));
        let a_bf16: Vec<u16> = (0..k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
            .collect();
        // &0x77 masks out NaN/huge-exponent E4M3 codes; values don't affect
        // timing, and both kernels decode identical bytes for the cosine check.
        let b_fp8: Vec<u8> = (0..n * k).map(|_| (rng.next_u64() & 0x77) as u8).collect();
        let scales: Vec<f32> = (0..n).map(|_| rng.uniform(0.5, 2.0)).collect();

        let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
        let s_ptr = upload_bytes(gpu, &f32s_to_le(&scales))?;
        let c_ptr = gpu.alloc(n * 2)?;

        // L2-defeating rotation: enough weight copies to exceed AGG_BYTES.
        let rot = (AGG_BYTES / (n * k)).clamp(2, 96);
        let b_ptrs: Vec<DevicePtr> = (0..rot)
            .map(|_| upload_bytes(gpu, &b_fp8))
            .collect::<Result<_>>()?;

        // Correctness cross-check: GEMV is the incumbent reference.
        launch(
            gpu,
            Variant::Gemv,
            h_gemv,
            a_ptr,
            b_ptrs[0],
            s_ptr,
            c_ptr,
            n as u32,
            k as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let ref_row = read_row_f64(gpu, c_ptr, n)?;

        for (name, v, h) in [
            ("gemv", Variant::Gemv, h_gemv),
            ("mtile8", Variant::Mtile8, h8),
            ("mtile8n32", Variant::Mtile8N32, h8n32),
        ] {
            // Poison C so "kernel wrote nothing" can't pass the cosine gate.
            gpu.copy_h2d(&vec![0x7Fu8; n * 2], c_ptr)?;
            launch(
                gpu, v, h, a_ptr, b_ptrs[0], s_ptr, c_ptr, n as u32, k as u32, stream,
            )?;
            gpu.synchronize(stream)?;
            let row = read_row_f64(gpu, c_ptr, n)?;
            let cos = cosine(&ref_row, &row);
            let pass = cos >= COSINE_GATE && row.iter().all(|x| x.is_finite());
            all_pass &= pass;

            // Warmup (touch every rotation buffer), then timed cold loop.
            for i in 0..2 * rot as u32 {
                launch(
                    gpu,
                    v,
                    h,
                    a_ptr,
                    b_ptrs[i as usize % rot],
                    s_ptr,
                    c_ptr,
                    n as u32,
                    k as u32,
                    stream,
                )?;
            }
            gpu.synchronize(stream)?;
            let t0 = std::time::Instant::now();
            for i in 0..ITERS {
                launch(
                    gpu,
                    v,
                    h,
                    a_ptr,
                    b_ptrs[i as usize % rot],
                    s_ptr,
                    c_ptr,
                    n as u32,
                    k as u32,
                    stream,
                )?;
            }
            gpu.synchronize(stream)?;
            let secs = t0.elapsed().as_secs_f64() / ITERS as f64;
            let gbs = (n * k) as f64 / secs / 1e9; // FP8 weight bytes dominate
            println!(
                "{tag:<8} {name:<10} {:>10.1} {gbs:>12.1} {cos:>10.6} {}",
                secs * 1e6,
                if pass { "PASS" } else { "FAIL" }
            );
        }
        for p in b_ptrs {
            gpu.free(p).ok();
        }
        for p in [a_ptr, s_ptr, c_ptr] {
            gpu.free(p).ok();
        }
    }

    if all_pass {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}
