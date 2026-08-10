// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness + throughput oracle for `w8a8_gemm_pipelined` (FP8-native
//! m16n8k32 MMA + per-row activation quant) vs the shipping
//! `w8a16_gemm_pipelined` (BF16 MMA after LUT dequant), at the prefill
//! projection shapes that make up the ~1.55 s FP8-GEMM class @N=2410.
//!
//! Gate: cosine >= 0.999 per shape (activation E4M3 quantization is lossy by
//! design; per-row scales bound the relative error) AND w8a8 not slower.
//! Final arbiter for shipping remains tool-eval-bench 90/100.
//!
//! Usage: cargo run --release -p spark-model --example w8a8_gemm_microtest \
//!            --features cuda,gpu-examples

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const ITERS: u32 = 30;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

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
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// E4M3 bytes for weights, avoiding NaN encodings (top exponent).
fn gen_fp8(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n).map(|_| (rng.next_u64() & 0x7F) as u8 % 0x76).collect()
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let w8a16_h = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    let w8a8_h = gpu.kernel("w8a8_gemm_pipelined", "w8a8_gemm_pipelined")?;
    let quant_h = gpu.kernel("w8a8_gemm_pipelined", "quantize_a_fp8_rows")?;
    let mut rng = Rng(0xF8F8);

    // (label, M, N, K) — the FP8-GEMM class at N=2410.
    let shapes = [
        ("wq_b        ", 2410usize, 32768usize, 1024usize),
        ("wo_a (group)", 2410, 1024, 4096),
        ("wo_b        ", 2410, 4096, 8192),
    ];

    let mut all_pass = true;
    for &(label, m, n, k) in &shapes {
        let a_bf16: Vec<u16> = (0..m * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-2.0, 2.0)))
            .collect();
        let a_bytes: Vec<u8> = a_bf16.iter().flat_map(|x| x.to_le_bytes()).collect();
        let b_fp8 = gen_fp8(&mut rng, n * k);
        let scales: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| rng.uniform(0.002, 0.02).to_le_bytes())
            .collect();

        let a_d = upload(&gpu, &a_bytes)?;
        let b_d = upload(&gpu, &b_fp8)?;
        let s_d = upload(&gpu, &scales)?;
        let a8_d = gpu.alloc(m * k)?;
        let rs_d = gpu.alloc(m * 4)?;
        let c16_d = gpu.alloc(m * n * 2)?;
        let c8_d = gpu.alloc(m * n * 2)?;

        let launch_a16 = |c_out: DevicePtr| -> Result<()> {
            KernelLaunch::new(&gpu, w8a16_h)
                .grid([(n.div_ceil(32)) as u32, (m.div_ceil(128)) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a_d)
                .arg_ptr(b_d)
                .arg_ptr(s_d)
                .arg_ptr(c_out)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        let launch_a8 = |c_out: DevicePtr| -> Result<()> {
            KernelLaunch::new(&gpu, quant_h)
                .grid([m as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a_d)
                .arg_ptr(a8_d)
                .arg_ptr(rs_d)
                .arg_u32(m as u32)
                .arg_u32(k as u32)
                .launch(stream)?;
            KernelLaunch::new(&gpu, w8a8_h)
                .grid([(n.div_ceil(32)) as u32, (m.div_ceil(128)) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a8_d)
                .arg_ptr(rs_d)
                .arg_ptr(b_d)
                .arg_ptr(s_d)
                .arg_ptr(c_out)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };

        launch_a16(c16_d)?;
        launch_a8(c8_d)?;
        gpu.synchronize(stream)?;

        let mut r16 = vec![0u8; m * n * 2];
        let mut r8 = vec![0u8; m * n * 2];
        gpu.copy_d2h(c16_d, &mut r16)?;
        gpu.copy_d2h(c8_d, &mut r8)?;
        let f = |b: &[u8]| -> Vec<f64> {
            b.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
                .collect()
        };
        let (x, y) = (f(&r16), f(&r8));
        let nz = x.iter().filter(|v| **v != 0.0).count();
        if nz == 0 {
            bail!("{label}: dead reference output");
        }
        if y.iter().any(|v| !v.is_finite()) {
            bail!("{label}: non-finite w8a8 output");
        }
        let (mut dot, mut nx, mut ny) = (0f64, 0f64, 0f64);
        for i in 0..x.len() {
            dot += x[i] * y[i];
            nx += x[i] * x[i];
            ny += y[i] * y[i];
        }
        let cos = dot / (nx.sqrt() * ny.sqrt());

        let time = |a8: bool| -> Result<f64> {
            for _ in 0..3 {
                if a8 { launch_a8(c8_d)?; } else { launch_a16(c16_d)?; }
            }
            gpu.synchronize(stream)?;
            let (mut e0, mut e1) = (0u64, 0u64);
            unsafe {
                cuEventCreate(&mut e0, 0);
                cuEventCreate(&mut e1, 0);
                cuEventRecord(e0, stream);
            }
            for _ in 0..ITERS {
                if a8 { launch_a8(c8_d)?; } else { launch_a16(c16_d)?; }
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
        let t16 = time(false)?;
        let t8 = time(true)?;
        let tf = |t_ms: f64| 2.0 * (m as f64) * (n as f64) * (k as f64) / (t_ms / 1e3) / 1e12;
        let ok = cos >= 0.999 && t8 <= t16 * 1.05;
        all_pass &= ok;
        println!(
            "  {label} M={m} N={n} K={k}: cos={cos:.6} | a16 {t16:.3} ms ({:.1} TF) | a8(incl quant) {t8:.3} ms ({:.1} TF) [{:.2}x] {}",
            tf(t16),
            tf(t8),
            t16 / t8,
            if ok { "PASS" } else { "FAIL" }
        );

        for p in [a_d, b_d, s_d, a8_d, rs_d, c16_d, c8_d] {
            let _ = gpu.free(p);
        }
    }
    if !all_pass {
        bail!("w8a8 oracle FAIL");
    }
    println!("PASS");
    Ok(())
}
