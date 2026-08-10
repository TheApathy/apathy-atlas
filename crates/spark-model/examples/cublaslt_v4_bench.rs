// SPDX-License-Identifier: AGPL-3.0-only

//! cuBLASLt-vs-custom GEMM bench at the DeepSeek-V4 prefill shapes.
//!
//! Every custom prefill GEMM in this tree measures 26-35 TFLOPS where the
//! GB10 does ~250 dense FP8 / ~125 BF16 — a family-wide gap. cuBLASLt's
//! BF16 path (`bf16_gemm_act_weight_t`) already serves the Laguna/Qwen
//! prefill; this bench asks whether routing the V4 projections through it
//! (which requires keeping the BF16 mirrors resident — reversing
//! ATLAS_V4_ATTN_RELEASE_BF16's 8 GiB saving) would buy enough TFLOPS to
//! be the 973→1100 lever, before any dispatch work is spent.
//!
//! Correctness: cosine vs dense_gemm_bf16_pipelined (>= 0.999; different
//! accumulation order).

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const ITERS: u32 = 30;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn bf16s(rng: &mut Rng, n: usize) -> Vec<u8> {
    // bounded-exponent bf16 bit patterns; values irrelevant for timing,
    // bounded so the cosine check stays finite
    (0..n)
        .flat_map(|_| {
            let m = (rng.next() & 0x7F) as u16;
            let s = ((rng.next() & 1) as u16) << 15;
            (s | 0x3E00 | m).to_le_bytes()
        })
        .collect()
}

fn read_f64(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f64>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
        .collect())
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut d, mut x, mut y) = (0f64, 0f64, 0f64);
    for (p, q) in a.iter().zip(b) {
        d += p * q;
        x += p * p;
        y += q * q;
    }
    d / (x.sqrt() * y.sqrt() + 1e-30)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.default_stream();
    let pipelined = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;

    // (label, M, N, K): the prefill projection shapes from the 973 waterfall.
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("wq_b     ", 2410, 32768, 1024),
        ("wo_b     ", 2410, 4096, 8192),
        ("wo_a_grp ", 2410, 1024, 4096),
        ("wq_a     ", 2410, 1024, 4096),
        ("kv_proj  ", 2410, 512, 4096),
    ];

    println!("{:<10} {:>14} {:>22} {:>10}", "shape", "pipelined", "cuBLASLt bf16", "cosine");
    for &(tag, m, n, k) in shapes {
        let mut rng = Rng(0xB1A5 ^ ((m * 31 + n * 7 + k) as u64));
        let a = upload(gpu, &bf16s(&mut rng, m * k))?;
        let b = upload(gpu, &bf16s(&mut rng, n * k))?;
        let c1 = gpu.alloc(m * n * 2)?;
        let c2 = gpu.alloc(m * n * 2)?;
        let w = crate_weight(b);

        // correctness pass
        spark_model_ops_dense_gemm_bf16_pipelined(gpu, pipelined, a, &w, c1, m, n, k, stream)?;
        spark_runtime::cublaslt::bf16_gemm_act_weight_t(a.0, b.0, c2.0, m as u32, n as u32, k as u32, stream)?;
        gpu.synchronize(stream)?;
        let r1 = read_f64(gpu, c1, m * n)?;
        let r2 = read_f64(gpu, c2, m * n)?;
        let cos = cosine(&r1, &r2);

        // timing
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let t_pipe = {
            for _ in 0..3 {
                spark_model_ops_dense_gemm_bf16_pipelined(gpu, pipelined, a, &w, c1, m, n, k, stream)?;
            }
            gpu.synchronize(stream)?;
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                spark_model_ops_dense_gemm_bf16_pipelined(gpu, pipelined, a, &w, c1, m, n, k, stream)?;
            }
            gpu.synchronize(stream)?;
            t0.elapsed().as_secs_f64() / ITERS as f64
        };
        let t_lt = {
            for _ in 0..3 {
                spark_runtime::cublaslt::bf16_gemm_act_weight_t(a.0, b.0, c2.0, m as u32, n as u32, k as u32, stream)?;
            }
            gpu.synchronize(stream)?;
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                spark_runtime::cublaslt::bf16_gemm_act_weight_t(a.0, b.0, c2.0, m as u32, n as u32, k as u32, stream)?;
            }
            gpu.synchronize(stream)?;
            t0.elapsed().as_secs_f64() / ITERS as f64
        };
        println!(
            "{tag:<10} {:>7.3} ms {:>4.0}TF {:>10.3} ms {:>6.0}TF [{:>4.2}x] {cos:>9.6}",
            t_pipe * 1e3,
            flop / t_pipe / 1e12,
            t_lt * 1e3,
            flop / t_lt / 1e12,
            t_pipe / t_lt,
        );
        for p in [a, b, c1, c2] {
            gpu.free(p).ok();
        }
    }
    Ok(())
}

// dense_gemm_bf16_pipelined wrapper shim (examples can't see pub(crate) ops)
fn crate_weight(w: DevicePtr) -> ShimWeight {
    ShimWeight { weight: w }
}
struct ShimWeight {
    weight: DevicePtr,
}
#[allow(clippy::too_many_arguments)]
fn spark_model_ops_dense_gemm_bf16_pipelined(
    gpu: &dyn GpuBackend,
    kernel: spark_runtime::gpu::KernelHandle,
    input: DevicePtr,
    weight: &ShimWeight,
    output: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

