// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone correctness oracle + bench for `dense_gemm_bf16_mtile16`
//! (module `dense_gemm_bf16_mtile16`) — the small-M (M ≤ 16) BF16
//! weight-streaming GEMM the DFlash drafter propose path launches when
//! `ATLAS_DFLASH_DRAFTER_FASTGEMM=1`.
//!
//!   C[M,N] = A[M,K] (BF16) · B[N,K]^T (BF16), FP32 accumulate, BF16 out.
//!
//! Part 1 — correctness at the REAL Laguna drafter shapes (γ=16) + edges:
//!   (a) BITWISE equality vs `dense_gemm_bf16_pipelined` (the production
//!       kernel it replaces — same m16n8k16 ascending-K accumulate chain,
//!       so outputs must match bit-for-bit), and
//!   (b) cosine vs a CPU FP32 reference (gate 0.99) on the shapes small
//!       enough to reference on CPU.
//! Part 2 — bench: mtile16 vs pipelined at every drafter propose shape
//!   (q/k/v/g/o/gate/up/down, fc, fused-KV, lm_head), reporting effective
//!   weight-read GB/s (= N*K*2 bytes / time).
//!
//! Usage: cargo run --release -p spark-model --example dflash_bf16gemm_smallm_microtest \
//!          --features cuda,gpu-examples
//! Exit 0 = PASS, 1 = FAIL — scriptable.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const COSINE_GATE: f64 = 0.99;
// CPU reference cap: shapes with M*N*K above this only run the bitwise gate.
const CPU_REF_MAC_CAP: usize = 800_000_000;

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
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

#[allow(clippy::too_many_arguments)]
fn launch_mtile16(
    gpu: &dyn GpuBackend,
    h: spark_runtime::gpu::KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([div_ceil(n, 64), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_mtile16_n128(
    gpu: &dyn GpuBackend,
    h: spark_runtime::gpu::KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([div_ceil(n, 128), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_pipelined(
    gpu: &dyn GpuBackend,
    h: spark_runtime::gpu::KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let h16 = gpu.kernel("dense_gemm_bf16_mtile16", "dense_gemm_bf16_mtile16")?;
    let h128 = gpu.kernel("dense_gemm_bf16_mtile16", "dense_gemm_bf16_mtile16_n128")?;
    let hpipe = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?;

    // ── Part 1: correctness — Laguna drafter propose shapes (γ=16) + edges ──
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("q_proj  ", 16, 9216, 3072),
        ("kv_proj ", 16, 1024, 3072),
        ("g_proj  ", 16, 72, 3072),
        ("o_proj  ", 16, 3072, 9216),
        ("gate/up ", 16, 12288, 3072),
        ("down    ", 16, 3072, 12288),
        ("fc      ", 1, 3072, 18432),
        ("fused_kv", 1, 12288, 3072),
        ("lm_head ", 16, 100352, 3072),
        // Edges: M<16, ragged N (not a multiple of 64), K%64 != 0 (K%8==0
        // contract still holds), single N tile.
        ("edge-a  ", 7, 1000, 2048),
        ("edge-b  ", 3, 192, 104),
        ("edge-c  ", 16, 64, 8),
        ("edge-d  ", 1, 3072, 3072),
    ];
    let mut all_pass = true;
    for &(tag, m, n, k) in shapes {
        let mut rng = Rng(0xD16 ^ ((m * 31 + n * 7 + k) as u64));
        let a_bf16: Vec<u16> = (0..m * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
            .collect();
        let b_bf16: Vec<u16> = (0..n * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-0.5, 0.5)))
            .collect();

        let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
        let b_ptr = upload_bytes(gpu, &u16s_to_le(&b_bf16))?;
        let c_new = gpu.alloc(m * n * 2)?;
        let c_wide = gpu.alloc(m * n * 2)?;
        let c_old = gpu.alloc(m * n * 2)?;
        // Poison outputs so "kernel wrote nothing" can't pass the gates.
        gpu.copy_h2d(&vec![0x7Fu8; m * n * 2], c_new)?;
        gpu.copy_h2d(&vec![0x33u8; m * n * 2], c_wide)?;
        gpu.copy_h2d(&vec![0x11u8; m * n * 2], c_old)?;

        launch_mtile16(
            gpu, h16, a_ptr, b_ptr, c_new, m as u32, n as u32, k as u32, stream,
        )?;
        launch_mtile16_n128(
            gpu, h128, a_ptr, b_ptr, c_wide, m as u32, n as u32, k as u32, stream,
        )?;
        launch_pipelined(
            gpu, hpipe, a_ptr, b_ptr, c_old, m as u32, n as u32, k as u32, stream,
        )?;
        gpu.synchronize(stream)?;

        let mut raw_new = vec![0u8; m * n * 2];
        let mut raw_wide = vec![0u8; m * n * 2];
        let mut raw_old = vec![0u8; m * n * 2];
        gpu.copy_d2h(c_new, &mut raw_new)?;
        gpu.copy_d2h(c_wide, &mut raw_wide)?;
        gpu.copy_d2h(c_old, &mut raw_old)?;
        let bitwise = raw_new == raw_old && raw_wide == raw_old;

        // CPU FP32 reference cosine (skipped on the huge shapes — the
        // bitwise gate against the trusted production kernel covers them).
        let macs = m * n * k;
        let (cosine, ref_ran) = if macs <= CPU_REF_MAC_CAP {
            let c_gpu: Vec<u16> = raw_new
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let (mut dot, mut ng, mut nc) = (0f64, 0f64, 0f64);
            for row in 0..m {
                for col in 0..n {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc += bf16_bits_to_f32(a_bf16[row * k + kk])
                            * bf16_bits_to_f32(b_bf16[col * k + kk]);
                    }
                    let r = acc as f64;
                    let g = bf16_bits_to_f32(c_gpu[row * n + col]) as f64;
                    dot += g * r;
                    ng += g * g;
                    nc += r * r;
                }
            }
            (dot / (ng.sqrt() * nc.sqrt() + 1e-30), true)
        } else {
            (1.0, false)
        };

        let pass = bitwise && cosine >= COSINE_GATE;
        all_pass &= pass;
        println!(
            "mtile16 {tag} M={m:<2} N={n:<6} K={k:<5} bitwise-vs-pipelined={} cosine={} {}",
            if bitwise { "YES" } else { "NO " },
            if ref_ran {
                format!("{cosine:.6}")
            } else {
                "(skipped)".to_string()
            },
            if pass { "PASS" } else { "FAIL" }
        );
        for p in [a_ptr, b_ptr, c_new, c_wide, c_old] {
            gpu.free(p).ok();
        }
    }

    // ── Part 2: bench old vs new at the drafter's real propose shapes ──
    let bench_shapes: &[(&str, usize, usize, usize)] = &[
        ("q_proj  ", 16, 9216, 3072),
        ("kv_proj ", 16, 1024, 3072),
        ("o_proj  ", 16, 3072, 9216),
        ("gate/up ", 16, 12288, 3072),
        ("down    ", 16, 3072, 12288),
        ("fc      ", 1, 3072, 18432),
        ("fused_kv", 1, 12288, 3072),
        ("lm_head ", 16, 100352, 3072),
    ];
    for &(tag, m, n, k) in bench_shapes {
        let mut rng = Rng(0xBEEF);
        let a_bf16: Vec<u16> = (0..m * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
            .collect();
        // Bench weights: raw bytes are fine (values don't affect timing).
        let b_raw: Vec<u8> = (0..n * k * 2)
            .map(|_| (rng.next_u64() & 0x3F) as u8)
            .collect();
        let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
        let b_ptr = upload_bytes(gpu, &b_raw)?;
        let c_ptr = gpu.alloc(m * n * 2)?;

        for (name, which) in [("mtile16", 0), ("m16n128", 1), ("pipelined", 2)] {
            let launch = |s| match which {
                0 => launch_mtile16(
                    gpu, h16, a_ptr, b_ptr, c_ptr, m as u32, n as u32, k as u32, s,
                ),
                1 => launch_mtile16_n128(
                    gpu, h128, a_ptr, b_ptr, c_ptr, m as u32, n as u32, k as u32, s,
                ),
                _ => launch_pipelined(
                    gpu, hpipe, a_ptr, b_ptr, c_ptr, m as u32, n as u32, k as u32, s,
                ),
            };
            for _ in 0..10 {
                launch(stream)?;
            }
            gpu.synchronize(stream)?;
            let iters = 100u32;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch(stream)?;
            }
            gpu.synchronize(stream)?;
            let secs = t0.elapsed().as_secs_f64() / iters as f64;
            let wbytes = (n * k * 2) as f64; // BF16 weight bytes = dominant traffic
            println!(
                "BENCH {tag} {name:<9} M={m:<2} N={n:<6} K={k:<5}: {:>8.1} us  weight-read {:>6.1} GB/s",
                secs * 1e6,
                wbytes / secs / 1e9
            );
        }
        for p in [a_ptr, b_ptr, c_ptr] {
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
