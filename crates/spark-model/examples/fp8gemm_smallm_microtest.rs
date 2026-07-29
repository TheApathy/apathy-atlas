// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone correctness oracle + bench for `fp8_gemm_t_row_scaled_mtile8`
//! (module `w4a16`) — the weight-read-bound M ≤ 8 row-scaled FP8 GEMM the
//! attention FP8-mirror verify projections launch via
//! `ops::fp8_gemm_row_scaled_smallm`.
//!
//!   C[M,N] = A[M,K] (BF16) · decode_e4m3(B[N,K])^T · diag(row_scale[N])
//!   (FP32 accumulation, BF16 write-out)
//!
//! Part 1 — correctness: mtile8 vs a CPU FP32 reference across edge shapes
//!   (M 1..8, ragged N, all Laguna mirror K's). Cosine gate 0.99 per shape.
//! Part 2 — bench: mtile8 vs fp8_gemm_t_row_scaled (M_TILE=64) and
//!   fp8_gemm_t_row_scaled_m16 at the real attention-mirror shapes
//!   (M=7: qkv N=6144/9216 K=3072, oproj N=3072 K=6144/9216), reporting
//!   effective weight-read GB/s (= N*K bytes / time).
//!
//! Usage: cargo run --release -p spark-model --example fp8gemm_smallm_microtest \
//!          --features cuda,gpu-examples
//! Exit 0 = PASS (all shapes >= gate), 1 = FAIL — scriptable.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const COSINE_GATE: f64 = 0.99;

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

// Standard OCP E4M3 (1-4-3, bias 7) decode.
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        0.0 // NaN -> 0
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

fn f32_to_e4m3(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for b in 0..=255u8 {
        let d = e4m3_to_f32(b);
        if !d.is_finite() {
            continue;
        }
        let e = (d - v).abs();
        if e < best_err {
            best_err = e;
            best = b;
        }
    }
    best
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

#[allow(clippy::too_many_arguments)]
fn launch_ntile(
    gpu: &dyn GpuBackend,
    h: spark_runtime::gpu::KernelHandle,
    n_tile: u32,
    a: DevicePtr,
    b: DevicePtr,
    s: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([div_ceil(n, n_tile), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(s)
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
    let h8 = gpu.kernel("w4a16", "fp8_gemm_t_row_scaled_mtile8")?;
    let h8n32 = gpu.kernel("w4a16", "fp8_gemm_t_row_scaled_mtile8_n32")?;
    let h64 = gpu.kernel("w4a16", "fp8_gemm_t_row_scaled")?;
    let h16 = gpu.kernel("w4a16", "fp8_gemm_t_row_scaled_m16")?;

    // ── Part 1: correctness vs CPU reference ──────────────────────────
    // Laguna mirror shapes (M=7) + edge cases: M=1/8, ragged N (not a
    // multiple of 64), K=32 minimum, K%64==32 (odd 32-step tail).
    let shapes: &[(usize, usize, usize)] = &[
        (7, 6144, 3072), // qkv q-proj, 48-head layer
        (7, 9216, 3072), // qkv q-proj, 72-head layer
        (7, 3072, 6144), // oproj, 48-head layer
        (7, 3072, 9216), // oproj, 72-head layer
        (8, 1024, 3072), // kv-proj-ish, full M tile
        (1, 3072, 3072), // single row
        (5, 1000, 2048), // ragged N
        (3, 192, 96),    // tiny, K%64==32 tail
        (8, 64, 32),     // single N tile, single K step
    ];
    let mut all_pass = true;
    for &(m, n, k) in shapes {
        let mut rng = Rng(0x51A7 ^ ((m * 31 + n * 7 + k) as u64));
        let a_bf16: Vec<u16> = (0..m * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
            .collect();
        let b_fp8: Vec<u8> = (0..n * k)
            .map(|_| f32_to_e4m3(rng.uniform(-0.5, 0.5)))
            .collect();
        let scales: Vec<f32> = (0..n).map(|_| rng.uniform(0.5, 2.0)).collect();

        let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
        let b_ptr = upload_bytes(gpu, &b_fp8)?;
        let s_ptr = upload_bytes(gpu, &f32s_to_le(&scales))?;
        let c_ptr = gpu.alloc(m * n * 2)?;

        // CPU reference: C[m,n] = row_scale[n] * sum_k A[m,k]*e4m3(B[n,k]).
        // Computed once per shape, checked against both N-tile variants.
        let mut cpu_ref = vec![0f64; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc +=
                        bf16_bits_to_f32(a_bf16[row * k + kk]) * e4m3_to_f32(b_fp8[col * k + kk]);
                }
                cpu_ref[row * n + col] = (acc * scales[col]) as f64;
            }
        }

        for (name, handle, n_tile) in [("mtile8", h8, 64u32), ("mtile8n32", h8n32, 32u32)] {
            // Poison C so "kernel wrote nothing" can't pass the gate.
            gpu.copy_h2d(&vec![0x7Fu8; m * n * 2], c_ptr)?;
            launch_ntile(
                gpu, handle, n_tile, a_ptr, b_ptr, s_ptr, c_ptr, m as u32, n as u32, k as u32,
                stream,
            )?;
            gpu.synchronize(stream)?;

            let mut c_raw = vec![0u8; m * n * 2];
            gpu.copy_d2h(c_ptr, &mut c_raw)?;
            let c_gpu: Vec<u16> = c_raw
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();

            let (mut dot, mut ng, mut nc, mut max_rel) = (0f64, 0f64, 0f64, 0f64);
            let mut nan_count = 0usize;
            for row in 0..m {
                for col in 0..n {
                    let r = cpu_ref[row * n + col];
                    let g = bf16_bits_to_f32(c_gpu[row * n + col]) as f64;
                    if !g.is_finite() {
                        nan_count += 1;
                    }
                    dot += g * r;
                    ng += g * g;
                    nc += r * r;
                    max_rel = max_rel.max((g - r).abs() / r.abs().max(1e-3));
                }
            }
            let cosine = dot / (ng.sqrt() * nc.sqrt() + 1e-30);
            let pass = cosine >= COSINE_GATE && nan_count == 0;
            all_pass &= pass;
            println!(
                "{name:<9} M={m:<2} N={n:<5} K={k:<5} cosine={cosine:.6} max_rel={max_rel:.3e} \
                 nan={nan_count} {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        for p in [a_ptr, b_ptr, s_ptr, c_ptr] {
            gpu.free(p).ok();
        }
    }

    // ── Part 2: COLD bench vs the M64 / _m16 tiles at the mirror shapes ──
    //
    // Every iteration reads a DIFFERENT copy of the FP8 weight (round-robin
    // over ROT buffers, ≥150MB aggregate footprint) so the weight bytes are
    // never L2-resident — matching the serve-side reality where each of the
    // 48 layers' mirrors is cold every verify step. (The earlier single-buffer
    // bench was L2-hot and overstated GB/s at these ≤28MB weights.)
    // FOOTPRINT KNOB (`ATLAS_MICRO_ROT`, default 8). ROT controls the aggregate
    // resident footprint the bench walks: ROT copies of an N*K FP8 weight.
    //
    // Why this is the interesting variable, not a tuning detail. At ROT=8 these
    // kernels hit 200-215 GB/s (82-88% of GB10's 245 GB/s usable wall), but the
    // SAME kernel at the SAME shape measures ~1.4-1.5x slower inside the serve
    // loop (nsys vgp.sqlite, decode steady state: q_proj p50 = 140us here vs
    // 92us standalone). Sustained-throttle was falsified -- 8 back-to-back
    // rounds showed zero decay. The remaining structural difference is
    // FOOTPRINT: the bench walks ~150 MB while the serve loop walks a ~47 GB
    // resident weight set in expert-selection order. 47 GB does not fit any
    // plausible GPU TLB reach, so every weight access risks a page walk that
    // the 150 MB bench never pays.
    //
    // Sweeping ROT interpolates between those two worlds on ONE kernel with
    // everything else held fixed. If GB/s decays as the footprint grows toward
    // tens of GB, the 46% stack-wide efficiency gap is address-translation, not
    // arithmetic and not bandwidth -- which would explain why cutting expert
    // BYTES (W3, -22%) lost while leaving the footprint's page count intact.
    let rot_env: usize = std::env::var("ATLAS_MICRO_ROT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let rot = rot_env.max(1);
    #[allow(non_snake_case)]
    let ROT = rot;
    let bench_shapes: &[(&str, usize, usize, usize)] = &[
        ("qkv-q48", 7, 6144, 3072),
        ("qkv-q72", 7, 9216, 3072),
        ("oproj48", 7, 3072, 6144),
        ("oproj72", 7, 3072, 9216),
        // ── The DFlash DRAFTER-step shapes (M=5, 94% of steps) ──
        // The four above are all M=7 (the RETRIEVAL regime, 6% of steps) and
        // all have N >= 3072. They MISS the K/V projections entirely: Laguna
        // has 8 kv-heads x 128 = N=1024, so `fp8_mirror_gemm` sends k_proj and
        // v_proj to the N_TILE=64 kernel (k=3072 fails its `k >= 4096` n32
        // gate) at a grid of ceil(1024/64) = 16 CTAs — one third of GB10's 48
        // SMs, and 1/6 of the in-flight cp.async depth the q/o shapes get.
        // That is the exact pathology the _n32 variant was written for, at a
        // shape the dispatch predicate does not route to it.
        ("kv-m5", 5, 1024, 3072),
        ("q-m5", 5, 6144, 3072),
        ("o-m5", 5, 3072, 6144),
    ];
    // Shape filter (`ATLAS_MICRO_SHAPE`, default all). A footprint sweep only
    // needs one shape, and at large ROT the upload alone is tens of GB — so
    // restrict rather than pay it four times over.
    let shape_filter = std::env::var("ATLAS_MICRO_SHAPE").ok();
    for &(tag, m, n, k) in bench_shapes {
        if let Some(f) = shape_filter.as_deref()
            && tag != f
        {
            continue;
        }
        let mut rng = Rng(0xBEEF);
        let a_bf16: Vec<u16> = (0..m * k)
            .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
            .collect();
        // Bench weights: raw bytes are fine (values don't affect timing);
        // avoid the O(256·N·K) nearest-code search on 28MB matrices.
        let b_fp8: Vec<u8> = (0..n * k).map(|_| (rng.next_u64() & 0x77) as u8).collect();
        let scales: Vec<f32> = (0..n).map(|_| rng.uniform(0.5, 2.0)).collect();
        let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
        let b_ptrs: Vec<DevicePtr> = (0..ROT)
            .map(|_| upload_bytes(gpu, &b_fp8))
            .collect::<Result<_>>()?;
        let s_ptr = upload_bytes(gpu, &f32s_to_le(&scales))?;
        let c_ptr = gpu.alloc(m * n * 2)?;

        for (name, which) in [("mtile8", 0), ("mtile8n32", 3), ("m64tile", 1), ("m16", 2)] {
            let launch = |s, b_ptr| match which {
                0 => launch_ntile(
                    gpu, h8, 64, a_ptr, b_ptr, s_ptr, c_ptr, m as u32, n as u32, k as u32, s,
                ),
                3 => launch_ntile(
                    gpu, h8n32, 32, a_ptr, b_ptr, s_ptr, c_ptr, m as u32, n as u32, k as u32, s,
                ),
                1 => KernelLaunch::new(gpu, h64)
                    .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 64), 1])
                    .block([128, 1, 1])
                    .arg_ptr(a_ptr)
                    .arg_ptr(b_ptr)
                    .arg_ptr(s_ptr)
                    .arg_ptr(c_ptr)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(s),
                _ => KernelLaunch::new(gpu, h16)
                    .grid([div_ceil(n as u32, 128), 1, 1])
                    .block([32, 1, 1])
                    .arg_ptr(a_ptr)
                    .arg_ptr(b_ptr)
                    .arg_ptr(s_ptr)
                    .arg_ptr(c_ptr)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(s),
            };
            for i in 0..2 * ROT as u32 {
                launch(stream, b_ptrs[i as usize % ROT])?;
            }
            gpu.synchronize(stream)?;
            let iters = 200u32;
            let t0 = std::time::Instant::now();
            for i in 0..iters {
                launch(stream, b_ptrs[i as usize % ROT])?;
            }
            gpu.synchronize(stream)?;
            let secs = t0.elapsed().as_secs_f64() / iters as f64;
            let wbytes = (n * k) as f64; // FP8 weight bytes = dominant traffic
            println!(
                "BENCH-COLD {tag} {name:<9} M={m} N={n} K={k}: {:.1} us  weight-read {:.1} GB/s",
                secs * 1e6,
                wbytes / secs / 1e9
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
