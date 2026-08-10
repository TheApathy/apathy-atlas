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

/// CPU E4M3 decode (bias 7, 3 mantissa bits; subnormals at exp=0). The
/// generators avoid the NaN encodings (0x_F7/0x_FF class), so no NaN arm.
fn e4m3_to_f64(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let mant = (b & 7) as f64;
    if exp == 0 {
        sign * (mant / 8.0) * 2f64.powi(-6)
    } else {
        sign * (1.0 + mant / 8.0) * 2f64.powi(exp - 7)
    }
}

fn bf16_bits_to_f64(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let w8a16_h = gpu.kernel("w8a16_gemm_pipelined", "w8a16_gemm_pipelined")?;
    let w8a8_h = gpu.kernel("w8a8_gemm_pipelined", "w8a8_gemm_pipelined")?;
    let w8a8_ld_h = gpu.kernel("w8a8_gemm_pipelined", "w8a8_gemm_pipelined_ld")?;
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

    // ── w8a8_gemm_pipelined_ld: strided-A/C sibling ──
    // (a) lda=K/ldc=N must be BYTE-IDENTICAL to the packed entry (shared impl).
    // (b) grouped-slice case (the V4 wo_a shape class, G groups over one
    //     full-row quantization): each group's strided run must be
    //     byte-identical to the packed kernel on a host-gathered copy of its
    //     A slice with the SAME full-row scales — proving the kernel applies
    //     a_row_scale[m] uniformly so a full-row scale is exact on a k-slice —
    //     and cosine >= 0.999 vs an f64 reference from the ORIGINAL BF16
    //     activations (bounding the full-row-absmax quantization coarsening).
    {
        const GROUPS: usize = 2;
        let (m, n, k) = (640usize, 1024usize, 4096usize);
        let kw = GROUPS * k;
        let nw = GROUPS * n;

        let a_bf16: Vec<u16> = (0..m * kw)
            .map(|_| f32_to_bf16_bits(rng.uniform(-2.0, 2.0)))
            .collect();
        let a_bytes: Vec<u8> = a_bf16.iter().flat_map(|x| x.to_le_bytes()).collect();
        let b_fp8 = gen_fp8(&mut rng, nw * k); // GROUPS stacked [n, k] weights
        let scales: Vec<u8> = (0..(nw / 128) * (k / 128))
            .flat_map(|_| rng.uniform(0.002, 0.02).to_le_bytes())
            .collect();
        let scales_f32: Vec<f32> = scales
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let a_d = upload(&gpu, &a_bytes)?;
        let b_d = upload(&gpu, &b_fp8)?;
        let s_d = upload(&gpu, &scales)?;
        let a8_d = gpu.alloc(m * kw)?;
        let rs_d = gpu.alloc(m * 4)?;
        let c_wide_d = gpu.alloc(m * nw * 2)?;
        let c_ref_d = gpu.alloc(m * n * 2)?;
        let a8_slice_d = gpu.alloc(m * k)?;

        // Quantize the FULL wide rows once (K = kw).
        KernelLaunch::new(&gpu, quant_h)
            .grid([m as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_d)
            .arg_ptr(a8_d)
            .arg_ptr(rs_d)
            .arg_u32(m as u32)
            .arg_u32(kw as u32)
            .launch(stream)?;

        // (a) Packed identity: group 0's gathered slice through BOTH entries.
        let mut a8_host = vec![0u8; m * kw];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(a8_d, &mut a8_host)?;
        let launch_ld =
            |a: DevicePtr, c: DevicePtr, lda: usize, ldc: usize, sg: DevicePtr, bg: DevicePtr| {
                KernelLaunch::new(&gpu, w8a8_ld_h)
                    .grid([(n.div_ceil(32)) as u32, (m.div_ceil(128)) as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(a)
                    .arg_ptr(rs_d)
                    .arg_ptr(bg)
                    .arg_ptr(sg)
                    .arg_ptr(c)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .arg_u32(lda as u32)
                    .arg_u32(ldc as u32)
                    .launch(stream)
            };
        let mut ld_pass = true;
        for g in 0..GROUPS {
            // Host-gather group g's A slice into a packed [m, k] buffer.
            let mut a8_slice = vec![0u8; m * k];
            for row in 0..m {
                a8_slice[row * k..(row + 1) * k]
                    .copy_from_slice(&a8_host[row * kw + g * k..row * kw + (g + 1) * k]);
            }
            gpu.copy_h2d(&a8_slice, a8_slice_d)?;
            let b_g = b_d.offset(g * n * k);
            let s_g = s_d.offset(g * (n / 128) * (k / 128) * 4);
            // Packed reference on the gathered slice (same full-row scales).
            KernelLaunch::new(&gpu, w8a8_h)
                .grid([(n.div_ceil(32)) as u32, (m.div_ceil(128)) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a8_slice_d)
                .arg_ptr(rs_d)
                .arg_ptr(b_g)
                .arg_ptr(s_g)
                .arg_ptr(c_ref_d)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)?;
            // Strided run straight off the wide quantized A, in place in C.
            launch_ld(
                a8_d.offset(g * k),
                c_wide_d.offset(g * n * 2),
                kw,
                nw,
                s_g,
                b_g,
            )?;
            gpu.synchronize(stream)?;

            let mut c_ref = vec![0u8; m * n * 2];
            let mut c_wide = vec![0u8; m * nw * 2];
            gpu.copy_d2h(c_ref_d, &mut c_ref)?;
            gpu.copy_d2h(c_wide_d, &mut c_wide)?;
            let ident = (0..m).all(|row| {
                c_wide[row * nw * 2 + g * n * 2..row * nw * 2 + (g + 1) * n * 2]
                    == c_ref[row * n * 2..(row + 1) * n * 2]
            });

            // f64 reference from the ORIGINAL BF16 activations. Subsampled
            // (co-prime strides cover all 128-blocks and MMA lane roles) to
            // keep the CPU triple loop tractable; the packed-identity check
            // above already covers every element bit-for-bit.
            let (mut dot, mut nx, mut ny) = (0f64, 0f64, 0f64);
            for row in (0..m).step_by(7) {
                for col in (0..n).step_by(13) {
                    let mut acc = 0f64;
                    for kk in 0..k {
                        let a = bf16_bits_to_f64(a_bf16[row * kw + g * k + kk]);
                        let b = e4m3_to_f64(b_fp8[(g * n + col) * k + kk]);
                        let bs =
                            scales_f32[(g * (n / 128) + col / 128) * (k / 128) + kk / 128] as f64;
                        acc += a * b * bs;
                    }
                    let got = {
                        let off = row * nw * 2 + (g * n + col) * 2;
                        bf16_bits_to_f64(u16::from_le_bytes([c_wide[off], c_wide[off + 1]]))
                    };
                    dot += acc * got;
                    nx += acc * acc;
                    ny += got * got;
                }
            }
            let cos = dot / (nx.sqrt() * ny.sqrt());
            let ok = ident && cos >= 0.999;
            ld_pass &= ok;
            println!(
                "  w8a8_ld group {g}/{GROUPS} M={m} N={n} K={k} lda={kw} ldc={nw}: \
                 packed-identity={ident} cos_f64={cos:.6} {}",
                if ok { "PASS" } else { "FAIL" }
            );
        }
        // (b) Degenerate strides lda=K/ldc=N == packed, byte-identical.
        launch_ld(a8_slice_d, c_ref_d, k, n, s_d, b_d)?;
        let mut c_ld = vec![0u8; m * n * 2];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(c_ref_d, &mut c_ld)?;
        KernelLaunch::new(&gpu, w8a8_h)
            .grid([(n.div_ceil(32)) as u32, (m.div_ceil(128)) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a8_slice_d)
            .arg_ptr(rs_d)
            .arg_ptr(b_d)
            .arg_ptr(s_d)
            .arg_ptr(c_ref_d)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        let mut c_packed = vec![0u8; m * n * 2];
        gpu.synchronize(stream)?;
        gpu.copy_d2h(c_ref_d, &mut c_packed)?;
        let degenerate_ident = c_ld == c_packed;
        ld_pass &= degenerate_ident;
        println!(
            "  w8a8_ld lda=K ldc=N degenerate identity: {} {}",
            degenerate_ident,
            if degenerate_ident { "PASS" } else { "FAIL" }
        );
        all_pass &= ld_pass;

        for p in [a_d, b_d, s_d, a8_d, rs_d, c_wide_d, c_ref_d, a8_slice_d] {
            let _ = gpu.free(p);
        }
    }

    if !all_pass {
        bail!("w8a8 oracle FAIL");
    }
    println!("PASS");
    Ok(())
}
