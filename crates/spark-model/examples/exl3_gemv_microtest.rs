// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 trellis (3.0 bpw) decode-GEMV microtest — SYNTHETIC-data oracle.
//!
//! Validates `kernels/gb10/common/exl3_gemv.cu` without the real checkpoint:
//! random i16 trellis words are a valid EXL3 payload by construction (the
//! format has no metadata — every 96-B tile is a self-contained circular bit
//! stream), so the decode path is exercised bit-for-bit with zero quant runs.
//!
//! Gates (exit code reflects all of them):
//!   1. DEQUANT BIT-EXACT: `exl3_dequant_dump` output == CPU reference decode
//!      (exact u16 compare over all N*K weights). The CPU decode is an EXACT
//!      port: u32 wrap-mul 0xCBAC1FED, (x & 0x8fff8fff) ^ 0x3b603b60, then an
//!      exact IEEE fp16 add (f64 is wide enough to hold any two-fp16 sum
//!      exactly; single RN-even rounding back to fp16 == GPU `__hadd`).
//!   2. GEMV COSINE >= 0.99999 vs an f64 reference of the FULL pipeline
//!      (suh -> blockwise-128 Sylvester Hadamard -> GEMV -> Hadamard -> svh),
//!      at SPLIT_K = 1 and the production split.
//!   3. COLD-ROTATION GB/s at the expert decode shapes, weights rotated
//!      through a >=512 MB ring so every iteration streams from DRAM.
//!
//!   cargo run -p spark-model --release --example exl3_gemv_microtest \
//!       --features cuda,gpu-examples
//!
//! Trellis bytes counted for GB/s: N*K*3/8 + (N+K)*2 (payload + suh/svh).

use anyhow::{Result, bail};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const PASS_COS: f64 = 0.99999;
const MCG_MULT: u32 = 0xCBAC_1FED;

/// (label, N, K) — the expert matrix decode shapes from EXPERT-3BPW-PLAN §3.
const SHAPES: &[(&str, usize, usize)] = &[
    ("w1/w3 N=2048 K=4096", 2048, 4096),
    ("w2    N=4096 K=2048", 4096, 2048),
];

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64 — no rand dependency)
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn u16(&mut self) -> u16 {
        (self.next() >> 40) as u16
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn sign_f16(&mut self) -> u16 {
        // +1.0 / -1.0 in fp16 — the EXL3 suh/svh are random sign vectors.
        if self.next() & 1 == 0 { 0x3C00 } else { 0xBC00 }
    }
}

// ---------------------------------------------------------------------------
// Exact fp16 <-> f64 (self-contained; avoids any double-rounding hazard)
// ---------------------------------------------------------------------------

/// Exact 2^e for e in [-1022, 1023] (bit construction — no libm rounding).
fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

fn f16_to_f64(b: u16) -> f64 {
    let s = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 10) & 0x1F) as i32;
    let m = (b & 0x3FF) as f64;
    match e {
        0 => s * m * pow2(-24),
        31 => {
            if m == 0.0 {
                s * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => s * (1024.0 + m) * pow2(e - 25),
    }
}

/// Correct RN-even f64 -> fp16 via nearest-neighbour search over the (finite,
/// monotonic) fp16 value lattice. Exact by construction; decoded EXL3 values
/// lie in (-4, 4) so the overflow-to-inf corner never triggers.
fn f64_to_f16(v: f64) -> u16 {
    if v.is_nan() {
        return 0x7E00;
    }
    let s: u16 = if v.is_sign_negative() { 0x8000 } else { 0 };
    let a = v.abs();
    if a >= f16_to_f64(0x7BFF) {
        // >= max finite: RN rounds to inf past 65520 (midpoint), else max
        return if a >= 65520.0 { s | 0x7C00 } else { s | 0x7BFF };
    }
    let (mut lo, mut hi) = (0u16, 0x7BFFu16);
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if f16_to_f64(mid) <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let b0 = lo;
    let b1 = b0 + 1; // <= 0x7C00 never reached (handled above)
    let d0 = a - f16_to_f64(b0);
    let d1 = f16_to_f64(b1) - a;
    let b = if d0 < d1 {
        b0
    } else if d1 < d0 {
        b1
    } else if b0 & 1 == 0 {
        b0
    } else {
        b1
    };
    s | b
}

// ---------------------------------------------------------------------------
// CPU reference: exact EXL3 trellis decode (port of codebook.cuh cb=1 +
// exl3_dq.cuh window geometry, vendored ExLlamaV3, MIT, Turboderp 2025)
// ---------------------------------------------------------------------------

/// 3INST decode of one 16-bit window -> fp16 bits. Pure function of the
/// window, so the microtest builds a 64 Ki LUT once.
fn decode_3inst(w16: u16) -> u16 {
    let x = (w16 as u32).wrapping_mul(MCG_MULT);
    let x = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
    let lo = f16_to_f64(x as u16);
    let hi = f16_to_f64((x >> 16) as u16);
    // Two fp16 values sum exactly in f64 (span <= 51 significand bits);
    // one RN-even rounding == hardware __hadd.
    f64_to_f16(lo + hi)
}

/// Tile linear index t -> (k_in_tile, n_in_tile). This is the m16n8k16 MMA
/// B-fragment order the EXL3 quantizer packs for (exl3_gemm_inner.cuh
/// `dq_dispatch(shb, lane_id << 3, frag_b[n2], frag_b[n2+1])` + the PTX
/// fragment layout): lane = t/8, s = t%8.
fn tile_kn(t: usize) -> (usize, usize) {
    let lane = t >> 3;
    let s = t & 7;
    let n = 8 * (s >> 2) + (lane >> 2);
    let k = 2 * (lane & 3) + (s & 1) + 8 * ((s & 3) >> 1);
    (k, n)
}

/// Decode one 96-B tile (48 LE u16 words) into fp16 bit patterns laid out
/// [k_in_tile][n_in_tile]. Weight t is the 16-bit window ENDING at bit
/// (t+257)*3 of the circular 768-bit stream; bit g of the stream lives in
/// LE-u32 word g/32 at bit position 31 - g%32 (MSB-first within words).
fn decode_tile(words: &[u16], lut: &[u16; 65536], out: &mut [[u16; 16]; 16]) {
    let mut w32 = [0u32; 24];
    for (j, w) in w32.iter_mut().enumerate() {
        *w = words[2 * j] as u32 | ((words[2 * j + 1] as u32) << 16);
    }
    let bit = |g: usize| -> u16 {
        let g = g % 768;
        ((w32[g >> 5] >> (31 - (g & 31))) & 1) as u16
    };
    for t in 0..256 {
        let b0 = (t + 257) * 3 - 16;
        let mut w16 = 0u16;
        for i in 0..16 {
            w16 = (w16 << 1) | bit(b0 + i);
        }
        let (k, n) = tile_kn(t);
        out[k][n] = lut[w16 as usize];
    }
}

/// Blockwise-128 Sylvester Hadamard in f64: y[i] = sum_j (-1)^pc(i&j) x[j],
/// per aligned 128-chunk, scaled 1/sqrt(128).
fn had128(x: &[f64]) -> Vec<f64> {
    let rs = 1.0 / 128f64.sqrt();
    let mut y = vec![0.0; x.len()];
    for (cy, cx) in y.chunks_mut(128).zip(x.chunks(128)) {
        for (i, yi) in cy.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (j, xj) in cx.iter().enumerate() {
                if (i & j).count_ones() & 1 == 0 {
                    acc += xj;
                } else {
                    acc -= xj;
                }
            }
            *yi = acc * rs;
        }
    }
    y
}

// ---------------------------------------------------------------------------
// GPU plumbing
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn cos(a: &[f64], b: &[f64]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-300)
}

#[allow(clippy::too_many_arguments)]
fn launch_gemv(
    g: &dyn GpuBackend,
    kh: spark_runtime::gpu::KernelHandle,
    stream: u64,
    split: u32,
    a: DevicePtr,
    trellis: DevicePtr,
    suh: DevicePtr,
    svh: DevicePtr,
    c: DevicePtr,
    ws: DevicePtr,
    counters: DevicePtr,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([(n / 128) as u32, split, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(trellis)
        .arg_ptr(suh)
        .arg_ptr(svh)
        .arg_ptr(c)
        .arg_ptr(ws)
        .arg_ptr(counters)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let stream = 0u64;

    let kh_gemv = g.kernel("exl3_gemv", "exl3_gemv_m1")?;
    let kh_dump = g.kernel("exl3_gemv", "exl3_dequant_dump")?;

    // 3INST LUT: decode is a pure function of the 16-bit window.
    let mut lut = Box::new([0u16; 65536]);
    for (w, l) in lut.iter_mut().enumerate() {
        *l = decode_3inst(w as u16);
    }

    let mut all_ok = true;

    for &(label, n, k) in SHAPES {
        let mut r = Rng(0xE7_1337 ^ ((n as u64) << 20) ^ k as u64);
        let tiles_k = k / 16;
        let tiles_n = n / 16;
        let trellis_words = tiles_k * tiles_n * 48;
        let payload_bytes = n * k * 3 / 8 + (n + k) * 2;

        // ---- synthetic tensors ----
        let trellis_host: Vec<u16> = (0..trellis_words).map(|_| r.u16()).collect();
        let suh_host: Vec<u16> = (0..k).map(|_| r.sign_f16()).collect();
        let svh_host: Vec<u16> = (0..n).map(|_| r.sign_f16()).collect();
        let a_host: Vec<u16> = (0..k)
            .map(|_| bf16::from_f32((r.unit() - 0.5) * 0.5).to_bits())
            .collect();

        // ---- CPU reference decode: W bits [N][K] ----
        let mut w_bits = vec![0u16; n * k];
        let mut tile = [[0u16; 16]; 16];
        for kb in 0..tiles_k {
            for nb in 0..tiles_n {
                let base = (kb * tiles_n + nb) * 48;
                decode_tile(&trellis_host[base..base + 48], &lut, &mut tile);
                for (kit, row) in tile.iter().enumerate() {
                    for (nit, bits) in row.iter().enumerate() {
                        w_bits[(nb * 16 + nit) * k + kb * 16 + kit] = *bits;
                    }
                }
            }
        }

        // ---- CPU reference GEMV (f64, full pipeline) ----
        let x: Vec<f64> = a_host
            .iter()
            .zip(&suh_host)
            .map(|(&a, &s)| bf16::from_bits(a).to_f64() * f16_to_f64(s))
            .collect();
        let xp = had128(&x);
        let mut y0 = vec![0.0f64; n];
        for (nn, y) in y0.iter_mut().enumerate() {
            let row = &w_bits[nn * k..(nn + 1) * k];
            let mut acc = 0.0;
            for (kk, &wb) in row.iter().enumerate() {
                acc += f16_to_f64(wb) * xp[kk];
            }
            *y = acc;
        }
        let y_ref: Vec<f64> = had128(&y0)
            .iter()
            .zip(&svh_host)
            .map(|(&v, &s)| v * f16_to_f64(s))
            .collect();

        // ---- upload ----
        let to_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let d_trellis = up(g, &to_bytes(&trellis_host))?;
        let d_suh = up(g, &to_bytes(&suh_host))?;
        let d_svh = up(g, &to_bytes(&svh_host))?;
        let d_a = up(g, &to_bytes(&a_host))?;
        let d_c = g.alloc(n * 2)?;
        let d_w = g.alloc(n * k * 2)?;
        // Split sweep widened 4 -> 12: at N=2048 the strip grid is 16 CTAs, and
        // with 2 blocks/SM the GB10 has 96 slots — split 6 fills them exactly.
        // First hardware run measured 135-151 GB/s at splits 1-3 (underfilled).
        let max_split = 12usize;
        let d_ws = g.alloc(max_split * n * 4)?;
        let d_cnt = g.alloc((n / 128) * 4)?;
        g.memset(d_cnt, 0, (n / 128) * 4)?;

        // ---- GATE 1: dequant bit-exact ----
        KernelLaunch::new(g, kh_dump)
            .grid([tiles_n as u32, tiles_k as u32, 1])
            .block([32, 1, 1])
            .arg_ptr(d_trellis)
            .arg_ptr(d_w)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        g.synchronize(stream)?;
        let mut w_gpu = vec![0u8; n * k * 2];
        g.copy_d2h(d_w, &mut w_gpu)?;
        let bitdiff = w_gpu
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .zip(&w_bits)
            .filter(|(a, b)| a != *b)
            .count();
        let g1 = bitdiff == 0;
        all_ok &= g1;
        eprintln!(
            "{label}  GATE1 dequant bit-exact: bitdiff={bitdiff}/{}  {}",
            n * k,
            if g1 { "PASS" } else { "FAIL" }
        );

        // ---- GATE 2: GEMV cosine, SPLIT_K = 1 and production split ----
        let strips = n / 128;
        let prod_split = (48usize.div_ceil(strips)).clamp(1, max_split) as u32;
        for split in [1u32, prod_split] {
            launch_gemv(
                g, kh_gemv, stream, split, d_a, d_trellis, d_suh, d_svh, d_c, d_ws, d_cnt, n, k,
            )?;
            g.synchronize(stream)?;
            let mut c_gpu = vec![0u8; n * 2];
            g.copy_d2h(d_c, &mut c_gpu)?;
            let y_gpu: Vec<f64> = c_gpu
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
                .collect();
            let cs = cos(&y_gpu, &y_ref);
            let max_d = y_gpu
                .iter()
                .zip(&y_ref)
                .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
            let ok = cs >= PASS_COS;
            all_ok &= ok;
            eprintln!(
                "{label}  GATE2 gemv split={split}: cos={cs:.8} max|d|={max_d:.5}  {}",
                if ok { "PASS" } else { "FAIL" }
            );
            if split == prod_split && split > 1 {
                // determinism probe: relaunch, must byte-match itself
                launch_gemv(
                    g, kh_gemv, stream, split, d_a, d_trellis, d_suh, d_svh, d_c, d_ws, d_cnt, n, k,
                )?;
                g.synchronize(stream)?;
                let mut c2 = vec![0u8; n * 2];
                g.copy_d2h(d_c, &mut c2)?;
                let det = c2 == c_gpu;
                all_ok &= det;
                eprintln!(
                    "{label}  GATE2b split={split} relaunch byte-identical: {}",
                    if det { "PASS" } else { "FAIL" }
                );
            }
        }

        // ---- GATE 3 (informational): cold-rotation GB/s ----
        // Ring >= 512 MB of distinct weight instances so every iteration is a
        // cold DRAM stream (GB10 L2 is 24 MB; a single hot buffer measures L2).
        let ring_len = ((512usize << 20) / payload_bytes).max(2);
        let ring: Vec<(DevicePtr, DevicePtr, DevicePtr)> = (0..ring_len)
            .map(|_| -> Result<_> {
                let t = g.alloc(trellis_words * 2)?;
                let su = g.alloc(k * 2)?;
                let sv = g.alloc(n * 2)?;
                Ok((t, su, sv))
            })
            .collect::<Result<_>>()?;
        // contents don't matter for timing; leave garbage (sized correctly)
        let iters = 400u32;
        let time_split = |split: u32| -> Result<f64> {
            for i in 0..20u32 {
                let (t, su, sv) = ring[i as usize % ring_len];
                launch_gemv(
                    g, kh_gemv, stream, split, d_a, t, su, sv, d_c, d_ws, d_cnt, n, k,
                )?;
            }
            g.synchronize(stream)?;
            let (mut ev0, mut ev1): (u64, u64) = (0, 0);
            unsafe {
                if cuEventCreate(&mut ev0, 0) != 0 || cuEventCreate(&mut ev1, 0) != 0 {
                    bail!("cuEventCreate failed");
                }
                if cuEventRecord(ev0, stream) != 0 {
                    bail!("cuEventRecord failed");
                }
            }
            for i in 0..iters {
                let (t, su, sv) = ring[i as usize % ring_len];
                launch_gemv(
                    g, kh_gemv, stream, split, d_a, t, su, sv, d_c, d_ws, d_cnt, n, k,
                )?;
            }
            let mut ms = 0f32;
            unsafe {
                if cuEventRecord(ev1, stream) != 0 || cuEventSynchronize(ev1) != 0 {
                    bail!("event sync failed");
                }
                if cuEventElapsedTime(&mut ms, ev0, ev1) != 0 {
                    bail!("cuEventElapsedTime failed");
                }
                cuEventDestroy_v2(ev0);
                cuEventDestroy_v2(ev1);
            }
            Ok(ms as f64 / iters as f64)
        };
        for split in 1u32..=(max_split as u32) {
            let ms = time_split(split)?;
            let gbs = payload_bytes as f64 / (ms * 1e-3) / 1e9;
            eprintln!(
                "{label}  GATE3 split={split}: {:.1} us/iter  {gbs:.1} GB/s (ring={ring_len}, ceiling 229)",
                ms * 1e3
            );
        }
        eprintln!();

        for (t, su, sv) in ring {
            let _ = g.free(t);
            let _ = g.free(su);
            let _ = g.free(sv);
        }
        for p in [d_trellis, d_suh, d_svh, d_a, d_c, d_w, d_ws, d_cnt] {
            let _ = g.free(p);
        }
    }

    eprintln!(
        "EXL3 GEMV GATE (bit-exact dequant + cos>={PASS_COS} + determinism): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
