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
//! P1 prefill-path gates (M=64 rows over 4 experts, chunk=2 — exercises the
//! sub-range grouped-GEMM launches exactly as forward_prefill_exl3.rs):
//!   4. `exl3_h128_pre_rows` BIT-EXACT vs a CPU f32 replica of the GPU
//!      Hadamard op order (4-point in-register stage + 5 fma xor-stages),
//!      including the sorted_token_ids gather and per-expert suh.
//!   5. `exl3_dequant_chunk_bf16` BIT-EXACT vs CPU decode + fp16→bf16 RN.
//!   6. `exl3_h128_post_rows` BIT-EXACT via download-before/after: the CPU
//!      replica applied to the pre-post GEMM output must byte-match the
//!      kernel's in-place result.
//!   7. FULL PREFILL PATH (pre → chunked dequant + sub-range
//!      `moe_bf16_grouped_gemm` → post) COSINE >= 0.999 vs the f64 full-
//!      pipeline reference at M=64 (per-row expert routing, gathered A).
//!   P2. DIRECT GROUPED PREFILL == P1, BYTE-IDENTICAL for K2 and K3 at
//!      1/63/64/65/129 routed rows per expert. Different poison bytes ensure
//!      unwritten output cannot compare equal.
//!   P2-POST. FUSED H128-POST + SWIGLU + DOWN H128-PRE == the four-kernel
//!      composition, BYTE-IDENTICAL with distinct per-expert sign vectors.
//!   P2-TAIL. FUSED DOWN H128-POST + INDEXED UNPERMUTE == the two-kernel
//!      composition, BYTE-IDENTICAL with nontrivial routing and weights.
//!
//! Fused decode-dispatch gate (the launch collapse in `exl3_decode.rs`):
//!   8. FUSED == PER-SLOT, BYTE-IDENTICAL. The same synthetic experts and
//!      routing are pushed through both the per-slot bring-up chain
//!      (`exl3_gemv_m1_idx` × 3·top_k + `moe_silu_mul` × top_k) and the fused
//!      pair (`exl3_gemv_m1_fused_gate_up` + one flat `moe_silu_mul` +
//!      `exl3_gemv_m1_fused_down`); all three output buffers must byte-match,
//!      the fused path must be self-byte-identical across a relaunch (split-K
//!      determinism with per-slot scratch regions), and the launch counts are
//!      asserted (4·top_k → 3 per layer).
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
///
/// The N=128 shape is a DRAM-pattern DIAGNOSTIC, not a production shape: at
/// N=128 there is one tile-strip, so each CTA's trellis stream is perfectly
/// contiguous (row stride == strip width == 768 B). The production shapes
/// read 768-B islands with an N/16*96-B jump between them; if the ~156 GB/s
/// plateau is island-pattern inefficiency, this shape clears it with the
/// SAME kernel. K=32768 keeps the payload comparable (1.57 MB).
const SHAPES: &[(&str, usize, usize)] = &[
    ("w1/w3 N=2048 K=4096", 2048, 4096),
    ("w2    N=4096 K=2048", 4096, 2048),
    ("diag  N=128  K=32768", 128, 32768),
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
fn decode_tile(words: &[u16], bits: usize, lut: &[u16; 65536], out: &mut [[u16; 16]; 16]) {
    let words32 = 8 * bits;
    let mut w32 = [0u32; 24];
    for (j, w) in w32.iter_mut().take(words32).enumerate() {
        *w = words[2 * j] as u32 | ((words[2 * j + 1] as u32) << 16);
    }
    let bit = |g: usize| -> u16 {
        let g = g % (256 * bits);
        ((w32[g >> 5] >> (31 - (g & 31))) & 1) as u16
    };
    for t in 0..256 {
        let b0 = (t + 257) * bits - 16;
        let mut w16 = 0u16;
        for i in 0..16 {
            w16 = (w16 << 1) | bit(b0 + i);
        }
        let (k, n) = tile_kn(t);
        out[k][n] = lut[w16 as usize];
    }
}

/// CPU f32 replica of the GPU `exl3_had128` op ORDER over one 128-chunk
/// (element i owned by lane i/4, slot i%4): the in-register 4-point stage,
/// then 5 xor shuffle-stages computed as `fma(sgn, h_old, p_old)`. Bit-exact
/// vs the kernel by construction (`f32::mul_add` == `__fmaf_rn`; sgn = ±1 so
/// the product is exact; both round once per op).
fn had128_f32_gpu(x: &mut [f32; 128]) {
    for l in 0..32 {
        let (a, b, c, d) = (x[4 * l], x[4 * l + 1], x[4 * l + 2], x[4 * l + 3]);
        let (s0, d0, s1, d1) = (a + b, a - b, c + d, c - d);
        x[4 * l] = s0 + s1;
        x[4 * l + 1] = d0 + d1;
        x[4 * l + 2] = s0 - s1;
        x[4 * l + 3] = d0 - d1;
    }
    for i in [1usize, 2, 4, 8, 16] {
        let old = *x;
        for l in 0..32 {
            let sgn: f32 = if l & i != 0 { -1.0 } else { 1.0 };
            for r in 0..4 {
                x[4 * l + r] = sgn.mul_add(old[4 * l + r], old[4 * (l ^ i) + r]);
            }
        }
    }
}

/// GPU constant from exl3_gemv.cu (EXL3_RSQRT128) — must match bit-for-bit
/// for the H128 bit-exact gates.
const RSQRT128_F32: f32 = 0.088388347648;

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
        .arg_u32(3)
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

    // K2 contract gate: validate the 64-byte tile geometry independently of
    // the established K3 end-to-end suite below.
    {
        let (n, k, bits) = (128usize, 128usize, 2usize);
        let mut r = Rng(0x4B32_0002);
        let trellis: Vec<u16> = (0..(n / 16) * (k / 16) * 16 * bits)
            .map(|_| r.u16())
            .collect();
        let mut expected = vec![0u16; n * k];
        let mut tile = [[0u16; 16]; 16];
        for kb in 0..k / 16 {
            for nb in 0..n / 16 {
                let base = (kb * (n / 16) + nb) * 16 * bits;
                decode_tile(&trellis[base..base + 16 * bits], bits, &lut, &mut tile);
                for (ki, row) in tile.iter().enumerate() {
                    for (ni, value) in row.iter().enumerate() {
                        expected[(nb * 16 + ni) * k + kb * 16 + ki] = *value;
                    }
                }
            }
        }
        let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let d_t = up(g, &bytes(&trellis))?;
        let d_w = g.alloc(n * k * 2)?;
        KernelLaunch::new(g, kh_dump)
            .grid([(n / 16) as u32, (k / 16) as u32, 1])
            .block([32, 1, 1])
            .arg_ptr(d_t)
            .arg_ptr(d_w)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(bits as u32)
            .launch(stream)?;
        g.synchronize(stream)?;
        let mut got = vec![0u8; n * k * 2];
        g.copy_d2h(d_w, &mut got)?;
        let diff = got
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .zip(expected)
            .filter(|(a, b)| *a != *b)
            .count();
        eprintln!(
            "K2 GATE0 dequant bit-exact: diff={diff} {}",
            if diff == 0 { "PASS" } else { "FAIL" }
        );
        all_ok &= diff == 0;
        g.free(d_t)?;
        g.free(d_w)?;
    }

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
                decode_tile(&trellis_host[base..base + 48], 3, &lut, &mut tile);
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
        // with the round-2 kernel at 4 blocks/SM the GB10 has 192 slots —
        // split 12 fills them exactly (N=4096: 32 strips, split 6).
        // First hardware run measured 135-151 GB/s at splits 1-3 (underfilled);
        // round 1 plateaued at ~156 GB/s across splits 4-12 (issue-bound).
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
            .arg_u32(3)
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
        // Contract: the kernel's s_x staging holds at most 4096 K per block
        // K-slice (EXL3_MAX_XCHUNKS) — splits below K/4096 are illegal.
        let min_split = k.div_ceil(4096).max(1) as u32;
        let prod_split = (48usize.div_ceil(strips)).clamp(1, max_split) as u32;
        let prod_split = prod_split.max(min_split);
        for split in [min_split, prod_split] {
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
        for split in min_split..=(max_split as u32).max(min_split) {
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

    all_ok &= prefill_gates(g, &lut)?;
    all_ok &= direct_prefill_parity_gate(g)?;
    all_ok &= fused_post_silu_gate(g)?;
    all_ok &= fused_post_unpermute_gate(g)?;
    all_ok &= fused_decode_gate(g)?;
    all_ok &= mrow_verify_gate(g)?;

    eprintln!(
        "EXL3 GEMV GATE (bit-exact dequant + cos>={PASS_COS} + determinism + P1 prefill \
         + P2 direct/post byte-identity + fused decode byte-identity + m-row verify byte-identity): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Fused final down-post + indexed unpermute parity. The legacy composition
// materializes rotated BF16 rows before weighting; the fused kernel must keep
// that rounding boundary and the top-k accumulation order exactly.
// ---------------------------------------------------------------------------

fn fused_post_unpermute_gate(g: &dyn GpuBackend) -> Result<bool> {
    let stream = 0u64;
    let (tokens, topk, rows, h, ne) = (3usize, 4usize, 12usize, 256usize, 5usize);
    let sorted_ids = [4i32, 1, 3, 0, 2, 4, 1, 0, 3, 2, 1, 4];
    let token_to_perm = [3i32, 8, 1, 10, 0, 11, 5, 2, 9, 4, 7, 6];
    let weights = [
        0.11f32, 0.29, 0.37, 0.23, 0.41, 0.07, 0.31, 0.21, 0.19, 0.17, 0.47, 0.17,
    ];
    let ints = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let floats = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let d_sorted_ids = up(g, &ints(&sorted_ids))?;
    let d_token_to_perm = up(g, &ints(&token_to_perm))?;
    let d_weights = up(g, &floats(&weights))?;

    let mut rng = Rng(0x7A11_D518_2026_0827);
    let svh: Vec<Vec<u16>> = (0..ne)
        .map(|_| (0..h).map(|_| rng.sign_f16()).collect())
        .collect();
    let bf16_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let d_svh: Vec<DevicePtr> = svh
        .iter()
        .map(|v| up(g, &bf16_bytes(v)))
        .collect::<Result<_>>()?;
    let svh_tab: Vec<u8> = d_svh.iter().flat_map(|p| p.0.to_le_bytes()).collect();
    let d_svh_tab = up(g, &svh_tab)?;
    let raw: Vec<u16> = (0..rows * h)
        .map(|_| bf16::from_f32((rng.unit() - 0.5) * 20.0).to_bits())
        .collect();
    let raw_b = bf16_bytes(&raw);
    let d_composed_rows = up(g, &raw_b)?;
    let d_fused_rows = up(g, &raw_b)?;
    let out_bytes = tokens * h * 2;
    let d_composed_out = g.alloc(out_bytes)?;
    let d_fused_out = g.alloc(out_bytes)?;
    g.memset(d_composed_out, 0xA5, out_bytes)?;
    g.memset(d_fused_out, 0x5A, out_bytes)?;

    let kh_post = g.kernel("exl3_gemv", "exl3_h128_post_rows")?;
    let kh_unpermute = g.kernel("moe", "moe_unpermute_reduce_indexed")?;
    let kh_fused = g.kernel("exl3_gemv", "exl3_h128_post_unpermute_rows")?;
    KernelLaunch::new(g, kh_post)
        .grid([rows as u32, (h as u32).div_ceil(1024), 1])
        .block([256, 1, 1])
        .arg_ptr(d_composed_rows)
        .arg_ptr(d_sorted_ids)
        .arg_ptr(d_svh_tab)
        .arg_u32(h as u32)
        .launch(stream)?;
    KernelLaunch::new(g, kh_unpermute)
        .grid([tokens as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(d_composed_rows)
        .arg_ptr(d_composed_out)
        .arg_ptr(d_token_to_perm)
        .arg_ptr(d_weights)
        .arg_u32(h as u32)
        .arg_u32(tokens as u32)
        .arg_u32(topk as u32)
        .launch(stream)?;
    KernelLaunch::new(g, kh_fused)
        .grid([tokens as u32, (h as u32).div_ceil(1024), 1])
        .block([256, 1, 1])
        .arg_ptr(d_fused_rows)
        .arg_ptr(d_fused_out)
        .arg_ptr(d_token_to_perm)
        .arg_ptr(d_weights)
        .arg_ptr(d_sorted_ids)
        .arg_ptr(d_svh_tab)
        .arg_u32(h as u32)
        .arg_u32(tokens as u32)
        .arg_u32(topk as u32)
        .launch(stream)?;
    g.synchronize(stream)?;

    let mut composed = vec![0u8; out_bytes];
    let mut fused = vec![0u8; out_bytes];
    g.copy_d2h(d_composed_out, &mut composed)?;
    g.copy_d2h(d_fused_out, &mut fused)?;
    let diff = composed.iter().zip(&fused).filter(|(a, b)| a != b).count();
    let nontrivial = composed.iter().any(|&x| x != 0xA5) && fused.iter().any(|&x| x != 0x5A);
    let pass = diff == 0 && nontrivial;
    eprintln!(
        "P2 TAIL post-unpermute fused==composed: diff={diff}/{out_bytes} \
         tokens={tokens} topk={topk} nontrivial={nontrivial} {}",
        if pass { "PASS" } else { "FAIL" }
    );

    for p in d_svh.into_iter().chain([
        d_sorted_ids,
        d_token_to_perm,
        d_weights,
        d_svh_tab,
        d_composed_rows,
        d_fused_rows,
        d_composed_out,
        d_fused_out,
    ]) {
        let _ = g.free(p);
    }
    Ok(pass)
}

// ---------------------------------------------------------------------------
// Fused P2 post+SwiGLU gate: compare the new single pass with the exact
// post(gate/up) + moe_silu_mul + down pre-rotation composition it replaces.
// ---------------------------------------------------------------------------

const FUSED_POST_GUARD_BYTES: usize = 256;

struct FusedPostGuard {
    base: DevicePtr,
    data: DevicePtr,
    data_bytes: usize,
}

fn fused_post_guarded_up(g: &dyn GpuBackend, data: &[u8]) -> Result<FusedPostGuard> {
    let total = data.len() + 2 * FUSED_POST_GUARD_BYTES;
    let base = g.alloc(total)?;
    g.memset(base, 0xA5, FUSED_POST_GUARD_BYTES)?;
    let data_ptr = DevicePtr(base.0 + FUSED_POST_GUARD_BYTES as u64);
    g.copy_h2d(data, data_ptr)?;
    let suffix = DevicePtr(data_ptr.0 + data.len() as u64);
    g.memset(suffix, 0x5A, FUSED_POST_GUARD_BYTES)?;
    Ok(FusedPostGuard {
        base,
        data: data_ptr,
        data_bytes: data.len(),
    })
}

fn fused_post_canary_ok(g: &dyn GpuBackend, guarded: &FusedPostGuard) -> Result<bool> {
    let mut prefix = vec![0u8; FUSED_POST_GUARD_BYTES];
    let mut suffix = vec![0u8; FUSED_POST_GUARD_BYTES];
    g.copy_d2h(guarded.base, &mut prefix)?;
    g.copy_d2h(
        DevicePtr(guarded.data.0 + guarded.data_bytes as u64),
        &mut suffix,
    )?;
    Ok(prefix.iter().all(|&byte| byte == 0xA5) && suffix.iter().all(|&byte| byte == 0x5A))
}

fn fused_post_silu_case(
    g: &dyn GpuBackend,
    label: &str,
    tokens: usize,
    topk: usize,
    n: usize,
    ne: usize,
    seed: u64,
) -> Result<bool> {
    let stream = 0u64;
    let rows = tokens * topk;
    let ids: Vec<i32> = (0..rows)
        .map(|row| ((row * 131 + row / 7) % ne) as i32)
        .collect();
    let ids_b: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d_ids = up(g, &ids_b)?;
    let mut rng = Rng(seed);
    let gate_svh: Vec<Vec<u16>> = (0..ne)
        .map(|_| (0..n).map(|_| rng.sign_f16()).collect())
        .collect();
    let up_svh: Vec<Vec<u16>> = (0..ne)
        .map(|_| (0..n).map(|_| rng.sign_f16()).collect())
        .collect();
    let down_suh: Vec<Vec<u16>> = (0..ne)
        .map(|_| (0..n).map(|_| rng.sign_f16()).collect())
        .collect();
    let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let d_gate_svh: Vec<DevicePtr> = gate_svh
        .iter()
        .map(|v| up(g, &bytes(v)))
        .collect::<Result<_>>()?;
    let d_up_svh: Vec<DevicePtr> = up_svh
        .iter()
        .map(|v| up(g, &bytes(v)))
        .collect::<Result<_>>()?;
    let d_down_suh: Vec<DevicePtr> = down_suh
        .iter()
        .map(|v| up(g, &bytes(v)))
        .collect::<Result<_>>()?;
    let ptrs = |v: &[DevicePtr]| -> Vec<u8> { v.iter().flat_map(|p| p.0.to_le_bytes()).collect() };
    let d_gate_svh_tab = up(g, &ptrs(&d_gate_svh))?;
    let d_up_svh_tab = up(g, &ptrs(&d_up_svh))?;
    let d_down_suh_tab = up(g, &ptrs(&d_down_suh))?;

    // Wide raw GEMM-like values exercise both clamp edges after H128.
    let gate_h: Vec<u16> = (0..rows * n)
        .map(|_| bf16::from_f32((rng.unit() - 0.5) * 40.0).to_bits())
        .collect();
    let up_h: Vec<u16> = (0..rows * n)
        .map(|_| bf16::from_f32((rng.unit() - 0.5) * 40.0).to_bits())
        .collect();
    let gate_b = bytes(&gate_h);
    let up_b = bytes(&up_h);
    let d_comp_gate = fused_post_guarded_up(g, &gate_b)?;
    let d_comp_up = fused_post_guarded_up(g, &up_b)?;
    let d_fused_gate = fused_post_guarded_up(g, &gate_b)?;
    let d_fused_up = fused_post_guarded_up(g, &up_b)?;
    let kh_post = g.kernel("exl3_gemv", "exl3_h128_post_rows")?;
    let kh_pre = g.kernel("exl3_gemv", "exl3_h128_pre_rows")?;
    let kh_fused = g.kernel("exl3_gemv", "exl3_h128_post_silu_pre_rows")?;
    let kh_silu = g.kernel("moe_silu_mul", "moe_silu_mul")?;

    for (buf, tab) in [
        (d_comp_gate.data, d_gate_svh_tab),
        (d_comp_up.data, d_up_svh_tab),
    ] {
        KernelLaunch::new(g, kh_post)
            .grid([rows as u32, (n as u32).div_ceil(1024), 1])
            .block([256, 1, 1])
            .arg_ptr(buf)
            .arg_ptr(d_ids)
            .arg_ptr(tab)
            .arg_u32(n as u32)
            .launch(stream)?;
    }
    KernelLaunch::new(g, kh_silu)
        .grid([((rows * n) as u32).div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(d_comp_gate.data)
        .arg_ptr(d_comp_up.data)
        .arg_ptr(d_comp_gate.data)
        .arg_u32((rows * n) as u32)
        .launch(stream)?;
    KernelLaunch::new(g, kh_pre)
        .grid([rows as u32, (n as u32).div_ceil(1024), 1])
        .block([256, 1, 1])
        .arg_ptr(d_comp_gate.data)
        .arg_ptr(DevicePtr(0))
        .arg_ptr(d_ids)
        .arg_ptr(d_down_suh_tab)
        .arg_ptr(d_comp_gate.data)
        .arg_u32(n as u32)
        .launch(stream)?;
    KernelLaunch::new(g, kh_fused)
        .grid([rows as u32, (n as u32).div_ceil(1024), 1])
        .block([256, 1, 1])
        .arg_ptr(d_fused_gate.data)
        .arg_ptr(d_fused_up.data)
        .arg_ptr(d_ids)
        .arg_ptr(d_gate_svh_tab)
        .arg_ptr(d_up_svh_tab)
        .arg_ptr(d_down_suh_tab)
        .arg_u32(n as u32)
        .launch(stream)?;
    g.synchronize(stream)?;

    let mut composed = vec![0u8; rows * n * 2];
    let mut fused = vec![0u8; rows * n * 2];
    g.copy_d2h(d_comp_gate.data, &mut composed)?;
    g.copy_d2h(d_fused_gate.data, &mut fused)?;
    let diff = composed.iter().zip(&fused).filter(|(a, b)| a != b).count();
    let nontrivial = composed != gate_b;
    let canary_ok = [&d_comp_gate, &d_comp_up, &d_fused_gate, &d_fused_up]
        .into_iter()
        .try_fold(true, |ok, guarded| {
            Ok::<_, anyhow::Error>(ok && fused_post_canary_ok(g, guarded)?)
        })?;
    let pass = diff == 0 && nontrivial && canary_ok;
    eprintln!(
        "P2 POST {label} fused==composed byte-identical: tokens={tokens} topk={topk} \
         experts={ne} rows={rows} n={n} diff={diff}/{} nontrivial={nontrivial} \
         canary={}: {}",
        composed.len(),
        if canary_ok { "clean" } else { "CORRUPT" },
        if pass { "PASS" } else { "FAIL" }
    );

    for p in d_gate_svh
        .into_iter()
        .chain(d_up_svh)
        .chain(d_down_suh)
        .chain([d_ids, d_gate_svh_tab, d_up_svh_tab, d_down_suh_tab])
    {
        let _ = g.free(p);
    }
    for guarded in [d_comp_gate, d_comp_up, d_fused_gate, d_fused_up] {
        let _ = g.free(guarded.base);
    }
    Ok(pass)
}

fn fused_post_silu_gate(g: &dyn GpuBackend) -> Result<bool> {
    let small = fused_post_silu_case(g, "hostile-small", 5, 1, 256, 3, 0xF05E_D518_2026_0827)?;
    let production = fused_post_silu_case(
        g,
        "production-n2048-topk6-e256",
        2048,
        6,
        2048,
        256,
        0xF05E_D518_2026_0908,
    )?;
    Ok(small && production)
}

// ---------------------------------------------------------------------------
// P2 direct grouped-prefill parity: compare the global-BF16 P1 composition
// with direct trellis-to-fragment P2. The compact N=64, K=128 shape preserves
// the production tile geometry while covering both encodings and M64 edges.
// This is compiled offline but intentionally runs only in the GPU microtest.
// ---------------------------------------------------------------------------

fn direct_prefill_parity_gate(g: &dyn GpuBackend) -> Result<bool> {
    let stream = 0u64;
    let counts = [1usize, 63, 64, 65, 129];
    let ne = counts.len();
    let (n, k) = (64usize, 128usize);
    let rows: usize = counts.iter().sum();
    let mut offsets = Vec::with_capacity(ne + 1);
    offsets.push(0i32);
    for count in counts {
        offsets.push(offsets.last().copied().unwrap() + count as i32);
    }
    let offsets_b: Vec<u8> = offsets.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d_offsets = up(g, &offsets_b)?;
    let mut rng = Rng(0xD1EC_7F11_2026_0827);
    let a: Vec<u16> = (0..rows * k)
        .map(|_| bf16::from_f32((rng.unit() - 0.5) * 0.5).to_bits())
        .collect();
    let a_b: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d_a = up(g, &a_b)?;
    let out_bytes = rows * n * 2;
    let d_p1 = g.alloc(out_bytes)?;
    let d_p2 = g.alloc(out_bytes)?;
    let kh_dq = g.kernel("exl3_gemv", "exl3_dequant_chunk_bf16")?;
    let kh_gemm = g.kernel("moe_bf16_grouped_gemm", "moe_bf16_grouped_gemm")?;
    let kh_direct = g.kernel("exl3_grouped_prefill", "exl3_grouped_prefill")?;
    let kh_direct_k2 = g.kernel("exl3_grouped_prefill_k2", "exl3_grouped_prefill_k2")?;
    let kh_direct_k64 = g.kernel("exl3_grouped_prefill_k64", "exl3_grouped_prefill_k64")?;
    let kh_direct_k64_k2 =
        g.kernel("exl3_grouped_prefill_k64_k2", "exl3_grouped_prefill_k64_k2")?;
    let kh_direct_m128 = g.kernel("exl3_grouped_prefill_m128", "exl3_grouped_prefill_m128")?;
    let max_m_tiles = counts.into_iter().max().unwrap().div_ceil(64) as u32;
    let max_m128_tiles = counts.into_iter().max().unwrap().div_ceil(128) as u32;
    let mut ok = true;

    for bits in [2usize, 3] {
        let words_per_expert = (k / 16) * (n / 16) * (16 * bits);
        let trellis_h: Vec<Vec<u16>> = (0..ne)
            .map(|_| (0..words_per_expert).map(|_| rng.u16()).collect())
            .collect();
        let trellis_b = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let d_trellis: Vec<DevicePtr> = trellis_h
            .iter()
            .map(|v| up(g, &trellis_b(v)))
            .collect::<Result<_>>()?;
        let ptr_tab: Vec<u8> = d_trellis.iter().flat_map(|p| p.0.to_le_bytes()).collect();
        let d_trellis_tab = up(g, &ptr_tab)?;
        let slot_bytes = n * k * 2;
        let d_scratch = g.alloc(ne * slot_bytes)?;
        let slot_tab: Vec<u8> = (0..ne)
            .flat_map(|e| (d_scratch.0 + (e * slot_bytes) as u64).to_le_bytes())
            .collect();
        let d_slot_tab = up(g, &slot_tab)?;

        g.memset(d_p1, 0xA5, out_bytes)?;
        g.memset(d_p2, 0x5A, out_bytes)?;
        KernelLaunch::new(g, kh_dq)
            .grid([(n / 16) as u32, (k / 16) as u32, ne as u32])
            .block([32, 1, 1])
            .arg_ptr(d_trellis_tab)
            .arg_u32(0)
            .arg_u32(ne as u32)
            .arg_ptr(d_scratch)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(bits as u32)
            .launch(stream)?;
        KernelLaunch::new(g, kh_gemm)
            .grid([(n / 64) as u32, max_m_tiles, ne as u32])
            .block([128, 1, 1])
            .arg_ptr(d_a)
            .arg_ptr(d_slot_tab)
            .arg_ptr(d_p1)
            .arg_ptr(d_offsets)
            .arg_ptr(DevicePtr(0))
            .arg_u32(ne as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        KernelLaunch::new(g, kh_direct)
            .grid([(n / 64) as u32, max_m_tiles, ne as u32])
            .block([128, 1, 1])
            .arg_ptr(d_a)
            .arg_ptr(d_trellis_tab)
            .arg_ptr(d_p2)
            .arg_ptr(d_offsets)
            .arg_ptr(DevicePtr(0))
            .arg_u32(ne as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(bits as u32)
            .arg_u32(0)
            .launch(stream)?;
        g.synchronize(stream)?;

        let mut p1 = vec![0u8; out_bytes];
        let mut p2 = vec![0u8; out_bytes];
        g.copy_d2h(d_p1, &mut p1)?;
        g.copy_d2h(d_p2, &mut p2)?;
        let diff = p1.iter().zip(&p2).filter(|(a, b)| a != b).count();
        let nontrivial = p1.iter().any(|&x| x != 0xA5) && p2.iter().any(|&x| x != 0x5A);
        let pass = diff == 0 && nontrivial;
        ok &= pass;
        eprintln!(
            "P2 PREFILL K{bits} direct==P1: diff={diff}/{out_bytes} \
             rows=1/63/64/65/129 nontrivial={nontrivial} {}",
            if pass { "PASS" } else { "FAIL" }
        );

        // Cover both the deliberately undersubscribed grid-stride mapping and
        // production's one-CTA-per-strip compact grid.
        for (label, grid, sentinel) in [
            ("persistent-strided", [7, 1, 1], 0x3C),
            ("persistent-full", [(ne * (n / 64)) as u32, 1, 1], 0x2C),
        ] {
            g.memset(d_p2, sentinel, out_bytes)?;
            KernelLaunch::new(g, kh_direct)
                .grid(grid)
                .block([128, 1, 1])
                .arg_ptr(d_a)
                .arg_ptr(d_trellis_tab)
                .arg_ptr(d_p2)
                .arg_ptr(d_offsets)
                .arg_ptr(DevicePtr(0))
                .arg_u32(ne as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(bits as u32)
                .arg_u32(1)
                .launch(stream)?;
            g.synchronize(stream)?;
            g.copy_d2h(d_p2, &mut p2)?;
            let persistent_diff = p1.iter().zip(&p2).filter(|(a, b)| a != b).count();
            let persistent_nontrivial = p2.iter().any(|&x| x != sentinel);
            let persistent_pass = persistent_diff == 0 && persistent_nontrivial;
            ok &= persistent_pass;
            eprintln!(
                "P2 PREFILL K{bits} {label}==P1: diff={persistent_diff}/{out_bytes} \
                 ctas={} rows=1/63/64/65/129 nontrivial={persistent_nontrivial} {}",
                grid[0],
                if persistent_pass { "PASS" } else { "FAIL" }
            );
        }

        for (label, grid, persistent_mode) in [
            ("k64-exact", [(n / 64) as u32, max_m_tiles, ne as u32], 0),
            ("k64-persistent-strided", [7, 1, 1], 1),
            ("k64-persistent-full", [(ne * (n / 64)) as u32, 1, 1], 1),
        ] {
            g.memset(d_p2, 0x4D, out_bytes)?;
            KernelLaunch::new(g, kh_direct_k64)
                .grid(grid)
                .block([128, 1, 1])
                .arg_ptr(d_a)
                .arg_ptr(d_trellis_tab)
                .arg_ptr(d_p2)
                .arg_ptr(d_offsets)
                .arg_ptr(DevicePtr(0))
                .arg_u32(ne as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(bits as u32)
                .arg_u32(persistent_mode)
                .launch(stream)?;
            g.synchronize(stream)?;
            g.copy_d2h(d_p2, &mut p2)?;
            let k64_diff = p1.iter().zip(&p2).filter(|(a, b)| a != b).count();
            let k64_nontrivial = p2.iter().any(|&x| x != 0x4D);
            let k64_pass = k64_diff == 0 && k64_nontrivial;
            ok &= k64_pass;
            eprintln!(
                "P2 PREFILL K{bits} {label}==P1: diff={k64_diff}/{out_bytes} \
                 rows=1/63/64/65/129 nontrivial={k64_nontrivial} {}",
                if k64_pass { "PASS" } else { "FAIL" }
            );
        }

        if bits == 2 {
            for (label, kernel, grid, persistent_mode) in [
                (
                    "k2-fixed-exact",
                    kh_direct_k2,
                    [(n / 64) as u32, max_m_tiles, ne as u32],
                    0,
                ),
                (
                    "k2-fixed-persistent-full",
                    kh_direct_k2,
                    [(ne * (n / 64)) as u32, 1, 1],
                    1,
                ),
                (
                    "k64-k2-fixed-exact",
                    kh_direct_k64_k2,
                    [(n / 64) as u32, max_m_tiles, ne as u32],
                    0,
                ),
                (
                    "k64-k2-fixed-persistent-full",
                    kh_direct_k64_k2,
                    [(ne * (n / 64)) as u32, 1, 1],
                    1,
                ),
            ] {
                g.memset(d_p2, 0x6B, out_bytes)?;
                KernelLaunch::new(g, kernel)
                    .grid(grid)
                    .block([128, 1, 1])
                    .arg_ptr(d_a)
                    .arg_ptr(d_trellis_tab)
                    .arg_ptr(d_p2)
                    .arg_ptr(d_offsets)
                    .arg_ptr(DevicePtr(0))
                    .arg_u32(ne as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .arg_u32(bits as u32)
                    .arg_u32(persistent_mode)
                    .launch(stream)?;
                g.synchronize(stream)?;
                g.copy_d2h(d_p2, &mut p2)?;
                let fixed_diff = p1.iter().zip(&p2).filter(|(a, b)| a != b).count();
                let fixed_nontrivial = p2.iter().any(|&x| x != 0x6B);
                let fixed_pass = fixed_diff == 0 && fixed_nontrivial;
                ok &= fixed_pass;
                eprintln!(
                    "P2 PREFILL {label}==P1: diff={fixed_diff}/{out_bytes} \
                     rows=1/63/64/65/129 nontrivial={fixed_nontrivial} {}",
                    if fixed_pass { "PASS" } else { "FAIL" }
                );
            }
        }

        for (label, grid, persistent_mode) in [
            (
                "m128-exact",
                [(n / 64) as u32, max_m128_tiles, ne as u32],
                0,
            ),
            ("m128-persistent", [7, 1, 1], 1),
            ("m128-persistent-full", [(ne * (n / 64)) as u32, 1, 1], 1),
        ] {
            g.memset(d_p2, 0xC3, out_bytes)?;
            KernelLaunch::new(g, kh_direct_m128)
                .grid(grid)
                .block([256, 1, 1])
                .arg_ptr(d_a)
                .arg_ptr(d_trellis_tab)
                .arg_ptr(d_p2)
                .arg_ptr(d_offsets)
                .arg_ptr(DevicePtr(0))
                .arg_u32(ne as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(bits as u32)
                .arg_u32(persistent_mode)
                .launch(stream)?;
            g.synchronize(stream)?;
            g.copy_d2h(d_p2, &mut p2)?;
            let m128_diff = p1.iter().zip(&p2).filter(|(a, b)| a != b).count();
            let m128_nontrivial = p2.iter().any(|&x| x != 0xC3);
            let m128_pass = m128_diff == 0 && m128_nontrivial;
            ok &= m128_pass;
            eprintln!(
                "P2 PREFILL K{bits} {label}==P1: diff={m128_diff}/{out_bytes} \
                 rows=1/63/64/65/129 nontrivial={m128_nontrivial} {}",
                if m128_pass { "PASS" } else { "FAIL" }
            );
        }

        for p in d_trellis
            .into_iter()
            .chain([d_trellis_tab, d_scratch, d_slot_tab])
        {
            let _ = g.free(p);
        }
    }

    for p in [d_offsets, d_a, d_p1, d_p2] {
        let _ = g.free(p);
    }
    Ok(ok)
}

// ---------------------------------------------------------------------------
// P1 prefill-path gates (4-7): H128 pre/post row kernels, chunked BF16
// dequant, and the full scratch-dequant prefill pipeline vs the f64 oracle.
// Mirrors forward_prefill_exl3.rs exactly: 4 experts, chunk = 2 (two
// sub-range grouped-GEMM launches), sorted-layout expansion with a real
// token gather.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn prefill_gates(g: &dyn GpuBackend, lut: &[u16; 65536]) -> Result<bool> {
    let stream = 0u64;
    let (ne, n, k) = (4usize, 2048usize, 4096usize);
    let (tokens, rows, chunk) = (32usize, 64usize, 2usize);
    let rows_per_expert = rows / ne; // 16
    let mut ok = true;

    let kh_pre = g.kernel("exl3_gemv", "exl3_h128_pre_rows")?;
    let kh_post = g.kernel("exl3_gemv", "exl3_h128_post_rows")?;
    let kh_dq = g.kernel("exl3_gemv", "exl3_dequant_chunk_bf16")?;
    let kh_gemm = g.kernel("moe_bf16_grouped_gemm", "moe_bf16_grouped_gemm")?;

    let mut r = Rng(0x91E7_2026_0810u64); // P1 prefill-gate seed
    let tiles_k = k / 16;
    let tiles_n = n / 16;
    let trellis_words = tiles_k * tiles_n * 48;

    // ---- synthetic experts + tokens + routing ----
    let trellis_h: Vec<Vec<u16>> = (0..ne)
        .map(|_| (0..trellis_words).map(|_| r.u16()).collect())
        .collect();
    let suh_h: Vec<Vec<u16>> = (0..ne)
        .map(|_| (0..k).map(|_| r.sign_f16()).collect())
        .collect();
    let svh_h: Vec<Vec<u16>> = (0..ne)
        .map(|_| (0..n).map(|_| r.sign_f16()).collect())
        .collect();
    let a_h: Vec<u16> = (0..tokens * k)
        .map(|_| bf16::from_f32((r.unit() - 0.5) * 0.5).to_bits())
        .collect();
    // Row r → token (r*7+3)%tokens (repeats = one token, several experts),
    // expert r/16; offsets are the absolute sorted-layout prefix sums.
    let sti: Vec<i32> = (0..rows).map(|rr| ((rr * 7 + 3) % tokens) as i32).collect();
    let sei: Vec<i32> = (0..rows).map(|rr| (rr / rows_per_expert) as i32).collect();
    let offs: Vec<i32> = (0..=ne).map(|e| (e * rows_per_expert) as i32).collect();

    // ---- CPU: per-expert decoded W bits [N][K] ----
    let mut w_bits: Vec<Vec<u16>> = Vec::with_capacity(ne);
    for th in &trellis_h {
        let mut wb = vec![0u16; n * k];
        let mut tile = [[0u16; 16]; 16];
        for kb in 0..tiles_k {
            for nb in 0..tiles_n {
                let base = (kb * tiles_n + nb) * 48;
                decode_tile(&th[base..base + 48], 3, lut, &mut tile);
                for (kit, row) in tile.iter().enumerate() {
                    for (nit, bits) in row.iter().enumerate() {
                        wb[(nb * 16 + nit) * k + kb * 16 + kit] = *bits;
                    }
                }
            }
        }
        w_bits.push(wb);
    }

    // ---- CPU f64 oracle: y_ref[row][n] through the full pipeline ----
    let mut y_ref = vec![0.0f64; rows * n];
    for rr in 0..rows {
        let (tok, e) = (sti[rr] as usize, sei[rr] as usize);
        let x: Vec<f64> = (0..k)
            .map(|kk| bf16::from_bits(a_h[tok * k + kk]).to_f64() * f16_to_f64(suh_h[e][kk]))
            .collect();
        let xp = had128(&x);
        let mut y0 = vec![0.0f64; n];
        for (nn, y) in y0.iter_mut().enumerate() {
            let row = &w_bits[e][nn * k..(nn + 1) * k];
            *y = row
                .iter()
                .zip(&xp)
                .map(|(&wb, &v)| f16_to_f64(wb) * v)
                .sum();
        }
        for (nn, &v) in had128(&y0).iter().enumerate() {
            y_ref[rr * n + nn] = v * f16_to_f64(svh_h[e][nn]);
        }
    }

    // ---- upload ----
    let to_b = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let to_bi = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let d_trellis: Vec<DevicePtr> = trellis_h
        .iter()
        .map(|t| up(g, &to_b(t)))
        .collect::<Result<_>>()?;
    let d_suh: Vec<DevicePtr> = suh_h
        .iter()
        .map(|t| up(g, &to_b(t)))
        .collect::<Result<_>>()?;
    let d_svh: Vec<DevicePtr> = svh_h
        .iter()
        .map(|t| up(g, &to_b(t)))
        .collect::<Result<_>>()?;
    let tab = |ps: &[DevicePtr]| -> Vec<u8> { ps.iter().flat_map(|p| p.0.to_le_bytes()).collect() };
    let d_trellis_tab = up(g, &tab(&d_trellis))?;
    let d_suh_tab = up(g, &tab(&d_suh))?;
    let d_svh_tab = up(g, &tab(&d_svh))?;
    let d_a = up(g, &to_b(&a_h))?;
    let d_sti = up(g, &to_bi(&sti))?;
    let d_sei = up(g, &to_bi(&sei))?;
    let d_offs = up(g, &to_bi(&offs))?;
    let d_arot = g.alloc(rows * k * 2)?;
    let slot_bytes = n * k * 2;
    let d_scratch = g.alloc(chunk * slot_bytes)?;
    let slot_tab: Vec<u8> = (0..chunk)
        .flat_map(|z| (d_scratch.0 + (z * slot_bytes) as u64).to_le_bytes())
        .collect();
    let d_slot_tab = up(g, &slot_tab)?;
    let d_c = g.alloc(rows * n * 2)?;

    // ---- GATE 4: exl3_h128_pre_rows bit-exact vs the f32 CPU replica ----
    KernelLaunch::new(g, kh_pre)
        .grid([rows as u32, (k as u32).div_ceil(1024), 1])
        .block([256, 1, 1])
        .arg_ptr(d_a)
        .arg_ptr(d_sti)
        .arg_ptr(d_sei)
        .arg_ptr(d_suh_tab)
        .arg_ptr(d_arot)
        .arg_u32(k as u32)
        .launch(stream)?;
    g.synchronize(stream)?;
    let mut arot_gpu = vec![0u8; rows * k * 2];
    g.copy_d2h(d_arot, &mut arot_gpu)?;
    let mut arot_ref = vec![0u16; rows * k];
    for rr in 0..rows {
        let (tok, e) = (sti[rr] as usize, sei[rr] as usize);
        for c in 0..k / 128 {
            let mut buf = [0f32; 128];
            for (j, b) in buf.iter_mut().enumerate() {
                let kk = c * 128 + j;
                *b = bf16::from_bits(a_h[tok * k + kk]).to_f32()
                    * half::f16::from_bits(suh_h[e][kk]).to_f32();
            }
            had128_f32_gpu(&mut buf);
            for (j, &v) in buf.iter().enumerate() {
                arot_ref[rr * k + c * 128 + j] = bf16::from_f32(v * RSQRT128_F32).to_bits();
            }
        }
    }
    let prediff = arot_gpu
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .zip(&arot_ref)
        .filter(|(a, b)| a != *b)
        .count();
    let g4 = prediff == 0;
    ok &= g4;
    eprintln!(
        "P1 GATE4 h128_pre bit-exact: diff={prediff}/{} {}",
        rows * k,
        if g4 { "PASS" } else { "FAIL" }
    );

    // ---- GATES 5 + GEMM: chunked dequant (bit-exact) + sub-range GEMM ----
    let mut dqdiff = 0usize;
    for e0 in (0..ne).step_by(chunk) {
        let cnt = chunk.min(ne - e0);
        KernelLaunch::new(g, kh_dq)
            .grid([tiles_n as u32, tiles_k as u32, cnt as u32])
            .block([32, 1, 1])
            .arg_ptr(d_trellis_tab)
            .arg_u32(e0 as u32)
            .arg_u32(cnt as u32)
            .arg_ptr(d_scratch)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(3)
            .launch(stream)?;
        g.synchronize(stream)?;
        let mut sc = vec![0u8; cnt * slot_bytes];
        g.copy_d2h(d_scratch, &mut sc)?;
        for z in 0..cnt {
            let slot = &sc[z * slot_bytes..(z + 1) * slot_bytes];
            dqdiff += slot
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .zip(&w_bits[e0 + z])
                .filter(|(got, want)| {
                    *got != bf16::from_f32(half::f16::from_bits(**want).to_f32()).to_bits()
                })
                .count();
        }
        // Sub-range grouped GEMM: offsets + e0, num_experts = cnt,
        // sorted_token_ids = NULL (A already expanded + rotated).
        let max_m_tiles = (rows_per_expert as u32).div_ceil(64).max(1);
        KernelLaunch::new(g, kh_gemm)
            .grid([(n as u32).div_ceil(64), max_m_tiles, cnt as u32])
            .block([128, 1, 1])
            .arg_ptr(d_arot)
            .arg_ptr(d_slot_tab)
            .arg_ptr(d_c)
            .arg_ptr(d_offs.offset(e0 * 4))
            .arg_ptr(DevicePtr(0))
            .arg_u32(cnt as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        g.synchronize(stream)?;
    }
    let g5 = dqdiff == 0;
    ok &= g5;
    eprintln!(
        "P1 GATE5 dequant_chunk_bf16 bit-exact: diff={dqdiff}/{} {}",
        ne * n * k,
        if g5 { "PASS" } else { "FAIL" }
    );

    // ---- GATE 6: exl3_h128_post_rows bit-exact (download before/after) ----
    let mut c_pre = vec![0u8; rows * n * 2];
    g.copy_d2h(d_c, &mut c_pre)?;
    KernelLaunch::new(g, kh_post)
        .grid([rows as u32, (n as u32).div_ceil(1024), 1])
        .block([256, 1, 1])
        .arg_ptr(d_c)
        .arg_ptr(d_sei)
        .arg_ptr(d_svh_tab)
        .arg_u32(n as u32)
        .launch(stream)?;
    g.synchronize(stream)?;
    let mut c_post = vec![0u8; rows * n * 2];
    g.copy_d2h(d_c, &mut c_post)?;
    let mut postdiff = 0usize;
    for rr in 0..rows {
        let e = sei[rr] as usize;
        for c in 0..n / 128 {
            let mut buf = [0f32; 128];
            for (j, b) in buf.iter_mut().enumerate() {
                let idx = (rr * n + c * 128 + j) * 2;
                *b = bf16::from_bits(u16::from_le_bytes([c_pre[idx], c_pre[idx + 1]])).to_f32();
            }
            had128_f32_gpu(&mut buf);
            for (j, &v) in buf.iter().enumerate() {
                let nn = c * 128 + j;
                let want =
                    bf16::from_f32(v * RSQRT128_F32 * half::f16::from_bits(svh_h[e][nn]).to_f32())
                        .to_bits();
                let idx = (rr * n + nn) * 2;
                let got = u16::from_le_bytes([c_post[idx], c_post[idx + 1]]);
                if got != want {
                    postdiff += 1;
                }
            }
        }
    }
    let g6 = postdiff == 0;
    ok &= g6;
    eprintln!(
        "P1 GATE6 h128_post bit-exact: diff={postdiff}/{} {}",
        rows * n,
        if g6 { "PASS" } else { "FAIL" }
    );

    // ---- GATE 7: full prefill path cosine vs the f64 oracle ----
    let y_gpu: Vec<f64> = c_post
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect();
    let cs = cos(&y_gpu, &y_ref);
    let g7 = cs >= 0.999;
    ok &= g7;
    eprintln!(
        "P1 GATE7 prefill path (pre+dequant+GEMM+post) M={rows}: cos={cs:.8} {}",
        if g7 { "PASS" } else { "FAIL" }
    );

    for p in d_trellis.into_iter().chain(d_suh).chain(d_svh).chain([
        d_trellis_tab,
        d_suh_tab,
        d_svh_tab,
        d_a,
        d_sti,
        d_sei,
        d_offs,
        d_arot,
        d_scratch,
        d_slot_tab,
        d_c,
    ]) {
        let _ = g.free(p);
    }
    Ok(ok)
}

// ---------------------------------------------------------------------------
// GATE 8: fused decode dispatch == per-slot chain, byte for byte.
//
// The production decode dispatch (`layers/moe/exl3_decode.rs`) collapses the
// routed FFN from 4·top_k launches per layer (gate, up, SwiGLU, down per slot)
// to 3 (`exl3_gemv_m1_fused_gate_up`, one flat `moe_silu_mul`,
// `exl3_gemv_m1_fused_down`). Fusion moves only WHICH CTA owns which
// (slot, strip, split) triple — the per-output accumulation order, the
// per-128-k-chunk fp32 combine and the fixed split-order combine are the same
// device code — so the two paths must agree BIT FOR BIT at equal SPLIT_K.
// That is the gate below; it also re-runs the fused path to prove the split-K
// election still lands deterministically now that every launch group carries
// its own `ws`/`counters` region and the groups run concurrently.
//
// Output buffers are poisoned with a DIFFERENT byte before each path (0x00 vs
// 0xFF), so a slot or strip that no CTA writes cannot pass by accident.
// ---------------------------------------------------------------------------

/// Routed slots (DeepSeek-V4 tp1 routes 8; 6 keeps the gate quick and still
/// exercises 12 concurrent gate+up launch groups).
const FUSED_TOP_K: usize = 6;
/// Distinct synthetic experts the routing draws from.
const FUSED_NE: usize = 8;

/// One EXL3 projection's device pointer tables: (trellis, suh, svh).
type ProjTabs = (DevicePtr, DevicePtr, DevicePtr);

#[allow(clippy::too_many_arguments)]
fn launch_gemv_idx(
    g: &dyn GpuBackend,
    kh: spark_runtime::gpu::KernelHandle,
    stream: u64,
    split: u32,
    a: DevicePtr,
    tab: ProjTabs,
    idx: DevicePtr,
    slot: u32,
    c: DevicePtr,
    ws: DevicePtr,
    cnt: DevicePtr,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([(n / 128) as u32, split, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(tab.0)
        .arg_ptr(tab.1)
        .arg_ptr(tab.2)
        .arg_ptr(idx)
        .arg_u32(slot)
        .arg_ptr(c)
        .arg_ptr(ws)
        .arg_ptr(cnt)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(3)
        .launch(stream)
}

fn launch_silu(
    g: &dyn GpuBackend,
    kh: spark_runtime::gpu::KernelHandle,
    stream: u64,
    gate: DevicePtr,
    upp: DevicePtr,
    out: DevicePtr,
    total: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([total.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(upp)
        .arg_ptr(out)
        .arg_u32(total)
        .launch(stream)
}

#[allow(clippy::too_many_lines)]
fn fused_decode_gate(g: &dyn GpuBackend) -> Result<bool> {
    let stream = 0u64;
    let (h, inter) = (4096usize, 2048usize);
    let (top_k, ne) = (FUSED_TOP_K, FUSED_NE);
    // Distinct routed ids, deliberately NOT slot-ordered: a fused CTA must
    // resolve its expert from indices[slot] on device, not from blockIdx.
    let indices: Vec<u32> = vec![5, 0, 7, 2, 6, 1];
    assert_eq!(
        indices.len(),
        top_k,
        "GATE8 index list must have top_k entries"
    );
    assert!(indices.iter().all(|&e| (e as usize) < ne));

    let kh_idx = g.kernel("exl3_gemv", "exl3_gemv_m1_idx")?;
    let kh_gu = g.kernel("exl3_gemv", "exl3_gemv_m1_fused_gate_up")?;
    let kh_dn = g.kernel("exl3_gemv", "exl3_gemv_m1_fused_down")?;
    let kh_silu = g.kernel("moe_silu_mul", "moe_silu_mul")?;

    let to_b = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut r = Rng(0x5EED_F05E_2026_0810);
    let mut owned: Vec<DevicePtr> = Vec::new();

    // Build the [ne] pointer tables for one projection of shape [n, k].
    let build = |r: &mut Rng, owned: &mut Vec<DevicePtr>, n: usize, k: usize| -> Result<ProjTabs> {
        let words = (k / 16) * (n / 16) * 48;
        let (mut tp, mut sup, mut svp) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..ne {
            let t: Vec<u16> = (0..words).map(|_| r.u16()).collect();
            let su: Vec<u16> = (0..k).map(|_| r.sign_f16()).collect();
            let sv: Vec<u16> = (0..n).map(|_| r.sign_f16()).collect();
            tp.push(up(g, &to_b(&t))?);
            sup.push(up(g, &to_b(&su))?);
            svp.push(up(g, &to_b(&sv))?);
        }
        let tab =
            |ps: &[DevicePtr]| -> Vec<u8> { ps.iter().flat_map(|p| p.0.to_le_bytes()).collect() };
        let out = (up(g, &tab(&tp))?, up(g, &tab(&sup))?, up(g, &tab(&svp))?);
        owned.extend(tp);
        owned.extend(sup);
        owned.extend(svp);
        owned.extend([out.0, out.1, out.2]);
        Ok(out)
    };
    let gate_t = build(&mut r, &mut owned, inter, h)?;
    let up_t = build(&mut r, &mut owned, inter, h)?;
    let down_t = build(&mut r, &mut owned, h, inter)?;

    let a_host: Vec<u16> = (0..h)
        .map(|_| bf16::from_f32((r.unit() - 0.5) * 0.5).to_bits())
        .collect();
    let d_a = up(g, &to_b(&a_host))?;
    let idx_bytes: Vec<u8> = indices.iter().flat_map(|x| x.to_le_bytes()).collect();
    let d_idx = up(g, &idx_bytes)?;

    let gu_bytes = top_k * inter * 2;
    let dn_bytes = top_k * h * 2;
    let d_gate = g.alloc(gu_bytes)?;
    let d_upo = g.alloc(gu_bytes)?;
    let d_down = g.alloc(dn_bytes)?;

    // Scratch sized exactly as `set_exl3_experts` sizes it: one private
    // [SPLIT_K, N] fp32 region + [N/128] counters per launch GROUP, widest
    // group count = 2·top_k (gate+up).
    let (max_split, max_n) = (12usize, 4096usize);
    let groups = 2 * top_k;
    let cnt_bytes = groups * (max_n / 128) * 4;
    let d_ws = g.alloc(groups * max_split * max_n * 4)?;
    let d_cnt = g.alloc(cnt_bytes)?;
    g.memset(d_cnt, 0, cnt_bytes)?;

    // Production split policy (mirrors Exl3MoeState::split_for): ~96 CTAs per
    // launch group, rounded up to the first split that makes
    // `strips · split · groups` an exact multiple of a 192-CTA GB10 wave.
    let split_for = |n: usize, grp: usize| -> u32 {
        let strips = (n / 128).max(1);
        let base = (96 / strips).clamp(1, max_split);
        (base..=max_split)
            .find(|s| (strips * s * grp.max(1)) % 192 == 0)
            .unwrap_or(base) as u32
    };
    let (split_gu, split_dn) = (split_for(inter, 2 * top_k), split_for(h, top_k));

    // ---- Path A: per-slot bring-up chain (4·top_k launches) ----
    let mut launches_a = 0usize;
    g.memset(d_gate, 0x00, gu_bytes)?;
    g.memset(d_upo, 0x00, gu_bytes)?;
    g.memset(d_down, 0x00, dn_bytes)?;
    for slot in 0..top_k as u32 {
        let gate_row = d_gate.offset(slot as usize * inter * 2);
        let up_row = d_upo.offset(slot as usize * inter * 2);
        let down_row = d_down.offset(slot as usize * h * 2);
        launch_gemv_idx(
            g, kh_idx, stream, split_gu, d_a, gate_t, d_idx, slot, gate_row, d_ws, d_cnt, inter, h,
        )?;
        launch_gemv_idx(
            g, kh_idx, stream, split_gu, d_a, up_t, d_idx, slot, up_row, d_ws, d_cnt, inter, h,
        )?;
        launch_silu(g, kh_silu, stream, gate_row, up_row, gate_row, inter as u32)?;
        launch_gemv_idx(
            g, kh_idx, stream, split_dn, gate_row, down_t, d_idx, slot, down_row, d_ws, d_cnt, h,
            inter,
        )?;
        launches_a += 4;
    }
    g.synchronize(stream)?;
    let (mut a_gate, mut a_up, mut a_down) = (
        vec![0u8; gu_bytes],
        vec![0u8; gu_bytes],
        vec![0u8; dn_bytes],
    );
    g.copy_d2h(d_gate, &mut a_gate)?;
    g.copy_d2h(d_upo, &mut a_up)?;
    g.copy_d2h(d_down, &mut a_down)?;

    // ---- Path B: fused (3 launches), poisoned with the opposite byte ----
    let fused = |g: &dyn GpuBackend| -> Result<usize> {
        g.memset(d_gate, 0xFF, gu_bytes)?;
        g.memset(d_upo, 0xFF, gu_bytes)?;
        g.memset(d_down, 0xFF, dn_bytes)?;
        KernelLaunch::new(g, kh_gu)
            .grid([(inter / 128) as u32, split_gu, 2 * top_k as u32])
            .block([256, 1, 1])
            .arg_ptr(d_a)
            .arg_ptr(gate_t.0)
            .arg_ptr(gate_t.1)
            .arg_ptr(gate_t.2)
            .arg_ptr(up_t.0)
            .arg_ptr(up_t.1)
            .arg_ptr(up_t.2)
            .arg_ptr(d_idx)
            .arg_ptr(d_gate)
            .arg_ptr(d_upo)
            .arg_ptr(d_ws)
            .arg_ptr(d_cnt)
            .arg_u32(inter as u32)
            .arg_u32(h as u32)
            .arg_u32(3)
            .launch(stream)?;
        launch_silu(
            g,
            kh_silu,
            stream,
            d_gate,
            d_upo,
            d_gate,
            (top_k * inter) as u32,
        )?;
        KernelLaunch::new(g, kh_dn)
            .grid([(h / 128) as u32, split_dn, top_k as u32])
            .block([256, 1, 1])
            .arg_ptr(d_gate)
            .arg_ptr(down_t.0)
            .arg_ptr(down_t.1)
            .arg_ptr(down_t.2)
            .arg_ptr(d_idx)
            .arg_ptr(d_down)
            .arg_ptr(d_ws)
            .arg_ptr(d_cnt)
            .arg_u32(h as u32)
            .arg_u32(inter as u32)
            .arg_u32(3)
            .launch(stream)?;
        g.synchronize(stream)?;
        Ok(3)
    };
    let launches_b = fused(g)?;
    let (mut b_gate, mut b_up, mut b_down) = (
        vec![0u8; gu_bytes],
        vec![0u8; gu_bytes],
        vec![0u8; dn_bytes],
    );
    g.copy_d2h(d_gate, &mut b_gate)?;
    g.copy_d2h(d_upo, &mut b_up)?;
    g.copy_d2h(d_down, &mut b_down)?;

    let diff = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(a, b)| a != b).count();
    let (dg, du, dd) = (
        diff(&a_gate, &b_gate),
        diff(&a_up, &b_up),
        diff(&a_down, &b_down),
    );
    // Guard against "both paths wrote nothing": each path starts from a
    // DIFFERENT poison byte, so an unwritten region cannot compare equal —
    // and the fused result must not still be the poison.
    let nontrivial = b_down.iter().any(|&x| x != 0xFF) && b_gate.iter().any(|&x| x != 0xFF);
    let g8 = dg == 0 && du == 0 && dd == 0 && nontrivial;
    eprintln!(
        "FUSED GATE8 fused==per-slot byte-identical: gate_diff={dg}/{gu_bytes} \
         up_diff={du}/{gu_bytes} down_diff={dd}/{dn_bytes} nontrivial={nontrivial}  {}",
        if g8 { "PASS" } else { "FAIL" }
    );

    // Relaunch determinism on the fused path (per-group split-K scratch +
    // self-resetting counters must survive back-to-back launches).
    let _ = fused(g)?;
    let (mut c_gate, mut c_up, mut c_down) = (
        vec![0u8; gu_bytes],
        vec![0u8; gu_bytes],
        vec![0u8; dn_bytes],
    );
    g.copy_d2h(d_gate, &mut c_gate)?;
    g.copy_d2h(d_upo, &mut c_up)?;
    g.copy_d2h(d_down, &mut c_down)?;
    let g8b = c_gate == b_gate && c_up == b_up && c_down == b_down;
    eprintln!(
        "FUSED GATE8b fused relaunch byte-identical (split={split_gu}/{split_dn}, \
         {groups} concurrent groups): {}",
        if g8b { "PASS" } else { "FAIL" }
    );

    // Launch-count assertion: this is the whole point of the change.
    let g8c = launches_a == 4 * top_k && launches_b == 3;
    eprintln!(
        "FUSED GATE8c launches/layer (top_k={top_k}): per-slot={launches_a} fused={launches_b} \
         ({:.1}x fewer; +4 for the NVFP4 shared expert in both)  {}",
        launches_a as f64 / launches_b as f64,
        if g8c { "PASS" } else { "FAIL" }
    );

    // ---- GATE 9 (informational): GB/s at the PRODUCTION fused geometry ----
    //
    // Why this exists (round-4 diagnosis). GATE3 times ONE matrix: grid =
    // (N/128, SPLIT_K) = 128-192 CTAs, i.e. a SINGLE 192-CTA wave, so the
    // per-CTA prologue (x' input pass) and epilogue (split-K fence + atomic
    // election + ws re-read) are fully exposed and are amortised over only
    // 32/SPLIT_K stages. Production NEVER runs that shape: the fused launch
    // carries `2·top_k` / `top_k` groups on grid.z, so gate+up is
    // 16·6·2·top_k = 192·top_k CTAs = `top_k` FULL WAVES, where a retiring
    // CTA's tail overlaps the next CTA's prologue. Tuning against GATE3 alone
    // optimises a regime the model does not run in — GATE9 is the number the
    // end-to-end tok/s arithmetic should be built on.
    //
    // Cache: gate+up streams `2·top_k` distinct expert matrices (16 × 3.15 MB
    // = 50 MB at top_k = 8) against a 24 MB L2 in a cyclic pattern, so the
    // re-read hit rate across iterations is ~0 — cold enough without a ring.
    {
        let iters = 200u32;
        let time_it = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..10 {
                f()?;
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
            for _ in 0..iters {
                f()?;
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
        let launch_gu = || -> Result<()> {
            KernelLaunch::new(g, kh_gu)
                .grid([(inter / 128) as u32, split_gu, 2 * top_k as u32])
                .block([256, 1, 1])
                .arg_ptr(d_a)
                .arg_ptr(gate_t.0)
                .arg_ptr(gate_t.1)
                .arg_ptr(gate_t.2)
                .arg_ptr(up_t.0)
                .arg_ptr(up_t.1)
                .arg_ptr(up_t.2)
                .arg_ptr(d_idx)
                .arg_ptr(d_gate)
                .arg_ptr(d_upo)
                .arg_ptr(d_ws)
                .arg_ptr(d_cnt)
                .arg_u32(inter as u32)
                .arg_u32(h as u32)
                .arg_u32(3)
                .launch(stream)
        };
        let launch_dn = || -> Result<()> {
            KernelLaunch::new(g, kh_dn)
                .grid([(h / 128) as u32, split_dn, top_k as u32])
                .block([256, 1, 1])
                .arg_ptr(d_gate)
                .arg_ptr(down_t.0)
                .arg_ptr(down_t.1)
                .arg_ptr(down_t.2)
                .arg_ptr(d_idx)
                .arg_ptr(d_down)
                .arg_ptr(d_ws)
                .arg_ptr(d_cnt)
                .arg_u32(h as u32)
                .arg_u32(inter as u32)
                .arg_u32(3)
                .launch(stream)
        };
        // Payload = trellis + suh/svh actually streamed by each launch.
        let mat = |n: usize, k: usize| n * k * 3 / 8 + (n + k) * 2;
        let by_gu = 2 * top_k * mat(inter, h);
        let by_dn = top_k * mat(h, inter);
        let ms_gu = time_it(&launch_gu)?;
        let ms_dn = time_it(&launch_dn)?;
        let ctas_gu = (inter / 128) * split_gu as usize * 2 * top_k;
        let ctas_dn = (h / 128) * split_dn as usize * top_k;
        eprintln!(
            "FUSED GATE9 gate+up: {:.1} us  {:.1} GB/s  ({ctas_gu} CTAs = {:.2} waves of 192, \
             split={split_gu}, {} chunks/CTA)",
            ms_gu * 1e3,
            by_gu as f64 / (ms_gu * 1e-3) / 1e9,
            ctas_gu as f64 / 192.0,
            (h / 128) as f64 / split_gu as f64,
        );
        eprintln!(
            "FUSED GATE9 down:    {:.1} us  {:.1} GB/s  ({ctas_dn} CTAs = {:.2} waves of 192, \
             split={split_dn}, {} chunks/CTA)",
            ms_dn * 1e3,
            by_dn as f64 / (ms_dn * 1e-3) / 1e9,
            ctas_dn as f64 / 192.0,
            (inter / 128) as f64 / split_dn as f64,
        );
        eprintln!(
            "FUSED GATE9 routed FFN per layer: {:.1} us for {:.2} MB  =>  {:.1} GB/s \
             (ceiling 229)",
            (ms_gu + ms_dn) * 1e3,
            (by_gu + by_dn) as f64 / 1e6,
            (by_gu + by_dn) as f64 / ((ms_gu + ms_dn) * 1e-3) / 1e9,
        );
    }

    for p in owned
        .into_iter()
        .chain([d_a, d_idx, d_gate, d_upo, d_down, d_ws, d_cnt])
    {
        let _ = g.free(p);
    }
    Ok(g8 && g8b && g8c)
}

// ---------------------------------------------------------------------------
// GATE 9: the m-row (speculative verify) dispatch == the m=1 fused path, row by
// row, byte for byte.
//
// THE EXACT-GEMV LAW. docs/DECODE-WATERFALL-2026-08-10.md and the o-proj A/B
// (memory `oproj-grouped-kernels-ab-2026-08-09`) measured that a PARTIALLY
// exact verify chain is WORSE than either extreme: o-proj-only exactness scored
// 2.54 tok/step against 2.83 for none and 2.92-3.01 for full. So the m-row
// expert output for a verify row must equal, BIT FOR BIT, what the m=1 fused
// decode path computes for that same token — not "close", not "cosine 0.9999".
// This gate is the only thing standing between that law and a regression, so it
// is deliberately hostile:
//
//   * m in {2, 4, 6, 8} — every compiled ladder rung the host can dispatch.
//   * the index list is NOT slot-ordered (a CTA must resolve its expert from
//     `indices[y]` on device, never from blockIdx) and is duplicate-HEAVY
//     across rows, including one row that repeats another row's set verbatim
//     (every leader gathers M = m) and one that reverses it (the same expert
//     lands in a different slot position on different rows). Within a row the
//     ids stay distinct, which is the routing invariant the gather relies on to
//     bound M by num_tokens.
//   * the two paths are poisoned with DIFFERENT bytes (0xAA vs 0x55), so a
//     (row, slot) that no CTA writes cannot pass by both paths agreeing on
//     garbage, and the result must additionally not still BE the poison.
//   * the m-row path is relaunched and must be byte-identical to itself —
//     split-K determinism with many concurrent, per-slot-keyed `ws` regions and
//     self-resetting counters is the thing most likely to break silently.
// ---------------------------------------------------------------------------

/// Ladder rungs the host may dispatch (`EXL3_MROW_ARMS` in exl3_decode.rs,
/// minus the m1 reference rung which GATE8 already covers).
const MROW_WIDTHS: [usize; 5] = [2, 4, 6, 8, 16];

#[allow(clippy::too_many_lines)]
fn mrow_verify_gate(g: &dyn GpuBackend) -> Result<bool> {
    let mut ok = true;
    for (module, bits, quant_label) in [("exl3_gemv", 3usize, "K3"), ("exl3_gemv_k2", 2usize, "K2")]
    {
        // Each quant lane allocates, validates, and frees its synthetic expert
        // tables before the next lane starts, so K2 does not double peak VRAM.
        ok &= mrow_verify_quant_gate(g, module, bits, quant_label)?;
    }
    Ok(ok)
}

#[allow(clippy::too_many_lines)]
fn mrow_verify_quant_gate(
    g: &dyn GpuBackend,
    module: &str,
    bits: usize,
    quant_label: &str,
) -> Result<bool> {
    let stream = 0u64;
    let (h, inter) = (4096usize, 2048usize);
    let (top_k, ne) = (FUSED_TOP_K, FUSED_NE);

    let kh_gu_1 = g.kernel(module, "exl3_gemv_m1_fused_gate_up")?;
    let kh_dn_1 = g.kernel(module, "exl3_gemv_m1_fused_down")?;
    let kh_work_build_m6 = g.kernel(module, "exl3_build_m6_worklist")?;
    let kh_work_gu_m6 = g.kernel(module, "exl3_gemv_mrow_persistent_gate_up_m6")?;
    let kh_work_dn_m6 = g.kernel(module, "exl3_gemv_mrow_persistent_down_m6")?;
    let kh_work_build_m16 = g.kernel(module, "exl3_build_m16_worklist")?;
    let kh_work_gu_m16 = g.kernel(module, "exl3_gemv_mrow_persistent_gate_up_m16")?;
    let kh_work_dn_m16 = g.kernel(module, "exl3_gemv_mrow_persistent_down_m16")?;
    let kh_silu = g.kernel("moe_silu_mul", "moe_silu_mul")?;

    let to_b = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut r = Rng(0x9A17_C0DE_2026_0812);
    let mut owned: Vec<DevicePtr> = Vec::new();

    let build = |r: &mut Rng, owned: &mut Vec<DevicePtr>, n: usize, k: usize| -> Result<ProjTabs> {
        let words = (k / 16) * (n / 16) * (16 * bits);
        let (mut tp, mut sup, mut svp) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..ne {
            let t: Vec<u16> = (0..words).map(|_| r.u16()).collect();
            let su: Vec<u16> = (0..k).map(|_| r.sign_f16()).collect();
            let sv: Vec<u16> = (0..n).map(|_| r.sign_f16()).collect();
            tp.push(up(g, &to_b(&t))?);
            sup.push(up(g, &to_b(&su))?);
            svp.push(up(g, &to_b(&sv))?);
        }
        let tab =
            |ps: &[DevicePtr]| -> Vec<u8> { ps.iter().flat_map(|p| p.0.to_le_bytes()).collect() };
        let out = (up(g, &tab(&tp))?, up(g, &tab(&sup))?, up(g, &tab(&svp))?);
        owned.extend(tp);
        owned.extend(sup);
        owned.extend(svp);
        owned.extend([out.0, out.1, out.2]);
        Ok(out)
    };
    let gate_t = build(&mut r, &mut owned, inter, h)?;
    let up_t = build(&mut r, &mut owned, inter, h)?;
    let down_t = build(&mut r, &mut owned, h, inter)?;

    // Per-row expert sets. Row 1 repeats row 0 verbatim (maximum dedup: every
    // leader gathers all m rows); row 2 reverses it (same union, different slot
    // positions); the rest rotate so leaders land at assorted flat slots and
    // gather assorted subsets.
    let base: [u32; FUSED_TOP_K] = [5, 0, 7, 2, 6, 1];
    let row_ids = |t: usize| -> Vec<u32> {
        match t {
            0 | 1 => base.to_vec(),
            2 => base.iter().rev().copied().collect(),
            _ => (0..top_k)
                .map(|s| base[(s + t) % top_k] ^ ((t as u32) & 1))
                .collect(),
        }
    };
    // Sanity: within a row the ids must be distinct (the routing invariant that
    // bounds a leader's gather by num_tokens) and inside [0, ne).
    for t in 0..*MROW_WIDTHS.last().unwrap() {
        let ids = row_ids(t);
        assert!(
            ids.iter().all(|&e| (e as usize) < ne),
            "row {t} id out of range"
        );
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            top_k,
            "row {t} must route to top_k DISTINCT experts"
        );
    }

    let max_m = *MROW_WIDTHS.last().unwrap();
    let a_host: Vec<u16> = (0..max_m * h)
        .map(|_| bf16::from_f32((r.unit() - 0.5) * 0.5).to_bits())
        .collect();
    let d_a = up(g, &to_b(&a_host))?;

    let gu_bytes_max = max_m * top_k * inter * 2;
    let dn_bytes_max = max_m * top_k * h * 2;
    let d_gate = g.alloc(gu_bytes_max)?;
    let d_upo = g.alloc(gu_bytes_max)?;
    let d_down = g.alloc(dn_bytes_max)?;

    // Production split policy — the m-row dispatch MUST pass the same splits
    // the m=1 path passes, or the K-slice per split moves and bit-identity is
    // gone by construction. Using one `split_for` for both paths encodes that.
    let max_split = 12usize;
    let split_for = |n: usize| -> u32 { (96 / (n / 128).max(1)).clamp(1, max_split) as u32 };
    let (split_gu, split_dn) = (split_for(inter), split_for(h));

    // Scratch sized exactly as `set_exl3_experts` sizes it (`exl3_ws_floats`):
    // the m-row `ws` is keyed by the flat routed SLOT (`2*slot + proj` for
    // gate+up, `slot` for down), so it must span every slot of the widest row
    // count; `counters` stays keyed by launch group.
    let slots_max = max_m * top_k;
    let ws_floats = (2 * slots_max * split_gu as usize * inter)
        .max(slots_max * split_dn as usize * h)
        .max(2 * top_k * max_split * 4096);
    let cnt_ints = (2 * slots_max).max(2 * top_k) * (4096 / 128);
    let d_ws = g.alloc(ws_floats * 4)?;
    let d_cnt = g.alloc(cnt_ints * 4)?;
    g.memset(d_cnt, 0, cnt_ints * 4)?;
    // One allocation serves both exact ABIs. M6 still uses capacity 36 and a
    // 32-byte record stride; M16 uses capacity 96 and an 80-byte stride.
    let d_worklist = g.alloc(96 * 80)?;
    let d_work_state = g.alloc(16)?;

    let mut ok = true;
    for &m in &MROW_WIDTHS {
        let total_routed = m * top_k;
        let gu_bytes = total_routed * inter * 2;
        let dn_bytes = total_routed * h * 2;
        let indices: Vec<u32> = (0..m).flat_map(row_ids).collect();
        let d_idx = up(
            g,
            &indices
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;

        let kh_gu_m = g.kernel(module, &format!("exl3_gemv_mrow_fused_gate_up_m{m}"))?;
        let kh_dn_m = g.kernel(module, &format!("exl3_gemv_mrow_fused_down_m{m}"))?;

        // ---- Path A: the m-row dedup'd dispatch (3 launches for ALL rows) ----
        let mrow = |g: &dyn GpuBackend| -> Result<()> {
            g.memset(d_gate, 0xAA, gu_bytes)?;
            g.memset(d_upo, 0xAA, gu_bytes)?;
            g.memset(d_down, 0xAA, dn_bytes)?;
            KernelLaunch::new(g, kh_gu_m)
                .grid([(inter / 128) as u32, split_gu, 2 * total_routed as u32])
                .block([256, 1, 1])
                .arg_ptr(d_a)
                .arg_ptr(gate_t.0)
                .arg_ptr(gate_t.1)
                .arg_ptr(gate_t.2)
                .arg_ptr(up_t.0)
                .arg_ptr(up_t.1)
                .arg_ptr(up_t.2)
                .arg_ptr(d_idx)
                .arg_ptr(d_gate)
                .arg_ptr(d_upo)
                .arg_ptr(d_ws)
                .arg_ptr(d_cnt)
                .arg_u32(inter as u32)
                .arg_u32(h as u32)
                .arg_u32(top_k as u32)
                .arg_u32(m as u32)
                .arg_u32(bits as u32)
                .launch(stream)?;
            launch_silu(
                g,
                kh_silu,
                stream,
                d_gate,
                d_upo,
                d_gate,
                (total_routed * inter) as u32,
            )?;
            KernelLaunch::new(g, kh_dn_m)
                .grid([(h / 128) as u32, split_dn, total_routed as u32])
                .block([256, 1, 1])
                .arg_ptr(d_gate)
                .arg_ptr(down_t.0)
                .arg_ptr(down_t.1)
                .arg_ptr(down_t.2)
                .arg_ptr(d_idx)
                .arg_ptr(d_down)
                .arg_ptr(d_ws)
                .arg_ptr(d_cnt)
                .arg_u32(h as u32)
                .arg_u32(inter as u32)
                .arg_u32(top_k as u32)
                .arg_u32(m as u32)
                .arg_u32(bits as u32)
                .launch(stream)?;
            g.synchronize(stream)
        };
        mrow(g)?;
        let (mut a_gate, mut a_up, mut a_down) = (
            vec![0u8; gu_bytes],
            vec![0u8; gu_bytes],
            vec![0u8; dn_bytes],
        );
        g.copy_d2h(d_gate, &mut a_gate)?;
        g.copy_d2h(d_upo, &mut a_up)?;
        g.copy_d2h(d_down, &mut a_down)?;

        // ---- Path B: the shipping m=1 fused path, once per row (what the
        //      per-row `forward_batched` fallback runs today). Poisoned with
        //      the OPPOSITE byte. ----
        g.memset(d_gate, 0x55, gu_bytes)?;
        g.memset(d_upo, 0x55, gu_bytes)?;
        g.memset(d_down, 0x55, dn_bytes)?;
        for t in 0..m {
            let a_row = d_a.offset(t * h * 2);
            let idx_row = d_idx.offset(t * top_k * 4);
            let gate_row = d_gate.offset(t * top_k * inter * 2);
            let up_row = d_upo.offset(t * top_k * inter * 2);
            let down_row = d_down.offset(t * top_k * h * 2);
            KernelLaunch::new(g, kh_gu_1)
                .grid([(inter / 128) as u32, split_gu, 2 * top_k as u32])
                .block([256, 1, 1])
                .arg_ptr(a_row)
                .arg_ptr(gate_t.0)
                .arg_ptr(gate_t.1)
                .arg_ptr(gate_t.2)
                .arg_ptr(up_t.0)
                .arg_ptr(up_t.1)
                .arg_ptr(up_t.2)
                .arg_ptr(idx_row)
                .arg_ptr(gate_row)
                .arg_ptr(up_row)
                .arg_ptr(d_ws)
                .arg_ptr(d_cnt)
                .arg_u32(inter as u32)
                .arg_u32(h as u32)
                .arg_u32(bits as u32)
                .launch(stream)?;
            launch_silu(
                g,
                kh_silu,
                stream,
                gate_row,
                up_row,
                gate_row,
                (top_k * inter) as u32,
            )?;
            KernelLaunch::new(g, kh_dn_1)
                .grid([(h / 128) as u32, split_dn, top_k as u32])
                .block([256, 1, 1])
                .arg_ptr(gate_row)
                .arg_ptr(down_t.0)
                .arg_ptr(down_t.1)
                .arg_ptr(down_t.2)
                .arg_ptr(idx_row)
                .arg_ptr(down_row)
                .arg_ptr(d_ws)
                .arg_ptr(d_cnt)
                .arg_u32(h as u32)
                .arg_u32(inter as u32)
                .arg_u32(bits as u32)
                .launch(stream)?;
        }
        g.synchronize(stream)?;
        let (mut b_gate, mut b_up, mut b_down) = (
            vec![0u8; gu_bytes],
            vec![0u8; gu_bytes],
            vec![0u8; dn_bytes],
        );
        g.copy_d2h(d_gate, &mut b_gate)?;
        g.copy_d2h(d_upo, &mut b_up)?;
        g.copy_d2h(d_down, &mut b_down)?;

        // Per-ROW diff, so a failure names the row that drifted rather than a
        // byte count over the whole block.
        let row_diffs = |x: &[u8], y: &[u8], row_bytes: usize| -> Vec<usize> {
            x.chunks_exact(row_bytes)
                .zip(y.chunks_exact(row_bytes))
                .map(|(a, b)| a.iter().zip(b).filter(|(p, q)| p != q).count())
                .collect()
        };
        let dg = row_diffs(&a_gate, &b_gate, top_k * inter * 2);
        let du = row_diffs(&a_up, &b_up, top_k * inter * 2);
        let dd = row_diffs(&a_down, &b_down, top_k * h * 2);
        let bad: Vec<usize> = (0..m)
            .filter(|&t| dg[t] != 0 || du[t] != 0 || dd[t] != 0)
            .collect();
        // Neither path may have left its poison behind.
        let nontrivial = a_gate.iter().any(|&x| x != 0xAA)
            && a_down.iter().any(|&x| x != 0xAA)
            && b_gate.iter().any(|&x| x != 0x55)
            && b_down.iter().any(|&x| x != 0x55);
        let g9 = bad.is_empty() && nontrivial;
        ok &= g9;
        eprintln!(
            "MROW GATE9 {quant_label} m={m}: every row byte-identical to the m=1 fused path \
             (rows={m} slots={total_routed}, drifted rows={bad:?}, \
             gate={dg:?} up={du:?} down={dd:?}, nontrivial={nontrivial})  {}",
            if g9 { "PASS" } else { "FAIL" }
        );

        // Relaunch determinism: per-slot-keyed ws regions + self-resetting
        // counters across many concurrent groups must land the same every time.
        mrow(g)?;
        let (mut c_gate, mut c_up, mut c_down) = (
            vec![0u8; gu_bytes],
            vec![0u8; gu_bytes],
            vec![0u8; dn_bytes],
        );
        g.copy_d2h(d_gate, &mut c_gate)?;
        g.copy_d2h(d_upo, &mut c_up)?;
        g.copy_d2h(d_down, &mut c_down)?;
        let g9b = c_gate == a_gate && c_up == a_up && c_down == a_down;
        ok &= g9b;
        eprintln!(
            "MROW GATE9b {quant_label} m={m}: relaunch byte-identical \
             (split={split_gu}/{split_dn}, {} concurrent gate+up groups): {}",
            2 * total_routed,
            if g9b { "PASS" } else { "FAIL" }
        );

        if m == 6 || m == 16 {
            let (kh_work_build, kh_work_gu, kh_work_dn, work_capacity, work_label) = if m == 6 {
                (kh_work_build_m6, kh_work_gu_m6, kh_work_dn_m6, 36u32, "M6")
            } else {
                (
                    kh_work_build_m16,
                    kh_work_gu_m16,
                    kh_work_dn_m16,
                    96u32,
                    "M16",
                )
            };
            let persistent = |g: &dyn GpuBackend, route: DevicePtr, poison: u8| -> Result<()> {
                g.memset(d_gate, poison, gu_bytes)?;
                g.memset(d_upo, poison, gu_bytes)?;
                g.memset(d_down, poison, dn_bytes)?;
                KernelLaunch::new(g, kh_work_build)
                    .grid([1, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(route)
                    .arg_ptr(d_worklist)
                    .arg_ptr(d_work_state)
                    .arg_u32(m as u32)
                    .arg_u32(top_k as u32)
                    .arg_u32(ne as u32)
                    .arg_u32(work_capacity)
                    .launch(stream)?;
                KernelLaunch::new(g, kh_work_gu)
                    .grid([96, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_a)
                    .arg_ptr(gate_t.0)
                    .arg_ptr(gate_t.1)
                    .arg_ptr(gate_t.2)
                    .arg_ptr(up_t.0)
                    .arg_ptr(up_t.1)
                    .arg_ptr(up_t.2)
                    .arg_ptr(d_worklist)
                    .arg_ptr(d_work_state)
                    .arg_ptr(d_gate)
                    .arg_ptr(d_upo)
                    .arg_ptr(d_ws)
                    .arg_ptr(d_cnt)
                    .arg_u32(inter as u32)
                    .arg_u32(h as u32)
                    .arg_u32(top_k as u32)
                    .arg_u32(split_gu)
                    .arg_u32(bits as u32)
                    .launch(stream)?;
                launch_silu(
                    g,
                    kh_silu,
                    stream,
                    d_gate,
                    d_upo,
                    d_gate,
                    (total_routed * inter) as u32,
                )?;
                KernelLaunch::new(g, kh_work_dn)
                    .grid([96, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_gate)
                    .arg_ptr(down_t.0)
                    .arg_ptr(down_t.1)
                    .arg_ptr(down_t.2)
                    .arg_ptr(d_worklist)
                    .arg_ptr(d_work_state)
                    .arg_ptr(d_down)
                    .arg_ptr(d_ws)
                    .arg_ptr(d_cnt)
                    .arg_u32(h as u32)
                    .arg_u32(inter as u32)
                    .arg_u32(top_k as u32)
                    .arg_u32(split_dn)
                    .arg_u32(bits as u32)
                    .launch(stream)?;
                g.synchronize(stream)
            };

            persistent(g, d_idx, 0x3C)?;
            let (mut p_gate, mut p_up, mut p_down) = (
                vec![0u8; gu_bytes],
                vec![0u8; gu_bytes],
                vec![0u8; dn_bytes],
            );
            g.copy_d2h(d_gate, &mut p_gate)?;
            g.copy_d2h(d_upo, &mut p_up)?;
            g.copy_d2h(d_down, &mut p_down)?;
            let mut state = [0u8; 16];
            g.copy_d2h(d_work_state, &mut state)?;
            let valid_work_count = u32::from_le_bytes(state[0..4].try_into().unwrap());
            let valid_work_status = u32::from_le_bytes(state[4..8].try_into().unwrap());
            let compact_ok = p_gate == a_gate
                && p_up == a_up
                && p_down == a_down
                && valid_work_count > 0
                && valid_work_count <= work_capacity
                && valid_work_status == 0
                && p_gate.iter().any(|&byte| byte != 0x3C)
                && p_down.iter().any(|&byte| byte != 0x3C);
            ok &= compact_ok;

            persistent(g, d_idx, 0xC3)?;
            let (mut p2_gate, mut p2_up, mut p2_down) = (
                vec![0u8; gu_bytes],
                vec![0u8; gu_bytes],
                vec![0u8; dn_bytes],
            );
            g.copy_d2h(d_gate, &mut p2_gate)?;
            g.copy_d2h(d_upo, &mut p2_up)?;
            g.copy_d2h(d_down, &mut p2_down)?;
            let compact_replay = p2_gate == p_gate && p2_up == p_up && p2_down == p_down;
            ok &= compact_replay;

            // A duplicate in the final row used to let the first leader cross
            // the record boundary before the builder reached that row. Both
            // exact-width builders must reject before emission, and both
            // consumers must zero every routed output rather than blend stale
            // scratch.
            let mut malformed = indices.clone();
            let final_row = (m - 1) * top_k;
            malformed[final_row + 1] = malformed[final_row];
            let d_malformed = up(
                g,
                &malformed
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )?;
            persistent(g, d_malformed, 0xA5)?;
            let (mut bad_gate, mut bad_up, mut bad_down) = (
                vec![0u8; gu_bytes],
                vec![0u8; gu_bytes],
                vec![0u8; dn_bytes],
            );
            g.copy_d2h(d_gate, &mut bad_gate)?;
            g.copy_d2h(d_upo, &mut bad_up)?;
            g.copy_d2h(d_down, &mut bad_down)?;
            g.copy_d2h(d_work_state, &mut state)?;
            let work_count = u32::from_le_bytes(state[0..4].try_into().unwrap());
            let work_status = u32::from_le_bytes(state[4..8].try_into().unwrap());
            let malformed_zeroed = work_count == 0
                && work_status == 3
                && bad_gate.iter().all(|&byte| byte == 0)
                && bad_up.iter().all(|&byte| byte == 0)
                && bad_down.iter().all(|&byte| byte == 0);
            ok &= malformed_zeroed;
            let _ = g.free(d_malformed);

            // The last routed id is outside the pointer tables. This must be
            // rejected by the complete-route validation before any record is
            // consumed, with the same full-output zeroing guarantee.
            let mut out_of_range = indices.clone();
            out_of_range[final_row + top_k - 1] = ne as u32;
            let d_out_of_range = up(
                g,
                &out_of_range
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )?;
            persistent(g, d_out_of_range, 0x5A)?;
            let (mut range_gate, mut range_up, mut range_down) = (
                vec![0u8; gu_bytes],
                vec![0u8; gu_bytes],
                vec![0u8; dn_bytes],
            );
            g.copy_d2h(d_gate, &mut range_gate)?;
            g.copy_d2h(d_upo, &mut range_up)?;
            g.copy_d2h(d_down, &mut range_down)?;
            g.copy_d2h(d_work_state, &mut state)?;
            let range_count = u32::from_le_bytes(state[0..4].try_into().unwrap());
            let range_status = u32::from_le_bytes(state[4..8].try_into().unwrap());
            let out_of_range_zeroed = range_count == 0
                && range_status == 2
                && range_gate.iter().all(|&byte| byte == 0)
                && range_up.iter().all(|&byte| byte == 0)
                && range_down.iter().all(|&byte| byte == 0);
            ok &= out_of_range_zeroed;
            let _ = g.free(d_out_of_range);
            eprintln!(
                "MROW GATE9d {quant_label} compact {work_label}: valid_count={valid_work_count} \
                 valid_status={valid_work_status} invalid_count={work_count} \
                 invalid_status={work_status} \
                 range_count={range_count} range_status={range_status} \
                 byte-identical={compact_ok} replay={compact_replay} \
                 malformed_zeroed={malformed_zeroed} \
                 out_of_range_zeroed={out_of_range_zeroed} {}",
                if compact_ok && compact_replay && malformed_zeroed && out_of_range_zeroed {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
        }

        // Launch-count assertion: this is the whole point of the change. The
        // per-row fallback runs the WHOLE routed expert set once per row; the
        // m-row path streams each distinct expert's trellis once for the block.
        let distinct: usize = {
            let mut v = indices.clone();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        eprintln!(
            "MROW GATE9c {quant_label} m={m}: launches {} -> 3, expert-trellis reads {total_routed} -> \
             {distinct} ({:.2}x fewer bytes on the union)",
            3 * m,
            total_routed as f64 / distinct as f64
        );

        let _ = g.free(d_idx);
    }

    for p in owned.into_iter().chain([
        d_a,
        d_gate,
        d_upo,
        d_down,
        d_ws,
        d_cnt,
        d_worklist,
        d_work_state,
    ]) {
        let _ = g.free(p);
    }
    Ok(ok)
}
