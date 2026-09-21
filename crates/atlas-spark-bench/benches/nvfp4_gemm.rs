// SPDX-License-Identifier: AGPL-3.0-only

//! NVFP4×NVFP4 (W4A4) tensor-core GEMM correctness + timing bench.
//!
//! Compares two prefill FFN GEMM paths:
//!   A. `w4a16_gemm_t_m128` — BF16 A × NVFP4 B (production path)
//!   B. `nvfp4_nvfp4_gemm_t_m64` — NVFP4 A × NVFP4 B with hardware
//!      block-scaled MMA (`mma.sync.kind::mxf4nvf4.scale_vec::4X.m16n8k64`)
//!   C. `nvfp4_nvfp4_gemm_kmajor_m128` — the same W4A4 arithmetic over
//!      transformed K-major weights, shared across 128 rows and required to
//!      match B bit-for-bit
//!   D. `nvfp4_nvfp4_gemm_kmajor_m256` — the long-prefill shadow, shared
//!      across 256 rows and required to match B and C bit-for-bit
//!   E. `moe_silu_mul` + `quantize_bf16_to_nvfp4` — the production separate
//!      SwiGLU/activation-quantization boundary
//!   F. `quantize_silu_mul_bf16_to_nvfp4` — the fused boundary, required to
//!      match E byte-for-byte for both packed E2M1 values and E4M3 scales
//!
//! Correctness strategy
//! --------------------
//! The two production kernels expect DIFFERENT B-weight layouts:
//!   - w4a16_t_m128 reads `B_packed[K/2][N]` + `B_scale[K/16][N]` (K-major)
//!   - nvfp4_nvfp4_t_m64 reads `B_packed[N][K/2]` + `B_scale[N][K/16]` (HF layout)
//!
//! To get a meaningful apples-to-apples comparison, we:
//!   1. Generate a canonical BF16 weight `B_bf16[N][K]` on host.
//!   2. Pack two layouts of the same data with shared per-tensor scale2.
//!   3. Generate canonical BF16 activations `A_bf16[M][K]` on host.
//!   4. Compute a HOST reference `C_ref = A_bf16 @ dequant(B_bf16)^T`.
//!   5. Run both kernels with their respective layouts.
//!   6. Cosine-sim each kernel's output against `C_ref`.
//!
//! Pass threshold: cosine ≥ 0.99 (both kernels carry NVFP4 quant error;
//! that's expected). Path B's score should be in the same ballpark as
//! Path A's — if it's catastrophically lower we've shipped a scale-gather
//! bug. The classic NVFP4 aliasing symptom is cos ≈ 0 or cos ≈ -1 with
//! plausible magnitudes (i.e. not NaN/Inf).

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

unsafe extern "C" {
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> i32;
}

fn h2d<T: Copy>(dev: u64, host: &[T]) {
    let bytes = std::mem::size_of_val(host);
    unsafe {
        let rc = cuMemcpyHtoD_v2(dev, host.as_ptr() as *const c_void, bytes);
        assert_eq!(rc, 0, "cuMemcpyHtoD failed: {rc}");
    }
}

fn d2h<T: Copy>(dst: &mut [T], dev: u64) {
    let bytes = std::mem::size_of_val(dst);
    unsafe {
        let rc = cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, dev, bytes);
        assert_eq!(rc, 0, "cuMemcpyDtoH failed: {rc}");
    }
}

const OUTPUT_CANARY_BYTES: usize = 4096;
const OUTPUT_CANARY_PREFIX: u8 = 0xA5;
const OUTPUT_CANARY_SUFFIX: u8 = 0x5A;

fn guarded_output_alloc(stream: u64, output_bytes: usize) -> (u64, u64) {
    let total = output_bytes
        .checked_add(2 * OUTPUT_CANARY_BYTES)
        .expect("guarded output allocation overflow");
    let raw = gpu::gpu_alloc_zeroed(stream, total).expect("allocate guarded output");
    // `gpu_alloc_zeroed` enqueues an asynchronous memset on this stream,
    // while h2d below is a synchronous driver copy with no ordering edge to
    // that non-default stream. Finish the memset before publishing canaries.
    gpu::gpu_sync(stream).expect("finish guarded output initialization");
    h2d(raw, &vec![OUTPUT_CANARY_PREFIX; OUTPUT_CANARY_BYTES]);
    h2d(
        raw + (OUTPUT_CANARY_BYTES + output_bytes) as u64,
        &vec![OUTPUT_CANARY_SUFFIX; OUTPUT_CANARY_BYTES],
    );
    (raw, raw + OUTPUT_CANARY_BYTES as u64)
}

fn assert_output_canaries(raw: u64, output_bytes: usize, label: &str) {
    let mut prefix = vec![0u8; OUTPUT_CANARY_BYTES];
    let mut suffix = vec![0u8; OUTPUT_CANARY_BYTES];
    d2h(&mut prefix, raw);
    d2h(
        &mut suffix,
        raw + (OUTPUT_CANARY_BYTES + output_bytes) as u64,
    );
    assert!(
        prefix.iter().all(|&byte| byte == OUTPUT_CANARY_PREFIX),
        "{label} wrote before its output buffer"
    );
    assert!(
        suffix.iter().all(|&byte| byte == OUTPUT_CANARY_SUFFIX),
        "{label} wrote after its output buffer"
    );
}

fn fnv1a_bf16(values: &[u16]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn fnv1a_bytes(values: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &byte in values {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn assert_bytes_identical(left_label: &str, left: &[u8], right_label: &str, right: &[u8]) {
    assert_eq!(
        left.len(),
        right.len(),
        "{left_label}/{right_label} output lengths differ"
    );
    if left != right {
        let first = left
            .iter()
            .zip(right)
            .position(|(left_value, right_value)| left_value != right_value)
            .expect("different output vectors must contain a mismatch");
        panic!(
            "{left_label}/{right_label} byte mismatch at flat index {first}: {left_label}=0x{:02x} {right_label}=0x{:02x}",
            left[first], right[first]
        );
    }
}

fn assert_bf16_identical(left_label: &str, left: &[u16], right_label: &str, right: &[u16]) {
    assert_eq!(
        left.len(),
        right.len(),
        "{left_label}/{right_label} output lengths differ"
    );
    if left != right {
        let first = left
            .iter()
            .zip(right)
            .position(|(left_value, right_value)| left_value != right_value)
            .expect("different output vectors must contain a mismatch");
        panic!(
            "{left_label}/{right_label} bitwise mismatch at flat index {first}: {left_label}=0x{:04x} {right_label}=0x{:04x}",
            left[first], right[first]
        );
    }
}

static W4A16_M128_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static NVFP4_GEMM_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static NVFP4_KMAJOR_M128_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static NVFP4_KMAJOR_M256_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static NVFP4_GEMM_SK_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static NVFP4_SK_REDUCE_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static ABSMAX_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static QUANT_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static SILU_MUL_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static FUSED_SILU_QUANT_FN: OnceLock<RawCudaFunc> = OnceLock::new();

// Tunable shape. Default is a small validation shape so the CPU reference
// matmul (used for cos-sim) finishes in ~0.3s. For production timing
// override via env vars:
//   ATLAS_BENCH_M=128 ATLAS_BENCH_K=5120 ATLAS_BENCH_N=17408
// (gate_proj/up_proj shape on Qwen3.6-27B prefill chunk size 128).
// CPU reference at the production shape would take ~10 minutes — we
// skip validation entirely when M*N*K > 1e9.
const GROUP_SIZE: u32 = 16;

fn bench_shape() -> (u32, u32, u32) {
    let m = std::env::var("ATLAS_BENCH_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128u32);
    let k = std::env::var("ATLAS_BENCH_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048u32);
    let n = std::env::var("ATLAS_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048u32);
    (m, k, n)
}

// ── LCG random + bf16 helpers ──
fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    *seed
}

fn bf16_from_f32(v: f32) -> u16 {
    // RNE rounding
    let bits = v.to_bits();
    let bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(bias)) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn random_bf16(elements: usize, seed: u32) -> Vec<u16> {
    let mut s = seed;
    (0..elements)
        .map(|_| {
            let r = (lcg(&mut s) >> 8) as f32 / ((1u32 << 24) as f32);
            bf16_from_f32(r * 2.0 - 1.0)
        })
        .collect()
}

// NVFP4 E2M1 LUT (matches kernels/.../cutlass_nvfp4_gemm.cu:156)
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

// Float → E2M1 nibble (matches quantize_bf16_to_nvfp4.cu:80-108)
fn float_to_e2m1(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 8u8 } else { 0u8 };
    let absv = v.abs();
    let idx: u8 = if absv < 0.25 {
        0
    } else if absv <= 0.75 {
        1
    } else if absv <= 1.25 {
        2
    } else if absv <= 1.75 {
        3
    } else if absv <= 2.5 {
        4
    } else if absv <= 3.5 {
        5
    } else if absv <= 5.0 {
        6
    } else {
        7
    };
    sign | idx
}

// Float → FP8 E4M3 byte (matches the production helper in quantize_bf16_to_nvfp4.cu:62)
fn float_to_fp8_e4m3(v: f32) -> u8 {
    if v == 0.0 || v.is_nan() {
        return 0;
    }
    let absv = v.abs();
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    // E4M3 max = 448. Saturate.
    let clamped = absv.min(448.0);
    // Naive encoding: exp bias 7, 3 mantissa bits.
    let bits = clamped.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_man = bits & 0x7FFFFF;
    let exp_unbiased = f32_exp - 127;
    let e4m3_exp = exp_unbiased + 7;
    if e4m3_exp <= 0 {
        // subnormal: mantissa = round(clamped / 2^-9)
        let m = (clamped / (1.0 / 512.0)).round() as u32;
        let m = m.min(7);
        return sign | (m as u8);
    }
    if e4m3_exp > 15 {
        return sign | 0x7E; // S.1111.110 = 448
    }
    // RNE on 3 mantissa bits
    let man3 = (f32_man + (1 << 19)) >> 20;
    let (final_exp, final_man) = if man3 > 7 {
        (e4m3_exp as u32 + 1, 0u32)
    } else {
        (e4m3_exp as u32, man3)
    };
    if final_exp > 15 {
        return sign | 0x7E;
    }
    sign | ((final_exp << 3) | final_man) as u8
}

// Decode FP8 E4M3 byte → float (matches kernel logic in quantize_bf16_to_nvfp4.cu:196-209)
fn fp8_e4m3_to_float(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0xF;
    let man = b & 0x7;
    let mag = if exp == 0 {
        (man as f32) * (1.0 / 512.0) // 2^-9 per mantissa unit
    } else if exp == 15 && man == 7 {
        0.0 // NaN → 0
    } else {
        let f32_bits = ((exp as u32 + 120) << 23) | ((man as u32) << 20);
        f32::from_bits(f32_bits)
    };
    if sign == 1 { -mag } else { mag }
}

// Quantize a BF16 row of length K to NVFP4 with per-group e4m3 scales.
// Returns (packed_nibbles_len_K/2, fp8_scales_len_K/16).
// Replicates quantize_bf16_to_nvfp4.cu exactly so we can match its output
// when validating Path B.
fn quant_bf16_row_to_nvfp4(row_bf16: &[u16], scale2: f32) -> (Vec<u8>, Vec<u8>) {
    let k = row_bf16.len();
    assert!(k.is_multiple_of(GROUP_SIZE as usize));
    let num_groups = k / GROUP_SIZE as usize;
    let mut packed = vec![0u8; k / 2];
    let mut scales = vec![0u8; num_groups];
    let inv_s2 = if scale2 > 0.0 { 1.0 / scale2 } else { 0.0 };
    for g in 0..num_groups {
        let base = g * GROUP_SIZE as usize;
        let mut gmax: f32 = 0.0;
        for i in 0..GROUP_SIZE as usize {
            let v = bf16_to_f32(row_bf16[base + i]).abs();
            if v > gmax {
                gmax = v;
            }
        }
        let fp8_float = if gmax > 0.0 { gmax * inv_s2 / 6.0 } else { 0.0 };
        let fp8_byte = float_to_fp8_e4m3(fp8_float);
        scales[g] = fp8_byte;
        let eff = fp8_e4m3_to_float(fp8_byte) * scale2;
        let inv_eff = if eff > 0.0 { 1.0 / eff } else { 0.0 };
        for i in (0..GROUP_SIZE as usize).step_by(2) {
            let v0 = bf16_to_f32(row_bf16[base + i]) * inv_eff;
            let v1 = bf16_to_f32(row_bf16[base + i + 1]) * inv_eff;
            let n0 = float_to_e2m1(v0);
            let n1 = float_to_e2m1(v1);
            packed[(base + i) / 2] = (n1 << 4) | n0;
        }
    }
    (packed, scales)
}

// Dequantize one row of NVFP4 back to BF16 for use in the reference matmul.
fn dequant_nvfp4_row(packed: &[u8], scales: &[u8], scale2: f32) -> Vec<f32> {
    let k = packed.len() * 2;
    let mut out = vec![0.0f32; k];
    for g in 0..scales.len() {
        let eff = fp8_e4m3_to_float(scales[g]) * scale2;
        for i in 0..GROUP_SIZE as usize {
            let pos = g * GROUP_SIZE as usize + i;
            let byte = packed[pos / 2];
            let nib = if pos & 1 == 0 { byte & 0xF } else { byte >> 4 };
            out[pos] = E2M1_LUT[nib as usize] * eff;
        }
    }
    out
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn bench_fused_silu_quant(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let silu_kernel = gpu::get_kernel(reg, &SILU_MUL_FN, "moe_silu_mul", "moe_silu_mul");
    let quant_kernel = gpu::get_kernel(reg, &QUANT_FN, "quantize_nvfp4", "quantize_bf16_to_nvfp4");
    let fused_kernel = gpu::get_kernel(
        reg,
        &FUSED_SILU_QUANT_FN,
        "quantize_nvfp4",
        "quantize_silu_mul_bf16_to_nvfp4",
    );

    let (m, k, _) = bench_shape();
    assert!(m > 0, "ATLAS_BENCH_M must be positive");
    assert!(
        k >= 16 && k.is_multiple_of(16),
        "ATLAS_BENCH_K must be at least 16 and a multiple of 16"
    );
    let total_elements = m.checked_mul(k).expect("fused SiLU element count overflow");
    let scale2 = std::env::var("ATLAS_BENCH_SCALE2")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.001);
    assert!(
        scale2.is_finite() && scale2 > 0.0,
        "ATLAS_BENCH_SCALE2 must be finite and positive"
    );

    let mut host_gate = random_bf16(total_elements as usize, 0x51A051A0);
    let mut host_up = random_bf16(total_elements as usize, 0xA15EA15E);
    let edge_values = [-16.0, -8.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 8.0, 16.0];
    for (index, &value) in edge_values.iter().enumerate() {
        host_gate[index] = bf16_from_f32(value);
        host_up[index] = bf16_from_f32(edge_values[edge_values.len() - 1 - index]);
        let tail = host_gate.len() - 1 - index;
        host_gate[tail] = bf16_from_f32(value * 0.5);
        host_up[tail] = bf16_from_f32(edge_values[edge_values.len() - 1 - index] * 0.5);
    }

    let bf16_bytes = total_elements as usize * std::mem::size_of::<u16>();
    let packed_bytes = total_elements as usize / 2;
    let scale_bytes = total_elements as usize / GROUP_SIZE as usize;
    let (gate_raw, gate) = guarded_output_alloc(stream, bf16_bytes);
    let (up_raw, up) = guarded_output_alloc(stream, bf16_bytes);
    let (silu_raw, silu) = guarded_output_alloc(stream, bf16_bytes);
    let (separate_packed_raw, separate_packed) = guarded_output_alloc(stream, packed_bytes);
    let (separate_scale_raw, separate_scale) = guarded_output_alloc(stream, scale_bytes);
    let (fused_packed_raw, fused_packed) = guarded_output_alloc(stream, packed_bytes);
    let (fused_scale_raw, fused_scale) = guarded_output_alloc(stream, scale_bytes);
    h2d(gate, &host_gate);
    h2d(up, &host_up);
    h2d(silu, &vec![0x3Cu8; bf16_bytes]);
    h2d(separate_packed, &vec![0x11u8; packed_bytes]);
    h2d(separate_scale, &vec![0x22u8; scale_bytes]);
    h2d(fused_packed, &vec![0xD4u8; packed_bytes]);
    h2d(fused_scale, &vec![0xE8u8; scale_bytes]);
    gpu::gpu_sync(stream).expect("finish fused SiLU input upload");

    let assert_inputs_unchanged = |phase: &str| {
        let mut gate_after = vec![0u16; host_gate.len()];
        let mut up_after = vec![0u16; host_up.len()];
        d2h(&mut gate_after, gate);
        d2h(&mut up_after, up);
        assert_eq!(gate_after, host_gate, "{phase} mutated gate input");
        assert_eq!(up_after, host_up, "{phase} mutated up input");
        assert_output_canaries(gate_raw, bf16_bytes, "fused SiLU gate input");
        assert_output_canaries(up_raw, bf16_bytes, "fused SiLU up input");
    };

    let silu_grid = (total_elements.div_ceil(256), 1, 1);
    let quant_grid = (m, 1, 1);
    let launch_separate = || {
        let mut silu_params: Vec<*mut c_void> = vec![
            &gate as *const u64 as *mut c_void,
            &up as *const u64 as *mut c_void,
            &silu as *const u64 as *mut c_void,
            &total_elements as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                silu_kernel,
                silu_grid,
                (256, 1, 1),
                0,
                stream,
                &mut silu_params,
            )
            .unwrap();
        }
        let mut quant_params: Vec<*mut c_void> = vec![
            &silu as *const u64 as *mut c_void,
            &separate_packed as *const u64 as *mut c_void,
            &separate_scale as *const u64 as *mut c_void,
            &scale2 as *const f32 as *mut c_void,
            &m as *const u32 as *mut c_void,
            &k as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                quant_kernel,
                quant_grid,
                (256, 1, 1),
                0,
                stream,
                &mut quant_params,
            )
            .unwrap();
        }
    };
    let launch_fused = || {
        let mut params: Vec<*mut c_void> = vec![
            &gate as *const u64 as *mut c_void,
            &up as *const u64 as *mut c_void,
            &fused_packed as *const u64 as *mut c_void,
            &fused_scale as *const u64 as *mut c_void,
            &scale2 as *const f32 as *mut c_void,
            &m as *const u32 as *mut c_void,
            &k as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                fused_kernel,
                quant_grid,
                (256, 1, 1),
                0,
                stream,
                &mut params,
            )
            .unwrap();
        }
    };

    launch_separate();
    gpu::gpu_sync(stream).expect("finish separate SiLU quantization arm");
    assert_inputs_unchanged("separate SiLU quantization arm");
    assert_output_canaries(silu_raw, bf16_bytes, "separate SiLU output");
    assert_output_canaries(separate_packed_raw, packed_bytes, "separate packed output");
    assert_output_canaries(separate_scale_raw, scale_bytes, "separate scale output");

    launch_fused();
    gpu::gpu_sync(stream).expect("finish fused SiLU quantization arm");
    assert_inputs_unchanged("fused SiLU quantization arm");
    assert_output_canaries(fused_packed_raw, packed_bytes, "fused packed output");
    assert_output_canaries(fused_scale_raw, scale_bytes, "fused scale output");

    let mut separate_packed_host = vec![0u8; packed_bytes];
    let mut separate_scale_host = vec![0u8; scale_bytes];
    let mut fused_packed_host = vec![0u8; packed_bytes];
    let mut fused_scale_host = vec![0u8; scale_bytes];
    d2h(&mut separate_packed_host, separate_packed);
    d2h(&mut separate_scale_host, separate_scale);
    d2h(&mut fused_packed_host, fused_packed);
    d2h(&mut fused_scale_host, fused_scale);
    assert_bytes_identical(
        "separate packed",
        &separate_packed_host,
        "fused packed",
        &fused_packed_host,
    );
    assert_bytes_identical(
        "separate scale",
        &separate_scale_host,
        "fused scale",
        &fused_scale_host,
    );
    eprintln!(
        "[fused_silu_nvfp4 validation] M={m} K={k} scale2={scale2:.9} bytewise PASS; packed_fingerprint=fnv1a64:{:016x}; scale_fingerprint=fnv1a64:{:016x}",
        fnv1a_bytes(&fused_packed_host),
        fnv1a_bytes(&fused_scale_host)
    );

    let mut group = c.benchmark_group("fused_silu_nvfp4");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("E_separate_silu_plus_quantize", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 10, iters as usize, &launch_separate);
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });
    group.bench_function("F_fused_silu_quantize", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 10, iters as usize, &launch_fused);
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });
    group.finish();

    gpu::gpu_sync(stream).expect("finish fused SiLU timing launches");
    for (raw, bytes, label) in [
        (gate_raw, bf16_bytes, "fused SiLU gate input"),
        (up_raw, bf16_bytes, "fused SiLU up input"),
        (silu_raw, bf16_bytes, "separate SiLU output"),
        (separate_packed_raw, packed_bytes, "separate packed output"),
        (separate_scale_raw, scale_bytes, "separate scale output"),
        (fused_packed_raw, packed_bytes, "fused packed output"),
        (fused_scale_raw, scale_bytes, "fused scale output"),
    ] {
        assert_output_canaries(raw, bytes, label);
    }
    assert_inputs_unchanged("timed fused/separate launches");
    eprintln!("[fused_silu_nvfp4 validation] canaries=PASS inputs_immutable=PASS");

    for ptr in [
        gate_raw,
        up_raw,
        silu_raw,
        separate_packed_raw,
        separate_scale_raw,
        fused_packed_raw,
        fused_scale_raw,
    ] {
        gpu::gpu_free(ptr);
    }
}

fn bench_nvfp4_gemm(c: &mut Criterion) {
    if std::env::var("ATLAS_BENCH_FUSED_SILU_ONLY").as_deref() == Ok("1") {
        bench_fused_silu_quant(c);
        return;
    }
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();

    let w4a16_kernel = gpu::get_kernel(reg, &W4A16_M128_FN, "w4a16", "w4a16_gemm_t_m128");
    let nvfp4_kernel = gpu::get_kernel(
        reg,
        &NVFP4_GEMM_FN,
        "nvfp4_cutlass",
        "nvfp4_nvfp4_gemm_t_m64",
    );
    let nvfp4_kmajor_m128_kernel = gpu::get_kernel(
        reg,
        &NVFP4_KMAJOR_M128_FN,
        "nvfp4_cutlass",
        "nvfp4_nvfp4_gemm_kmajor_m128",
    );
    let nvfp4_kmajor_m256_kernel = gpu::get_kernel(
        reg,
        &NVFP4_KMAJOR_M256_FN,
        "nvfp4_cutlass",
        "nvfp4_nvfp4_gemm_kmajor_m256",
    );
    let nvfp4_sk_kernel = gpu::get_kernel(
        reg,
        &NVFP4_GEMM_SK_FN,
        "nvfp4_cutlass",
        "nvfp4_nvfp4_gemm_t_m64_splitk",
    );
    let nvfp4_sk_reduce = gpu::get_kernel(
        reg,
        &NVFP4_SK_REDUCE_FN,
        "nvfp4_cutlass",
        "nvfp4_splitk_reduce",
    );
    let absmax_kernel = gpu::get_kernel(reg, &ABSMAX_FN, "quantize_nvfp4", "nvfp4_global_absmax");
    let quant_kernel = gpu::get_kernel(reg, &QUANT_FN, "quantize_nvfp4", "quantize_bf16_to_nvfp4");

    let (mm, kk, nn) = bench_shape();
    // Bind to local aliases used in unsafe param pointers; CUDA kernels
    // expect u32 by value, and we need the address.
    let m: u32 = mm;
    let k: u32 = kk;
    let n: u32 = nn;
    assert!(m > 0, "ATLAS_BENCH_M must be positive");
    assert!(
        n >= 128 && n.is_multiple_of(128),
        "ATLAS_BENCH_N must be a positive multiple of 128"
    );
    assert!(
        k >= 64 && k.is_multiple_of(64),
        "ATLAS_BENCH_K must be a positive multiple of 64"
    );
    let skip_ref = (m as u64) * (k as u64) * (n as u64) > 1_000_000_000;

    eprintln!(
        "[nvfp4_gemm] shape M={m} K={k} N={n}  skip_cpu_ref={skip_ref} \
         (override via ATLAS_BENCH_M / ATLAS_BENCH_K / ATLAS_BENCH_N)"
    );

    // ──────────────────────────────────────────────────────────────────
    // Host-side: build canonical inputs and reference output.
    // ──────────────────────────────────────────────────────────────────
    // BF16 activation A[M, K] in [-1, 1]
    let host_a = random_bf16(m as usize * k as usize, 0xA1A2A3A4);
    // BF16 reference weight B_bf16[N, K] in [-1, 1]
    let host_b_bf16 = random_bf16(n as usize * k as usize, 0xB1B2B3B4);

    // Per-tensor weight scale = global_max / (6.0 * 448.0)
    let b_absmax: f32 = host_b_bf16
        .iter()
        .map(|&b| bf16_to_f32(b).abs())
        .fold(0.0f32, f32::max);
    let scale2_w = if b_absmax > 0.0 {
        b_absmax / (6.0 * 448.0)
    } else {
        1.0
    };

    // Quantize each weight row to NVFP4 (per-row e4m3 scales, single scale2_w).
    let mut weight_packed_per_row: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
    let mut weight_scale_per_row: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
    for r in 0..n as usize {
        let row = &host_b_bf16[r * k as usize..(r + 1) * k as usize];
        let (p, s) = quant_bf16_row_to_nvfp4(row, scale2_w);
        weight_packed_per_row.push(p);
        weight_scale_per_row.push(s);
    }

    let c_ref: Vec<f32> = if !skip_ref {
        // Build B_dequant[N, K] BF16 (lossy round-trip — matches what the
        // kernel sees during MMA).
        let mut b_dequant_bf16 = vec![0u16; n as usize * k as usize];
        for r in 0..n as usize {
            let row_f = dequant_nvfp4_row(
                &weight_packed_per_row[r],
                &weight_scale_per_row[r],
                scale2_w,
            );
            for kk_i in 0..k as usize {
                b_dequant_bf16[r * k as usize + kk_i] = bf16_from_f32(row_f[kk_i]);
            }
        }
        eprintln!("[nvfp4_gemm validation] computing CPU reference C_ref ...");
        let t0 = std::time::Instant::now();
        let mut c_ref = vec![0.0f32; m as usize * n as usize];
        for mi in 0..m as usize {
            for ni in 0..n as usize {
                let mut acc = 0.0f32;
                for kk_i in 0..k as usize {
                    acc += bf16_to_f32(host_a[mi * k as usize + kk_i])
                        * bf16_to_f32(b_dequant_bf16[ni * k as usize + kk_i]);
                }
                c_ref[mi * n as usize + ni] = acc;
            }
        }
        eprintln!(
            "[nvfp4_gemm validation] CPU reference done in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        c_ref
    } else {
        eprintln!("[nvfp4_gemm validation] CPU reference SKIPPED (shape too big)");
        Vec::new()
    };

    // ──────────────────────────────────────────────────────────────────
    // Pack weights into the two GPU layouts expected by Path A vs Path B.
    // ──────────────────────────────────────────────────────────────────
    // Path B (HF layout): B_packed[N][K/2], B_scale[N][K/16].
    let mut b_packed_hf = vec![0u8; n as usize * k as usize / 2];
    let mut b_scale_hf = vec![0u8; n as usize * k as usize / GROUP_SIZE as usize];
    for r in 0..n as usize {
        let dst_p = &mut b_packed_hf[r * k as usize / 2..(r + 1) * k as usize / 2];
        dst_p.copy_from_slice(&weight_packed_per_row[r]);
        let dst_s = &mut b_scale_hf
            [r * k as usize / GROUP_SIZE as usize..(r + 1) * k as usize / GROUP_SIZE as usize];
        dst_s.copy_from_slice(&weight_scale_per_row[r]);
    }
    // Path A (K-major transposed): B_packed_t[K/2][N], B_scale_t[K/16][N].
    let mut b_packed_t = vec![0u8; n as usize * k as usize / 2];
    let mut b_scale_t = vec![0u8; n as usize * k as usize / GROUP_SIZE as usize];
    for r in 0..n as usize {
        let row_p = &weight_packed_per_row[r];
        for c in 0..k as usize / 2 {
            b_packed_t[c * n as usize + r] = row_p[c];
        }
        let row_s = &weight_scale_per_row[r];
        for g in 0..k as usize / GROUP_SIZE as usize {
            b_scale_t[g * n as usize + r] = row_s[g];
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Allocate GPU buffers and upload.
    // ──────────────────────────────────────────────────────────────────
    let a_bf16_bytes = m as usize * k as usize * 2;
    let a_packed_bytes = m as usize * k as usize / 2;
    let a_scale_bytes = m as usize * k as usize / GROUP_SIZE as usize;
    let b_bytes = n as usize * k as usize / 2;
    let s_bytes = n as usize * k as usize / GROUP_SIZE as usize;
    let c_bytes = m as usize * n as usize * 2;

    let g_a_bf16 = gpu::gpu_alloc_zeroed(stream, a_bf16_bytes).unwrap();
    let g_a_packed = gpu::gpu_alloc_zeroed(stream, a_packed_bytes).unwrap();
    let g_a_scale = gpu::gpu_alloc_zeroed(stream, a_scale_bytes).unwrap();
    let g_a_max = gpu::gpu_alloc_zeroed(stream, 4).unwrap();
    let g_b_hf = gpu::gpu_alloc_zeroed(stream, b_bytes).unwrap();
    let g_s_hf = gpu::gpu_alloc_zeroed(stream, s_bytes).unwrap();
    let g_b_t = gpu::gpu_alloc_zeroed(stream, b_bytes).unwrap();
    let g_s_t = gpu::gpu_alloc_zeroed(stream, s_bytes).unwrap();
    let g_c_a = gpu::gpu_alloc_zeroed(stream, c_bytes).unwrap();
    let (g_c_b_raw, g_c_b) = guarded_output_alloc(stream, c_bytes);
    let (g_c_km128_raw, g_c_km128) = guarded_output_alloc(stream, c_bytes);
    let (g_c_km256_raw, g_c_km256) = guarded_output_alloc(stream, c_bytes);
    // Split-K scratch: F32 [K_SPLITS, M, N]. Allocate at max K_SPLITS=8.
    const MAX_K_SPLITS: u32 = 8;
    let scratch_bytes = MAX_K_SPLITS as usize * m as usize * n as usize * 4;
    let g_scratch = gpu::gpu_alloc_zeroed(stream, scratch_bytes).unwrap();
    let g_c_sk = gpu::gpu_alloc_zeroed(stream, c_bytes).unwrap();
    gpu::gpu_sync(stream).unwrap();

    h2d(g_a_bf16, &host_a);
    h2d(g_b_hf, &b_packed_hf);
    h2d(g_s_hf, &b_scale_hf);
    h2d(g_b_t, &b_packed_t);
    h2d(g_s_t, &b_scale_t);
    gpu::gpu_sync(stream).unwrap();

    // ──────────────────────────────────────────────────────────────────
    // Run Path A once for correctness
    // w4a16_gemm_t_m128 (BF16 A × NVFP4 B_t)
    // ──────────────────────────────────────────────────────────────────
    let grid_a = (n.div_ceil(128), m.div_ceil(128), 1);
    {
        let mut params: Vec<*mut c_void> = vec![
            &g_a_bf16 as *const u64 as *mut c_void,
            &g_b_t as *const u64 as *mut c_void,
            &g_s_t as *const u64 as *mut c_void,
            &scale2_w as *const f32 as *mut c_void,
            &g_c_a as *const u64 as *mut c_void,
            &m as *const u32 as *mut c_void,
            &n as *const u32 as *mut c_void,
            &k as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                w4a16_kernel,
                grid_a,
                (128, 1, 1),
                0,
                stream,
                &mut params,
            )
            .unwrap();
        }
    }
    gpu::gpu_sync(stream).unwrap();

    // ──────────────────────────────────────────────────────────────────
    // Run Path B once for correctness
    //   (1) absmax → scale2_a
    //   (2) quantize_bf16_to_nvfp4 → A_packed, A_scale
    //   (3) nvfp4_nvfp4_gemm_t_m64
    // ──────────────────────────────────────────────────────────────────
    let total_elems = m * k;
    let abs_grid = ((total_elems / 256).clamp(1, 1024), 1, 1);
    h2d(g_a_max, &[0u8; 4]);
    {
        let mut params: Vec<*mut c_void> = vec![
            &g_a_bf16 as *const u64 as *mut c_void,
            &g_a_max as *const u64 as *mut c_void,
            &total_elems as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                absmax_kernel,
                abs_grid,
                (256, 1, 1),
                0,
                stream,
                &mut params,
            )
            .unwrap();
        }
    }
    gpu::gpu_sync(stream).unwrap();
    let mut amb = [0u8; 4];
    d2h(&mut amb, g_a_max);
    let a_absmax = f32::from_le_bytes(amb);
    let scale2_a = if a_absmax > 0.0 {
        a_absmax / (6.0 * 448.0)
    } else {
        1.0
    };
    let scale2_ab = scale2_a * scale2_w;
    {
        let mut params: Vec<*mut c_void> = vec![
            &g_a_bf16 as *const u64 as *mut c_void,
            &g_a_packed as *const u64 as *mut c_void,
            &g_a_scale as *const u64 as *mut c_void,
            &scale2_a as *const f32 as *mut c_void,
            &m as *const u32 as *mut c_void,
            &k as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                quant_kernel,
                (m, 1, 1),
                (256, 1, 1),
                0,
                stream,
                &mut params,
            )
            .unwrap();
        }
    }
    gpu::gpu_sync(stream).unwrap();
    let grid_b = (n.div_ceil(128), m.div_ceil(64), 1);
    {
        let mut params: Vec<*mut c_void> = vec![
            &g_a_packed as *const u64 as *mut c_void,
            &g_a_scale as *const u64 as *mut c_void,
            &g_b_hf as *const u64 as *mut c_void,
            &g_s_hf as *const u64 as *mut c_void,
            &scale2_ab as *const f32 as *mut c_void,
            &g_c_b as *const u64 as *mut c_void,
            &m as *const u32 as *mut c_void,
            &n as *const u32 as *mut c_void,
            &k as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                nvfp4_kernel,
                grid_b,
                (128, 1, 1),
                0,
                stream,
                &mut params,
            )
            .unwrap();
        }
    }
    gpu::gpu_sync(stream).unwrap();

    // ──────────────────────────────────────────────────────────────────
    // Run the two K-major W4A4 kernels on the same already-quantized A bytes
    // and the same weights represented in the K-major transform. All three
    // W4A4 paths promise bitwise BF16 identity: every warp retains the same
    // m16n128 fragment and K64 accumulation order; only weight staging and
    // row sharing differ.
    // ──────────────────────────────────────────────────────────────────
    let grid_km128 = (n / 128, m.div_ceil(128), 1);
    {
        let mut params: Vec<*mut c_void> = vec![
            &g_a_packed as *const u64 as *mut c_void,
            &g_a_scale as *const u64 as *mut c_void,
            &g_b_t as *const u64 as *mut c_void,
            &g_s_t as *const u64 as *mut c_void,
            &scale2_ab as *const f32 as *mut c_void,
            &g_c_km128 as *const u64 as *mut c_void,
            &m as *const u32 as *mut c_void,
            &n as *const u32 as *mut c_void,
            &k as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                nvfp4_kmajor_m128_kernel,
                grid_km128,
                (256, 1, 1),
                0,
                stream,
                &mut params,
            )
            .unwrap();
        }
    }
    let grid_km256 = (n / 128, m.div_ceil(256), 1);
    {
        let mut params: Vec<*mut c_void> = vec![
            &g_a_packed as *const u64 as *mut c_void,
            &g_a_scale as *const u64 as *mut c_void,
            &g_b_t as *const u64 as *mut c_void,
            &g_s_t as *const u64 as *mut c_void,
            &scale2_ab as *const f32 as *mut c_void,
            &g_c_km256 as *const u64 as *mut c_void,
            &m as *const u32 as *mut c_void,
            &n as *const u32 as *mut c_void,
            &k as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                nvfp4_kmajor_m256_kernel,
                grid_km256,
                (512, 1, 1),
                0,
                stream,
                &mut params,
            )
            .unwrap();
        }
    }
    gpu::gpu_sync(stream).unwrap();

    // ──────────────────────────────────────────────────────────────────
    // Run Path B-SplitK once for correctness (K_SPLITS=2)
    // ──────────────────────────────────────────────────────────────────
    let k_splits: u32 = std::env::var("ATLAS_BENCH_K_SPLITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2u32);
    // SplitK kernel zero-initializes only its outputs (the partials it writes).
    // To be safe and deterministic with predicated cp.async fallthrough,
    // zero the scratch slice we use.
    let scratch_used_bytes = k_splits as usize * m as usize * n as usize * 4;
    let _ = scratch_used_bytes; // kernel writes all valid (r,c); no zero needed
    let grid_b_sk = (n.div_ceil(128), m.div_ceil(64), k_splits);
    let scale_zero: f32 = 0.0_f32;
    let _ = scale_zero;
    // Phase 1: partials (no scale2_ab — applied in reduce).
    {
        let mut params: Vec<*mut c_void> = vec![
            &g_a_packed as *const u64 as *mut c_void,
            &g_a_scale as *const u64 as *mut c_void,
            &g_b_hf as *const u64 as *mut c_void,
            &g_s_hf as *const u64 as *mut c_void,
            &g_scratch as *const u64 as *mut c_void,
            &m as *const u32 as *mut c_void,
            &n as *const u32 as *mut c_void,
            &k as *const u32 as *mut c_void,
            &k_splits as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                nvfp4_sk_kernel,
                grid_b_sk,
                (128, 1, 1),
                0,
                stream,
                &mut params,
            )
            .unwrap();
        }
    }
    // Phase 2: reduce + scale + BF16 cast.
    let red_grid = (n.div_ceil(256), m, 1);
    {
        let mut params: Vec<*mut c_void> = vec![
            &g_scratch as *const u64 as *mut c_void,
            &g_c_sk as *const u64 as *mut c_void,
            &scale2_ab as *const f32 as *mut c_void,
            &m as *const u32 as *mut c_void,
            &n as *const u32 as *mut c_void,
            &k_splits as *const u32 as *mut c_void,
        ];
        unsafe {
            gpu::launch(
                reg,
                nvfp4_sk_reduce,
                red_grid,
                (256, 1, 1),
                0,
                stream,
                &mut params,
            )
            .unwrap();
        }
    }
    gpu::gpu_sync(stream).unwrap();

    // ──────────────────────────────────────────────────────────────────
    // Compare both kernel outputs to the CPU reference.
    // ──────────────────────────────────────────────────────────────────
    let mut out_a = vec![0u16; m as usize * n as usize];
    let mut out_b = vec![0u16; m as usize * n as usize];
    let mut out_b_sk = vec![0u16; m as usize * n as usize];
    let mut out_km128 = vec![0u16; m as usize * n as usize];
    let mut out_km256 = vec![0u16; m as usize * n as usize];
    d2h(&mut out_a, g_c_a);
    d2h(&mut out_b, g_c_b);
    d2h(&mut out_b_sk, g_c_sk);
    d2h(&mut out_km128, g_c_km128);
    d2h(&mut out_km256, g_c_km256);
    assert_output_canaries(g_c_b_raw, c_bytes, "row-major M64");
    assert_output_canaries(g_c_km128_raw, c_bytes, "K-major M128");
    assert_output_canaries(g_c_km256_raw, c_bytes, "K-major M256");
    assert_bf16_identical("row-major M64", &out_b, "K-major M128", &out_km128);
    assert_bf16_identical("K-major M128", &out_km128, "K-major M256", &out_km256);
    let w4a4_fingerprint = fnv1a_bf16(&out_b);
    eprintln!(
        "[nvfp4_gemm validation] row-major M64/K-major M128/K-major M256 bitwise PASS; output_fingerprint=fnv1a64:{w4a4_fingerprint:016x}; canaries=PASS"
    );
    let f_a: Vec<f32> = out_a.iter().map(|&b| bf16_to_f32(b)).collect();
    let f_b: Vec<f32> = out_b.iter().map(|&b| bf16_to_f32(b)).collect();
    let f_b_sk: Vec<f32> = out_b_sk.iter().map(|&b| bf16_to_f32(b)).collect();
    let cos_ab = cosine_sim(&f_a, &f_b);
    let cos_b_sk_vs_b = cosine_sim(&f_b, &f_b_sk);

    if !skip_ref {
        let cos_a = cosine_sim(&f_a, &c_ref);
        let cos_b = cosine_sim(&f_b, &c_ref);
        let ref_norm: f32 = (c_ref.iter().map(|x| x * x).sum::<f32>() / c_ref.len() as f32).sqrt();
        let rel_err = |out: &[f32]| -> f32 {
            let mut sum = 0.0f64;
            for i in 0..out.len() {
                let d = out[i] - c_ref[i];
                sum += (d * d) as f64;
            }
            ((sum / out.len() as f64).sqrt() as f32) / ref_norm
        };
        let rel_a = rel_err(&f_a);
        let rel_b = rel_err(&f_b);
        eprintln!(
            "[nvfp4_gemm validation] M={m} K={k} N={n}  a_absmax={a_absmax:.4}  \
             scale2_w={scale2_w:.5} scale2_a={scale2_a:.5}"
        );
        eprintln!(
            "[nvfp4_gemm validation]   Path A (w4a16_t_m128):    cos_vs_ref={cos_a:.4}  rel_rms_err={rel_a:.4}"
        );
        eprintln!(
            "[nvfp4_gemm validation]   Path B (nvfp4_nvfp4_t_m64): cos_vs_ref={cos_b:.4}  rel_rms_err={rel_b:.4}"
        );
        let cos_b_sk = cosine_sim(&f_b_sk, &c_ref);
        let rel_b_sk = rel_err(&f_b_sk);
        eprintln!(
            "[nvfp4_gemm validation]   Path B-splitK (K_SPLITS={k_splits}): cos_vs_ref={cos_b_sk:.4}  rel_rms_err={rel_b_sk:.4}"
        );
        eprintln!(
            "[nvfp4_gemm validation]   cos(A, B) = {cos_ab:.4}   cos(B, B-splitK) = {cos_b_sk_vs_b:.4}"
        );
        let pass_a = cos_a >= 0.99;
        let pass_b = cos_b >= 0.99;
        let pass_b_sk = cos_b_sk >= 0.99;
        eprintln!(
            "[nvfp4_gemm validation] verdict: Path A {} ; Path B {} ; Path B-splitK {}",
            if pass_a { "PASS" } else { "FAIL" },
            if pass_b { "PASS" } else { "FAIL" },
            if pass_b_sk { "PASS" } else { "FAIL" }
        );
    } else {
        // Without a CPU ref the only sanity check we can do is "outputs
        // are finite and have plausible magnitudes" — useful for catching
        // NaN/Inf bugs at production shapes.
        let max_abs_a = f_a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let max_abs_b = f_b.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let nan_a = f_a.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
        let nan_b = f_b.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
        eprintln!(
            "[nvfp4_gemm validation] M={m} K={k} N={n}  a_absmax={a_absmax:.4}  \
             scale2_w={scale2_w:.5} scale2_a={scale2_a:.5}"
        );
        eprintln!("[nvfp4_gemm validation]   Path A max_abs={max_abs_a:.2} nan/inf={nan_a}");
        eprintln!("[nvfp4_gemm validation]   Path B max_abs={max_abs_b:.2} nan/inf={nan_b}");
        let max_abs_b_sk = f_b_sk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let nan_b_sk = f_b_sk
            .iter()
            .filter(|v| v.is_nan() || v.is_infinite())
            .count();
        eprintln!(
            "[nvfp4_gemm validation]   Path B-splitK (K_SPLITS={k_splits}) max_abs={max_abs_b_sk:.2} nan/inf={nan_b_sk}"
        );
        eprintln!(
            "[nvfp4_gemm validation]   cos(A, B) = {cos_ab:.4}   cos(B, B-splitK) = {cos_b_sk_vs_b:.4}"
        );
    }

    // ──────────────────────────────────────────────────────────────────
    // Timing
    // ──────────────────────────────────────────────────────────────────
    let mut group = c.benchmark_group("nvfp4_ffn");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("A_w4a16_t_m128", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &g_a_bf16 as *const u64 as *mut c_void,
                    &g_b_t as *const u64 as *mut c_void,
                    &g_s_t as *const u64 as *mut c_void,
                    &scale2_w as *const f32 as *mut c_void,
                    &g_c_a as *const u64 as *mut c_void,
                    &m as *const u32 as *mut c_void,
                    &n as *const u32 as *mut c_void,
                    &k as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        w4a16_kernel,
                        grid_a,
                        (128, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    group.bench_function("B_nvfp4_nvfp4_t_m64_mma_only", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &g_a_packed as *const u64 as *mut c_void,
                    &g_a_scale as *const u64 as *mut c_void,
                    &g_b_hf as *const u64 as *mut c_void,
                    &g_s_hf as *const u64 as *mut c_void,
                    &scale2_ab as *const f32 as *mut c_void,
                    &g_c_b as *const u64 as *mut c_void,
                    &m as *const u32 as *mut c_void,
                    &n as *const u32 as *mut c_void,
                    &k as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        nvfp4_kernel,
                        grid_b,
                        (128, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    group.bench_function("C_nvfp4_kmajor_m128_mma_only", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &g_a_packed as *const u64 as *mut c_void,
                    &g_a_scale as *const u64 as *mut c_void,
                    &g_b_t as *const u64 as *mut c_void,
                    &g_s_t as *const u64 as *mut c_void,
                    &scale2_ab as *const f32 as *mut c_void,
                    &g_c_km128 as *const u64 as *mut c_void,
                    &m as *const u32 as *mut c_void,
                    &n as *const u32 as *mut c_void,
                    &k as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        nvfp4_kmajor_m128_kernel,
                        grid_km128,
                        (256, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    group.bench_function("D_nvfp4_kmajor_m256_mma_only", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &g_a_packed as *const u64 as *mut c_void,
                    &g_a_scale as *const u64 as *mut c_void,
                    &g_b_t as *const u64 as *mut c_void,
                    &g_s_t as *const u64 as *mut c_void,
                    &scale2_ab as *const f32 as *mut c_void,
                    &g_c_km256 as *const u64 as *mut c_void,
                    &m as *const u32 as *mut c_void,
                    &n as *const u32 as *mut c_void,
                    &k as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        nvfp4_kmajor_m256_kernel,
                        grid_km256,
                        (512, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    group.bench_function("B_splitk_mma_plus_reduce", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut p1: Vec<*mut c_void> = vec![
                    &g_a_packed as *const u64 as *mut c_void,
                    &g_a_scale as *const u64 as *mut c_void,
                    &g_b_hf as *const u64 as *mut c_void,
                    &g_s_hf as *const u64 as *mut c_void,
                    &g_scratch as *const u64 as *mut c_void,
                    &m as *const u32 as *mut c_void,
                    &n as *const u32 as *mut c_void,
                    &k as *const u32 as *mut c_void,
                    &k_splits as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        nvfp4_sk_kernel,
                        grid_b_sk,
                        (128, 1, 1),
                        0,
                        stream,
                        &mut p1,
                    )
                    .unwrap();
                }
                let mut p2: Vec<*mut c_void> = vec![
                    &g_scratch as *const u64 as *mut c_void,
                    &g_c_sk as *const u64 as *mut c_void,
                    &scale2_ab as *const f32 as *mut c_void,
                    &m as *const u32 as *mut c_void,
                    &n as *const u32 as *mut c_void,
                    &k_splits as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        nvfp4_sk_reduce,
                        red_grid,
                        (256, 1, 1),
                        0,
                        stream,
                        &mut p2,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    group.bench_function("B_full_with_prequant_d2h_sync", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 10, iters as usize, || {
                h2d(g_a_max, &[0u8; 4]);
                let mut p1: Vec<*mut c_void> = vec![
                    &g_a_bf16 as *const u64 as *mut c_void,
                    &g_a_max as *const u64 as *mut c_void,
                    &total_elems as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        absmax_kernel,
                        abs_grid,
                        (256, 1, 1),
                        0,
                        stream,
                        &mut p1,
                    )
                    .unwrap();
                }
                gpu::gpu_sync(stream).unwrap();
                let mut amb2 = [0u8; 4];
                d2h(&mut amb2, g_a_max);
                let gm = f32::from_le_bytes(amb2);
                let sa = if gm > 0.0 { gm / (6.0 * 448.0) } else { 1.0 };
                let sab = sa * scale2_w;
                let mut p2: Vec<*mut c_void> = vec![
                    &g_a_bf16 as *const u64 as *mut c_void,
                    &g_a_packed as *const u64 as *mut c_void,
                    &g_a_scale as *const u64 as *mut c_void,
                    &sa as *const f32 as *mut c_void,
                    &m as *const u32 as *mut c_void,
                    &k as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        quant_kernel,
                        (m, 1, 1),
                        (256, 1, 1),
                        0,
                        stream,
                        &mut p2,
                    )
                    .unwrap();
                }
                let mut p3: Vec<*mut c_void> = vec![
                    &g_a_packed as *const u64 as *mut c_void,
                    &g_a_scale as *const u64 as *mut c_void,
                    &g_b_hf as *const u64 as *mut c_void,
                    &g_s_hf as *const u64 as *mut c_void,
                    &sab as *const f32 as *mut c_void,
                    &g_c_b as *const u64 as *mut c_void,
                    &m as *const u32 as *mut c_void,
                    &n as *const u32 as *mut c_void,
                    &k as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(reg, nvfp4_kernel, grid_b, (128, 1, 1), 0, stream, &mut p3)
                        .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    group.finish();

    gpu::gpu_free(g_a_bf16);
    gpu::gpu_free(g_a_packed);
    gpu::gpu_free(g_a_scale);
    gpu::gpu_free(g_a_max);
    gpu::gpu_free(g_b_hf);
    gpu::gpu_free(g_s_hf);
    gpu::gpu_free(g_b_t);
    gpu::gpu_free(g_s_t);
    gpu::gpu_free(g_c_a);
    gpu::gpu_free(g_c_b_raw);
    gpu::gpu_free(g_c_km128_raw);
    gpu::gpu_free(g_c_km256_raw);
    gpu::gpu_free(g_scratch);
    gpu::gpu_free(g_c_sk);
}

criterion_group!(benches, bench_nvfp4_gemm);
criterion_main!(benches);
