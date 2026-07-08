// SPDX-License-Identifier: AGPL-3.0-only

//! W3A16 (3-bit weight) kernel dispatch + host-side format mirror.
//!
//! Mixed-precision byte-reduction lane: selected FFN layers drop from NVFP4
//! (4-bit) to a 3-bit format, cutting packed weight bytes 25% on the
//! weight-bandwidth-bound single-stream decode. Gated by
//! `ATLAS_FFN_W3_LAYERS` + a repacked sidecar file (see
//! `weight_map::w3_sidecar` and `local/tools/repack_w3.py`). Quality is
//! ABBA-eval-gated, NOT md5-gated: W3 weights differ from W4 by
//! construction, so the token stream on gated layers cannot be
//! byte-identical — but the default path (no gate / no sidecar) is
//! untouched and stays on the md5 constitution.
//!
//! W3 FORMAT (v1) — single source of truth shared by:
//!   * `local/tools/repack_w3.py` (offline repack + simulator)
//!   * `kernels/gb10/common/w3a16_gemv.cu` / `w3a16_gemm.cu` (GPU dequant)
//!   * the host mirror functions in this module (golden-vector tested)
//!
//!   LUT:      `W3_LUT[8] = {0, 1, 2, 4, -0, -1, -2, -4}`
//!             (e2m1-subset magnitudes, sign in bit 2 — empirically ~15%
//!             lower relMSE than the linear {0,1,2,3} codebook on
//!             AEON-Q36-27B FFN weights)
//!   Packing:  8 weights -> 3 bytes little-endian: `u24 = Σ code_i << 3i`
//!             GEMV layout `[N, 3K/8]`; GEMM (transposed) layout
//!             `[3K/8, N_pad64]` (row `3j+b` = byte-plane `b` of octet `j`)
//!   Scales:   unchanged NVFP4 scheme — FP8-E4M3 per 16 K-values
//!             (`[N, K/16]` / transposed `[K/16, N_pad64]`) + per-tensor
//!             f32 `scale2` (sidecar value = 1.5x the W4 scale2)
//!   Dequant:  `sv = f32(e4m3(scale_byte)) * scale2; w = W3_LUT[code] * sv`
//!             (two f32 multiplies in exactly this association)

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// The global 8-entry W3 dequant LUT. `code = (sign << 2) | mag`.
pub const W3_LUT: [f32; 8] = [0.0, 1.0, 2.0, 4.0, -0.0, -1.0, -2.0, -4.0];

/// Group size of the FP8-E4M3 block scales (same as NVFP4).
pub const W3_GROUP_SIZE: usize = 16;

/// Packed bytes per row for a K-column W3 tensor.
pub const fn w3_row_bytes(k: usize) -> usize {
    k / 8 * 3
}

// ── Kernel dispatch ─────────────────────────────────────────────────────────

/// W3A16 GEMV (M=1): `C[n] = Σ_k A[k] * dequant3(B[n, k])`.
///
/// Kernel: `w3a16_gemv(A, B_packed3, B_scale, scale2, C, N, K)` —
/// same grid/block geometry as `w4a16_gemv`.
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w3a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W3A16 dual GEMV: gate + up projections sharing one BF16 input.
///
/// blockIdx.z selects projection 0 vs 1 (mirrors `w4a16_gemv_dual`).
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w3a16_gemv_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    weight2: &QuantizedWeight,
    output2: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_ptr(weight2.weight)
        .arg_ptr(weight2.weight_scale)
        .arg_f32(weight2.weight_scale_2)
        .arg_ptr(output2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W3A16 GEMV with fused SiLU input: `silu(gate)*up` as activation, GEMV
/// against W3 down weights (mirrors `w4a16_gemv_silu_input`).
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w3a16_gemv_silu_input(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// W3A16 GEMM, M_TILE=32 / N_TILE=64 — 3-bit clone of
/// `w4a16_gemm_t_m32_n64` for the K=γ verify FFN. `weight` must hold the
/// TRANSPOSED W3 layout (`[3K/8, N_pad64]` packed + `[K/16, N_pad64]`
/// scales, built by `w3_transpose_host` at load).
/// Grid: (ceil(N/64), ceil(M/32), 1)  Block: (128, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w3a16_gemm_n64_m32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // ldb = 64-padded N (identical padding rule to transpose_for_gemm; all
    // FFN dims here are already multiples of 64 so ldb == n in practice).
    let ldb = n.div_ceil(64) * 64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 32), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(ldb)
        .launch(stream)
}

// ── Host-side format mirror (golden-vector tested) ─────────────────────────

/// Decode an FP8 E4M3 (fn variant: bias 7, no inf, max 448) byte to f32.
/// NaN bytes (0x7F / 0xFF) decode to f32 NaN.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xF;
    let m = (b & 0x7) as f32;
    if e == 0xF && (b & 0x7) == 0x7 {
        f32::NAN
    } else if e == 0 {
        s * (m / 8.0) * 2.0f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2.0f32.powi(e as i32 - 7)
    }
}

/// Encode f32 -> nearest finite E4M3 byte (round-to-nearest-even,
/// saturating at ±448). Matches CUDA `cvt.rn.satfinite.e4m3x2.f32` on
/// finite inputs — verified against torch's float8_e4m3fn cast in the
/// Python simulator (exhaustive decode + 200k-point encode agreement).
pub fn f32_to_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs().min(448.0);
    // All finite non-negative magnitudes in byte order 0x00..=0x7E are
    // monotonically increasing, so binary-search the nearest.
    let (mut lo, mut hi) = (0u8, 0x7Eu8);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if e4m3_to_f32(mid) <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    // lo = largest byte with value <= a; candidate above is lo+1.
    let below = e4m3_to_f32(lo);
    let byte = if lo == 0x7E {
        lo
    } else {
        let above = e4m3_to_f32(lo + 1);
        let (d_below, d_above) = (a - below, above - a);
        if d_below < d_above || (d_below == d_above && lo % 2 == 0) {
            lo
        } else {
            lo + 1
        }
    };
    byte | sign
}

/// Unpack a W3 row: `[3K/8]` packed bytes -> `k` 3-bit codes.
pub fn w3_unpack_codes(packed: &[u8], k: usize) -> Vec<u8> {
    assert_eq!(packed.len(), w3_row_bytes(k));
    let mut codes = Vec::with_capacity(k);
    for j in 0..k / 8 {
        let u24 = packed[3 * j] as u32
            | ((packed[3 * j + 1] as u32) << 8)
            | ((packed[3 * j + 2] as u32) << 16);
        for i in 0..8 {
            codes.push(((u24 >> (3 * i)) & 7) as u8);
        }
    }
    codes
}

/// Pack `k` 3-bit codes into `3K/8` bytes (inverse of `w3_unpack_codes`).
pub fn w3_pack_codes(codes: &[u8]) -> Vec<u8> {
    assert_eq!(codes.len() % 8, 0);
    let mut out = Vec::with_capacity(w3_row_bytes(codes.len()));
    for octet in codes.chunks_exact(8) {
        let mut u24 = 0u32;
        for (i, &c) in octet.iter().enumerate() {
            debug_assert!(c < 8);
            u24 |= (c as u32) << (3 * i);
        }
        out.extend_from_slice(&[(u24 & 0xFF) as u8, ((u24 >> 8) & 0xFF) as u8, (u24 >> 16) as u8]);
    }
    out
}

/// Host mirror of the GPU W3 dequant (f32 path — `w3a16_gemv*` contract):
/// `sv = e4m3(scale) * scale2; w = W3_LUT[code] * sv`, both f32 multiplies
/// in exactly this association. Bit-exact vs the kernels and
/// `repack_w3.py::dequant_w3`.
pub fn w3_dequant_row(packed: &[u8], scale_bytes: &[u8], scale2: f32, k: usize) -> Vec<f32> {
    assert_eq!(scale_bytes.len(), k / W3_GROUP_SIZE);
    let codes = w3_unpack_codes(packed, k);
    codes
        .iter()
        .enumerate()
        .map(|(kk, &c)| {
            let sv = e4m3_to_f32(scale_bytes[kk / W3_GROUP_SIZE]) * scale2;
            W3_LUT[c as usize] * sv
        })
        .collect()
}

/// Host mirror of the m32_n64 GEMM dequant: the f32 dequant above PLUS the
/// FP8-E4M3 round-trip (`cvt.rn.satfinite.e4m3x2`) the kernel applies
/// before the MMA.
pub fn w3_dequant_row_e4m3(packed: &[u8], scale_bytes: &[u8], scale2: f32, k: usize) -> Vec<f32> {
    w3_dequant_row(packed, scale_bytes, scale2, k)
        .into_iter()
        .map(|w| e4m3_to_f32(f32_to_e4m3(w)))
        .collect()
}

/// Host transpose `[N, row_len]` u8 -> `[row_len, N_pad64]` u8 (tail
/// columns zero). Same padding rule as `QuantizedWeight::transpose_for_gemm`.
/// Used to build the W3 GEMM layouts from sidecar tensors at load time.
pub fn w3_transpose_host(src: &[u8], n: usize, row_len: usize) -> (Vec<u8>, usize) {
    assert_eq!(src.len(), n * row_len);
    let n_pad = n.div_ceil(64) * 64;
    let mut t = vec![0u8; row_len * n_pad];
    for i in 0..n {
        for j in 0..row_len {
            t[j * n_pad + i] = src[i * row_len + j];
        }
    }
    (t, n_pad)
}

/// Parse an `ATLAS_FFN_W3_LAYERS`-style layer set: comma-separated indices
/// with optional `a-b` inclusive ranges, whitespace-tolerant. Invalid
/// entries are ignored with a warning (fail-open to the W4 default path).
pub fn parse_w3_layer_set(spec: &str) -> std::collections::BTreeSet<usize> {
    let mut set = std::collections::BTreeSet::new();
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        if let Some((a, b)) = tok.split_once('-') {
            match (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                (Ok(a), Ok(b)) if a <= b => set.extend(a..=b),
                _ => tracing::warn!("ATLAS_FFN_W3_LAYERS: ignoring invalid range '{tok}'"),
            }
        } else {
            match tok.parse::<usize>() {
                Ok(v) => {
                    set.insert(v);
                }
                Err(_) => tracing::warn!("ATLAS_FFN_W3_LAYERS: ignoring invalid entry '{tok}'"),
            }
        }
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden vectors generated by `local/tools/repack_w3.py --golden`
    // (K=32, 2 scale groups). The Python simulator, this host mirror, and
    // the CUDA kernels share one dequant contract; these constants pin it.
    const G_CODES: [u8; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0, 2, 2, 5, 0, 1, 7, 3, 6, 4, 0, 2, 1, 6, 5,
        7, 3,
    ];
    const G_PACKED: [u8; 12] = [136, 198, 250, 119, 57, 5, 82, 145, 207, 132, 226, 126];
    const G_SCALES: [u8; 2] = [0x44, 0x2E]; // e4m3: 6.0, 0.21875
    const G_SCALE2_BITS: u32 = 0x3c4a4533;
    const G_DEQ_F32_BITS: [u32; 32] = [
        0x0, 0x3d17b3e6, 0x3d97b3e6, 0x3e17b3e6, 0x80000000, 0xbd17b3e6, 0xbd97b3e6, 0xbe17b3e6,
        0xbe17b3e6, 0xbd97b3e6, 0xbd17b3e6, 0x80000000, 0x3e17b3e6, 0x3d97b3e6, 0x3d17b3e6, 0x0,
        0x3c30fc8d, 0x3c30fc8d, 0xbbb0fc8d, 0x0, 0x3bb0fc8d, 0xbcb0fc8d, 0x3cb0fc8d, 0xbc30fc8d,
        0x80000000, 0x0, 0x3c30fc8d, 0x3bb0fc8d, 0xbc30fc8d, 0xbbb0fc8d, 0xbcb0fc8d, 0x3cb0fc8d,
    ];
    const G_DEQ_E4M3_BITS: [u32; 32] = [
        0x0, 0x3d100000, 0x3d900000, 0x3e100000, 0x80000000, 0xbd100000, 0xbd900000, 0xbe100000,
        0xbe100000, 0xbd900000, 0xbd100000, 0x80000000, 0x3e100000, 0x3d900000, 0x3d100000, 0x0,
        0x3c400000, 0x3c400000, 0xbbc00000, 0x0, 0x3bc00000, 0xbcb00000, 0x3cb00000, 0xbc400000,
        0x80000000, 0x0, 0x3c400000, 0x3bc00000, 0xbc400000, 0xbbc00000, 0xbcb00000, 0x3cb00000,
    ];

    #[test]
    fn golden_pack_matches_python() {
        assert_eq!(w3_pack_codes(&G_CODES), G_PACKED.to_vec());
        assert_eq!(w3_unpack_codes(&G_PACKED, 32), G_CODES.to_vec());
    }

    #[test]
    fn golden_dequant_f32_bit_exact() {
        let scale2 = f32::from_bits(G_SCALE2_BITS);
        let deq = w3_dequant_row(&G_PACKED, &G_SCALES, scale2, 32);
        let bits: Vec<u32> = deq.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, G_DEQ_F32_BITS.to_vec());
    }

    #[test]
    fn golden_dequant_e4m3_bit_exact() {
        let scale2 = f32::from_bits(G_SCALE2_BITS);
        let deq = w3_dequant_row_e4m3(&G_PACKED, &G_SCALES, scale2, 32);
        let bits: Vec<u32> = deq.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, G_DEQ_E4M3_BITS.to_vec());
    }

    #[test]
    fn e4m3_decode_spot_checks() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        // 0x44 = e:1000 (2^1) m:100 (1.5) -> 3.0; 6.0 is 0x4C (e:1001 m:100).
        assert_eq!(e4m3_to_f32(0x44), 3.0);
        assert_eq!(e4m3_to_f32(0x4C), 6.0);
        // 0x2E = e:0101 (2^-2) m:110 (1.75) -> 0.4375.
        assert_eq!(e4m3_to_f32(0x2E), 0.4375);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0xC4), -3.0);
        assert!(e4m3_to_f32(0x7F).is_nan());
        // smallest subnormal
        assert_eq!(e4m3_to_f32(0x01), 2.0f32.powi(-9));
    }

    #[test]
    fn e4m3_encode_roundtrip_and_rne() {
        // Every finite byte round-trips through decode -> encode.
        for b in 0u8..=0xFF {
            if (b & 0x7F) == 0x7F {
                continue; // NaN
            }
            let v = e4m3_to_f32(b);
            let rt = f32_to_e4m3(v);
            // -0.0 encodes to 0x80; 0.0 to 0x00 — both decode equal.
            assert_eq!(e4m3_to_f32(rt).to_bits(), v.to_bits(), "byte {b:#x}");
        }
        // Saturation
        assert_eq!(f32_to_e4m3(1e6), 0x7E);
        assert_eq!(f32_to_e4m3(-1e6), 0xFE);
        // Tie between 6.0 (0x4C) and 6.5 (0x4D) is 6.25 -> even byte 0x4C.
        assert_eq!(f32_to_e4m3(6.25), 0x4C);
        // Tie between 6.5 (0x4D) and 7.0 (0x4E) is 6.75 -> even byte 0x4E.
        assert_eq!(f32_to_e4m3(6.75), 0x4E);
    }

    #[test]
    fn pack_roundtrip_random() {
        // Deterministic LCG so the test is reproducible without rand.
        let mut state = 0x12345678u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) & 7) as u8
        };
        let codes: Vec<u8> = (0..1024).map(|_| next()).collect();
        let packed = w3_pack_codes(&codes);
        assert_eq!(packed.len(), w3_row_bytes(1024));
        assert_eq!(w3_unpack_codes(&packed, 1024), codes);
    }

    #[test]
    fn transpose_host_pads_and_maps() {
        // 3 rows x 4 cols -> 4 rows x 64 (padded) cols.
        let src: Vec<u8> = (0..12).collect();
        let (t, n_pad) = w3_transpose_host(&src, 3, 4);
        assert_eq!(n_pad, 64);
        assert_eq!(t.len(), 4 * 64);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(t[j * 64 + i], src[i * 4 + j]);
            }
        }
        // Padding stays zero.
        assert_eq!(t[3], 0);
        assert_eq!(t[64 + 63], 0);
    }

    #[test]
    fn parse_layer_set_variants() {
        assert!(parse_w3_layer_set("").is_empty());
        assert!(parse_w3_layer_set("  ,  ").is_empty());
        let s = parse_w3_layer_set("3,7,12");
        assert_eq!(s.into_iter().collect::<Vec<_>>(), vec![3, 7, 12]);
        let s = parse_w3_layer_set(" 1 , 5-8 ,63");
        assert_eq!(s.into_iter().collect::<Vec<_>>(), vec![1, 5, 6, 7, 8, 63]);
        // Invalid entries are ignored, valid ones kept.
        let s = parse_w3_layer_set("2,x,9-7,4");
        assert_eq!(s.into_iter().collect::<Vec<_>>(), vec![2, 4]);
        // Duplicates collapse.
        assert_eq!(parse_w3_layer_set("5,5,5").len(), 1);
    }
}
