// SPDX-License-Identifier: AGPL-3.0-only

//! CPU property oracle for direct EXL3 K2-window to packed-E4M3 conversion.
//!
//! The incumbent `w2f_decode2`/`w2a8_decode2` path decodes each 16-bit trellis
//! window to binary16, widens it to f32, multiplies by 16, and uses
//! `cvt.rn.satfinite.e4m3x2.f32`. CUDA 13 can keep the pair packed instead:
//! `__hmul2(decoded, half2(16))`, then
//! `cvt.rn.satfinite.e4m3x2.f16x2`. The tests below exhaust every legal K2
//! window and independently model the packed conversion with integer shifts.

use std::collections::HashSet;

use half::f16;

const MCG_MULT: u32 = 0xCBAC_1FED;
const WEIGHT_SCALE: f32 = 16.0;

fn decode_k2_window(window: u16) -> f16 {
    let mixed = (window as u32).wrapping_mul(MCG_MULT);
    let mixed = (mixed & 0x8FFF_8FFF) ^ 0x3B60_3B60;
    f16::from_f32(
        f16::from_bits(mixed as u16).to_f32() + f16::from_bits((mixed >> 16) as u16).to_f32(),
    )
}

fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    if exponent == 0 {
        return sign * mantissa as f32 * 2.0f32.powi(-9);
    }
    sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
}

/// Independent numerical oracle for `cvt.rn.satfinite.e4m3*.f32`.
fn f32_to_e4m3_reference(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return sign | 0x7f;
    }
    if magnitude >= 448.0 {
        return sign | 0x7e;
    }

    let mut best = 0u8;
    let mut best_distance = f64::INFINITY;
    for code in 0u8..=0x7e {
        let distance = (e4m3_to_f32(code) as f64 - magnitude as f64).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0 && best & 1 != 0)
        {
            best = code;
            best_distance = distance;
        }
    }
    sign | best
}

fn round_shift_rne(value: u32, shift: u32) -> u32 {
    let rounded = value >> shift;
    let remainder = value & ((1 << shift) - 1);
    let midpoint = 1 << (shift - 1);
    rounded + u32::from(remainder > midpoint || (remainder == midpoint && rounded & 1 != 0))
}

/// Compact integer oracle for CUDA's packed `e4m3x2.f16x2` conversion.
///
/// A normal half has eleven significand bits; E4M3 keeps four, so the normal
/// path is one RNE shift by seven. Values below E4M3's minimum normal instead
/// round in units of 2^-9. Only the two saturation/NaN cases need branches.
fn f16_bits_to_e4m3_rne_satfinite(bits: u16) -> u8 {
    let sign = ((bits >> 8) as u8) & 0x80;
    let magnitude = bits & 0x7fff;
    let exponent = u32::from(magnitude >> 10);
    let mantissa = u32::from(magnitude & 0x03ff);

    if exponent == 0x1f {
        return sign | if mantissa == 0 { 0x7e } else { 0x7f };
    }
    // Even the largest binary16 subnormal is below half of E4M3's smallest
    // subnormal. Preserve the sign when RNE produces zero.
    if exponent == 0 {
        return sign;
    }

    let significand = 1024 + mantissa;
    let code = if exponent <= 8 {
        round_shift_rne(significand, 16 - exponent)
    } else {
        let mut e4m3_exponent = exponent - 8;
        let mut rounded_significand = round_shift_rne(significand, 7);
        if rounded_significand == 16 {
            e4m3_exponent += 1;
            rounded_significand = 8;
        }
        (e4m3_exponent * 8 + rounded_significand - 8).min(0x7e)
    };
    sign | code.min(0x7e) as u8
}

fn half_mul_16_rn(bits: u16) -> u16 {
    f16::from_f32(f16::from_bits(bits).to_f32() * WEIGHT_SCALE).to_bits()
}

/// Every nonzero K2 result is normal and remains finite after a 2^4 scale, so
/// the packed half multiply is also exactly this per-lane exponent increment.
fn scale_legal_k2_half_by_16_bits(bits: u16) -> u16 {
    let magnitude = bits & 0x7fff;
    if magnitude == 0 {
        return bits;
    }
    let exponent = (magnitude >> 10) & 0x1f;
    assert!((1..=26).contains(&exponent));
    bits + (4 << 10)
}

fn incumbent_code(window: u16) -> u8 {
    f32_to_e4m3_reference(decode_k2_window(window).to_f32() * WEIGHT_SCALE)
}

fn packed_f16_candidate_code(window: u16) -> u8 {
    let decoded = decode_k2_window(window).to_bits();
    let scaled = half_mul_16_rn(decoded);
    assert_eq!(scaled, scale_legal_k2_half_by_16_bits(decoded));
    f16_bits_to_e4m3_rne_satfinite(scaled)
}

fn pack_pair(low: u8, high: u8) -> u16 {
    u16::from(low) | (u16::from(high) << 8)
}

fn is_e4m3_rounding_tie(bits: u16) -> bool {
    let magnitude = bits & 0x7fff;
    let exponent = u32::from(magnitude >> 10);
    if exponent == 0 || exponent == 0x1f {
        return false;
    }
    let significand = 1024 + u32::from(magnitude & 0x03ff);
    let shift = if exponent <= 8 { 16 - exponent } else { 7 };
    significand & ((1 << shift) - 1) == 1 << (shift - 1)
}

#[test]
fn every_legal_k2_window_matches_packed_f16x2_conversion_byte_exactly() {
    let mut codes = HashSet::new();
    let mut zero_windows = 0usize;
    let mut tie_windows = 0usize;

    for window in 0u16..=u16::MAX {
        let decoded = decode_k2_window(window);
        let scaled_bits = half_mul_16_rn(decoded.to_bits());
        let incumbent = incumbent_code(window);
        let candidate = packed_f16_candidate_code(window);

        assert!(decoded.is_finite(), "window={window:#06x}");
        assert_eq!(candidate, incumbent, "window={window:#06x}");
        assert_ne!(candidate & 0x7f, 0x7f, "window={window:#06x}");
        zero_windows += usize::from(decoded.to_bits() & 0x7fff == 0);
        if is_e4m3_rounding_tie(scaled_bits) {
            tie_windows += 1;
            assert_eq!(candidate & 1, 0, "RNE tie window={window:#06x}");
        }
        codes.insert(candidate);
    }

    assert_eq!(zero_windows, 12);
    assert_eq!(tie_windows, 1_131);
    assert_eq!(codes.len(), 207);
}

#[test]
fn packed_pair_exhausts_every_window_in_both_byte_lanes() {
    // Complement is a bijection, so every legal window appears once in each
    // packed byte without attempting an unnecessary 2^32 Cartesian product.
    for low_window in 0u16..=u16::MAX {
        let high_window = !low_window;
        let incumbent = pack_pair(incumbent_code(low_window), incumbent_code(high_window));
        let candidate = pack_pair(
            packed_f16_candidate_code(low_window),
            packed_f16_candidate_code(high_window),
        );
        assert_eq!(
            candidate, incumbent,
            "windows={low_window:#06x}/{high_window:#06x}"
        );
    }
}

#[test]
fn packed_f16_oracle_covers_signed_zero_subnormals_and_overflow() {
    let mut signed_zeros = 0usize;
    let mut subnormals = 0usize;

    // This broader converter proof includes cases the legal K2 codebook does
    // not produce, particularly negative zero and binary16 subnormals.
    for bits in 0u16..=u16::MAX {
        let value = f16::from_bits(bits);
        if value.is_nan() {
            continue;
        }
        let incumbent = f32_to_e4m3_reference(value.to_f32() * WEIGHT_SCALE);
        let candidate = f16_bits_to_e4m3_rne_satfinite(half_mul_16_rn(bits));
        assert_eq!(candidate, incumbent, "half={bits:#06x}");
        signed_zeros += usize::from(bits & 0x7fff == 0);
        subnormals += usize::from((1..=0x03ff).contains(&(bits & 0x7fff)));
    }

    assert_eq!(signed_zeros, 2);
    assert_eq!(subnormals, 2_046);
    assert_eq!(f16_bits_to_e4m3_rne_satfinite(0), 0x00);
    assert_eq!(f16_bits_to_e4m3_rne_satfinite(0x8000), 0x80);
}
