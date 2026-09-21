// SPDX-License-Identifier: AGPL-3.0-only

//! CPU numeric and fragment contracts for a future EXL3 K2 W2A8 prefill arm.
//!
//! This deliberately does not enable a serving path. It fixes the native
//! provisional E4M3/RNE contract and K32 fragment ownership before a native
//! SM121 conversion dump, GPU cosine, and timing gate can justify host wiring.

use std::collections::HashSet;

use half::{bf16, f16};

const MCG_MULT: u32 = 0xCBAC_1FED;
const WEIGHT_SCALE: f32 = 16.0;

fn decode_3inst(window: u16) -> f16 {
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

/// Reference for native `cvt.rn.satfinite.e4m3*.f32` on finite inputs.
/// Searching the 127 positive finite encodings keeps the tie-to-even rule
/// explicit instead of inheriting a host conversion library's FP8 variant.
fn f32_to_e4m3_rne_satfinite(value: f32) -> u8 {
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
        let decoded = e4m3_to_f32(code) as f64;
        let distance = (decoded - magnitude as f64).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0 && best & 1 != 0)
        {
            best = code;
            best_distance = distance;
        }
    }
    sign | best
}

fn tile_coord(lane: usize, sequence: usize) -> (usize, usize) {
    let n = 8 * (sequence / 4) + lane / 4;
    let k = 2 * (lane % 4) + (sequence % 2) + 8 * ((sequence % 4) / 2);
    (k, n)
}

fn pack_fp8_pairs(first: [u8; 2], second: [u8; 2]) -> u32 {
    u32::from_le_bytes([first[0], first[1], second[0], second[1]])
}

fn repacked_b_coordinates(
    group: usize,
    n_half: usize,
    native_tid: usize,
    k_half: usize,
) -> [(usize, usize); 4] {
    let source_tid = 2 * (native_tid % 2);
    let source_sequence = 4 * n_half + 2 * (native_tid / 2);
    let coordinate = |tid, sequence| {
        let (k, n) = tile_coord(4 * group + tid, sequence);
        (k_half + k, n)
    };
    [
        coordinate(source_tid, source_sequence),
        coordinate(source_tid, source_sequence + 1),
        coordinate(source_tid + 1, source_sequence),
        coordinate(source_tid + 1, source_sequence + 1),
    ]
}

fn activation_group_scale(group: &[f32]) -> f32 {
    (group
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()))
        / 448.0)
        .max(1.0e-12)
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

#[test]
fn exhaustive_k2_codebook_is_finite_and_bounded() {
    let mut distinct = HashSet::new();
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for window in 0u16..=u16::MAX {
        let value = decode_3inst(window);
        assert!(value.is_finite(), "window={window:#06x}");
        distinct.insert(value.to_bits());
        minimum = minimum.min(value.to_f32());
        maximum = maximum.max(value.to_f32());
    }
    assert_eq!(distinct.len(), 10_746);
    assert_eq!(minimum, -3.949_218_8);
    assert_eq!(maximum, 3.949_218_8);
}

#[test]
fn scale16_e4m3_recast_never_saturates_or_erases_nonzero_codebook_values() {
    let mut maximum_error = 0.0f32;
    for window in 0u16..=u16::MAX {
        let value = decode_3inst(window).to_f32();
        let scaled = value * WEIGHT_SCALE;
        assert!(scaled.abs() < 448.0);
        let encoded = f32_to_e4m3_rne_satfinite(scaled);
        assert_ne!(encoded & 0x7f, 0x7f, "window={window:#06x}");
        if value != 0.0 {
            assert_ne!(encoded & 0x7f, 0, "window={window:#06x}");
        }
        let reconstructed = e4m3_to_f32(encoded) / WEIGHT_SCALE;
        maximum_error = maximum_error.max((reconstructed - value).abs());
    }
    assert!(maximum_error <= 0.125, "maximum_error={maximum_error}");
}

#[test]
fn e4m3_reference_has_only_the_two_nan_encodings_and_rne_ties() {
    let nan_codes: Vec<_> = (0u8..=u8::MAX)
        .filter(|&code| e4m3_to_f32(code).is_nan())
        .collect();
    assert_eq!(nan_codes, [0x7f, 0xff]);
    assert_eq!(f32_to_e4m3_rne_satfinite(448.0), 0x7e);
    assert_eq!(f32_to_e4m3_rne_satfinite(-448.0), 0xfe);
    // Midpoint between 1.0 (0x38, even LSB) and 1.125 (0x39).
    assert_eq!(f32_to_e4m3_rne_satfinite(1.0625), 0x38);
    // Midpoint between 1.125 (odd LSB) and 1.25 (even LSB).
    assert_eq!(f32_to_e4m3_rne_satfinite(1.1875), 0x3a);
}

#[test]
fn e4m3_reference_round_trips_finite_codes_and_all_positive_midpoints() {
    for code in 0u8..=u8::MAX {
        let value = e4m3_to_f32(code);
        if value.is_finite() {
            assert_eq!(f32_to_e4m3_rne_satfinite(value), code, "code={code:#04x}");
        }
    }

    for lower in 0u8..0x7e {
        let midpoint = (e4m3_to_f32(lower) + e4m3_to_f32(lower + 1)) * 0.5;
        let expected = if lower & 1 == 0 { lower } else { lower + 1 };
        assert_eq!(
            f32_to_e4m3_rne_satfinite(midpoint),
            expected,
            "lower={lower:#04x}"
        );
        assert_eq!(
            f32_to_e4m3_rne_satfinite(-midpoint),
            expected | 0x80,
            "negative lower={lower:#04x}"
        );
    }

    assert_eq!(f32_to_e4m3_rne_satfinite(f32::INFINITY), 0x7e);
    assert_eq!(f32_to_e4m3_rne_satfinite(f32::NEG_INFINITY), 0xfe);
}

#[test]
fn paired_k16_weight_fragments_cover_k32_n16_once() {
    let mut coordinates = HashSet::new();
    for group in 0..8 {
        for n_half in 0..2 {
            for native_tid in 0..4 {
                for k_half in [0, 16] {
                    for coordinate in repacked_b_coordinates(group, n_half, native_tid, k_half) {
                        assert!(coordinates.insert(coordinate));
                    }
                }
            }
        }
    }
    assert_eq!(coordinates.len(), 32 * 16);
    assert!(coordinates.iter().all(|&(k, n)| k < 32 && n < 16));
}

#[test]
fn paired_k16_weight_registers_preserve_tile_and_sequence_order() {
    // The EXL3 decoder emits BF16-K16 lane fragments. Native FP8 K32 MMA
    // instead consumes four consecutive K bytes per register, so two source
    // lanes in each quad must be repacked before issuing the MMA.
    for n_half in 0..2 {
        let n = n_half * 8;
        for native_tid in 0..4 {
            let tags = |k_half| {
                repacked_b_coordinates(0, n_half, native_tid, k_half).map(|(k, actual_n)| {
                    assert_eq!(actual_n, n);
                    (actual_n * 16 + k) as u8
                })
            };
            let low = tags(0);
            let high = tags(16);
            let b0 = pack_fp8_pairs([low[0], low[1]], [low[2], low[3]]);
            let b1 = pack_fp8_pairs([high[0], high[1]], [high[2], high[3]]);
            let base = (n * 16 + native_tid * 4) as u8;
            assert_eq!(b0.to_le_bytes(), [base, base + 1, base + 2, base + 3]);
            assert_eq!(
                b1.to_le_bytes(),
                [base + 16, base + 17, base + 18, base + 19]
            );
        }
    }
}

#[test]
fn per_128_a8_scales_fold_each_group_before_the_bf16_boundary() {
    // Two short stand-ins for consecutive K128 groups. Their deliberately
    // different scales catch an incorrect single row-wide epilogue scale.
    let activations = [[1.0f32, -2.0], [4.0, -1.0]];
    let weights = [[0.5f32, -0.25], [0.125, -0.5]];
    let expected_scales = [1.0 / 224.0, 1.0 / 112.0];
    assert_eq!(activation_group_scale(&[0.0, -0.0]), 1.0e-12);

    let mut restored = 0.0f32;
    let mut reference = 0.0f32;
    for group in 0..2 {
        let activation_scale = activation_group_scale(&activations[group]);
        assert_eq!(activation_scale, expected_scales[group]);
        let activation_fp8 = activations[group]
            .map(|value| e4m3_to_f32(f32_to_e4m3_rne_satfinite(value / activation_scale)));
        let weight_fp8 = weights[group]
            .map(|value| e4m3_to_f32(f32_to_e4m3_rne_satfinite(value * WEIGHT_SCALE)));
        let fp8_accumulator = activation_fp8[0] * weight_fp8[0] + activation_fp8[1] * weight_fp8[1];
        restored += fp8_accumulator * (activation_scale / WEIGHT_SCALE);
        reference +=
            activations[group][0] * weights[group][0] + activations[group][1] * weights[group][1];
    }
    assert_eq!(restored, reference);

    // The scale-group contributions remain FP32 until the one output
    // boundary. These values make premature per-group BF16 rounding visible.
    let contributions = [0.0001f32, 0.00005];
    let final_boundary = bf16::from_f32(contributions.into_iter().sum()).to_bits();
    let premature = bf16::from_f32(
        contributions
            .into_iter()
            .map(|value| bf16::from_f32(value).to_f32())
            .sum(),
    )
    .to_bits();
    assert_ne!(final_boundary, premature);
}

#[test]
fn native_fp8_a_fragments_cover_m64_k32_once() {
    let mut coordinates = HashSet::new();
    for mt in 0..4 {
        for lane in 0..32 {
            let group = lane / 4;
            let tid = lane % 4;
            let row0 = mt * 16 + group;
            let row1 = row0 + 8;
            for (register, row, k_half) in
                [(0, row0, 0), (1, row1, 0), (2, row0, 16), (3, row1, 16)]
            {
                let register_coordinates = [
                    (row, k_half + 4 * tid),
                    (row, k_half + 4 * tid + 1),
                    (row, k_half + 4 * tid + 2),
                    (row, k_half + 4 * tid + 3),
                ];
                if mt == 0 && lane == 0 {
                    let expected = match register {
                        0 => [(0, 0), (0, 1), (0, 2), (0, 3)],
                        1 => [(8, 0), (8, 1), (8, 2), (8, 3)],
                        2 => [(0, 16), (0, 17), (0, 18), (0, 19)],
                        3 => [(8, 16), (8, 17), (8, 18), (8, 19)],
                        _ => unreachable!(),
                    };
                    assert_eq!(register_coordinates, expected);
                }
                for coordinate in register_coordinates {
                    assert!(coordinates.insert(coordinate));
                }
            }
        }
    }
    assert_eq!(coordinates.len(), 64 * 32);
    assert!(coordinates.iter().all(|&(m, k)| m < 64 && k < 32));
}

#[test]
fn native_fp8_accumulator_fragments_cover_m64_n64_once() {
    let mut outputs = HashSet::new();
    for warp in 0..4 {
        for mt in 0..4 {
            for n_half in 0..2 {
                for lane in 0..32 {
                    let group = lane / 4;
                    let tid = lane % 4;
                    let row0 = mt * 16 + group;
                    let row1 = row0 + 8;
                    let col0 = warp * 16 + n_half * 8 + tid * 2;
                    for coordinate in [
                        (row0, col0),
                        (row0, col0 + 1),
                        (row1, col0),
                        (row1, col0 + 1),
                    ] {
                        assert!(outputs.insert(coordinate));
                    }
                }
            }
        }
    }
    assert_eq!(outputs.len(), 64 * 64);
}

#[test]
fn compact_projection_oracle_bounds_w2a8_drift_and_reports_saturation() {
    // This is a mathematical end-to-end oracle, not a device-parity claim.
    // K=256 crosses two real K128 scale groups; the final row's first group
    // is all signed zero so the scale-floor path is load-bearing.
    const M: usize = 3;
    const K: usize = 256;
    const N: usize = 8;
    let mut state = 0x5732_4138_2026_0827u64;
    let mut activations = vec![0.0f32; M * K];
    for row in 0..M {
        for k in 0..K {
            let group = k / 128;
            let unit = (next_random(&mut state) >> 40) as f32 / (1u32 << 24) as f32;
            let amplitude = if group == 0 { 0.03125 } else { 0.5 };
            let value = if row == 2 && group == 0 {
                if k & 1 == 0 { 0.0 } else { -0.0 }
            } else {
                (unit * 2.0 - 1.0) * amplitude
            };
            activations[row * K + k] = bf16::from_f32(value).to_f32();
        }
    }
    let windows: Vec<u16> = (0..K * N)
        .map(|_| (next_random(&mut state) >> 48) as u16)
        .collect();

    let mut reference = Vec::with_capacity(M * N);
    let mut candidate = Vec::with_capacity(M * N);
    let mut saturated = 0usize;
    let mut floor_groups = 0usize;
    for row in 0..M {
        let mut quantized_a = [0u8; K];
        let mut scales = [0.0f32; K / 128];
        for group in 0..K / 128 {
            let start = row * K + group * 128;
            let values = &activations[start..start + 128];
            scales[group] = activation_group_scale(values);
            floor_groups += usize::from(scales[group] == 1.0e-12);
            for offset in 0..128 {
                let encoded = f32_to_e4m3_rne_satfinite(values[offset] / scales[group]);
                saturated += usize::from(encoded & 0x7f == 0x7e);
                quantized_a[group * 128 + offset] = encoded;
            }
        }

        for column in 0..N {
            let mut baseline_accumulator = 0.0f32;
            let mut w2a8_accumulator = 0.0f32;
            for group in 0..K / 128 {
                let mut inner = 0.0f32;
                for offset in 0..128 {
                    let k = group * 128 + offset;
                    let activation = activations[row * K + k];
                    let weight = decode_3inst(windows[k * N + column]).to_f32();
                    baseline_accumulator += activation * bf16::from_f32(weight).to_f32();
                    let activation_fp8 = e4m3_to_f32(quantized_a[k]);
                    let weight_fp8 = e4m3_to_f32(f32_to_e4m3_rne_satfinite(weight * WEIGHT_SCALE));
                    inner += activation_fp8 * weight_fp8;
                }
                w2a8_accumulator += inner * (scales[group] / WEIGHT_SCALE);
            }
            reference.push(bf16::from_f32(baseline_accumulator).to_f32());
            candidate.push(bf16::from_f32(w2a8_accumulator).to_f32());
        }
    }

    let dot: f64 = reference
        .iter()
        .zip(&candidate)
        .map(|(&lhs, &rhs)| lhs as f64 * rhs as f64)
        .sum();
    let reference_norm: f64 = reference.iter().map(|&value| (value as f64).powi(2)).sum();
    let candidate_norm: f64 = candidate.iter().map(|&value| (value as f64).powi(2)).sum();
    let error_norm: f64 = reference
        .iter()
        .zip(&candidate)
        .map(|(&lhs, &rhs)| ((lhs - rhs) as f64).powi(2))
        .sum();
    let cosine = dot / (reference_norm * candidate_norm).sqrt();
    let normalized_rmse = (error_norm / reference_norm).sqrt();
    assert_eq!(floor_groups, 1);
    assert!(saturated >= M * (K / 128) - floor_groups);
    assert!(cosine > 0.99, "cosine={cosine}");
    assert!(normalized_rmse < 0.15, "normalized_rmse={normalized_rmse}");
}

#[test]
fn producer_quantization_matches_standalone_after_the_bf16_boundary() {
    let standalone = |input: &[bf16]| {
        let values: Vec<_> = input.iter().map(|value| value.to_f32()).collect();
        let scale = activation_group_scale(&values);
        let codes = values
            .iter()
            .map(|value| f32_to_e4m3_rne_satfinite((value / scale).clamp(-448.0, 448.0)))
            .collect::<Vec<_>>();
        (scale, codes)
    };
    let producer = |raw: &[f32]| {
        let boundary = raw.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
        standalone(&boundary)
    };

    let mut raw = vec![0.0f32; 128];
    raw[0] = 448.0;
    raw[1] = -448.0;
    raw[2] = 1.0625;
    raw[3] = 1.1875;
    raw[4] = -0.0;
    let mut state = 0x6841_3238_2026_0827u64;
    for value in &mut raw[5..] {
        let unit = (next_random(&mut state) >> 40) as f32 / (1u32 << 24) as f32;
        *value = (unit * 2.0 - 1.0) * 400.0;
    }
    let boundary = raw.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
    let (scale, codes) = producer(&raw);
    assert_eq!((scale, codes.clone()), standalone(&boundary));
    assert_eq!(scale, 1.0);
    assert_eq!((codes[0], codes[1]), (0x7e, 0xfe));
    assert_eq!((codes[2], codes[3]), (0x38, 0x3a));

    let zeros = [0.0f32, -0.0].repeat(64);
    let (scale, codes) = producer(&zeros);
    assert_eq!(scale, 1.0e-12);
    assert_eq!((codes[0], codes[1]), (0x00, 0x80));

    // Quantizing the transform's FP32 result directly is observably different
    // from first crossing the incumbent BF16 store/load boundary.
    let fixed_max = 1.0f32;
    let midpoint: f32 = 1.0625 / 448.0;
    let raw_midpoint = f32::from_bits(midpoint.to_bits() + 1);
    let scale = fixed_max / 448.0;
    let before_boundary = f32_to_e4m3_rne_satfinite(raw_midpoint / scale);
    let after_boundary = f32_to_e4m3_rne_satfinite(bf16::from_f32(raw_midpoint).to_f32() / scale);
    assert_ne!(before_boundary, after_boundary);
}
