// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the fixed-shape K64 grouped-prefill BF16 store.

use half::bf16;
use std::collections::HashSet;

const KERNEL: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill.cu");
const GU_K16: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2_gu.cu");
const GU_K64: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_k2_gu.cu");
const DOWN_K16: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2_down.cu");
const DOWN_K64: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_k2_down.cu");
const GU_N128_K64: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_n128_k2_gu.cu");
const DOWN_N128_K64: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_n128_k2_down.cu");
const GENERIC_K16: &str = include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k2.cu");
const GENERIC_K64: &str =
    include_str!("../../../kernels/gb10/common/exl3_grouped_prefill_k64_k2.cu");

fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

#[test]
fn only_fixed_k64_wrappers_enable_packed_bf16_store() {
    for source in [GU_K64, DOWN_K64, GU_N128_K64, DOWN_N128_K64] {
        assert!(source.contains("#define EXL3_PF_PACKED_BF16_STORE 1"));
    }
    for source in [GU_K16, DOWN_K16, GENERIC_K16, GENERIC_K64] {
        assert!(!source.contains("#define EXL3_PF_PACKED_BF16_STORE 1"));
    }
}

#[test]
fn packed_store_compile_time_contract_is_fail_closed() {
    let source = compact(KERNEL);
    assert!(KERNEL.contains("#ifndef EXL3_PF_PACKED_BF16_STORE"));
    assert!(KERNEL.contains("#define EXL3_PF_PACKED_BF16_STORE 0"));
    assert!(
        source
            .contains("static_assert(EXL3_PF_PACKED_BF16_STORE==0||EXL3_PF_PACKED_BF16_STORE==1,")
    );
    assert!(source.contains("!EXL3_PF_PACKED_BF16_STORE||EXL3_PF_K_STEP==64"));
    assert!(source.contains("!EXL3_PF_PACKED_BF16_STORE||EXL3_PF_FIXED_N!=0"));
    assert!(source.contains("!EXL3_PF_PACKED_BF16_STORE||(EXL3_PF_FIXED_N&1)==0"));
    assert!(source.contains("!EXL3_PF_PACKED_BF16_STORE||EXL3_PF_FIXED_K!=0"));
    assert!(source.contains("!EXL3_PF_PACKED_BF16_STORE||EXL3_PF_EXACT_FULL_GRID==1"));
}

#[test]
fn packed_arm_vectorizes_independent_pairs_and_generic_arm_stays_scalar() {
    let source = compact(KERNEL);
    assert!(KERNEL.contains("#if EXL3_PF_PACKED_BF16_STORE"));
    assert!(source.contains("__floats2bfloat162_rn(acc[mt][nt][0],acc[mt][nt][1])"));
    assert!(source.contains("__floats2bfloat162_rn(acc[mt][nt][2],acc[mt][nt][3])"));
    assert_eq!(
        source
            .matches("*reinterpret_cast<__nv_bfloat162*>(out)=")
            .count(),
        2,
        "each live-row guard must contain one aligned packed store"
    );

    let row0_guard = source
        .find("if(m_start+row0<m_end){")
        .expect("existing row0 bounds guard");
    let row0_pair = source[row0_guard..]
        .find("__floats2bfloat162_rn(acc[mt][nt][0],acc[mt][nt][1])")
        .map(|offset| row0_guard + offset)
        .expect("row0 independent packed pair");
    let row1_guard = source
        .find("if(m_start+row1<m_end){")
        .expect("existing row1 bounds guard");
    let row1_pair = source[row1_guard..]
        .find("__floats2bfloat162_rn(acc[mt][nt][2],acc[mt][nt][3])")
        .map(|offset| row1_guard + offset)
        .expect("row1 independent packed pair");
    assert!(row0_guard < row0_pair && row0_pair < row1_guard && row1_guard < row1_pair);

    for scalar_store in [
        "out[0]=__float2bfloat16(acc[mt][nt][0]);",
        "out[1]=__float2bfloat16(acc[mt][nt][1]);",
        "out[0]=__float2bfloat16(acc[mt][nt][2]);",
        "out[1]=__float2bfloat16(acc[mt][nt][3]);",
    ] {
        assert!(
            source.contains(scalar_store),
            "missing generic store {scalar_store}"
        );
    }
}

#[test]
fn every_fixed_shape_output_pair_is_aligned_and_stays_within_its_row() {
    for n in [2048usize, 4096] {
        for n_tile_size in [64usize, 128] {
            let mut seen = HashSet::new();
            for n_tile in 0..n / n_tile_size {
                let n_base = n_tile * n_tile_size;
                for n_warp in 0..n_tile_size / 16 {
                    for nt in 0..2 {
                        for tid in 0..4 {
                            let col0 = n_base + n_warp * 16 + nt * 8 + tid * 2;
                            assert_eq!(col0 & 1, 0, "N={n} tile={n_tile_size} col0={col0}");
                            assert!(col0 + 1 < n, "N={n} tile={n_tile_size} col0={col0}");
                            assert_eq!(
                                col0 / n_tile_size,
                                (col0 + 1) / n_tile_size,
                                "N={n} tile={n_tile_size} col0={col0}"
                            );
                            assert!(seen.insert(col0));
                            assert!(seen.insert(col0 + 1));
                        }
                    }
                }
            }
            assert_eq!(
                seen,
                (0..n).collect(),
                "N={n} tile={n_tile_size} must cover every output column"
            );
        }
    }
}

fn independently_pack_bf16(low: f32, high: f32) -> u32 {
    u32::from(bf16::from_f32(low).to_bits()) | (u32::from(bf16::from_f32(high).to_bits()) << 16)
}

fn same_bf16(actual: bf16, expected: bf16) -> bool {
    (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
}

#[test]
fn packed_halves_equal_independent_round_to_nearest_for_edge_values() {
    let max_bf16 = bf16::from_bits(0x7f7f).to_f32();
    let min_bf16_subnormal = bf16::from_bits(0x0001).to_f32();
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        min_bf16_subnormal,
        -min_bf16_subnormal,
        max_bf16,
        -max_bf16,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];

    for &low in &values {
        for &high in &values {
            let packed = independently_pack_bf16(low, high);
            let packed_low = bf16::from_bits(packed as u16);
            let packed_high = bf16::from_bits((packed >> 16) as u16);
            let scalar_low = bf16::from_f32(low);
            let scalar_high = bf16::from_f32(high);
            assert!(same_bf16(packed_low, scalar_low), "low={low:?}");
            assert!(same_bf16(packed_high, scalar_high), "high={high:?}");
        }
    }
}
