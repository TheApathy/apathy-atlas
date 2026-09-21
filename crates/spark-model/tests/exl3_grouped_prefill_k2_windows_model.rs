// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for fixed K64/K2 EXL3 low-window extraction.

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
fn only_fixed_k64_k2_wrappers_enable_low_windows() {
    for source in [GU_K64, DOWN_K64, GU_N128_K64, DOWN_N128_K64] {
        assert!(source.contains("#define EXL3_PF_K2_LO_WINDOWS 1"));
    }
    for source in [GU_K16, DOWN_K16, GENERIC_K16, GENERIC_K64] {
        assert!(!source.contains("#define EXL3_PF_K2_LO_WINDOWS 1"));
    }
}

#[test]
fn low_window_compile_time_contract_is_fail_closed() {
    let source = compact(KERNEL);
    assert!(KERNEL.contains("#ifndef EXL3_PF_K2_LO_WINDOWS"));
    assert!(KERNEL.contains("#define EXL3_PF_K2_LO_WINDOWS 0"));
    assert!(source.contains("static_assert(EXL3_PF_K2_LO_WINDOWS==0||EXL3_PF_K2_LO_WINDOWS==1,"));
    assert!(source.contains("!EXL3_PF_K2_LO_WINDOWS||EXL3_PF_FIXED_BITS==2"));
    assert!(source.contains("!EXL3_PF_K2_LO_WINDOWS||EXL3_PF_K_STEP==64"));
}

#[test]
fn specialized_dq8_uses_only_low_word_while_generic_retains_funnel_windows() {
    let source = compact(KERNEL);
    let specialized_start = source
        .find("#ifEXL3_PF_K2_LO_WINDOWS")
        .expect("specialized K2 low-window arm");
    let generic_start = source[specialized_start..]
        .find("#else")
        .map(|offset| specialized_start + offset)
        .expect("generic window arm");
    let arm_end = source[generic_start..]
        .find("#endif")
        .map(|offset| generic_start + offset)
        .expect("window arm end");
    let specialized = &source[specialized_start..generic_start];
    let generic = &source[generic_start..arm_end];

    assert!(specialized.contains("constunsignedintw5=lo>>4;"));
    assert!(specialized.contains("constunsignedintw3=lo>>8;"));
    assert!(specialized.contains("constunsignedintw1=lo>>12;"));
    assert!(
        !specialized.contains("hi"),
        "fixed K2 low-window extraction must not consume the high word"
    );

    assert!(generic.contains("constunsignedinthi=a>>g.shift;"));
    assert!(generic.contains("constunsignedintw5=__funnelshift_r(lo,hi,2*bits);"));
    assert!(generic.contains("constunsignedintw3=__funnelshift_r(lo,hi,4*bits);"));
    assert!(generic.contains("constunsignedintw1=__funnelshift_r(lo,hi,6*bits);"));
}

#[derive(Clone, Copy)]
struct LaneGeom {
    shift: u32,
}

fn k2_lane_geom(lane: u32) -> LaneGeom {
    let bits = 2;
    let b1 = (lane * 8 + 257) * bits;
    let b2 = b1 + 7 * bits;
    let i2 = (b2 - 1) >> 5;
    LaneGeom {
        shift: (i2 + 1) * 32 - b2,
    }
}

fn funnelshift_r(low: u32, high: u32, shift: u32) -> u32 {
    let shift = shift & 31;
    if shift == 0 {
        low
    } else {
        (low >> shift) | (high << (32 - shift))
    }
}

fn generic_windows(a: u32, b: u32, lane: u32) -> [u32; 3] {
    let geom = k2_lane_geom(lane);
    let lo = funnelshift_r(b, a, geom.shift);
    let hi = if geom.shift >= 32 { 0 } else { a >> geom.shift };
    [
        funnelshift_r(lo, hi, 4),
        funnelshift_r(lo, hi, 8),
        funnelshift_r(lo, hi, 12),
    ]
}

fn specialized_windows(a: u32, b: u32, lane: u32) -> [u32; 3] {
    let lo = funnelshift_r(b, a, k2_lane_geom(lane).shift);
    [lo >> 4, lo >> 8, lo >> 12]
}

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn specialized_low_18_bits_match_generic_k2_windows_for_every_lane() {
    const CONSUMED: u32 = (1 << 18) - 1;
    let edges = [
        0,
        u32::MAX,
        1,
        1 << 31,
        0xaaaa_aaaa,
        0x5555_5555,
        0x0123_4567,
        0x89ab_cdef,
    ];
    let mut pairs = Vec::new();
    for &a in &edges {
        for &b in &edges {
            pairs.push((a, b));
        }
    }
    let mut state = 0x4b32_10f7;
    for _ in 0..256 {
        let a = xorshift32(&mut state);
        let b = xorshift32(&mut state);
        pairs.push((a, b));
    }

    for lane in 0..32 {
        for &(a, b) in &pairs {
            let generic = generic_windows(a, b, lane);
            let specialized = specialized_windows(a, b, lane);
            for window in 0..3 {
                assert_eq!(
                    specialized[window] & CONSUMED,
                    generic[window] & CONSUMED,
                    "lane={lane} window={window} a={a:#010x} b={b:#010x}"
                );
            }
        }
    }
}
