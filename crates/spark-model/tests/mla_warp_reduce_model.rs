// SPDX-License-Identifier: AGPL-3.0-only

const NUM_WARPS: usize = 8;
const WARP_SIZE: usize = 32;
const VEC: usize = 16;
const DIMS: usize = WARP_SIZE * VEC;
const KERNEL: &str =
    include_str!("../../../kernels/gb10/deepseek-v4-flash/nvfp4/mla_paged_decode_fp8.cu");

#[derive(Clone)]
struct Partials {
    m: [f32; NUM_WARPS],
    l: [f32; NUM_WARPS],
    o: Vec<[f32; DIMS]>,
}

fn merge_scales(partials: &mut Partials, mine: usize, other: usize) -> Option<(f32, f32)> {
    let other_l = partials.l[other];
    if other_l <= 0.0 {
        return None;
    }
    let mine_m = partials.m[mine];
    let other_m = partials.m[other];
    let mine_l = partials.l[mine];
    let new_m = mine_m.max(other_m);
    let mine_scale = (mine_m - new_m).exp();
    let other_scale = (other_m - new_m).exp();
    partials.l[mine] = mine_l * mine_scale + other_l * other_scale;
    partials.m[mine] = new_m;
    Some((mine_scale, other_scale))
}

fn merge_dims(
    partials: &mut Partials,
    mine: usize,
    other: usize,
    scales: (f32, f32),
    dims: impl Iterator<Item = usize>,
) {
    let (mine_scale, other_scale) = scales;
    for dim in dims {
        partials.o[mine][dim] =
            partials.o[mine][dim] * mine_scale + partials.o[other][dim] * other_scale;
    }
}

fn legacy_reduce(mut partials: Partials) -> Partials {
    for stride in [4, 2, 1] {
        for mine in 0..stride {
            let other = mine + stride;
            if let Some(scales) = merge_scales(&mut partials, mine, other) {
                merge_dims(&mut partials, mine, other, scales, 0..DIMS);
            }
        }
    }
    partials
}

fn lane_owned_reduce(mut partials: Partials) -> Partials {
    for stride in [4, 2, 1] {
        for mine in 0..stride {
            let other = mine + stride;
            if let Some(scales) = merge_scales(&mut partials, mine, other) {
                for lane in 0..WARP_SIZE {
                    merge_dims(
                        &mut partials,
                        mine,
                        other,
                        scales,
                        lane * VEC..(lane + 1) * VEC,
                    );
                }
            }
        }
    }
    partials
}

fn fixture() -> Partials {
    let mut state = 0xD33D_5EE5_CAFE_BABEu64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as u32) as f32 / u32::MAX as f32
    };
    Partials {
        m: std::array::from_fn(|warp| -3.0 + warp as f32 * 0.37),
        l: std::array::from_fn(|warp| if warp == 6 { 0.0 } else { 0.5 + next() }),
        o: (0..NUM_WARPS)
            .map(|_| std::array::from_fn(|_| next() * 2.0 - 1.0))
            .collect(),
    }
}

#[test]
fn lane_owned_merge_is_raw_bit_equivalent() {
    let legacy = legacy_reduce(fixture());
    let lane_owned = lane_owned_reduce(fixture());

    assert_eq!(legacy.m[0].to_bits(), lane_owned.m[0].to_bits());
    assert_eq!(legacy.l[0].to_bits(), lane_owned.l[0].to_bits());
    for dim in 0..DIMS {
        assert_eq!(legacy.o[0][dim].to_bits(), lane_owned.o[0][dim].to_bits());
    }
}

#[test]
fn production_reduction_assigns_one_vector_to_each_lane() {
    let reduction = KERNEL
        .split_once("// Reduce across warps")
        .unwrap()
        .1
        .split_once("// Write output")
        .unwrap()
        .0;

    assert!(!reduction.contains("for (int i = 0; i < 512; i++)"));
    assert!(reduction.contains("lane_id * VEC_BF16 + i"));
}
