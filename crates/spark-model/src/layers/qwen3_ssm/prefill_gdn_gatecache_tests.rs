// SPDX-License-Identifier: AGPL-3.0-only

use super::trait_prefill_gdn::{use_wy32_gatecache, wy32_dynamic_smem_bytes};

#[test]
fn gatecache_route_is_large_qwen_wy32_only_and_fails_closed() {
    assert_eq!(use_wy32_gatecache(true, true, 8192, 128, 128), Ok(true));
    assert_eq!(use_wy32_gatecache(false, true, 8192, 128, 128), Ok(false));
    assert_eq!(use_wy32_gatecache(true, false, 32, 128, 128), Ok(false));
    assert_eq!(use_wy32_gatecache(true, false, 8192, 64, 128), Ok(false));
    assert_eq!(use_wy32_gatecache(true, false, 8192, 128, 64), Ok(false));
    assert!(use_wy32_gatecache(true, false, 8192, 128, 128).is_err());
}

#[test]
fn monolithic_prefill_uses_the_fail_closed_gatecache_route_before_parent_wy32() {
    let monolithic = include_str!("trait_prefill.rs");
    let selector = monolithic
        .find("super::trait_prefill_gdn::use_wy32_gatecache(")
        .expect("monolithic prefill must use the shared gate-cache selector");
    let candidate = monolithic[selector..]
        .find("self.gdn_prefill_wy32_gatecache_k,")
        .map(|offset| selector + offset)
        .expect("monolithic prefill must launch the selected gate-cache kernel");
    let parent = monolithic[candidate..]
        .find("self.gdn_prefill_wy32_k,")
        .map(|offset| candidate + offset)
        .expect("monolithic prefill must retain the parent WY32 fallback");
    assert!(selector < candidate && candidate < parent);
    assert!(monolithic.contains("super::trait_prefill_gdn::wy32_dynamic_smem_bytes(kd, vd, true)"));
    assert!(monolithic.contains("super::trait_prefill_gdn::log_wy32_gatecache_engaged(k)"));

    let shared = include_str!("trait_prefill_gdn.rs");
    assert_eq!(
        shared
            .matches("ATLAS_GDN_PREFILL_GATECACHE engaged: WY32 cached gate products")
            .count(),
        1,
        "the two prefill implementations must share one process-wide marker"
    );
}

#[test]
fn dot_batched_gatecache_reserves_exact_partial_slab_and_fails_closed_on_overflow() {
    let parent_bytes = wy32_dynamic_smem_bytes(128, 128, false).unwrap();
    let dot_batched_bytes = wy32_dynamic_smem_bytes(128, 128, true).unwrap();
    assert_eq!(parent_bytes, 86_528);
    assert_eq!(dot_batched_bytes, 95_232);
    // The u16 pair map is 992 bytes, but it first consumes the prior
    // allocation's 240 bytes of alignment slack: rounded growth is 768 bytes.
    assert_eq!(496 * 2, 992);
    assert_eq!(dot_batched_bytes - parent_bytes, 8_704);
    assert_eq!(dot_batched_bytes - 94_464, 768);
    assert_eq!((86_288 + 7_936 + 992 + 255) / 256 * 256, 95_232);
    assert!(dot_batched_bytes < 99 * 1024);
    assert_eq!(wy32_dynamic_smem_bytes(usize::MAX, 128, true), None);
}

#[test]
fn cached_gate_products_are_bit_identical_to_parent_recomputation() {
    const C: usize = 32;
    let gates: [f32; C] = std::array::from_fn(|i| {
        // Non-powers-of-two make every intermediate rounding observable.
        f32::from_bits(0x3f70_0001u32.wrapping_sub((i as u32) * 0x0001_0101))
    });
    let mut cache = [[0.0f32; C]; C];
    let mut multiply_count = 0;
    let mut prefix = 1.0f32;
    for t in 0..C {
        cache[t][t] = prefix;
        if t + 1 < C {
            prefix *= gates[t];
            multiply_count += 1;
        }
    }
    for s in 0..C {
        let mut between = 1.0f32;
        for t in s + 1..C {
            cache[s][t] = between;
            if t + 1 < C {
                between *= gates[t];
                multiply_count += 1;
            }
        }
    }
    assert_eq!(multiply_count, 496);

    // The parent performs this work independently in every V-column thread:
    // 496 diagonal-prefix multiplies plus C(32, 3) = 4,960 between-token
    // multiplies. The candidate forms the shared coefficients once per CTA.
    let parent_prefix_multiplies: usize = (0..C).sum();
    let parent_between_multiplies: usize = (0..C)
        .flat_map(|t| (0..t).map(move |s| t.saturating_sub(s + 1)))
        .sum();
    assert_eq!(parent_prefix_multiplies, 496);
    assert_eq!(parent_between_multiplies, 4_960);
    let parent_per_thread = parent_prefix_multiplies + parent_between_multiplies;
    let parent_per_cta = parent_per_thread * 128;
    assert_eq!(parent_per_thread, 5_456);
    assert_eq!(parent_per_cta, 698_368);
    let saved_per_cta = parent_per_cta - multiply_count;
    assert_eq!(saved_per_cta, 697_872);
    assert_eq!(saved_per_cta * (8_192 / C), 178_655_232);
    assert_eq!(saved_per_cta * (8_192 / C) * 48, 8_575_451_136);

    for t in 0..C {
        let mut parent_prefix = 1.0f32;
        for &gate in gates.iter().take(t) {
            parent_prefix *= gate;
        }
        assert_eq!(cache[t][t].to_bits(), parent_prefix.to_bits());
        for s in 0..t {
            let mut parent_between = 1.0f32;
            for &gate in gates.iter().take(t).skip(s + 1) {
                parent_between *= gate;
            }
            assert_eq!(cache[s][t].to_bits(), parent_between.to_bits());
        }
    }
}

fn parent_warp_sum(mut lanes: [f32; 32]) -> f32 {
    for offset in [16, 8, 4, 2, 1] {
        let before = lanes;
        for lane in 0..offset {
            lanes[lane] = before[lane] + before[lane + offset];
        }
    }
    lanes[0]
}

#[test]
fn dot_batch_pair_order_and_four_warp_association_match_parent_bits() {
    const C: usize = 32;
    const K: usize = 128;
    let keys: [[f32; K]; C] = std::array::from_fn(|token| {
        std::array::from_fn(|dim| {
            let magnitude = 0x3e00u16 + ((token * 131 + dim * 17) % 0x180) as u16;
            let bits = if (token + dim).is_multiple_of(3) {
                magnitude | 0x8000
            } else {
                magnitude
            };
            f32::from_bits((bits as u32) << 16)
        })
    });

    let mut parent = [[0.0f32; C]; C];
    let mut batched_partials = [[0.0f32; 4]; 496];
    let mut pairs = [(0usize, 0usize); 496];
    let mut pair = 0;
    for i in 1..C {
        for j in 0..i {
            let warps: [f32; 4] = std::array::from_fn(|warp| {
                let lanes = std::array::from_fn(|lane| {
                    let dim = warp * 32 + lane;
                    keys[i][dim] * keys[j][dim]
                });
                parent_warp_sum(lanes)
            });
            parent[i][j] = warps[0] + warps[1] + warps[2] + warps[3];
            batched_partials[pair] = warps;
            pairs[pair] = (i, j);
            pair += 1;
        }
    }
    assert_eq!(pair, 496);

    let mut batched = [[0.0f32; C]; C];
    for (pair, &(i, j)) in pairs.iter().enumerate() {
        let partial = batched_partials[pair];
        batched[i][j] = partial[0] + partial[1] + partial[2] + partial[3];
    }
    for i in 1..C {
        for j in 0..i {
            assert_eq!(batched[i][j].to_bits(), parent[i][j].to_bits());
        }
    }
}

#[test]
fn packed_pair_map_matches_the_nested_strict_lower_order() {
    const C: usize = 32;
    const PAIRS: usize = C * (C - 1) / 2;
    let packed: [u16; PAIRS] = std::array::from_fn(|pair| {
        let mut i = 1usize;
        let mut row_start = 0usize;
        while pair >= row_start + i {
            row_start += i;
            i += 1;
        }
        let j = pair - row_start;
        ((i as u16) << 8) | j as u16
    });

    let mut pair = 0;
    for i in 1..C {
        for j in 0..i {
            assert_eq!(packed[pair], ((i as u16) << 8) | j as u16);
            pair += 1;
        }
    }
    assert_eq!(pair, PAIRS);
}

#[test]
fn cuda_shadow_caches_only_gate_coefficients_and_retains_wy_math() {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/common/gated_delta_rule_wy32_gatecache.cu"
    ));
    assert!(source.contains("smem_kd[t * C + t] = g_prefix"));
    assert!(source.contains("smem_kd[tid * C + t] = g_between"));
    assert!(source.contains("g_prefix *= smem_g[t]"));
    assert!(source.contains("g_between *= smem_g[t]"));
    assert!(!source.contains("g_prod_s"));
    assert!(source.contains("#define KD_PAIR_COUNT 496"));
    assert!(source.contains("smem_dot_partials = smem_bt + C"));
    assert!(source.contains("smem_pair_ij[pair] = (unsigned short)((i << 8) | j)"));
    assert!(source.contains("smem_dot_partials[pair_index * WARP_COUNT + warp_id] = partial"));
    assert!(source.contains("partial[0] + partial[1] + partial[2] + partial[3]"));
    assert!(source.contains("for (unsigned int pair = tid; pair < KD_PAIR_COUNT; pair += V_DIM)"));

    let input_load = source
        .split_once("for (unsigned int chunk_start")
        .expect("chunk loop is missing")
        .1
        .split_once("// Compute every strict-lower K-dot pair")
        .expect("dot-batch phase is missing")
        .0;
    assert!(!input_load.contains("__syncthreads();"));

    let dot_batch = source
        .split_once("// Compute every strict-lower K-dot pair")
        .expect("dot-batch phase is missing")
        .1
        .split_once("float hk_prev[C]")
        .expect("dot-batch phase boundary is missing")
        .0;
    assert_eq!(dot_batch.matches("__syncthreads();").count(), 2);
    assert!(!dot_batch.contains("gatecache_block_reduce"));

    let correction = source
        .split_once("float v_new_arr[C]")
        .expect("WY correction block is missing")
        .1
        .split_once("float o_out[C]")
        .expect("WY output block is missing")
        .0;
    assert!(correction.contains("smem_kd[t * C + t] * hk_prev[t]"));
    assert!(correction.contains("smem_kd[s * C + t] * smem_kd[t * C + s] * v_new_arr[s]"));
    assert!(!correction.contains("g_prefix"));
    assert!(!correction.contains("g_between"));

    assert!(source.contains("h_j = smem_g[t] * h_j + (float)smem_k[t * K_DIM + j] * v_new_arr[t]"));
    assert!(source.contains("o_out[t] += h_j * (float)smem_q[t * K_DIM + j]"));
    assert!(source.contains("__float2bfloat16(o_out[t] * inv_sqrt_d)"));
}
