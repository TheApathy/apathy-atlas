// SPDX-License-Identifier: AGPL-3.0-only

use std::cmp::Ordering;

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

#[path = "glm53_dsa_topk_visibility_tests.rs"]
mod visibility;

fn reference(
    scores: &[f32],
    pool_valid: &[bool],
    sequence_length: usize,
    tail_valid: &[bool; 3],
    query_position: usize,
    query_valid: bool,
) -> Vec<i32> {
    let mut output = vec![-1; OUTPUT_WIDTH as usize];
    if !query_valid || query_position >= sequence_length {
        return output;
    }
    let complete_pools = sequence_length / KPOOL as usize;
    let mut selected: Vec<usize> = (0..scores.len())
        .filter(|&pool| pool < complete_pools && pool_valid[pool] && pool * 4 + 3 <= query_position)
        .collect();
    selected.sort_by(|&left, &right| {
        if scores[left].is_nan() != scores[right].is_nan() {
            if scores[left].is_nan() {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        } else if scores[left] > scores[right] {
            Ordering::Less
        } else if scores[left] < scores[right] {
            Ordering::Greater
        } else {
            left.cmp(&right)
        }
    });
    selected.truncate(SELECTED_POOLS as usize);
    let tail_output = selected.len() * KPOOL as usize;
    for (rank, pool) in selected.into_iter().enumerate() {
        for offset in 0..KPOOL as usize {
            output[rank * KPOOL as usize + offset] = (pool * KPOOL as usize + offset) as i32;
        }
    }
    let visible_count = query_position + 1;
    let tail_count = visible_count % KPOOL as usize;
    let tail_start = visible_count - tail_count;
    for offset in 0..KPOOL as usize - 1 {
        let index = tail_start + offset;
        let pool = index / KPOOL as usize;
        let slot = index % KPOOL as usize;
        let valid = if pool < complete_pools {
            pool_valid[pool]
        } else {
            slot < tail_valid.len() && tail_valid[slot]
        };
        if offset < tail_count && index < sequence_length && valid {
            output[tail_output + offset] = index as i32;
        }
    }
    output
}

#[derive(Clone, Copy)]
struct Candidate {
    score: f32,
    pool: u32,
}

fn better(left: Candidate, right: Candidate) -> bool {
    if left.pool == u32::MAX || right.pool == u32::MAX {
        return left.pool != u32::MAX;
    }
    if left.score.is_nan() != right.score.is_nan() {
        return left.score.is_nan();
    }
    left.score > right.score || (left.score == right.score && left.pool < right.pool)
}

fn scalar_top512(scores: &[f32], valid: &[bool]) -> Vec<u32> {
    let mut values: Vec<_> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(pool, _)| valid[*pool])
        .map(|(pool, score)| Candidate {
            score,
            pool: pool as u32,
        })
        .collect();
    values.sort_by(|&left, &right| {
        if better(left, right) {
            Ordering::Less
        } else if better(right, left) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    values.truncate(SELECTED_POOLS as usize);
    values.into_iter().map(|value| value.pool).collect()
}

fn rolling_network(scores: &[f32], valid: &[bool]) -> Vec<u32> {
    let invalid = Candidate {
        score: 0.0,
        pool: u32::MAX,
    };
    let mut values = vec![invalid; 1_024];
    for chunk in (0..scores.len()).step_by(SELECTED_POOLS as usize) {
        for (item, slot) in values.iter_mut().enumerate().skip(512) {
            let pool = chunk + item - 512;
            *slot = if pool < scores.len() && valid[pool] {
                Candidate {
                    score: scores[pool],
                    pool: pool as u32,
                }
            } else {
                invalid
            };
        }
        let mut width = 2;
        while width <= values.len() {
            let mut stride = width / 2;
            while stride != 0 {
                for item in 0..values.len() {
                    let partner = item ^ stride;
                    if partner > item {
                        let descending = item & width == 0;
                        if (descending && better(values[partner], values[item]))
                            || (!descending && better(values[item], values[partner]))
                        {
                            values.swap(item, partner);
                        }
                    }
                }
                stride /= 2;
            }
            width *= 2;
        }
    }
    values[..512]
        .iter()
        .filter(|value| value.pool != u32::MAX)
        .map(|value| value.pool)
        .collect()
}

#[test]
fn plan_pins_exact_1m_geometry_and_extents() {
    let plan = Glm53DsaTopkPlan::new(1, MAX_QUERIES, 262_144, 1_048_576, 2_048, 4, true).unwrap();
    assert_eq!((plan.grid_y, plan.grid_z), (2_048, 1));
    assert_eq!(plan.score_bytes, 2_147_483_648);
    assert_eq!(plan.pool_validity_bytes, 262_144);
    assert_eq!(plan.sequence_length_bytes, 4);
    assert_eq!(plan.query_position_bytes, 8_192);
    assert_eq!(plan.query_validity_bytes, 2_048);
    assert_eq!(plan.tail_validity_bytes, 3);
    assert_eq!(plan.output_bytes, 16_801_792);
    assert_eq!((SELECTED_POOLS, OUTPUT_WIDTH), (512, 2_051));
    assert_eq!(
        Glm53DsaTopkPlan::new(70_000, 1, 1, 4, 2_048, 4, true)
            .unwrap()
            .grid_z,
        2
    );
    for bad in [
        Glm53DsaTopkPlan::new(0, 1, 1, 4, 2_048, 4, true),
        Glm53DsaTopkPlan::new(1, 0, 1, 4, 2_048, 4, true),
        Glm53DsaTopkPlan::new(1, MAX_QUERIES + 1, 1, 4, 2_048, 4, true),
        Glm53DsaTopkPlan::new(1, 1, 0, 4, 2_048, 4, true),
        Glm53DsaTopkPlan::new(1, 1, 2, 4, 2_048, 4, true),
        Glm53DsaTopkPlan::new(1, 1, 262_145, 1_048_576, 2_048, 4, true),
        Glm53DsaTopkPlan::new(1, 1, 1, 0, 2_048, 4, true),
        Glm53DsaTopkPlan::new(1, 1, 1, 1_048_577, 2_048, 4, true),
        Glm53DsaTopkPlan::new(1, 1, 1, 4, 2_047, 4, true),
        Glm53DsaTopkPlan::new(1, 1, 1, 4, 2_048, 4, false),
        Glm53DsaTopkPlan::new(u32::MAX, MAX_QUERIES, 1, 4, 2_048, 4, true),
    ] {
        assert!(bad.is_err());
    }
}

#[test]
fn reference_pins_ties_causality_tail_padding_and_score_preservation() {
    let scores = [9.0, 9.0, 100.0];
    let saved = scores;
    let valid = [true, true, false];
    let tail_valid = [true, true, false];
    let at_five = reference(&scores, &valid, 10, &tail_valid, 5, true);
    assert_eq!(&at_five[..8], &[0, 1, 2, 3, 4, 5, -1, -1]);
    assert!(at_five[8..].iter().all(|&x| x == -1));
    let at_seven = reference(&scores, &valid, 10, &tail_valid, 7, true);
    assert_eq!(&at_seven[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
    let high_index_tie_mutant = [4, 5, 6, 7, 0, 1, 2, 3];
    assert_ne!(&at_seven[..8], &high_index_tie_mutant);
    assert_eq!(scores, saved);
    assert!(
        reference(&scores, &valid, 10, &tail_valid, 7, false)
            .iter()
            .all(|&x| x == -1)
    );
    assert!(
        reference(&scores, &valid, 10, &tail_valid, 10, true)
            .iter()
            .all(|&x| x == -1)
    );
    let at_nine = reference(&scores, &valid, 10, &[true, false, false], 9, true);
    assert_eq!(&at_nine[8..11], &[8, -1, -1]);

    let many_scores = vec![1.0; 513];
    let many_valid = vec![true; 513];
    let capped = reference(&many_scores, &many_valid, 2_052, &[false; 3], 2_051, true);
    assert_eq!((capped[0], capped[2047]), (0, 2_047));
    assert!(!capped[..2048].contains(&2_048));
}

#[test]
fn late_winner_network_matches_scalar_through_max_pools() {
    for pools in [513, 4_097, 262_144] {
        let mut scores: Vec<_> = (0..pools).map(|pool| (pool % 97) as f32 - 48.0).collect();
        let mut valid: Vec<_> = (0..pools).map(|pool| pool % 17 != 0).collect();
        for pool in (0..pools).step_by(131) {
            scores[pool] = 7.0;
        }
        scores[0] = f32::NAN;
        valid[0] = false;
        scores[pools - 2] = f32::NAN;
        scores[pools - 1] = f32::NAN;
        valid[pools - 2] = true;
        valid[pools - 1] = true;
        let expected = scalar_top512(&scores, &valid);
        assert_eq!(&expected[..2], &[(pools - 2) as u32, (pools - 1) as u32]);
        assert_eq!(rolling_network(&scores, &valid), expected, "P={pools}");
    }
}

#[test]
fn boundary_positions_tail_validity_and_padding_are_exact() {
    let scores = [2.0, 1.0];
    let valid = [true, true];
    let tail = [true; 3];
    for (position, prefix) in [
        (0, &[0][..]),
        (2, &[0, 1, 2]),
        (3, &[0, 1, 2, 3]),
        (5, &[0, 1, 2, 3, 4, 5]),
        (7, &[0, 1, 2, 3, 4, 5, 6, 7]),
    ] {
        let output = reference(&scores, &valid, 8, &tail, position, true);
        assert_eq!(&output[..prefix.len()], prefix);
        assert!(output[prefix.len()..].iter().all(|&index| index == -1));
    }
    assert_eq!(
        &reference(&scores, &valid, 3, &[true, false, true], 2, true)[..4],
        &[0, -1, 2, -1]
    );
    for (position, query_valid) in [(0, false), (8, true)] {
        assert!(
            reference(&scores, &valid, 8, &tail, position, query_valid)
                .iter()
                .all(|&index| index == -1)
        );
    }
    let max_scores = vec![1.0; MAX_POOLS as usize];
    let max_valid = vec![true; MAX_POOLS as usize];
    let at_max = reference(
        &max_scores,
        &max_valid,
        MAX_POSITIONS as usize,
        &tail,
        MAX_POSITIONS as usize - 1,
        true,
    );
    assert_eq!(&at_max[..4], &[0, 1, 2, 3]);
    assert_eq!(at_max[2_050], -1);
    let before_max = reference(
        &max_scores,
        &max_valid,
        MAX_POSITIONS as usize - 1,
        &tail,
        MAX_POSITIONS as usize - 2,
        true,
    );
    assert_eq!(&before_max[2_048..], &[1_048_572, 1_048_573, 1_048_574]);
}

const CUDA_SEAMS: &[&str] = &[
    "return left_pool != GLM53_DSA_INVALID_POOL;",
    "return left_nan;",
    "return left_pool < right_pool;",
    "chunk += GLM53_DSA_SELECTED_POOLS",
    "lane + GLM53_DSA_SELECTED_POOLS;",
    "chunk + item - GLM53_DSA_SELECTED_POOLS",
    "const unsigned int partner = item ^ stride;",
    "const bool descending = (item & width) == 0U;",
    "(descending && partner_better) ||",
    "(!descending && item_better)",
    "const bool visible = pool_end <= query_position;",
    "output_indices[output_base + index] = -1;",
    "const unsigned int tail_output = selected_count * GLM53_DSA_KPOOL;",
    "const unsigned int tail_count = visible_count % GLM53_DSA_KPOOL;",
    "tail_validity[batch_index * (GLM53_DSA_KPOOL - 1U) + slot] != 0U",
    "offset < tail_count && index < sequence_length && valid",
    "pool * GLM53_DSA_KPOOL + offset",
];

fn cuda_contract(source: &str) -> bool {
    CUDA_SEAMS
        .iter()
        .all(|seam| source.matches(seam).count() == 1)
        && source.contains("#define GLM53_DSA_MAX_QUERIES 8192U")
        && source.contains("const float * __restrict__ scores")
        && !source.contains("scores[row * pool_count + pool] =")
}

#[test]
fn cuda_contract_rejects_each_network_causal_tail_and_sentinel_mutation() {
    const CUDA: &str =
        include_str!("../../../../../kernels/gb10/glm5.3-flash/iq3/glm53_dsa_topk.cu");
    assert!(cuda_contract(CUDA));
    for seam in CUDA_SEAMS {
        assert!(!cuda_contract(&CUDA.replacen(
            seam,
            "/* semantic mutant */",
            1
        )));
    }
}

#[test]
fn forged_extent_alignment_alias_and_overflow_fail_before_launch() {
    let gpu = MockGpuBackend::new();
    let kernel = Glm53DsaTopkKernel::load(&gpu).unwrap();
    let plan = Glm53DsaTopkPlan::new(1, 1, 1, 4, 2_048, 4, true).unwrap();
    let mut address = 0x10_0000u64;
    let mut next = |bytes: usize| {
        let buffer = GgmlIqBuffer {
            ptr: DevicePtr(address),
            bytes,
        };
        address += u64::try_from(bytes).unwrap().div_ceil(4) * 4 + 0x1000;
        buffer
    };
    let valid = Glm53DsaTopkBuffers {
        scores_f32: next(plan.score_bytes),
        pool_validity_u8: next(plan.pool_validity_bytes),
        sequence_lengths_u32: next(plan.sequence_length_bytes),
        query_positions_u32: next(plan.query_position_bytes),
        query_validity_u8: next(plan.query_validity_bytes),
        tail_validity_u8: next(plan.tail_validity_bytes),
        output_indices_i32: next(plan.output_bytes),
    };
    let mut forged = plan;
    forged.grid_z += 1;
    assert!(kernel.launch(&gpu, forged, valid, 0).is_err());
    assert!(
        kernel
            .launch(
                &gpu,
                plan,
                Glm53DsaTopkBuffers {
                    scores_f32: GgmlIqBuffer {
                        ptr: valid.scores_f32.ptr,
                        bytes: plan.score_bytes + 4,
                    },
                    ..valid
                },
                0,
            )
            .is_err()
    );
    assert!(
        kernel
            .launch(
                &gpu,
                plan,
                Glm53DsaTopkBuffers {
                    pool_validity_u8: GgmlIqBuffer {
                        ptr: DevicePtr::NULL,
                        bytes: plan.pool_validity_bytes,
                    },
                    ..valid
                },
                0,
            )
            .is_err()
    );
    assert!(
        kernel
            .launch(
                &gpu,
                plan,
                Glm53DsaTopkBuffers {
                    query_positions_u32: GgmlIqBuffer {
                        ptr: DevicePtr(valid.query_positions_u32.ptr.0 + 1),
                        bytes: plan.query_position_bytes,
                    },
                    ..valid
                },
                0,
            )
            .is_err()
    );
    assert!(
        kernel
            .launch(
                &gpu,
                plan,
                Glm53DsaTopkBuffers {
                    output_indices_i32: GgmlIqBuffer {
                        ptr: DevicePtr(u64::MAX - 1),
                        bytes: plan.output_bytes,
                    },
                    ..valid
                },
                0,
            )
            .is_err()
    );
    assert!(
        kernel
            .launch(
                &gpu,
                plan,
                Glm53DsaTopkBuffers {
                    output_indices_i32: GgmlIqBuffer {
                        ptr: valid.scores_f32.ptr,
                        bytes: plan.output_bytes,
                    },
                    ..valid
                },
                0,
            )
            .is_err()
    );
    assert_eq!(gpu.launch_count(), 0);
    kernel.launch(&gpu, plan, valid, 0).unwrap();
    assert_eq!(gpu.launch_count(), 1);
}
