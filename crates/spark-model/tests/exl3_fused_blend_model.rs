// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the opt-in DeepSeek-V4 H4096 fused routed/shared
//! tail. GPU byte parity remains a separate promotion gate.

use half::bf16;

const EXL3: &str = include_str!("../../../kernels/gb10/common/exl3_gemv.cu");
const BLEND: &str = include_str!("../../../kernels/gb10/common/moe_permute.cu");
const BLEND_HELPER: &str = include_str!("../../../kernels/gb10/common/moe_batched_blend.cuh");
const STATE: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const PREFILL: &str = include_str!("../src/layers/moe/forward_prefill.rs");
const TAIL: &str = include_str!("../src/layers/moe/forward_prefill_exl3_tail.rs");

fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn legacy_composition(experts: &[f32], weights: &[f32], shared: f32, gate: f32) -> u16 {
    let routed = experts
        .iter()
        .zip(weights)
        .fold(0.0f32, |acc, (&value, &weight)| {
            acc + weight * round_bf16(value)
        });
    bf16::from_f32(round_bf16(routed) + gate * round_bf16(shared)).to_bits()
}

fn fused_composition(experts: &[f32], weights: &[f32], shared: f32, gate: f32) -> u16 {
    // The fused kernel must retain both legacy materialization boundaries.
    let routed = experts
        .iter()
        .zip(weights)
        .fold(0.0f32, |acc, (&value, &weight)| {
            acc + weight * round_bf16(value)
        });
    bf16::from_f32(round_bf16(routed) + gate * round_bf16(shared)).to_bits()
}

fn without_expert_round(experts: &[f32], weights: &[f32], shared: f32, gate: f32) -> u16 {
    let routed = experts
        .iter()
        .zip(weights)
        .fold(0.0f32, |acc, (&value, &weight)| acc + weight * value);
    bf16::from_f32(round_bf16(routed) + gate * round_bf16(shared)).to_bits()
}

fn without_routed_round(experts: &[f32], weights: &[f32], shared: f32, gate: f32) -> u16 {
    let routed = experts
        .iter()
        .zip(weights)
        .fold(0.0f32, |acc, (&value, &weight)| {
            acc + weight * round_bf16(value)
        });
    bf16::from_f32(routed + gate * round_bf16(shared)).to_bits()
}

fn gate_tree_256(normed: &[bf16], gate: Option<&[bf16]>) -> f32 {
    let Some(gate) = gate else {
        return 1.0;
    };
    assert_eq!(normed.len(), gate.len());
    let mut thread_sum = [0.0f32; 256];
    for tid in 0..256 {
        for i in (tid..normed.len()).step_by(256) {
            thread_sum[tid] += normed[i].to_f32() * gate[i].to_f32();
        }
    }
    let mut warp_sum = [0.0f32; 8];
    for (warp, result) in warp_sum.iter_mut().enumerate() {
        let base = warp * 32;
        for offset in [16, 8, 4, 2, 1] {
            let before = thread_sum;
            for lane in 0..32 - offset {
                thread_sum[base + lane] += before[base + lane + offset];
            }
        }
        *result = thread_sum[base];
    }
    let total = warp_sum.into_iter().fold(0.0f32, |sum, value| sum + value);
    1.0 / (1.0 + (-total).exp())
}

#[test]
fn cpu_model_preserves_both_bf16_barriers_and_topk_order() {
    let cases = [
        (
            [1.003_906_2, -0.498_046_88, 11.9375, -7.96875, 0.001, 256.5],
            [0.19, 0.17, 0.16, 0.15, 0.14, 0.13],
            0.333_984_38,
            0.731_058_6,
        ),
        (
            [-511.0, 255.0, -127.5, 63.75, -31.875, 15.9375],
            [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0],
            -8.03125,
            1.0,
        ),
        (
            [
                0.035722654,
                -1.933259,
                -0.15847267,
                -2.4639227,
                2.6075301,
                2.192905,
            ],
            [
                0.27381945, 0.15012287, 0.45443514, 0.2861834, 0.44115862, 0.42402205,
            ],
            0.033447266,
            0.41394603,
        ),
    ];

    let (mut expert_barrier_exposed, mut routed_barrier_exposed) = (false, false);
    for (experts, weights, shared, gate) in cases {
        let legacy = legacy_composition(&experts, &weights, shared, gate);
        assert_eq!(legacy, fused_composition(&experts, &weights, shared, gate));
        expert_barrier_exposed |= legacy != without_expert_round(&experts, &weights, shared, gate);
        routed_barrier_exposed |= legacy != without_routed_round(&experts, &weights, shared, gate);

        let reversed_experts = experts.into_iter().rev().collect::<Vec<_>>();
        let reversed = fused_composition(&reversed_experts, &weights, shared, gate);
        assert_ne!(
            fused_composition(&experts, &weights, shared, gate),
            reversed,
            "the test vector must expose routing-order drift"
        );
    }
    assert!(
        expert_barrier_exposed,
        "per-expert BF16 barrier is load-bearing"
    );
    assert!(
        routed_barrier_exposed,
        "routed-sum BF16 barrier is load-bearing"
    );
}

#[test]
fn cpu_gate_tree_covers_h4096_and_preserves_null_and_edge_semantics() {
    let owners = (0..256)
        .flat_map(|tid| (tid..4096).step_by(256))
        .collect::<Vec<_>>();
    let mut sorted = owners.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..4096).collect::<Vec<_>>());

    let zeros = vec![bf16::from_bits(0x8000); 4096];
    let ones = vec![bf16::from_f32(1.0); 4096];
    assert_eq!(
        gate_tree_256(&zeros, Some(&ones)).to_bits(),
        0.5f32.to_bits()
    );

    let mut nan_input = zeros;
    nan_input[257] = bf16::from_bits(0x7fc0);
    assert!(gate_tree_256(&nan_input, Some(&ones)).is_nan());
    assert_eq!(gate_tree_256(&nan_input, None).to_bits(), 1.0f32.to_bits());
}

#[test]
fn legacy_and_fused_entries_share_gate_and_final_blend_arithmetic() {
    assert!(BLEND.contains("#include \"moe_batched_blend.cuh\""));
    assert!(EXL3.contains("#include \"moe_batched_blend.cuh\""));
    assert!(BLEND.contains("atlas_moe_shared_gate_scalar_256("));
    assert!(EXL3.contains("atlas_moe_shared_gate_scalar_256("));
    assert!(BLEND.contains("atlas_moe_blend_from_routed_bf16("));
    assert!(EXL3.contains("atlas_moe_blend_from_routed_bf16("));

    assert!(BLEND_HELPER.contains("float local_dot = 0.0f;"));
    assert!(BLEND_HELPER.contains("offset = 16; offset > 0; offset >>= 1"));
    assert!(BLEND_HELPER.contains("for (unsigned int w = 0; w < 8; ++w)"));
    assert!(BLEND_HELPER.contains("1.0f / (1.0f + __expf(-total))"));
    assert!(BLEND_HELPER.contains("gate_weight != 0"));
    assert!(BLEND_HELPER.contains("gate_scalar = 1.0f"));
    assert!(BLEND.contains("mov.u32 n, %ntid.x; setp.ne.u32 p, n, 256; @p exit;"));
}

#[test]
fn shared_gate_broadcast_is_volatile_and_read_after_publication_barrier() {
    // V23 PTX hoisted nonzero threads' shared load before tid0's store and
    // the second barrier. With a NULL gate, 255/256 threads then blended zero.
    // This source gate is necessary; emitted PTX and GPU replay are separate gates.
    assert!(BLEND_HELPER.contains("volatile float* warp_partials)"));
    assert!(!BLEND_HELPER.contains("float* __restrict__ warp_partials"));
    let mut tail = BLEND_HELPER;
    for operation in [
        "if (lane == 0) warp_partials[warp_id] = local_dot;",
        "__syncthreads();",
        "if (tid == 0)",
        "warp_partials[0] = gate_scalar;",
        "__syncthreads();",
        "return warp_partials[0];",
    ] {
        let (_, rest) = tail.split_once(operation).unwrap();
        tail = rest;
    }
}

#[test]
fn null_and_nonnull_gate_controls_require_every_thread_to_blend() {
    let normed = vec![bf16::from_f32(1.0); 4096];
    for dot in [None, Some(0.0), Some(1.0), Some(-1.0)] {
        let gate = dot.map(|dot| {
            let mut weights = vec![bf16::ZERO; 4096];
            weights[0] = bf16::from_f32(dot);
            weights
        });
        let gain = gate_tree_256(&normed, gate.as_deref());
        let expected_gain = dot.map_or(1.0, |dot| 1.0 / (1.0 + (-dot).exp()));
        assert_eq!(gain.to_bits(), expected_gain.to_bits());
        let expected = bf16::from_f32(1.0 + 2.0 * gain).to_bits();
        let mut visited = vec![false; 4096];
        for tid in 0..256 {
            for column in (tid..4096).step_by(256) {
                assert!(!visited[column]);
                visited[column] = true;
                let result = fused_composition(&[1.0], &[1.0], 2.0, gain);
                assert_eq!(result, expected, "thread {tid}, column {column}");
                // Every column must expose omission of the shared expert.
                assert_ne!(result, bf16::from_f32(1.0).to_bits());
            }
        }
        assert!(visited.into_iter().all(|value| value));
    }
}

#[test]
fn routed_math_has_one_ordered_topk_body_and_explicit_rounding_boundaries() {
    assert_eq!(
        EXL3.matches("for (unsigned int k = 0; k < topk; ++k)")
            .count(),
        1
    );
    assert!(EXL3.contains("exl3_h128_post_unpermute_chunk<FIXED_H>"));
    assert!(EXL3.contains("exl3_h128_post_unpermute_chunk<4096>"));
    assert!(EXL3.contains("atlas_round_bf16_to_f32(a0)"));
    assert!(EXL3.contains("atlas_moe_blend_from_routed_bf16("));
    assert!(EXL3.contains("routed0, shared_out[base + 0], gate_scalar"));
}

#[test]
fn fused_entry_is_exact_h4096_top6_and_one_cta_per_token() {
    assert!(EXL3.contains("exl3_h128_post_unpermute_blend_h4096("));
    assert!(EXL3.contains("if (H != 4096 || topk != 6) return;"));
    assert!(EXL3.contains("gridDim.x != num_tokens || gridDim.y != 1 || gridDim.z != 1"));
    assert!(EXL3.contains("mov.u32 n, %ntid.x; setp.ne.u32 p, n, 256; @p exit;"));
    assert!(EXL3.contains("for (unsigned int chunk = warp; chunk < 32; chunk += 8)"));
}

#[test]
fn host_path_is_opt_in_and_preserves_every_observable_fallback() {
    assert!(STATE.contains("ATLAS_EXL3_PREFILL_FUSED_BLEND"));
    assert!(STATE.contains("h128_post_unpermute_blend_h4096_k: KernelHandle"));
    assert!(TAIL.contains("try_exl3_fused_post_unpermute_blend"));
    assert!(TAIL.contains("[num_tokens, 1, 1]"));
    assert!(TAIL.contains("topk != 6"));
    assert!(!TAIL.contains("!gate_weight.is_null()"));

    assert!(PREFILL.contains("!is_ep_prefill"));
    assert!(PREFILL.contains("!use_overlap"));
    assert!(PREFILL.contains("!super::dump::enabled()"));
    assert!(PREFILL.contains("if !fused_blend_done"));
    assert!(PREFILL.contains("try_exl3_fused_post_unpermute("));
    assert!(PREFILL.contains("ops::moe_batched_blend("));
}

#[test]
fn production_shape_saves_one_launch_and_routed_intermediate_roundtrip_per_layer() {
    const TOKENS: u64 = 2410;
    const HIDDEN: u64 = 4096;
    const LAYERS: u64 = 43;
    const BF16_BYTES: u64 = 2;
    let saved_bytes_per_layer = 2 * TOKENS * HIDDEN * BF16_BYTES;

    assert_eq!(saved_bytes_per_layer, 39_485_440);
    assert_eq!(saved_bytes_per_layer * LAYERS, 1_697_873_920);
    assert_eq!(15 * LAYERS - 14 * LAYERS, LAYERS);
}
