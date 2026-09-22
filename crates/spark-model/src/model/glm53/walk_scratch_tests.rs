// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const TRANSIENT: u64 = 0x3_0000_0000;

fn scratch() -> Glm53WalkScratch {
    Glm53WalkScratch::bind(DevicePtr(TRANSIENT), Glm53WalkScratch::required_bytes()).unwrap()
}

/// Every buffer must carry the exact extent the op validates against, or the
/// op refuses at launch and the walk cannot run at all.
#[test]
fn every_buffer_carries_the_extent_the_serial_moe_op_requires() {
    let s = scratch();
    let input = GgmlIqBuffer {
        ptr: DevicePtr(0x9000_0000),
        bytes: 8_192,
    };
    let output = GgmlIqBuffer {
        ptr: DevicePtr(0x9001_0000),
        bytes: 8_192,
    };
    let b = s.moe_buffers(input, output);
    for (name, buffer, expected) in [
        ("input", b.input_bf16, 8_192),
        ("route ids", b.route_ids_u32, 32),
        ("route weights", b.route_weights_f32, 32),
        ("q8", b.q8_activation, 4_608),
        ("expert gate", b.expert_gate_bf16, 4_096),
        ("expert up", b.expert_up_bf16, 4_096),
        ("expert swiglu", b.expert_swiglu_bf16, 4_096),
        ("routed", b.routed_bf16, 65_536),
        ("shared gate", b.shared_gate_bf16, 4_096),
        ("shared up", b.shared_up_bf16, 4_096),
        ("shared swiglu", b.shared_swiglu_bf16, 4_096),
        ("shared", b.shared_bf16, 8_192),
        ("output", b.output_bf16, 8_192),
    ] {
        assert_eq!(buffer.bytes, expected, "{name} extent");
        assert!(!buffer.ptr.is_null(), "{name} is NULL");
    }
}

/// Overlap is the corruption class that yields fluent, wrong output rather than
/// a crash, and the op refuses any pair that aliases.
#[test]
fn scratch_buffers_never_overlap_each_other_or_the_walk_buffers() {
    let s = scratch();
    // Stand-ins for the workspace `collapsed` / `hidden_b`, placed far away.
    let input = GgmlIqBuffer {
        ptr: DevicePtr(0x9000_0000),
        bytes: 8_192,
    };
    let output = GgmlIqBuffer {
        ptr: DevicePtr(0x9001_0000),
        bytes: 8_192,
    };
    let b = s.moe_buffers(input, output);
    let spans: Vec<(u64, u64)> = [
        b.input_bf16,
        b.route_ids_u32,
        b.route_weights_f32,
        b.q8_activation,
        b.expert_gate_bf16,
        b.expert_up_bf16,
        b.expert_swiglu_bf16,
        b.routed_bf16,
        b.shared_gate_bf16,
        b.shared_up_bf16,
        b.shared_swiglu_bf16,
        b.shared_bf16,
        b.output_bf16,
    ]
    .iter()
    .map(|x| (x.ptr.0, x.ptr.0 + x.bytes as u64))
    .collect();
    for left in 0..spans.len() {
        for right in left + 1..spans.len() {
            let (l, r) = (spans[left], spans[right]);
            assert!(
                !(l.0 < r.1 && r.0 < l.1),
                "buffers {left} and {right} overlap: {l:?} vs {r:?}"
            );
        }
    }
}

/// Scratch is a dedicated allocation. The arena's transient region has exactly
/// 1,048,576 spare bytes, which is precisely the DSA score buffer at full 1M
/// context, so there was never room there for the working set.
#[test]
fn scratch_fits_its_own_allocation_which_the_arena_could_not_supply() {
    let s = scratch();
    let b = s.moe_buffers(
        GgmlIqBuffer {
            ptr: DevicePtr(0x9000_0000),
            bytes: 8_192,
        },
        GgmlIqBuffer {
            ptr: DevicePtr(0x9001_0000),
            bytes: 8_192,
        },
    );
    let end = TRANSIENT + Glm53WalkScratch::required_bytes();
    for buffer in [
        b.route_ids_u32,
        b.q8_activation,
        b.routed_bf16,
        b.shared_bf16,
    ] {
        assert!(buffer.ptr.0 >= TRANSIENT);
        assert!(buffer.ptr.0 + buffer.bytes as u64 <= end);
    }
    // Scratch is its own allocation, not a slice of the arena: the transient
    // region's 1,048,576 spare bytes are exactly the DSA score buffer, leaving
    // nothing for working buffers.
    assert!(
        Glm53WalkScratch::required_bytes() > 1_048_576,
        "the working set must exceed what the transient region could offer"
    );
    assert_eq!(
        Glm53WalkScratch::required_bytes(),
        Glm53WalkScratch::used_bytes()
    );
    assert!(
        Glm53WalkScratch::required_bytes() <= 6 * 1_024 * 1_024 * 1_024,
        "M2048 scratch must remain within its bounded six-GiB residency budget"
    );
}

#[test]
fn bind_refuses_a_null_base_or_an_undersized_allocation() {
    let need = Glm53WalkScratch::required_bytes();
    assert!(Glm53WalkScratch::bind(DevicePtr(0), need).is_err());
    assert!(Glm53WalkScratch::bind(DevicePtr(TRANSIENT), need - 1).is_err());
    assert!(Glm53WalkScratch::bind(DevicePtr(TRANSIENT + 1), need).is_err());
    assert!(Glm53WalkScratch::bind(DevicePtr(TRANSIENT), need).is_ok());
}

#[test]
fn exl3_projection_scratch_is_appended_and_exact() {
    let s = scratch();
    let x = s.exl3_projection_scratch();
    assert_eq!(x.input_f16.bytes, GLM53_EXL3_MAX_INPUT_F16_BYTES);
    assert_eq!(x.output_f16.bytes, GLM53_EXL3_MAX_OUTPUT_F16_BYTES);
    assert_eq!(x.locks_i32.bytes, GLM53_EXL3_LOCK_BYTES);
    assert_eq!(x.input_hadamard_f16.bytes, GLM53_EXL3_MAX_INPUT_F16_BYTES);
    let spans = [x.input_f16, x.output_f16, x.locks_i32, x.input_hadamard_f16]
        .map(|buffer| (buffer.ptr.0, buffer.ptr.0 + buffer.bytes as u64));
    for left in 0..spans.len() {
        for right in left + 1..spans.len() {
            assert!(spans[left].1 <= spans[right].0 || spans[right].1 <= spans[left].0);
        }
    }
}

#[test]
fn fused_exl3_moe_has_dedicated_full_abi_lock_storage() {
    let s = scratch();
    let projection = s.exl3_projection_scratch();
    let moe = s.exl3_moe_scratch();
    assert_eq!(moe.locks_i32.bytes, GLM53_EXL3_MOE_LOCK_BYTES);
    assert_eq!(moe.locks_i32.bytes, 4_202_760);
    assert!(
        projection.locks_i32.ptr.0 + projection.locks_i32.bytes as u64 <= moe.locks_i32.ptr.0
            || moe.locks_i32.ptr.0 + moe.locks_i32.bytes as u64 <= projection.locks_i32.ptr.0
    );
}

#[test]
fn k8_exl3_prefixes_remain_exact_append_only_and_disjoint() {
    let s = scratch();
    let t1 = s.exl3_projection_scratch();
    let wide = s.exl3_projection_scratch_rows(8).unwrap();
    assert_eq!(wide.input_f16.bytes, GLM53_EXL3_MAX_WIDE_INPUT_F16_BYTES);
    assert_eq!(wide.output_f16.bytes, GLM53_EXL3_MAX_WIDE_OUTPUT_F16_BYTES);
    assert_eq!(wide.locks_i32.bytes, GLM53_EXL3_LOCK_BYTES);
    assert_eq!(
        wide.input_hadamard_f16.bytes,
        GLM53_EXL3_MAX_WIDE_INPUT_F16_BYTES
    );
    assert!(
        t1.input_hadamard_f16.ptr.0 + t1.input_hadamard_f16.bytes as u64 <= wide.input_f16.ptr.0
    );

    let input = GgmlIqBuffer {
        ptr: DevicePtr(0x9000_0000),
        bytes: 8 * 4_096 * 2,
    };
    let output = GgmlIqBuffer {
        ptr: DevicePtr(0x9002_0000),
        bytes: 8 * 4_096 * 2,
    };
    let (logits, ids, weights) = s.router_scratch_rows(8).unwrap();
    let (probs, biased) = s.router_scores_rows(8).unwrap();
    assert_eq!(logits.bytes, 8 * 288 * 4);
    assert_eq!(ids.bytes, 8 * 8 * 4);
    assert_eq!(weights.bytes, 8 * 8 * 4);
    assert_eq!(probs.bytes, 8 * 288 * 4);
    assert_eq!(biased.bytes, 8 * 288 * 4);

    let dense = s.dense_buffers_rows(8).unwrap();
    assert_eq!(dense.gate_bf16.bytes, 8 * 12_288 * 2);
    assert_eq!(dense.up_bf16.bytes, 8 * 12_288 * 2);
    assert_eq!(dense.swiglu_bf16.bytes, 8 * 12_288 * 2);

    let moe = s.moe_buffers_rows(8, input, output).unwrap();
    assert_eq!(moe.input_bf16.bytes, 8 * 4_096 * 2);
    assert_eq!(moe.route_ids_u32.bytes, 8 * 8 * 4);
    assert_eq!(moe.route_weights_f32.bytes, 8 * 8 * 4);
    assert_eq!(moe.shared_gate_bf16.bytes, 8 * 2_048 * 2);
    assert_eq!(moe.shared_bf16.bytes, 8 * 4_096 * 2);
    assert_eq!(moe.output_bf16.bytes, 8 * 4_096 * 2);
}

#[test]
fn m2048_layer_major_scratch_exposes_exact_prefixes() {
    let s = scratch();
    let rows = MAX_WIDE_ROWS;
    let input = GgmlIqBuffer {
        ptr: DevicePtr(0x9000_0000),
        bytes: rows as usize * 4_096 * 2,
    };
    let output = GgmlIqBuffer {
        ptr: DevicePtr(0x9020_0000),
        bytes: rows as usize * 4_096 * 2,
    };

    let (logits, ids, weights) = s.router_scratch_rows(rows).unwrap();
    assert_eq!(logits.bytes, rows as usize * 288 * 4);
    assert_eq!(ids.bytes, rows as usize * 8 * 4);
    assert_eq!(weights.bytes, rows as usize * 8 * 4);
    let dense = s.dense_buffers_rows(rows).unwrap();
    assert_eq!(dense.gate_bf16.bytes, rows as usize * 12_288 * 2);
    let moe = s.moe_buffers_rows(rows, input, output).unwrap();
    assert_eq!(moe.shared_bf16.bytes, rows as usize * 4_096 * 2);
    let kda = s.kda_buffers_rows(rows).unwrap();
    assert_eq!(kda.combined_qkv_bf16.bytes, rows as usize * 3 * 16_384);
    let dsa = s.dsa_buffers_rows(rows).unwrap();
    assert_eq!(dsa.scores_f32.bytes, rows as usize * 1_048_576);
    assert_eq!(dsa.query_positions_u32.bytes, rows as usize * 4);
    assert_eq!(dsa.pool_keys_bf16.bytes, 512 * 128 * 2);
    assert_eq!(dsa.pool_validity_u8.bytes, 512);

    let fused = s.exl3_moe_scratch();
    assert_eq!(fused.token_sorted_i64.bytes, rows as usize * 8 * 8);
    assert_eq!(fused.temp_state_g_f16.bytes, 8 * rows as usize * 4_096 * 2);
    assert_eq!(fused.output_f32.bytes, rows as usize * 4_096 * 4);
    assert_eq!(fused.route_private_f32.bytes, 8 * 8 * 4_096 * 4);
    assert_eq!(fused.pair_expert_u32.bytes, rows as usize * 8 * 4);
    assert_eq!(
        fused.chunk_expert_u32.bytes,
        (rows as usize * 8).div_ceil(16) * 4 + 288 * 4
    );
    assert_eq!(fused.chunk_start_u32.bytes, fused.chunk_expert_u32.bytes);
    assert_eq!(fused.chunk_rows_u32.bytes, fused.chunk_expert_u32.bytes);
    assert_eq!(fused.chunk_count_u32.bytes, 4);
}

#[test]
fn wide_scratch_rejects_rows_outside_m2048() {
    let s = scratch();
    assert!(s.exl3_projection_scratch_rows(0).is_err());
    assert!(s.exl3_projection_scratch_rows(MAX_WIDE_ROWS + 1).is_err());
    assert!(s.router_scratch_rows(MAX_WIDE_ROWS + 1).is_err());
    assert!(s.dense_buffers_rows(MAX_WIDE_ROWS + 1).is_err());
    assert!(s.kda_buffers_rows(MAX_WIDE_ROWS + 1).is_err());
    assert!(s.dsa_buffers_rows(MAX_WIDE_ROWS + 1).is_err());
}
