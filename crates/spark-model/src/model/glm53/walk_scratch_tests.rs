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
}

#[test]
fn bind_refuses_a_null_base_or_an_undersized_allocation() {
    let need = Glm53WalkScratch::required_bytes();
    assert!(Glm53WalkScratch::bind(DevicePtr(0), need).is_err());
    assert!(Glm53WalkScratch::bind(DevicePtr(TRANSIENT), need - 1).is_err());
    assert!(Glm53WalkScratch::bind(DevicePtr(TRANSIENT + 1), need).is_err());
    assert!(Glm53WalkScratch::bind(DevicePtr(TRANSIENT), need).is_ok());
}
