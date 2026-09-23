// SPDX-License-Identifier: AGPL-3.0-only

//! Exercise the real scratch binder without allocating device memory.

#[test]
fn verify_schedule_is_latched_without_changing_any_arena_binding() {
    let base_policy = policy("1", "1");
    let required = Glm53WalkScratch::required_bytes_with_route_policy(base_policy);
    let old =
        Glm53WalkScratch::bind_with_route_policy(DevicePtr(BASE), required, base_policy).unwrap();
    for width in ["2", "4", "8"] {
        let p = Glm53Exl3RoutePolicy::parse_with_verify_group(
            Some(OsStr::new("1")),
            Some(OsStr::new("1")),
            Some(OsStr::new(width)),
            Some(OsStr::new("1")),
        )
        .unwrap();
        assert_eq!(
            Glm53WalkScratch::required_bytes_with_route_policy(p),
            required
        );
        let got = Glm53WalkScratch::bind_with_route_policy(DevicePtr(BASE), required, p).unwrap();
        assert_eq!(got.exl3_route_policy(), p);
        for (left, right) in all_bindings(&got).into_iter().zip(all_bindings(&old)) {
            assert_eq!(span(left), span(right));
        }
        let scratch = got.exl3_moe_scratch();
        assert!(scratch.temp_state_g_f16.bytes >= 24 * 8 * 4096 * 2);
        assert!(scratch.temp_state_u_f16.bytes >= 24 * 8 * 4096 * 2);
        assert!(scratch.temp_intermediate_g_f16.bytes >= 24 * 8 * 2048 * 2);
        assert!(scratch.temp_intermediate_u_f16.bytes >= 24 * 8 * 2048 * 2);
    }
}

use super::*;
use crate::layers::ops::glm53_exl3_route_policy::Glm53Exl3RoutePolicy;

#[test]
fn verify_staged_k32_preserves_every_owner_allocation_and_binding() {
    let base = Glm53Exl3RoutePolicy::parse(Some(std::ffi::OsStr::new("1")), None).unwrap();
    let selected = base
        .with_verify_staged_k32(
            Some(std::ffi::OsStr::new("1")),
            Some(std::ffi::OsStr::new("1")),
        )
        .unwrap();
    let bytes = Glm53WalkScratch::required_bytes_with_route_policy(base);
    assert_eq!(
        bytes,
        Glm53WalkScratch::required_bytes_with_route_policy(selected)
    );
    let old = Glm53WalkScratch::bind_with_route_policy(DevicePtr(BASE), bytes, base).unwrap();
    let new = Glm53WalkScratch::bind_with_route_policy(DevicePtr(BASE), bytes, selected).unwrap();
    assert_eq!(new.exl3_route_policy(), selected);
    for (a, b) in all_bindings(&old).into_iter().zip(all_bindings(&new)) {
        assert_eq!((a.ptr, a.bytes), (b.ptr, b.bytes));
    }
}
use std::ffi::OsStr;

const BASE: u64 = 0x3_0000_0000;
const MIB: u64 = 1024 * 1024;

fn policy(private: &str, prefill: &str) -> Glm53Exl3RoutePolicy {
    Glm53Exl3RoutePolicy::parse(Some(OsStr::new(private)), Some(OsStr::new(prefill))).unwrap()
}

// These are actual bound fields, not a second implementation of placement math.
fn all_bindings(s: &Glm53WalkScratch) -> Vec<GgmlIqBuffer> {
    let mut b = vec![
        s.router_logits_f32,
        s.route_ids_u32,
        s.route_weights_f32,
        s.q8_activation,
        s.expert_gate_bf16,
        s.expert_up_bf16,
        s.expert_swiglu_bf16,
        s.routed_bf16,
        s.shared_gate_bf16,
        s.shared_up_bf16,
        s.shared_swiglu_bf16,
        s.shared_bf16,
        s.dense_q8_activation,
        s.dense_gate_bf16,
        s.dense_up_bf16,
        s.dense_swiglu_bf16,
    ];
    b.extend(s.kda);
    b.extend(s.dsa);
    b.extend([s.router_probs_f32, s.router_biased_f32]);
    b.extend([
        s.grouped_gate_bf16,
        s.grouped_up_bf16,
        s.grouped_swiglu_bf16,
        s.grouped_down_q8,
    ]);
    b.extend(s.exl3);
    b.extend(s.exl3_moe);
    b.extend(s.exl3_wide);
    b.extend(s.exl3_wide_dense);
    b.extend(s.exl3_wide_router_moe);
    b.extend(s.exl3_wide_kda);
    b.extend(s.exl3_wide_dsa);
    b.extend(s.exl3_moe_staged);
    assert_eq!(b.len(), 138);
    b
}

fn span(buffer: GgmlIqBuffer) -> (u64, usize) {
    (buffer.ptr.0, buffer.bytes)
}

#[test]
fn disabled_prefill_preserves_every_legacy_binding_and_allocated_byte() {
    let required = Glm53WalkScratch::required_bytes();
    let old = Glm53WalkScratch::bind(DevicePtr(BASE), required).unwrap();
    let old_bindings = all_bindings(&old);
    assert_eq!(old_bindings[137].bytes as u64, MIB);
    assert_eq!(old_bindings[137].ptr.0 + MIB, BASE + required);
    for p in [policy("0", "0"), policy("1", "0")] {
        assert_eq!(
            Glm53WalkScratch::required_bytes_with_route_policy(p),
            required
        );
        let got = Glm53WalkScratch::bind_with_route_policy(DevicePtr(BASE), required, p).unwrap();
        assert_eq!(got.exl3_route_policy(), p);
        for (index, (left, right)) in all_bindings(&got)
            .into_iter()
            .zip(&old_bindings)
            .enumerate()
        {
            assert_eq!(span(left), span(*right), "legacy slot {index}");
        }
    }
}

#[test]
fn selected_prefill_grows_only_the_final_slot_by_exactly_255_mib() {
    let p = policy("1", "1");
    let old_required = Glm53WalkScratch::required_bytes();
    let required = Glm53WalkScratch::required_bytes_with_route_policy(p);
    assert_eq!(required, old_required + 255 * MIB);
    let old = Glm53WalkScratch::bind(DevicePtr(BASE), old_required).unwrap();
    let got = Glm53WalkScratch::bind_with_route_policy(DevicePtr(BASE), required, p).unwrap();
    let old_bindings = all_bindings(&old);
    let new_bindings = all_bindings(&got);
    for index in 0..137 {
        assert_eq!(
            span(new_bindings[index]),
            span(old_bindings[index]),
            "slot {index}"
        );
    }
    assert_eq!(new_bindings[137].ptr, old_bindings[137].ptr);
    assert_eq!(new_bindings[137].bytes as u64, 256 * MIB);
    assert_eq!(new_bindings[137].ptr.0 + 256 * MIB, BASE + required);
    assert_eq!(got.exl3_route_policy(), p);
    let selected = got.exl3_moe_scratch().route_private_f32;
    assert_eq!((selected.ptr.0, selected.bytes), span(new_bindings[137]));
}

#[test]
fn selected_slot_is_disjoint_bounded_and_covers_each_required_row_prefix() {
    let p = policy("1", "1");
    let required = Glm53WalkScratch::required_bytes_with_route_policy(p);
    let got = Glm53WalkScratch::bind_with_route_policy(DevicePtr(BASE), required, p).unwrap();
    let bindings = all_bindings(&got);
    for (index, b) in bindings.iter().enumerate() {
        assert!(b.ptr.0 >= BASE);
        assert!(b.ptr.0.is_multiple_of(256), "alignment slot {index}");
        assert!(b.ptr.0.checked_add(b.bytes as u64).unwrap() <= BASE + required);
    }
    for left in 0..bindings.len() {
        for right in left + 1..bindings.len() {
            let a = bindings[left];
            let b = bindings[right];
            assert!(
                a.ptr.0 + a.bytes as u64 <= b.ptr.0 || b.ptr.0 + b.bytes as u64 <= a.ptr.0,
                "overlap slots {left}/{right}"
            );
        }
    }
    let route = got.exl3_moe_scratch().route_private_f32;
    for rows in [8, 9, 106, 1023, 1024, 2038, 2048] {
        let bytes = p.private_bytes(rows).unwrap();
        assert!(bytes <= route.bytes);
        assert!(route.ptr.0 + bytes as u64 <= BASE + required);
    }
}

#[test]
fn conditional_bind_rejects_short_null_misaligned_and_overflowing_allocations() {
    for p in [policy("0", "0"), policy("1", "0"), policy("1", "1")] {
        let need = Glm53WalkScratch::required_bytes_with_route_policy(p);
        for (base, bytes) in [
            (0, need),
            (BASE + 1, need),
            (BASE, need - 1),
            (u64::MAX & !255, need),
        ] {
            assert!(Glm53WalkScratch::bind_with_route_policy(DevicePtr(base), bytes, p).is_err());
        }
        assert!(Glm53WalkScratch::bind_with_route_policy(DevicePtr(BASE), need, p).is_ok());
    }
    assert!(
        Glm53WalkScratch::bind_with_route_policy(
            DevicePtr(BASE),
            Glm53WalkScratch::required_bytes(),
            policy("1", "1")
        )
        .is_err()
    );
}

#[test]
fn independently_bound_policies_do_not_inherit_another_targets_capacity() {
    let full = policy("1", "1");
    let small = policy("1", "0");
    let a = Glm53WalkScratch::bind_with_route_policy(
        DevicePtr(BASE),
        Glm53WalkScratch::required_bytes_with_route_policy(full),
        full,
    )
    .unwrap();
    let b = Glm53WalkScratch::bind_with_route_policy(
        DevicePtr(BASE + 16 * 1024 * MIB),
        Glm53WalkScratch::required_bytes_with_route_policy(small),
        small,
    )
    .unwrap();
    assert_eq!(a.exl3_route_policy(), full);
    assert_eq!(b.exl3_route_policy(), small);
    assert_eq!(
        a.exl3_moe_scratch().route_private_f32.bytes as u64,
        256 * MIB
    );
    assert_eq!(b.exl3_moe_scratch().route_private_f32.bytes as u64, MIB);
}
