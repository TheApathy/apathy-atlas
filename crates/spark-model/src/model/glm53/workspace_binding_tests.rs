// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::{Glm53TargetGeometry, Glm53TargetSchedule};

fn ws() -> Glm53TargetWorkspace {
    Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1))
        .unwrap()
        .workspace
}

const BASE: u64 = 0x1_0000_0000; // 256-aligned

#[test]
fn binds_every_region_inside_the_arena() {
    let w = ws();
    let b = Glm53BoundWorkspace::bind(&w, DevicePtr(BASE), w.arena_bytes).unwrap();
    for (name, buf, region) in [
        ("hidden_a", b.hidden_a, w.hidden_a),
        ("hidden_b", b.hidden_b, w.hidden_b),
        ("collapsed", b.collapsed, w.collapsed),
        ("widened_hc", b.widened_hc, w.widened_hc),
        ("hyper_post", b.hyper_post, w.hyper_post),
        ("hyper_comb", b.hyper_comb, w.hyper_comb),
    ] {
        assert_eq!(buf.ptr.0, BASE + region.offset_bytes, "{name} address");
        // Ops get the payload, never the padded allocation.
        assert_eq!(buf.bytes as u64, region.payload_bytes, "{name} extent");
        assert!(buf.ptr.0 % 256 == 0, "{name} alignment");
        assert!(
            buf.ptr.0 + buf.bytes as u64 <= BASE + w.arena_bytes,
            "{name} escapes the arena"
        );
    }
}

/// An arena smaller than the plan must fail at bind time, not as a stray write
/// somewhere in the middle of a 234-event walk.
#[test]
fn refuses_an_arena_smaller_than_the_plan() {
    let w = ws();
    let err = Glm53BoundWorkspace::bind(&w, DevicePtr(BASE), w.arena_bytes - 1)
        .expect_err("undersized arena must be refused");
    assert!(err.to_string().contains("arena"), "{err}");
}

#[test]
fn refuses_a_null_or_misaligned_base() {
    let w = ws();
    assert!(Glm53BoundWorkspace::bind(&w, DevicePtr(0), w.arena_bytes).is_err());
    let err = Glm53BoundWorkspace::bind(&w, DevicePtr(BASE + 1), w.arena_bytes)
        .expect_err("misaligned base must be refused");
    assert!(err.to_string().contains("misaligned"), "{err}");
}

/// The regions the schedule hands us must not alias. If they ever do, two ops
/// share a buffer and the model emits fluent, wrong output instead of failing —
/// the Flash-Next corruption class.
#[test]
fn schedule_regions_do_not_overlap() {
    let w = ws();
    assert!(Glm53BoundWorkspace::bind(&w, DevicePtr(BASE), w.arena_bytes).is_ok());
    let r = [
        w.hidden_a,
        w.hidden_b,
        w.collapsed,
        w.widened_hc,
        w.hyper_post,
        w.hyper_comb,
    ];
    for i in 0..r.len() {
        for j in i + 1..r.len() {
            let (a, b) = (r[i], r[j]);
            let a_end = a.offset_bytes + a.allocation_bytes;
            let b_end = b.offset_bytes + b.allocation_bytes;
            assert!(
                !(a.offset_bytes < b_end && b.offset_bytes < a_end),
                "regions {i} and {j} overlap"
            );
        }
    }
}

/// Binding must be deterministic — same plan and base, same buffers.
#[test]
fn binding_is_deterministic() {
    let w = ws();
    let a = Glm53BoundWorkspace::bind(&w, DevicePtr(BASE), w.arena_bytes).unwrap();
    let b = Glm53BoundWorkspace::bind(&w, DevicePtr(BASE), w.arena_bytes).unwrap();
    let key = |x: &Glm53BoundWorkspace| {
        [
            (x.hidden_a.ptr.0, x.hidden_a.bytes),
            (x.hidden_b.ptr.0, x.hidden_b.bytes),
            (x.collapsed.ptr.0, x.collapsed.bytes),
            (x.widened_hc.ptr.0, x.widened_hc.bytes),
            (x.hyper_post.ptr.0, x.hyper_post.bytes),
            (x.hyper_comb.ptr.0, x.hyper_comb.bytes),
        ]
    };
    assert_eq!(key(&a), key(&b));
}

/// Larger chunks must still bind, since prefill uses multi-token geometry.
#[test]
fn binds_multi_token_geometry() {
    for tokens in [1u32, 64, 2048] {
        let w = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(tokens))
            .unwrap()
            .workspace;
        Glm53BoundWorkspace::bind(&w, DevicePtr(BASE), w.arena_bytes)
            .unwrap_or_else(|e| panic!("tokens={tokens}: {e}"));
    }
}
