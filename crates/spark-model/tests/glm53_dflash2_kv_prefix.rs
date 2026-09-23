// SPDX-License-Identifier: AGPL-3.0-only
//! RED: completed drafter KV-prefix ownership; no CUDA or numerical emulation.
#[path = "../src/model/glm53/dflash2_kv_prefix.rs"]
mod cache;
#[path = "glm53_dflash2_kv_prefix/fixture.rs"]
mod fixture;

use cache::{KvPrefix, parse_kv_prefix_flag};
use fixture::{DeferredIo, Fault, tag};

fn owner() -> KvPrefix {
    KvPrefix::new(5, 2047).unwrap()
}

fn completed(owner: &KvPrefix, expected: u32) {
    assert_eq!(owner.has_completed_rows(), expected != 0);
    for layer in 0..5 {
        assert_eq!(owner.completed_rows(layer).unwrap(), expected);
    }
    assert!(owner.completed_rows(5).is_err());
}

fn enqueue_all(owner: &mut KvPrefix, io: &mut DeferredIo) {
    for layer in 0..5 {
        owner.enqueue_layer(layer, io).unwrap();
    }
}

fn warm(owner: &mut KvPrefix, io: &mut DeferredIo, rows: u32) {
    owner.begin(rows, rows, 7, false).unwrap();
    enqueue_all(owner, io);
    owner.finish(io).unwrap();
}

#[test]
fn explicit_selector_and_dimensions_fail_closed() {
    for value in [None, Some("0")] {
        assert!(!parse_kv_prefix_flag(value).unwrap());
    }
    assert!(parse_kv_prefix_flag(Some("1")).unwrap());
    for value in ["", "true", "01", " 1", "2", "off"] {
        assert!(parse_kv_prefix_flag(Some(value)).is_err());
    }
    assert!(KvPrefix::new(0, 2047).is_err());
    assert!(KvPrefix::new(5, 0).is_err());
}

#[test]
fn bootstrap_receipts_and_actual_bytes_publish_only_after_all_layers_and_fence() {
    let (mut c, mut io) = (owner(), DeferredIo::new());
    c.begin(2000, 2000, 7, false).unwrap();
    assert!(c.pending());
    assert_eq!(c.pending_stream(), Some(7));
    for layer in 0..4 {
        c.enqueue_layer(layer, &mut io).unwrap();
        completed(&c, 0);
    }
    assert!(c.finish(&mut io).is_err());
    assert!(io.fences.is_empty());
    assert!(io.k.iter().flatten().all(|&x| x == 0));
    c.enqueue_layer(4, &mut io).unwrap();
    completed(&c, 0);
    c.finish(&mut io).unwrap();
    completed(&c, 2000);
    assert!(!c.pending());
    assert_eq!(io.fences, vec![7]);
    io.assert_committed(2000);
    for (_, plan, stream) in &io.calls {
        assert_eq!(
            (plan.source_row(), plan.retained_rows(), plan.new_rows()),
            (0, 0, 2000)
        );
        assert_eq!(plan.committed_end(), 2000);
        assert_eq!(*stream, 7);
    }
}

#[test]
fn repeated_context_refreshes_one_row_without_double_claiming_or_promoting_noise() {
    let (mut c, mut io) = (owner(), DeferredIo::new());
    warm(&mut c, &mut io, 31);
    let before = io.k.clone();
    c.begin(31, 31, 9, false).unwrap();
    enqueue_all(&mut c, &mut io);
    c.finish(&mut io).unwrap();
    completed(&c, 31);
    for (_, p, _) in &io.calls[5..] {
        assert_eq!(
            (p.source_row(), p.retained_rows(), p.new_rows()),
            (30, 30, 1)
        );
        assert_eq!(p.committed_end(), 31);
    }
    for (old, new) in before.iter().zip(&io.k) {
        assert_eq!(&old[..31], &new[..31]);
    }
    io.assert_committed(31);
}

#[test]
fn each_one_to_eight_committed_advance_preserves_prefix_and_overwrites_rejected_noise() {
    for delta in 1..=8u32 {
        let (mut c, mut io) = (owner(), DeferredIo::new());
        warm(&mut c, &mut io, 31);
        for layer in 0..5 {
            assert_ne!(io.k[layer][31], tag(layer, 31));
        }
        // Only target-confirmed rows count; the old proposal wrote eight noise rows.
        let end = 31 + delta;
        c.begin(end, end, 11, false).unwrap();
        enqueue_all(&mut c, &mut io);
        c.finish(&mut io).unwrap();
        completed(&c, end);
        io.assert_committed(end);
        for (_, p, _) in &io.calls[5..] {
            assert_eq!(
                (p.source_row(), p.retained_rows(), p.new_rows()),
                (31, 31, delta)
            );
            assert_eq!(p.committed_end(), end);
        }
    }
}

#[test]
fn deferred_proposal_can_catch_up_more_than_eight_actual_committed_rows() {
    let (mut c, mut io) = (owner(), DeferredIo::new());
    warm(&mut c, &mut io, 16);
    c.begin(33, 33, 7, false).unwrap();
    enqueue_all(&mut c, &mut io);
    c.finish(&mut io).unwrap();
    completed(&c, 33);
    assert_eq!(io.calls[5].1.new_rows(), 17);
    io.assert_committed(33);
}

#[test]
fn context_identity_bounds_layer_order_and_capture_are_effect_free_preflight() {
    let (mut c, mut io) = (owner(), DeferredIo::new());
    for (context, target, capturing) in [
        (0, 0, false),
        (2048, 2048, false),
        (20, 19, false),
        (19, 20, false),
        (20, 20, true),
    ] {
        assert!(c.begin(context, target, 7, capturing).is_err());
        assert!(!c.pending());
    }
    c.begin(20, 20, 7, false).unwrap();
    assert!(c.begin(20, 20, 8, false).is_err());
    assert!(c.enqueue_layer(1, &mut io).is_err());
    assert!(c.enqueue_layer(5, &mut io).is_err());
    assert!(io.calls.is_empty());
    c.enqueue_layer(0, &mut io).unwrap();
    assert!(c.enqueue_layer(0, &mut io).is_err());
    assert_eq!(io.calls.len(), 1);
    c.abort(&mut io).unwrap();
    assert!(c.poisoned());
    assert!(c.begin(20, 20, 7, false).is_err());
    c.reset(&mut io).unwrap();
    warm(&mut c, &mut io, 20);
    assert!(c.begin(19, 19, 7, false).is_err());
}

#[test]
fn final_capacity_row_is_valid_but_noise_extent_is_never_committed() {
    let (mut c, mut io) = (owner(), DeferredIo::new());
    warm(&mut c, &mut io, 2039);
    c.begin(2047, 2047, 7, false).unwrap();
    enqueue_all(&mut c, &mut io);
    c.finish(&mut io).unwrap();
    completed(&c, 2047);
    io.assert_committed(2047);
    assert!(c.begin(2048, 2048, 7, false).is_err());
    assert!(c.begin(2055, 2047, 7, false).is_err());
}

#[test]
fn enqueue_error_and_caught_panic_retain_pending_stream_until_explicit_recovery() {
    for fault in [Fault::Enqueue(2), Fault::Panic(2)] {
        let (mut c, mut io) = (owner(), DeferredIo::new());
        warm(&mut c, &mut io, 16);
        io.fault = fault;
        c.begin(18, 18, 0, false).unwrap();
        c.enqueue_layer(0, &mut io).unwrap();
        c.enqueue_layer(1, &mut io).unwrap();
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| c.enqueue_layer(2, &mut io)));
        assert!(result.is_err() || result.unwrap().is_err());
        completed(&c, 16);
        assert!(c.pending());
        assert_eq!(c.pending_stream(), Some(0));
        assert!(c.begin(18, 18, 7, false).is_err());
        assert!(c.finish(&mut io).is_err());
        io.fault = Fault::None;
        c.reset(&mut io).unwrap();
        assert_eq!(io.fences.last(), Some(&0));
        assert!(!c.pending());
        completed(&c, 0);
    }
}

#[test]
fn failed_completion_never_publishes_and_failed_reset_retains_all_pending_ownership() {
    let (mut c, mut io) = (owner(), DeferredIo::new());
    warm(&mut c, &mut io, 31);
    c.begin(39, 39, 17, false).unwrap();
    enqueue_all(&mut c, &mut io);
    io.fault = Fault::Fence;
    assert!(c.finish(&mut io).is_err());
    completed(&c, 31);
    assert!(c.pending());
    assert!(c.poisoned());
    assert!(c.reset(&mut io).is_err());
    assert_eq!(c.pending_stream(), Some(17));
    assert!(c.begin(39, 39, 18, false).is_err());
    io.fault = Fault::None;
    c.reset(&mut io).unwrap();
    completed(&c, 0);
    assert!(!c.poisoned());
    assert!(io.fences.iter().skip(1).all(|&stream| stream == 17));
}

#[test]
fn abort_after_all_enqueues_discards_cache_authority_and_reset_rebuilds_every_layer() {
    let (mut c, mut io) = (owner(), DeferredIo::new());
    warm(&mut c, &mut io, 31);
    c.begin(33, 33, 7, false).unwrap();
    enqueue_all(&mut c, &mut io);
    c.abort(&mut io).unwrap();
    completed(&c, 31);
    assert!(c.poisoned());
    assert!(c.begin(33, 33, 7, false).is_err());
    c.reset(&mut io).unwrap();
    warm(&mut c, &mut io, 5);
    completed(&c, 5);
    for (_, p, _) in &io.calls[10..] {
        assert_eq!((p.retained_rows(), p.source_row(), p.new_rows()), (0, 0, 5));
    }
    io.assert_committed(5);
}
