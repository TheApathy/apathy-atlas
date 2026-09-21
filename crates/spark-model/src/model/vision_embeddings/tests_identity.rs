// SPDX-License-Identifier: AGPL-3.0-only

use super::test_gpu::MemoryGpu;
use super::tests::{encode, geometry, image};
use super::*;

#[test]
fn a_forced_hash_collision_does_not_reuse_different_pixel_bits() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let destination = gpu.alloc(4).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    let original = [image(3.0, 2, 2)];
    let changed = [image(9.0, 2, 2)];
    state
        .prepare(&gpu, &original, geometry(), true, 0, 7, |p, h, w| {
            encode(&gpu, scratch, p, h, w)
        })
        .unwrap();
    let collision = layout::cache_key(&changed, geometry(), 0);
    state
        .published
        .as_mut()
        .unwrap()
        .1
        .as_mut()
        .unwrap()
        .fingerprint = collision;
    assert!(
        !state
            .prepare(&gpu, &changed, geometry(), true, 0, 7, |p, h, w| encode(
                &gpu, scratch, p, h, w
            ))
            .unwrap()
    );
    state
        .splice(&gpu, &[99], 99, 0, 1, destination, 2, 2, 7)
        .unwrap();
    assert_eq!(gpu.bytes(destination), vec![9; 4]);
    state.release(&gpu).unwrap();
}

#[test]
fn exact_cache_confirmation_is_bounded_and_preserves_negative_zero() {
    let original = [image(0.0, 2, 2)];
    assert!(cache::ExactCacheKey::capture(&original, 0, 1, 4).is_none());
    let exact = cache::ExactCacheKey::capture(&original, 0, 1, 32 * 1024).unwrap();
    let mut changed = original.clone();
    changed[0].0[1] = -0.0;
    assert!(!exact.matches(&changed, 0, 1));
    assert!(!exact.matches(&original, 1, 1));
    assert!(exact.matches(&original, 0, 1));
}

#[test]
fn encoder_offset_alias_and_overflow_fail_before_copy_without_publishing() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    let images = [image(3.0, 2, 4)];
    state
        .prepare(&gpu, &images, geometry(), false, 0, 7, |p, h, w| {
            encode(&gpu, scratch, p, h, w)
        })
        .unwrap();
    let aggregate = state.buffer;
    let intact = gpu.bytes(aggregate);
    let copies = gpu.copy_count();
    for bad_source in [
        aggregate.offset(2),
        DevicePtr(aggregate.0 - 2),
        DevicePtr(u64::MAX - 1),
    ] {
        assert!(
            state
                .prepare(&gpu, &images, geometry(), false, 0, 7, |_, _, _| Ok((
                    bad_source, 4
                )))
                .is_err()
        );
        assert!(!state.is_ready());
        assert_eq!(gpu.copy_count(), copies);
        assert_eq!(gpu.bytes(aggregate), intact);
    }
    state.release(&gpu).unwrap();
}
