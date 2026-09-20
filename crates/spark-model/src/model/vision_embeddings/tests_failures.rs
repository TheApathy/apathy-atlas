// SPDX-License-Identifier: AGPL-3.0-only

use super::test_gpu::MemoryGpu;
use super::tests::{encode, geometry, image};
use super::*;
use std::sync::atomic::Ordering;

#[test]
fn failed_copy_and_sync_do_not_publish_or_release_in_flight_allocation() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    let images = [image(3.0, 2, 2)];
    gpu.fail_copy.store(true, Ordering::SeqCst);
    assert!(
        state
            .prepare(&gpu, &images, geometry(), true, 0, 7, |p, h, w| encode(
                &gpu, scratch, p, h, w
            ))
            .is_err()
    );
    assert!(!state.is_ready());
    assert_eq!(gpu.alloc_count(), 2);
    state
        .prepare(&gpu, &images, geometry(), true, 0, 7, |p, h, w| {
            encode(&gpu, scratch, p, h, w)
        })
        .unwrap();
    gpu.fail_sync.store(true, Ordering::SeqCst);
    assert!(state.release(&gpu).is_err());
    assert!(!state.is_ready());
    assert_eq!(
        gpu.alloc_count(),
        2,
        "failed drain must not free device storage"
    );
    state.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 1);
}

#[test]
fn final_publish_sync_failure_and_wrong_encoder_rows_fail_closed() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    let images = [image(3.0, 2, 2)];
    assert!(
        state
            .prepare(&gpu, &images, geometry(), true, 0, 7, |p, h, w| {
                let result = encode(&gpu, scratch, p, h, w)?;
                gpu.fail_sync.store(true, Ordering::SeqCst);
                Ok(result)
            })
            .is_err()
    );
    assert!(!state.is_ready());
    let copies = gpu.copy_count();
    assert!(
        state
            .prepare(&gpu, &images, geometry(), true, 0, 7, |_, _, _| Ok((
                scratch, 3
            )))
            .is_err()
    );
    assert!(!state.is_ready());
    assert_eq!(gpu.copy_count(), copies);
    state.release(&gpu).unwrap();
}

#[test]
fn grow_and_reuse_drain_previous_consumer_stream_before_free_or_write() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let dst = gpu.alloc(128).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    state
        .prepare(
            &gpu,
            &[image(3.0, 2, 2)],
            geometry(),
            false,
            0,
            7,
            |p, h, w| encode(&gpu, scratch, p, h, w),
        )
        .unwrap();
    state.splice(&gpu, &[99], 99, 0, 1, dst, 2, 2, 11).unwrap();
    gpu.events.lock().unwrap().clear();
    state
        .prepare(
            &gpu,
            &[image(9.0, 2, 4)],
            geometry(),
            false,
            0,
            7,
            |p, h, w| encode(&gpu, scratch, p, h, w),
        )
        .unwrap();
    let events = gpu.events.lock().unwrap().clone();
    let drain = events.iter().position(|e| *e == ('s', 11)).unwrap();
    let free = events.iter().position(|e| e.0 == 'f').unwrap();
    let write = events.iter().position(|e| e.0 == 'c').unwrap();
    assert!(drain < free && free < write);
    assert_eq!(gpu.alloc_count(), 3);
    state.release(&gpu).unwrap();
}

#[test]
fn chunk_copy_bytes_match_full_copy_and_splice_failure_invalidates() {
    let gpu = MemoryGpu::new();
    let scratch = gpu.alloc(128).unwrap();
    let full = gpu.alloc(24).unwrap();
    let split = gpu.alloc(24).unwrap();
    let mut state = VisionEmbeddingState::new(128, 1);
    state
        .prepare(
            &gpu,
            &[image(3.0, 2, 2), image(9.0, 2, 4)],
            geometry(),
            false,
            0,
            7,
            |p, h, w| encode(&gpu, scratch, p, h, w),
        )
        .unwrap();
    let tokens = [1, 99, 2, 99, 99, 3];
    state
        .splice(&gpu, &tokens, 99, 0, 6, full, 2, 2, 7)
        .unwrap();
    state
        .splice(&gpu, &tokens, 99, 0, 4, split, 2, 2, 7)
        .unwrap();
    state
        .splice(&gpu, &tokens, 99, 4, 2, split.offset(16), 2, 2, 7)
        .unwrap();
    assert_eq!(gpu.bytes(full), gpu.bytes(split));
    gpu.fail_copy.store(true, Ordering::SeqCst);
    assert!(
        state
            .splice(&gpu, &tokens, 99, 0, 6, full, 2, 2, 7)
            .is_err()
    );
    assert!(!state.is_ready());
    state.release(&gpu).unwrap();
}
