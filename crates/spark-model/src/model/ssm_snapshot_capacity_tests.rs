// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

const NUM_SLOTS: usize = 5;
const H_BYTES: usize = 8;
const CONV_BYTES: usize = 4;
const NUM_LAYERS: usize = 2;
const RING_SLOTS: usize = 3;
const MAX_SEQS: usize = 8;
const EXPECTED_SNAPSHOT_BYTES: usize = 696;

fn expect_new_error(shape: (usize, usize, usize, usize, usize, usize), expected: &str) {
    let gpu = MockGpuBackend::new();
    let (num_slots, h_bytes, conv_bytes, num_layers, ring_slots, max_seqs) = shape;
    let error = match SsmSnapshotPool::new(
        num_slots, h_bytes, conv_bytes, num_layers, ring_slots, max_seqs, &gpu,
    ) {
        Ok(_) => panic!("overflow shape unexpectedly allocated: {shape:?}"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains(expected),
        "wrong error for {shape:?}: {error:#}"
    );
    assert_eq!(gpu.alloc_count(), 0, "overflow must fail before alloc");
}

#[test]
fn representative_pool_geometry_matches_the_preflight_fixture() {
    let capacity = checked_snapshot_capacity(
        NUM_SLOTS, H_BYTES, CONV_BYTES, NUM_LAYERS, RING_SLOTS, MAX_SEQS,
    )
    .unwrap();
    assert_eq!(capacity.marconi_h_bytes, 40);
    assert_eq!(capacity.marconi_conv_bytes, 20);
    assert_eq!(capacity.marconi_total_bytes, 120);
    assert_eq!(capacity.decode_region_slots, 24);
    assert_eq!(capacity.decode_h_bytes, 192);
    assert_eq!(capacity.decode_conv_bytes, 96);
    assert_eq!(capacity.decode_total_bytes, 576);
    assert_eq!(
        capacity.marconi_total_bytes + capacity.decode_total_bytes,
        EXPECTED_SNAPSHOT_BYTES
    );

    let gpu = MockGpuBackend::new();
    let pool = SsmSnapshotPool::new(
        NUM_SLOTS, H_BYTES, CONV_BYTES, NUM_LAYERS, RING_SLOTS, MAX_SEQS, &gpu,
    )
    .unwrap();
    assert_eq!(gpu.alloc_count(), 8);
    assert_eq!(pool.num_slots, NUM_SLOTS);
    assert_eq!(pool.decode_ring_slots, RING_SLOTS);
    assert_eq!(pool.decode_max_seqs, MAX_SEQS);
}

#[test]
fn disabled_shapes_skip_irrelevant_arithmetic_and_allocation() {
    let gpu = MockGpuBackend::new();
    let no_layers = SsmSnapshotPool::new(
        usize::MAX,
        usize::MAX,
        usize::MAX,
        0,
        usize::MAX,
        usize::MAX,
        &gpu,
    )
    .unwrap();
    assert!(!no_layers.is_enabled());
    assert!(!no_layers.decode_rollback_enabled());

    let no_regions =
        SsmSnapshotPool::new(0, usize::MAX, usize::MAX, 2, 0, usize::MAX, &gpu).unwrap();
    assert!(!no_regions.is_enabled());
    assert!(!no_regions.decode_rollback_enabled());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn enabled_zero_byte_regions_fail_before_allocation() {
    expect_new_error((1, 0, 4, 1, 0, 0), "h bytes must be positive");
    expect_new_error((1, 4, 0, 1, 0, 0), "conv bytes must be positive");
    expect_new_error((0, 0, 4, 1, 1, 1), "h bytes must be positive");
    expect_new_error((0, 4, 0, 1, 1, 1), "conv bytes must be positive");
}

#[test]
fn every_marconi_checked_boundary_fails_before_allocation() {
    expect_new_error((2, usize::MAX, 1, 1, 0, 0), "Marconi h bytes");
    expect_new_error((2, 1, usize::MAX, 1, 0, 0), "Marconi conv bytes");
    expect_new_error((1, usize::MAX, 1, 1, 0, 0), "Marconi per-layer bytes");
    expect_new_error((1, usize::MAX / 2 + 1, 1, 2, 0, 0), "Marconi total bytes");
}

#[test]
fn every_decode_checked_boundary_fails_before_allocation() {
    expect_new_error((0, 1, 1, 1, 2, usize::MAX), "decode region slots");
    expect_new_error((0, usize::MAX, 1, 1, 2, 1), "decode h bytes");
    expect_new_error((0, 1, usize::MAX, 1, 2, 1), "decode conv bytes");
    expect_new_error((0, usize::MAX, 1, 1, 1, 1), "decode per-layer bytes");
    expect_new_error((0, usize::MAX / 2 + 1, 1, 2, 1, 1), "decode total bytes");
}
