// SPDX-License-Identifier: AGPL-3.0-only

use super::{ImageSpan, RotaryPositions};

fn image_map() -> RotaryPositions {
    // text, 2x2 image, text, 2x3 image, text: 13 physical tokens,
    // but only eight logical rotary positions before continuation.
    RotaryPositions::from_image_spans(
        13,
        64,
        &[
            ImageSpan {
                start: 1,
                height: 2,
                width: 2,
            },
            ImageSpan {
                start: 6,
                height: 2,
                width: 3,
            },
        ],
    )
    .unwrap()
}

#[test]
fn image_continuation_shifts_rotary_not_physical_cache_address() {
    let map = image_map();
    let physical = 13usize;
    assert_eq!(map.position(physical).unwrap(), [8; 3]);
    assert_eq!(map.tail_scalar(physical).unwrap(), 8);
    assert_eq!((physical / 4, physical % 4, physical + 1), (3, 1, 14));
    assert!(!map.is_identity());
}

#[test]
fn replay_inside_image_keeps_all_three_axes_and_rejects_scalar_alias() {
    let map = image_map();
    assert_eq!(map.position(0).unwrap(), [0; 3]);
    assert_eq!(map.position(1).unwrap(), [1, 1, 1]);
    assert_eq!(map.position(2).unwrap(), [1, 1, 2]);
    assert_eq!(map.position(4).unwrap(), [1, 2, 2]);
    assert_eq!(map.position(5).unwrap(), [3; 3]);
    assert_eq!(map.position(11).unwrap(), [4, 5, 6]);
    assert_eq!(map.position(12).unwrap(), [7; 3]);
    // Even the first spatial row is part of a non-scalar prompt contract.
    assert!(map.tail_scalar(1).is_err());
    assert!(map.tail_scalar(12).is_err());
}

#[test]
fn chunk_ranges_concatenate_to_the_same_global_rotary_rows() {
    let map = image_map();
    let whole = map.range(0, 13).unwrap();
    let mut chunks = map.range(0, 3).unwrap();
    chunks.extend(map.range(3, 5).unwrap());
    chunks.extend(map.range(8, 5).unwrap());
    assert_eq!(whole, chunks);
    assert_eq!(whole[8], [4, 4, 6]);
    assert_eq!(whole.len(), 13);
}

#[test]
fn flat_and_tree_verify_positions_preserve_physical_depth_and_storage() {
    let map = image_map();
    let flat = map.verify_tail(13, &[0, 1, 2, 3]).unwrap();
    let tree = map.verify_tail(13, &[0, 1, 1, 2]).unwrap();
    assert_eq!(flat, [8, 9, 10, 11]);
    assert_eq!(tree, [8, 9, 9, 10]);
    let physical_storage: Vec<_> = (13..17).collect();
    assert_eq!(physical_storage, [13, 14, 15, 16]);
    assert!(map.verify_tail(12, &[0, 1]).is_err());
}

#[test]
fn rollback_and_proposer_arm_reseed_do_not_mutate_prompt_mapping() {
    let map = image_map();
    let parked = map.clone();
    assert_eq!(map.tail_scalar(19).unwrap(), 14);
    // Speculation advances/rolls back physical seq_len, never the map.
    assert_eq!(map.tail_scalar(15).unwrap(), 10);
    assert_eq!(parked.range(10, 5).unwrap(), map.range(10, 5).unwrap());
    let mut reused = map;
    assert!(!reused.is_identity());
    reused = RotaryPositions::identity();
    assert!(reused.is_identity());
    assert_eq!(reused.position(13).unwrap(), [13; 3]);
}

#[test]
fn native_mtp_last_k_uses_predict_into_rows_without_reindexing_cache() {
    let map = image_map();
    let input_start = 9usize;
    let input_rows = 4usize;
    let predict_into = map.range(input_start + 1, input_rows).unwrap();
    assert_eq!(predict_into, [[4, 5, 5], [4, 5, 6], [7; 3], [8; 3]]);
    let own_cache_rows: Vec<_> = (0..input_rows).collect();
    assert_eq!(own_cache_rows, [0, 1, 2, 3]);
}

#[test]
fn malformed_spans_and_capacity_fail_closed() {
    for spans in [
        vec![ImageSpan {
            start: 1,
            height: 0,
            width: 2,
        }],
        vec![ImageSpan {
            start: 12,
            height: 2,
            width: 2,
        }],
        vec![ImageSpan {
            start: 1,
            height: usize::MAX,
            width: 2,
        }],
        vec![
            ImageSpan {
                start: 1,
                height: 2,
                width: 2,
            },
            ImageSpan {
                start: 4,
                height: 1,
                width: 1,
            },
        ],
        vec![
            ImageSpan {
                start: 6,
                height: 1,
                width: 1,
            },
            ImageSpan {
                start: 1,
                height: 1,
                width: 1,
            },
        ],
    ] {
        assert!(RotaryPositions::from_image_spans(13, 64, &spans).is_err());
    }
    assert!(RotaryPositions::from_image_spans(13, 12, &[]).is_err());
    assert!(RotaryPositions::from_image_spans(1, usize::MAX, &[]).is_err());
    let map = image_map();
    assert!(map.position(64).is_err());
    assert!(map.range(63, 2).is_err());
    assert!(map.range(0, usize::MAX).is_err());
    assert!(map.verify_tail(63, &[0, 1]).is_err());
}

#[test]
fn identity_text_positions_remain_exact() {
    let map = RotaryPositions::identity();
    assert!(map.is_identity());
    assert_eq!(map.range(0, 4).unwrap(), [[0; 3], [1; 3], [2; 3], [3; 3]]);
    assert_eq!(map.tail_scalar(1_048_575).unwrap(), 1_048_575);
    assert_eq!(
        map.verify_tail(10, &[0, 1, 1, 2]).unwrap(),
        [10, 11, 11, 12]
    );
    assert!(map.position(u32::MAX as usize + 1).is_err());
}
