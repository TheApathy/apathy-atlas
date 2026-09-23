// SPDX-License-Identifier: AGPL-3.0-only

//! RED: pure physical-cover admission, without CUDA, dereferences, or I/O.

#[path = "../src/model/glm53/state_read_plan.rs"]
mod state_read_plan;

use state_read_plan::StateReadPlan;
use std::ops::Range;

fn accepted(
    input: (u64, usize, u64, usize, usize, usize),
    source: u64,
    physical_bytes: usize,
    logical_range: Range<usize>,
) {
    let (parent, parent_bytes, region, region_bytes, offset, bytes) = input;
    let plan = StateReadPlan::new(parent, parent_bytes, region, region_bytes, offset, bytes)
        .expect("valid bounded covering read");
    assert_eq!(plan.source(), source, "input={input:?}");
    assert_eq!(plan.physical_bytes(), physical_bytes, "input={input:?}");
    assert_eq!(plan.logical_range(), logical_range, "input={input:?}");
    assert_eq!(source % 2, 0);
    assert_eq!(physical_bytes % 2, 0);
    assert!((2..=65_536).contains(&physical_bytes));
    assert!(physical_bytes <= isize::MAX as usize);
    assert_eq!(plan.logical_range().len(), bytes);
}

fn rejected(input: (u64, usize, u64, usize, usize, usize)) {
    let (parent, parent_bytes, region, region_bytes, offset, bytes) = input;
    assert!(
        StateReadPlan::new(parent, parent_bytes, region, region_bytes, offset, bytes).is_err(),
        "invalid input admitted: {input:?}"
    );
}

#[test]
fn aligned_rows_and_full_physical_limit_preserve_exact_extents() {
    accepted(
        (0x1000, 131_072, 0x2000, 65_536, 0, 8192),
        0x2000,
        8192,
        0..8192,
    );
    accepted(
        (0x1000, 131_072, 0x2000, 65_536, 8192, 4096),
        0x4000,
        4096,
        0..4096,
    );
    accepted(
        (0x1000, 65_536, 0x1000, 65_536, 0, 65_536),
        0x1000,
        65_536,
        0..65_536,
    );
}

#[test]
fn odd_validity_span_uses_parent_cover_and_crops_only_requested_bytes() {
    let arena = [
        91u8, 17, 200, 1, 0, 1, 222, 31, 41, 51, 61, 71, 81, 101, 111, 121,
    ];
    let plan = StateReadPlan::new(0x1000, arena.len(), 0x1003, 3, 0, 3).unwrap();
    assert_eq!(plan.source(), 0x1002);
    assert_eq!(plan.physical_bytes(), 4);
    assert_eq!(plan.logical_range(), 1..4);
    let start = (plan.source() - 0x1000) as usize;
    let physical = &arena[start..start + plan.physical_bytes()];
    assert_eq!(physical, [200, 1, 0, 1]);
    assert_eq!(&physical[plan.logical_range()], [1, 0, 1]);
    accepted((0x1000, 16, 0x1004, 3, 0, 3), 0x1004, 4, 0..3);
    accepted((0x1000, 16, 0x1003, 3, 1, 1), 0x1004, 2, 0..1);
}

#[test]
fn isolated_three_byte_allocation_cannot_supply_missing_final_cover_byte() {
    accepted((0x1000, 3, 0x1000, 3, 0, 1), 0x1000, 2, 0..1);
    accepted((0x1000, 3, 0x1000, 3, 0, 2), 0x1000, 2, 0..2);
    rejected((0x1000, 3, 0x1000, 3, 0, 3));
    rejected((0x1000, 3, 0x1000, 3, 2, 1));
    // The same logical region succeeds when the real allocation supplies byte4.
    accepted((0x1000, 4, 0x1000, 3, 2, 1), 0x1002, 2, 0..1);
}

#[test]
fn unaligned_parent_rejects_only_covers_that_escape_its_real_start() {
    rejected((0x1001, 3, 0x1001, 3, 0, 1));
    rejected((0x1001, 3, 0x1001, 3, 0, 3));
    accepted((0x1001, 3, 0x1001, 3, 1, 2), 0x1002, 2, 0..2);
    accepted((0x1001, 3, 0x1001, 3, 2, 1), 0x1002, 2, 1..2);
    accepted((0x1001, 7, 0x1003, 3, 0, 3), 0x1002, 4, 1..4);
}

#[test]
fn last_logical_byte_may_end_at_parent_end_only_if_its_cover_fits() {
    accepted((0x1000, 8, 0x1007, 1, 0, 1), 0x1006, 2, 1..2);
    rejected((0x1000, 7, 0x1006, 1, 0, 1));
    rejected((0x1000, 1, 0x1000, 1, 0, 1));
    rejected((0x1001, 1, 0x1001, 1, 0, 1));
}

#[test]
fn physical_cap_includes_both_alignment_bytes_not_only_logical_length() {
    let parent = 0x1000;
    accepted(
        (parent, 131_072, parent, 65_536, 0, 65_535),
        parent,
        65_536,
        0..65_535,
    );
    accepted(
        (parent, 131_072, parent + 1, 65_536, 0, 65_535),
        parent,
        65_536,
        1..65_536,
    );
    rejected((parent, 131_072, parent + 1, 65_536, 0, 65_536));
    accepted(
        (parent, 131_072, parent + 1, 65_536, 1, 65_535),
        parent + 2,
        65_536,
        0..65_535,
    );
    rejected((parent, 131_072, parent, 65_537, 0, 65_537));
    rejected((parent, 131_072, parent + 1, 65_538, 0, 65_537));
    accepted(
        (parent, 131_072, parent, 65_537, 65_536, 1),
        parent + 65_536,
        2,
        0..1,
    );
}

#[test]
fn null_addresses_empty_extents_and_empty_requests_are_rejected() {
    for input in [
        (0, 8, 2, 2, 0, 2),
        (2, 8, 0, 2, 0, 2),
        (2, 0, 2, 2, 0, 2),
        (2, 8, 2, 0, 0, 1),
        (2, 8, 2, 2, 0, 0),
        (2, 8, 2, 2, 2, 0),
    ] {
        rejected(input);
    }
}

#[test]
fn entire_region_must_fit_parent_even_when_requested_subset_would_fit() {
    for input in [
        (0x1000, 8, 0x0fff, 4, 1, 2),
        (0x1000, 8, 0x1006, 4, 0, 2),
        (0x1000, 8, 0x1008, 1, 0, 1),
        (0x1000, 8, 0x1009, 1, 0, 1),
        (0x1000, 8, 0x1002, 2, 2, 1),
        (0x1000, 8, 0x1002, 2, 1, 2),
        (0x1000, 8, 0x1002, 2, 3, 1),
    ] {
        rejected(input);
    }
}

#[test]
fn integer_overflow_and_extents_beyond_isize_are_rejected() {
    // The signed-length limit applies to extents, not a blanket low-address cap.
    accepted((2, isize::MAX as usize, 2, 2, 0, 2), 2, 2, 0..2);
    accepted(
        (2, isize::MAX as usize, 2, isize::MAX as usize, 0, 2),
        2,
        2,
        0..2,
    );
    let too_large = isize::MAX as usize + 1;
    for input in [
        (u64::MAX, 1, u64::MAX, 1, 0, 1),
        (u64::MAX - 1, 4, u64::MAX - 1, 1, 0, 1),
        (0x1000, 8, u64::MAX - 1, 4, 0, 1),
        (2, usize::MAX, 2, 2, 0, 2),
        (2, too_large, 2, 2, 0, 2),
        (2, 8, 2, usize::MAX, 0, 2),
        (2, 8, 2, too_large, 0, 2),
        (2, 8, 2, 8, usize::MAX, 1),
        (2, 8, 2, 8, 1, usize::MAX),
        (2, 8, 2, 8, usize::MAX, usize::MAX),
        (2, 8, 2, 8, too_large, 1),
        (2, 8, 2, 8, 0, too_large),
    ] {
        rejected(input);
    }
}

#[test]
fn high_device_addresses_are_not_signed_host_lengths_but_cover_end_cannot_wrap() {
    let parent = u64::MAX - 7;
    accepted((parent, 6, parent + 1, 3, 0, 3), parent, 4, 1..4);
    // Logical end u64::MAX is representable; its even covering end is not.
    rejected((parent, 7, parent + 6, 1, 0, 1));
}

#[test]
fn exhaustive_small_regions_offsets_and_lengths_match_even_boundary_enumeration() {
    for parent in 0x1000..=0x1003 {
        for parent_bytes in 1usize..=9 {
            for region_offset in 0..parent_bytes {
                for region_bytes in 1..=parent_bytes - region_offset {
                    let region = parent + region_offset as u64;
                    for offset in 0..=region_bytes + 1 {
                        for bytes in 0..=region_bytes + 1 {
                            let input = (parent, parent_bytes, region, region_bytes, offset, bytes);
                            let requested_start = region + offset as u64;
                            let requested_end = requested_start + bytes as u64;
                            let mut expected = None;
                            if bytes > 0 && offset + bytes <= region_bytes {
                                // Enumerate legal physical boundaries independently; no
                                // production rounding expression or allocation is reused.
                                for first in 0..parent_bytes {
                                    let source = parent + first as u64;
                                    for last in first + 1..=parent_bytes {
                                        let end = parent + last as u64;
                                        if source % 2 == 0
                                            && end % 2 == 0
                                            && source <= requested_start
                                            && end >= requested_end
                                        {
                                            let width = last - first;
                                            if expected.is_none_or(|(_, old)| width < old) {
                                                expected = Some((source, width));
                                            }
                                        }
                                    }
                                }
                            }
                            match expected {
                                Some((source, physical_bytes)) => {
                                    let first = (requested_start - source) as usize;
                                    accepted(input, source, physical_bytes, first..first + bytes);
                                }
                                None => rejected(input),
                            }
                        }
                    }
                }
            }
        }
    }
}
