// SPDX-License-Identifier: AGPL-3.0-only
//! P12 additional RED: read-only family admission and real projection extents.
#[path = "../src/model/glm53/dflash2_kv_prefix.rs"]
#[allow(dead_code)]
mod kv_prefix;
#[path = "../src/model/glm53/dflash2_projection_contract.rs"]
#[allow(dead_code)]
mod projection;
use kv_prefix::KvPrefix;
use projection::{ProjectionBinding, ProjectionFamily, checked_spans};

const INPUT: u64 = 0x1000_0000;
const WEIGHT: u64 = 0x2000_0000;
const OUTPUT: u64 = 0x3000_0000;
const HIDDEN: u32 = 4096;
const WIDTH: u32 = 1024;
const MAX_ROWS: u32 = 2047;

fn operands(rows: u32) -> [(u64, usize); 3] {
    [
        (INPUT, rows as usize * HIDDEN as usize * 2),
        (WEIGHT, WIDTH as usize * HIDDEN as usize * 2),
        (OUTPUT, rows as usize * WIDTH as usize * 2),
    ]
}

#[test]
fn read_only_admit_does_not_bind_or_change_family_and_rejects_pending() {
    let mut prefix = KvPrefix::new(5, MAX_ROWS).unwrap();
    let mut binding = ProjectionBinding::new();
    binding.admit(ProjectionFamily::StableTc, &prefix).unwrap();
    binding.admit(ProjectionFamily::Original, &prefix).unwrap();
    assert_eq!(binding.family(), None);
    assert!(!prefix.pending());
    binding.select(ProjectionFamily::Original, &prefix).unwrap();
    binding.admit(ProjectionFamily::Original, &prefix).unwrap();
    assert!(binding.admit(ProjectionFamily::StableTc, &prefix).is_err());
    assert_eq!(binding.family(), Some(ProjectionFamily::Original));
    prefix.begin(1, 1, 47, false).unwrap();
    assert!(binding.admit(ProjectionFamily::Original, &prefix).is_err());
    assert_eq!(binding.family(), Some(ProjectionFamily::Original));
    assert_eq!(prefix.pending_stream(), Some(47));
}

#[test]
fn all_real_projection_shapes_and_source_row_offsets_are_admitted() {
    for rows in [1, 2, 8, 16, 17, 2047] {
        let [input, weight, output] = operands(rows);
        checked_spans(rows, MAX_ROWS, HIDDEN, WIDTH, input, weight, output).unwrap();
    }
    let [mut input, weight, output] = operands(8);
    input.0 += 17 * HIDDEN as u64 * 2;
    checked_spans(8, MAX_ROWS, HIDDEN, WIDTH, input, weight, output).unwrap();
}

#[test]
fn malformed_geometry_and_each_short_operand_fail_without_io() {
    let [input, weight, output] = operands(2);
    for (rows, max, hidden, width) in [
        (0, MAX_ROWS, HIDDEN, WIDTH),
        (2048, MAX_ROWS, HIDDEN, WIDTH),
        (1, 0, HIDDEN, WIDTH),
        (2, MAX_ROWS, 0, WIDTH),
        (2, MAX_ROWS, HIDDEN, 0),
        (u32::MAX, u32::MAX, u32::MAX, u32::MAX),
    ] {
        assert!(checked_spans(rows, max, hidden, width, input, weight, output).is_err());
    }
    for operand in 0..3 {
        let mut buffers = operands(2);
        buffers[operand].1 -= 2;
        assert!(
            checked_spans(
                2, MAX_ROWS, HIDDEN, WIDTH, buffers[0], buffers[1], buffers[2]
            )
            .is_err()
        );
    }
}

#[test]
fn null_unaligned_and_overflowing_addresses_fail_for_every_operand() {
    for operand in 0..3 {
        for address in [0, 1, INPUT + 1, u64::MAX - 1] {
            let mut buffers = operands(2);
            buffers[operand].0 = address;
            assert!(
                checked_spans(
                    2, MAX_ROWS, HIDDEN, WIDTH, buffers[0], buffers[1], buffers[2]
                )
                .is_err()
            );
        }
    }
}

#[test]
fn every_restrict_operand_pair_rejects_overlap_but_adjacent_spans_are_valid() {
    for (left, right) in [(0, 1), (0, 2), (1, 2)] {
        let mut buffers = operands(2);
        buffers[right].0 = buffers[left].0 + 2;
        assert!(
            checked_spans(
                2, MAX_ROWS, HIDDEN, WIDTH, buffers[0], buffers[1], buffers[2]
            )
            .is_err()
        );
    }
    let [input, weight, mut output] = operands(2);
    output.0 = input.0 + input.1 as u64;
    checked_spans(2, MAX_ROWS, HIDDEN, WIDTH, input, weight, output).unwrap();
}
