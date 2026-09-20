// SPDX-License-Identifier: AGPL-3.0-only
//! The actual bounded operator plan; no CUDA, numerical emulation or I/O.
#[path = "../examples/glm53_projection_probe/gemv_contract.rs"]
mod contract;
use contract::{GemvPlan, Span, metadata, validate_handles};

fn owners() -> [Span; 5] {
    [
        Span {
            ptr: 0x10000,
            bytes: 2 * 4096 * 2,
        },
        Span {
            ptr: 0x100000,
            bytes: 1024 * 4096 * 2,
        },
        Span {
            ptr: 0xa00000,
            bytes: 2 * 1024 * 2,
        },
        Span {
            ptr: 0xb00000,
            bytes: 2 * 4,
        },
        Span {
            ptr: 0xc00000,
            bytes: 8,
        },
    ]
}

fn plan(rows: u32, row: u32, p: [Span; 5]) -> anyhow::Result<GemvPlan> {
    GemvPlan::new(rows, row, p[0], p[1], p[2], p[3], p[4])
}

#[test]
fn binds_two_rows_and_nonzero_single_row_without_expanding_owners() {
    let p = owners();
    let full = plan(2, 0, p).unwrap();
    assert_eq!(full.rows(), 2);
    assert_eq!(full.input(), p[0]);
    assert_eq!(full.weight(), p[1]);
    assert_eq!(full.output(), p[2]);
    assert_eq!(full.slots(), p[3]);
    assert_eq!(full.table(), p[4]);
    let tail = plan(1, 1, p).unwrap();
    assert_eq!(tail.rows(), 1);
    assert_eq!(
        tail.input(),
        Span {
            ptr: p[0].ptr + 8192,
            bytes: 8192
        }
    );
    assert_eq!(
        tail.output(),
        Span {
            ptr: p[2].ptr + 2048,
            bytes: 2048
        }
    );
    assert_eq!(
        tail.slots(),
        Span {
            ptr: p[3].ptr + 4,
            bytes: 4
        }
    );
    assert_eq!(tail.weight(), p[1]);
    assert_eq!(tail.table(), p[4]);
}

#[test]
fn fixed_actual_input_geometry_rejects_empty_large_and_wrapped_rows() {
    for (rows, row) in [(0, 0), (3, 0), (2, 1), (1, 2), (2, u32::MAX)] {
        assert!(plan(rows, row, owners()).is_err(), "rows={rows} row={row}");
    }
    assert!(plan(1, 0, owners()).is_ok());
}

#[test]
fn validates_every_full_owner_capacity_even_for_a_single_row_call() {
    for index in 0..5 {
        let mut p = owners();
        p[index].bytes -= 1;
        assert!(plan(1, 0, p).is_err(), "short owner {index}");
    }
}

#[test]
fn validates_vector_input_weight_and_integer_metadata_alignment() {
    for (index, offset) in [(0, 2), (1, 2), (2, 1), (3, 2), (4, 4)] {
        let mut p = owners();
        p[index].ptr += offset;
        assert!(plan(2, 0, p).is_err(), "alignment owner {index}");
    }
    for index in 0..5 {
        let mut p = owners();
        p[index].ptr = 0;
        assert!(plan(2, 0, p).is_err(), "null owner {index}");
    }
}

#[test]
fn every_owner_pair_must_be_nonoverlapping() {
    for left in 0..5 {
        for right in left + 1..5 {
            let mut p = owners();
            p[right].ptr = p[left].ptr;
            assert!(plan(2, 0, p).is_err(), "overlap {left}/{right}");
        }
    }
}

#[test]
fn checks_full_span_end_and_isize_limits_before_slicing() {
    for index in 0..5 {
        let mut p = owners();
        p[index].ptr = u64::MAX - 15;
        p[index].bytes = 32;
        assert!(plan(1, 0, p).is_err(), "wrapped owner {index}");
        let mut p = owners();
        p[index].bytes = usize::MAX;
        assert!(plan(1, 0, p).is_err(), "oversized owner {index}");
    }
}

#[test]
fn metadata_is_two_zero_i32_slots_and_one_actual_weight_pointer() {
    let weight = owners()[1];
    let (slots, table) = metadata(weight).unwrap();
    assert_eq!(slots, [0; 8]);
    assert_eq!(table, weight.ptr.to_le_bytes());
    assert_eq!(i32::from_le_bytes(slots[..4].try_into().unwrap()), 0);
    assert_eq!(i32::from_le_bytes(slots[4..].try_into().unwrap()), 0);
    for invalid in [
        Span { ptr: 0, ..weight },
        Span {
            ptr: weight.ptr + 2,
            ..weight
        },
        Span {
            bytes: weight.bytes - 1,
            ..weight
        },
        Span {
            ptr: u64::MAX - 15,
            ..weight
        },
    ] {
        assert!(metadata(invalid).is_err());
    }
}

#[test]
fn all_three_kernel_handles_are_required_before_projection() {
    validate_handles([1, 2, 3]).unwrap();
    for index in 0..3 {
        let mut handles = [1, 2, 3];
        handles[index] = 0;
        assert!(validate_handles(handles).is_err(), "missing handle {index}");
    }
}
