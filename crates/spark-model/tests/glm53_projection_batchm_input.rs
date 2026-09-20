// SPDX-License-Identifier: AGPL-3.0-only
//! Derived operator corpus is never mislabeled as a captured long model context.
#[path = "../examples/glm53_projection_probe/batchm_input.rs"]
mod batchm_input;
#[allow(dead_code)]
#[path = "../examples/glm53_projection_probe/contract.rs"]
mod contract;
#[allow(dead_code)]
#[path = "../examples/glm53_projection_probe/gemv_contract.rs"]
mod gemv_contract;
use batchm_input::{INPUT_DERIVATION, derive_input};

fn source() -> Vec<u8> {
    let mut bytes = [0x80, 0x3f].repeat(4096);
    bytes.extend([0x00, 0x40].repeat(4096));
    bytes
}

#[test]
fn expanded_rows_are_exact_alternating_pinned_rows_without_input_mutation() {
    let source = source();
    let original = source.clone();
    for rows in [1, 2, 3, 8, 9, 15, 16, 17, 31, 32, 33, 256, 2047, 2048] {
        let derived = derive_input(&source, rows).unwrap();
        assert_eq!(derived.len(), rows as usize * 8192);
        for (row, bytes) in derived.chunks_exact(8192).enumerate() {
            let start = (row % 2) * 8192;
            assert_eq!(bytes, &source[start..start + 8192]);
        }
        assert_eq!(source, original);
    }
    assert_eq!(INPUT_DERIVATION, "repeated-pinned-two-row-input");
}

#[test]
fn wrong_extent_and_nonfinite_source_are_rejected_before_expansion() {
    for bytes in [vec![], vec![0; 8192], vec![0; 16383], vec![0; 16386]] {
        assert!(derive_input(&bytes, 16).is_err());
    }
    for bits in [0x7f80u16, 0xff80, 0x7fc0, 0xffff] {
        let mut bytes = source();
        bytes[16382..16384].copy_from_slice(&bits.to_le_bytes());
        assert!(
            derive_input(&bytes, 1).is_err(),
            "unused source row must still validate"
        );
    }
}

#[test]
fn derived_owner_has_explicit_2048_row_cap_no_unbounded_allocation() {
    for rows in [0, 2049, u32::MAX] {
        assert!(derive_input(&source(), rows).is_err());
    }
}
