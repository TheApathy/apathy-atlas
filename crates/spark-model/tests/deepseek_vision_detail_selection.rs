// SPDX-License-Identifier: AGPL-3.0-only
//! Actual production selection logic; no CUDA, model loading, or I/O fixtures.
#[path = "../src/layers/deepseek_vision/detail_selection.rs"]
mod detail_selection;
use detail_selection::DetailSelection;

const SUFFIXES: [&str; 14] = [
    "norm1",
    "qkv",
    "query",
    "key",
    "value",
    "head-00-scores",
    "head-00-probs",
    "attention",
    "projection",
    "residual1",
    "norm2",
    "fc1",
    "swiglu",
    "fc2",
];

#[test]
fn selected_block_is_checked_against_the_actual_loaded_depth() {
    for (block, actual_blocks) in [(0, 0), (1, 1), (8, 8), (32, 32), (usize::MAX, 32)] {
        assert!(
            DetailSelection::new(block, actual_blocks).is_err(),
            "admitted block {block} with only {actual_blocks} loaded blocks"
        );
    }
    for (block, actual_blocks) in [(0, 1), (7, 8), (8, 9), (31, 32)] {
        assert!(DetailSelection::new(block, actual_blocks).is_ok());
    }
}

#[test]
fn selection_does_not_substitute_a_fixed_block_zero_or_a_fixed_depth() {
    for (block, actual_blocks) in [(0, 1), (0, 32), (8, 32), (31, 32)] {
        let selection = DetailSelection::new(block, actual_blocks).unwrap();
        let selected: Vec<_> = (0..actual_blocks)
            .filter(|&layer| selection.is_selected(layer))
            .collect();
        assert_eq!(selected, vec![block]);
        assert!(!selection.is_selected(actual_blocks));
        assert!(!selection.is_selected(usize::MAX));
    }
}

#[test]
fn legacy_block_zero_names_are_byte_identical() {
    let selection = DetailSelection::new(0, 32).unwrap();
    let expected = [
        "block-00-norm1",
        "block-00-qkv",
        "block-00-query",
        "block-00-key",
        "block-00-value",
        "block-00-head-00-scores",
        "block-00-head-00-probs",
        "block-00-attention",
        "block-00-projection",
        "block-00-residual1",
        "block-00-norm2",
        "block-00-fc1",
        "block-00-swiglu",
        "block-00-fc2",
    ];
    for (suffix, expected) in SUFFIXES.into_iter().zip(expected) {
        assert_eq!(selection.stage_name(suffix).as_bytes(), expected.as_bytes());
    }
}

#[test]
fn block_eight_names_are_complete_unique_and_never_relabel_block_zero() {
    let selection = DetailSelection::new(8, 32).unwrap();
    let names: Vec<_> = SUFFIXES
        .iter()
        .map(|suffix| selection.stage_name(suffix))
        .collect();
    assert_eq!(names.len(), 14);
    assert_eq!(
        names
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        14
    );
    for (name, suffix) in names.iter().zip(SUFFIXES) {
        assert_eq!(name, &format!("block-08-{suffix}"));
        assert!(!name.starts_with("block-00-"));
        assert!(name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'));
    }
}

#[test]
fn last_loaded_block_keeps_its_own_two_digit_stage_identity() {
    let selection = DetailSelection::new(31, 32).unwrap();
    assert_eq!(selection.stage_name("norm1"), "block-31-norm1");
    assert_eq!(selection.stage_name("fc2"), "block-31-fc2");
}
