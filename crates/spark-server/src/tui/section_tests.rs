// SPDX-License-Identifier: AGPL-3.0-only

//! `Section` is the SSOT three things read — the sidebar, `⇥`, and the mouse
//! handler. These pin the shape all three assume; the drift they exist to catch
//! is a section added to one list and not the others.

use super::*;

use std::collections::BTreeSet;

/// Every variant. The exhaustive match is the point: a section added to the
/// enum fails to compile here rather than quietly going uncovered.
fn every_variant() -> Vec<Section> {
    let all = vec![
        Section::Main,
        Section::Stats,
        Section::Network,
        Section::Library,
        Section::Terminal,
        Section::Help,
    ];
    for s in &all {
        match s {
            Section::Main
            | Section::Stats
            | Section::Network
            | Section::Library
            | Section::Terminal
            | Section::Help => {}
        }
    }
    all
}

#[test]
fn all_lists_every_section_exactly_once() {
    let listed: Vec<Section> = Section::ALL.to_vec();
    for s in every_variant() {
        assert_eq!(
            listed.iter().filter(|l| **l == s).count(),
            1,
            "{s:?} must appear exactly once in ALL"
        );
    }
    assert_eq!(listed.len(), every_variant().len());
}

#[test]
fn labels_and_icons_are_present_and_distinct() {
    // Two sections sharing a label is a navigation bug the sidebar cannot show.
    let mut labels = BTreeSet::new();
    let mut icons = BTreeSet::new();
    for s in Section::ALL {
        assert!(!s.label().is_empty(), "{s:?} has no label");
        assert!(!s.icon().is_empty(), "{s:?} has no icon");
        assert!(labels.insert(s.label()), "duplicate label {}", s.label());
        assert!(icons.insert(s.icon()), "duplicate icon {}", s.icon());
        assert_eq!(
            s.icon().chars().count(),
            1,
            "the sidebar draws the icon in one cell: {:?}",
            s.icon()
        );
    }
}

#[test]
fn subsections_are_either_absent_or_a_pair() {
    // `App::sub_index` maps each subsection to a bool, so a section with three
    // would have one row the keyboard could never reach.
    for s in Section::ALL {
        let subs = s.subs();
        assert!(
            subs.is_empty() || subs.len() == 2,
            "{s:?} has {} subsections",
            subs.len()
        );
        for name in subs {
            assert!(!name.is_empty(), "{s:?} has an unnamed subsection");
        }
    }
}

#[test]
fn the_navigable_row_count_is_what_the_sidebar_draws() {
    // One row per subsection, or a single row for a section with none — the
    // count `⇥` steps through. Three hardcoded copies of this is how `⇥` came
    // to skip past rows the sidebar was drawing.
    let rows: usize = Section::ALL.iter().map(|s| s.subs().len().max(1)).sum();
    assert_eq!(rows, 3 + 4 * 2, "3 plain sections and 4 with a pair each");
}

