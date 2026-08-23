// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn production_seam_is_exact(source: &str) -> bool {
    let Some(start) = source.find("pub fn step_verify_dflash(") else {
        return false;
    };
    let Some(end_offset) = source[start..].find("\n/// Whether any DFlash TREE") else {
        return false;
    };
    let body = &source[start..start + end_offset];
    let entry = "let (mut spec_cycle, spec_phase) = SpecCycle::begin(a, drafts);";
    if body.matches("SpecCycle::begin(").count() != 1 || body.matches(entry).count() != 1 {
        return false;
    }
    let Some(mut previous) = body.find(entry) else {
        return false;
    };
    let Some(first_clock) = body.find("let t_step = Instant::now();") else {
        return false;
    };
    if previous >= first_clock {
        return false;
    }
    for site in [
        "spec_cycle.secondary_wait_enqueue(spec_phase)",
        "spec_cycle.setup(spec_phase)",
        "spec_cycle.verify_complete(spec_phase)",
        "spec_cycle.accept(spec_phase)",
        "spec_cycle.commit_enqueue(spec_phase)",
        "spec_cycle.post_commit_enqueue(spec_phase)",
        "spec_cycle.proposer_state(spec_phase)",
        "spec_cycle.propose_complete(spec_phase)",
        "spec_cycle.finalize(spec_phase)",
    ] {
        if body.matches(site).count() != 1 {
            return false;
        }
        let Some(position) = body.find(site) else {
            return false;
        };
        if position <= previous {
            return false;
        }
        previous = position;
    }
    body.matches("spec_cycle.terminal(spec_phase, num_accepted")
        .count()
        == 3
        && body
            .matches("spec_cycle.complete(spec_phase, num_accepted, a.output_tokens.len())")
            .count()
            == 1
}

#[test]
fn disabled_path_never_reads_the_clock() {
    let (cycle, _) = SpecCycle::begin_with(
        false,
        8,
        4,
        1,
        None,
        |_| panic!("request tracker touched"),
        || panic!("clock read"),
    );
    assert!(cycle.0.is_none());
}

fn quarantined_tree_never_reads_clock_or_formats_rows(nodes: usize) {
    let (mut cycle, phase) = SpecCycle::begin_with(
        true,
        8,
        4,
        1,
        Some(nodes),
        |_| panic!("request tracker touched"),
        || panic!("clock read"),
    );
    assert!(cycle.0.is_none());
    let phase = cycle.secondary_wait_enqueue(phase);
    let phase = cycle.setup(phase);
    let phase = cycle.verify_complete(phase);
    let phase = cycle.accept(phase);
    let phase = cycle.commit_enqueue(phase);
    let phase = cycle.post_commit_enqueue(phase);
    let phase = cycle.proposer_state(phase);
    let phase = cycle.propose_complete(phase);
    let phase = cycle.finalize(phase);
    assert!(cycle.complete(phase, 0, 2).is_none());

    let (cycle, _) = SpecCycle::begin_with(
        true,
        8,
        4,
        1,
        Some(nodes),
        |_| panic!("request tracker touched"),
        || panic!("clock read"),
    );
    assert!(cycle.terminal(AwaitAccept, 0, 2).is_none());
}

#[test]
fn same_width_pending_tree_is_quarantined() {
    quarantined_tree_never_reads_clock_or_formats_rows(4);
}

#[test]
fn wide_pending_tree_is_quarantined() {
    quarantined_tree_never_reads_clock_or_formats_rows(5);
}

#[test]
fn absent_and_empty_pending_tree_keep_all_nine_phases() {
    for topology in [None, Some(0)] {
        std::thread::spawn(move || {
            let base = Instant::now();
            let (mut cycle, _) =
                SpecCycle::begin_with(true, 8, 4, 1, topology, begin_request, || base);
            let phase = mark_all(&mut cycle, base);
            assert!(cycle.complete(phase, 3, 5).is_some());
        })
        .join()
        .unwrap();
    }

    let base = Instant::now();
    let (cycle, _) = SpecCycle::begin_with(true, 12, 4, 5, None, begin_request, || base);
    assert!(cycle.terminal(AwaitAccept, 0, 6).is_some());
}

#[test]
fn production_entry_and_phase_seam_is_typed_and_forbids_work() {
    let _: fn(&ActiveSeq, &[u32]) -> (SpecCycle, AwaitSecondaryWait) = SpecCycle::begin;
    let verify = include_str!("verify_dflash_step.rs");
    assert!(production_seam_is_exact(verify));

    let wrong_argument = verify.replacen(
        "SpecCycle::begin(a, drafts)",
        "SpecCycle::begin(None, drafts)",
        1,
    );
    assert!(!production_seam_is_exact(&wrong_argument));
    let entry_line = "    let (mut spec_cycle, spec_phase) = SpecCycle::begin(a, drafts);\n";
    let clock_line = "    let t_step = Instant::now();\n";
    let moved_after_clock = verify.replacen(entry_line, "", 1).replacen(
        clock_line,
        &format!("{clock_line}{entry_line}"),
        1,
    );
    assert!(!production_seam_is_exact(&moved_after_clock));
    let reordered = verify
        .replacen(
            "spec_cycle.setup(spec_phase)",
            "spec_cycle.__phase_swap(spec_phase)",
            1,
        )
        .replacen(
            "spec_cycle.verify_complete(spec_phase)",
            "spec_cycle.setup(spec_phase)",
            1,
        )
        .replacen(
            "spec_cycle.__phase_swap(spec_phase)",
            "spec_cycle.verify_complete(spec_phase)",
            1,
        );
    assert!(!production_seam_is_exact(&reordered));

    let collector = include_str!("spec_timing.rs");
    let bracket_start = collector.find("pub(super) fn begin(").unwrap();
    let bracket_end = collector[bracket_start..]
        .find("fn begin_with")
        .map(|offset| bracket_start + offset)
        .unwrap();
    let real_entry_bracket = &collector[bracket_start..bracket_end];
    let compact_entry: String = real_entry_bracket
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    assert!(compact_entry.contains("active.pending_tree_payload"));
    for forbidden in ["cuda", "synchronize", "DevicePtr", "Vec<", "Box<"] {
        assert!(
            !real_entry_bracket.contains(forbidden),
            "forbidden production-entry work: {forbidden}"
        );
    }
}
