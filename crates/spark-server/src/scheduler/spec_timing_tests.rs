// SPDX-License-Identifier: AGPL-3.0-only

use std::time::{Duration, Instant};

use super::*;

fn isolated_cycle(base: Instant, pre: u64, gamma: u32, output_before: usize) -> SpecCycle {
    SpecCycle(Some(ActiveCycle {
        req: 0xabu64,
        pre,
        gamma,
        output_before,
        last: base,
        phases: [0; PHASE_COUNT],
        next_phase: 0,
        tracked: false,
    }))
}

fn mark_all(cycle: &mut SpecCycle, base: Instant) -> CompleteReady {
    let phases = [
        Phase::SecondaryWaitEnqueue,
        Phase::Setup,
        Phase::VerifyComplete,
        Phase::Accept,
        Phase::CommitEnqueue,
        Phase::PostCommitEnqueue,
        Phase::ProposerState,
        Phase::ProposeComplete,
        Phase::Finalize,
    ];
    let mut elapsed = 0;
    for (index, phase) in phases.into_iter().enumerate() {
        elapsed += index as u64 + 1;
        cycle.mark_at(phase, base + Duration::from_nanos(elapsed));
    }
    CompleteReady
}

#[test]
fn gate_is_cached_off_and_rejects_non_c1_or_async() {
    assert_eq!(Gate::resolve(None, None, 8), Ok(Gate { enabled: false }));
    assert_eq!(
        Gate::resolve(Some("0"), Some("1"), 8),
        Ok(Gate { enabled: false })
    );
    assert!(Gate::resolve(Some("1"), Some("0"), 2).is_err());
    assert!(Gate::resolve(Some("1"), None, 1).is_err());
    assert!(Gate::resolve(Some("1"), Some("1"), 1).is_err());
    assert_eq!(
        Gate::resolve(Some("1"), Some("0"), 1),
        Ok(Gate { enabled: true })
    );
    assert!(Gate::resolve(Some("yes"), Some("0"), 1).is_err());
}

#[test]
fn complete_record_has_exclusive_ordered_accounting() {
    let base = Instant::now();
    let mut cycle = isolated_cycle(base, 8, 4, 1);
    let phase = mark_all(&mut cycle, base);
    let row = cycle
        .complete(phase, 3, 5)
        .expect("complete row")
        .to_string();
    assert_eq!(
        row,
        "SPEC_CYCLE_V2 schema=2 req=00000000000000ab pre=8 k=5 gamma=4 accepted=3 emitted=4 c=1 async_requested=0 async_engaged=0 secondary_wait_enqueue_host_ns=1 setup_host_ns=2 verify_complete_host_ns=3 accept_host_ns=4 commit_enqueue_host_ns=5 post_commit_enqueue_host_ns=6 proposer_state_host_ns=7 propose_complete_host_ns=8 finalize_host_ns=9 total_host_ns=45 syncs_added=0"
    );
}

#[test]
fn incomplete_or_out_of_order_phases_publish_nothing() {
    let base = Instant::now();
    let mut cycle = isolated_cycle(base, 8, 4, 1);
    cycle.mark_at(Phase::Setup, base + Duration::from_nanos(1));
    assert!(cycle.complete(CompleteReady, 0, 2).is_none());

    let mut cycle = isolated_cycle(base, 8, 4, 1);
    cycle.mark_at(
        Phase::SecondaryWaitEnqueue,
        base.checked_sub(Duration::from_nanos(1)).unwrap(),
    );
    assert!(cycle.complete(CompleteReady, 0, 2).is_none());

    let mut cycle = isolated_cycle(base, 8, 4, 1);
    let phase = mark_all(&mut cycle, base);
    assert!(
        cycle.complete(phase, 2, 3).is_none(),
        "claimed emission was not observed"
    );
}

#[test]
fn terminal_records_distinguish_draft_and_bonus_returns() {
    let base = Instant::now();
    let draft = isolated_cycle(base, 8, 4, 10)
        .terminal(AwaitAccept, 4, 12)
        .expect("draft terminal")
        .to_string();
    assert_eq!(
        draft,
        "SPEC_CYCLE_V2_TERMINAL schema=2 req=00000000000000ab pre=8 k=5 gamma=4 verifier_accepted=4 accepted_emitted=2 bonus_emitted=0 emitted=2 terminal_branch=accepted_draft c=1 async_requested=0 async_engaged=0 syncs_added=0"
    );

    let bonus = isolated_cycle(base, 8, 4, 10)
        .terminal(AwaitAccept, 2, 13)
        .expect("bonus terminal")
        .to_string();
    assert!(bonus.contains(
        "verifier_accepted=2 accepted_emitted=2 bonus_emitted=1 emitted=3 terminal_branch=bonus"
    ));
    assert!(
        isolated_cycle(base, 8, 4, 10)
            .terminal(AwaitAccept, 2, 14)
            .is_none()
    );
}

#[test]
fn request_tracker_keeps_one_id_then_retires_it() {
    let base = Instant::now();
    let (mut first, _) = SpecCycle::begin_with(true, 8, 4, 1, None, begin_request, || base);
    let phase = mark_all(&mut first, base);
    let first_row = first.complete(phase, 2, 4).expect("first row").to_string();

    let (second, _) = SpecCycle::begin_with(true, 11, 4, 4, None, begin_request, || base);
    let terminal = second
        .terminal(AwaitAccept, 0, 5)
        .expect("terminal row")
        .to_string();
    let first_req = first_row.split_whitespace().nth(2).unwrap();
    let second_req = terminal.split_whitespace().nth(2).unwrap();
    assert_eq!(first_req, second_req);

    let (next, _) = SpecCycle::begin_with(true, 8, 4, 1, None, begin_request, || base);
    let next_terminal = next
        .terminal(AwaitAccept, 0, 2)
        .expect("next request")
        .to_string();
    assert_ne!(first_req, next_terminal.split_whitespace().nth(2).unwrap());
}

#[path = "spec_timing_entry_tests.rs"]
mod entry_tests;
