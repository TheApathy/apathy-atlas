// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::{Glm53TargetGeometry, Glm53TargetSchedule};

fn exec() -> Glm53Executor {
    Glm53Executor::new(Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1)).unwrap()).unwrap()
}

/// Every scheduled event must be classified; an unclassified event would read
/// as "fine" by omission.
#[test]
fn every_scheduled_event_has_a_seam() {
    let r = exec().readiness();
    assert_eq!(
        r.total_events,
        super::super::forward_one::GLM53_TARGET_EVENTS
    );
    assert_eq!(r.identified_events + r.unbound_events, r.total_events);
}

/// **The load-bearing test.** `execute()` must never return `Ok` while dispatch
/// is unimplemented. A census that returned success would be indistinguishable
/// from a real forward pass to every caller — the exact shape of the Flash-Next
/// batched-prefill failure, where a path that "ran" emitted fluent wrong output.
#[test]
fn execute_never_reports_success_without_dispatch() {
    let err = exec()
        .execute()
        .expect_err("must not claim a pass it did not run");
    let msg = err.to_string();
    assert!(
        msg.contains("dispatch is not implemented"),
        "must say dispatch is missing, got: {msg}"
    );
    assert!(
        msg.contains("buffer/weight binding"),
        "must name the remaining work, got: {msg}"
    );
}

/// Identifying an op is not calling it. This asserts the distinction survives
/// refactors: a full census must still not satisfy `execute`.
#[test]
fn full_census_does_not_imply_execution() {
    let r = exec().readiness();
    assert!(r.is_complete(), "all events should resolve to in-tree ops");
    assert!(
        exec().execute().is_err(),
        "complete census must still refuse"
    );
}

/// The seams must name real ops and the capabilities a run would prove, so
/// admission can be flipped on evidence rather than assertion.
#[test]
fn seams_name_real_ops_and_capabilities() {
    let r = exec().readiness();
    assert!(r.identified_events >= 34, "at least the 34 KDA layers");
    assert!(!r.proves.is_empty(), "must name capabilities");
}

/// Blockers are grouped with counts so remaining work reads as a component
/// list, not 100+ identical lines.
#[test]
fn blockers_are_grouped_with_counts() {
    let r = exec().readiness();
    let summed: usize = r.blockers.iter().map(|(_, c)| *c).sum();
    assert_eq!(summed, r.unbound_events);
}

/// Walking an unvalidated plan is how capture-order drift would reach the GPU.
#[test]
fn construction_revalidates_the_schedule() {
    let s = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1)).unwrap();
    assert!(Glm53Executor::new(s).is_ok());
}

/// Prints the census; not an assertion.
#[test]
fn dump_readiness_census() {
    let r = exec().readiness();
    println!("\n  GLM executor readiness");
    println!("  ----------------------");
    println!(
        "  events with an identified op : {}/{}",
        r.identified_events, r.total_events
    );
    println!("  events with no implementation: {}", r.unbound_events);
    println!("  capabilities a real run would prove: {:?}", r.proves);
    for (needs, count) in &r.blockers {
        println!("    [{count:>3} events] {needs}");
    }
    println!("  NOTE: identified != dispatched. execute() still refuses.");
}
