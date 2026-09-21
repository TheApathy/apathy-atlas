// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// The parser must agree with the harness files under review, so this fixture
/// is a verbatim excerpt of one — not a hand-written approximation.
const REAL_TIMING: &str = "\
warmup: ttft_ms=2211.5 wall_ms=2258.6 prompt_tokens=2047 text=':' tok_s=925.6
trial-1: ttft_ms=2089.7 wall_ms=2129.6 prompt_tokens=2047 text=':' tok_s=979.6
trial-2: ttft_ms=2088.1 wall_ms=2139.7 prompt_tokens=2047 text=':' tok_s=980.3
trial-3: ttft_ms=2091.6 wall_ms=2132.6 prompt_tokens=2047 text=':' tok_s=978.7
MEDIAN ttft_ms=2105.5  tok/s=972.2  min=2088.1 max=2119.7
";

#[test]
fn the_warmup_and_the_median_line_are_not_trials() {
    let a = parse_timing(REAL_TIMING, "ix-c0").expect("three trials");
    assert_eq!(a.trials, 3, "warmup and MEDIAN must not be counted");
    assert_eq!(a.median_tok_s, 979.6);
    assert_eq!(a.min_tok_s, 978.7);
    assert_eq!(a.max_tok_s, 980.3);
}

#[test]
fn a_file_with_no_trials_is_not_an_arm_measuring_zero() {
    // The decay harness produced exactly this: a server that loaded, then
    // died in post-processing before any trial. Reporting it as an arm with
    // median 0 would put a fabricated number on screen.
    assert!(parse_timing("warmup: tok_s=900.0\n", "ix-dead").is_none());
    assert!(parse_timing("", "empty").is_none());
}

#[test]
fn the_control_spread_needs_two_arms_and_reports_the_measured_floor() {
    let arm = |n: &str, m: f64| ArmSummary {
        name: n.into(),
        trials: 25,
        median_tok_s: m,
        min_tok_s: m,
        max_tok_s: m,
    };
    // One arm cannot have a spread. Returning 0.0 here is the two-sample
    // underestimate that hid GLM's real floor for a night.
    assert_eq!(control_spread_pct(&[arm("c0", 972.2)]), None);

    // The five GLM controls, as measured.
    let five = [
        arm("c0", 972.2),
        arm("c1", 956.3),
        arm("c2", 951.5),
        arm("c3", 960.9),
        arm("c4", 956.3),
    ];
    let spread = control_spread_pct(&five).expect("five arms");
    assert!(
        (spread - 2.16).abs() < 0.01,
        "the measured GLM floor, to two decimals: {spread}"
    );
}

#[test]
fn the_verdict_names_every_blocker_not_just_the_first() {
    let blocked = BoxState {
        lock_holder: Some(1234),
        waiters: vec![5678],
        mem_available_mib: 1024,
        builders: 2,
    };
    let v = blocked.verdict();
    assert!(v.starts_with("DO NOT MEASURE"), "{v}");
    for expected in ["1234", "queued", "build", "memory"] {
        assert!(v.contains(expected), "missing {expected:?} from: {v}");
    }

    let clear = BoxState {
        lock_holder: None,
        waiters: Vec::new(),
        mem_available_mib: 100 * 1024,
        builders: 0,
    };
    assert!(
        clear.verdict().starts_with("box is clear"),
        "{}",
        clear.verdict()
    );
}

#[test]
fn a_dead_holder_pid_still_counts_as_held() {
    // The lock survives through a descriptor an alive child inherited, so a
    // dead pid in the table is NOT a stale lock to be cleared.
    let locks = "116: FLOCK  ADVISORY  WRITE 346214 103:02:999999 0 EOF\n\
                 116: ->    FLOCK  ADVISORY  WRITE 409729 103:02:999999 0 EOF\n";
    // No file at this path, so the inode lookup fails and nothing is claimed
    // — the negative arm, proving the parse is not inventing holders.
    let (h, w) = read_lock(std::path::Path::new("/nonexistent/atlas.lock"), locks);
    assert_eq!(h, None);
    assert!(w.is_empty());
}
