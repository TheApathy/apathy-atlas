// SPDX-License-Identifier: AGPL-3.0-only

//! Pure-CPU state-machine tests for the N-arm throughput gate.
//!
//! No GPU, no model, no server: the gate only ever sees `(wall, emitted)`
//! pairs, so its whole policy is testable by driving it with synthetic step
//! timings. Ported from upstream `mtp_gate/tests.rs` and extended for the
//! N-arm cases our fork needs (three arms, round-robin probing).
//!
//! These tests read `ATLAS_MTP_GATE_*`; they assume those are unset, which is
//! the default in CI. `probe_windows` is read once per gate, so the 2-window
//! default (our deviation from upstream) is baked into the arithmetic below.

use super::*;

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// Two arms: index 0 external drafter (primary), index 1 native MTP head.
fn two_arms() -> Vec<ArmSpec> {
    vec![
        ArmSpec::spec("ddtree", PROPOSER_ARM_PRIMARY, 0, 0),
        ArmSpec::spec("mtp-k2", PROPOSER_ARM_ALT, 0, 1),
    ]
}

/// Three arms: the two above plus plain serial decode.
fn three_arms() -> Vec<ArmSpec> {
    vec![
        ArmSpec::spec("ddtree", PROPOSER_ARM_PRIMARY, 0, 0),
        ArmSpec::spec("mtp-k2", PROPOSER_ARM_ALT, 0, 1),
        ArmSpec::serial("serial"),
    ]
}

/// Feed `n` steps at a fixed (emitted, wall), regardless of which arm the
/// gate thinks it is running.
fn drive(g: &mut MtpGate, n: usize, emitted: usize, wall: Duration) {
    for _ in 0..n {
        g.record_step(wall, emitted);
    }
}

/// Drive the given arm until the gate opens a probe of a DIFFERENT arm,
/// then return the arm index it chose to probe.
fn run_until_probe(g: &mut MtpGate, arm: usize, emitted: usize, wall: Duration) -> usize {
    for _ in 0..100_000 {
        if g.is_probing() {
            return g.next_arm_index();
        }
        assert_eq!(
            g.next_arm_index(),
            arm,
            "gate left arm {arm} before opening a probe"
        );
        g.record_step(wall, emitted);
    }
    panic!("gate never opened a probe");
}

/// Run out a probe excursion (all its windows) at the given rate.
fn finish_probe(g: &mut MtpGate, emitted: usize, wall: Duration) {
    let guard = WINDOW_STEPS * 64;
    for _ in 0..guard {
        if !g.is_probing() {
            return;
        }
        g.record_step(wall, emitted);
    }
    panic!("probe never closed");
}

#[test]
fn starts_on_the_primary_arm() {
    let g = MtpGate::new(two_arms());
    assert_eq!(g.current_arm_index(), 0);
    assert_eq!(g.next_arm_index(), 0);
    assert!(!g.is_probing());
    match g.next_arm().kind {
        ArmKind::Spec { proposer_arm, .. } => assert_eq!(proposer_arm, PROPOSER_ARM_PRIMARY),
        other => panic!("expected the primary spec arm, got {other:?}"),
    }
}

#[test]
fn a_single_arm_gate_never_probes_and_never_switches() {
    // Degenerate but reachable: `--mtp-gate dflash|mtp` pins an arm, and
    // `--mtp-gate auto` lands here on a build with only one proposer. The gate
    // must be inert, not spin.
    let mut g = MtpGate::new(vec![ArmSpec::spec("ddtree", PROPOSER_ARM_PRIMARY, 0, 0)]);
    drive(&mut g, WINDOW_STEPS * 200, 2, ms(50));
    assert_eq!(g.current_arm_index(), 0);
    assert!(!g.is_probing());
    assert_eq!(g.take_fresh_switch(), None);
}

#[test]
fn a_single_arm_gate_reports_that_it_cannot_arbitrate() {
    // The scheduler hoists this out of its step loop and uses it to skip ALL
    // per-step measurement, so a pinned arm costs a branch. If this ever
    // returns true for one arm, `--mtp-gate dflash` starts paying for
    // scans and timing it can never act on.
    let one = MtpGate::new(vec![ArmSpec::spec("ddtree", PROPOSER_ARM_PRIMARY, 0, 0)]);
    assert!(!one.arbitrates());
    assert!(MtpGate::new(two_arms()).arbitrates());
    assert!(MtpGate::new(three_arms()).arbitrates());
}

#[test]
fn a_single_arm_gate_keeps_whole_windows_past_the_event_interval() {
    // Regression: `tokens_since_event` is only cleared when a probe opens or
    // completes, and a one-arm gate never probes — so it crosses
    // `event_interval` (~1024 tokens) once and stays over it forever. Without
    // the `arbitrates()` guard in `record_step` that closed a window on EVERY
    // subsequent step, collapsing the 16-step window to 1 step and turning the
    // tok/s EWMA into per-step noise.
    let mut g = MtpGate::new(vec![ArmSpec::spec("ddtree", PROPOSER_ARM_PRIMARY, 0, 0)]);
    // Well past 1024 emitted tokens at 4 tok/step.
    drive(&mut g, WINDOW_STEPS * 40, 4, ms(50));
    // A degenerate 1-step window would still average to the same rate here, so
    // assert on the accumulator directly: mid-window state must be non-empty
    // and must never exceed one whole window.
    drive(&mut g, 3, 4, ms(50));
    assert_eq!(
        g.window_steps_debug(),
        3,
        "steps must accumulate into a window instead of closing one each step"
    );
}

#[test]
fn no_switch_without_a_second_baseline() {
    let mut g = MtpGate::new(two_arms());
    // Plenty of slow primary-arm windows, but arm 1 was never measured, and
    // the gate must not switch to an arm it has no number for.
    for _ in 0..64 {
        if g.is_probing() {
            break;
        }
        g.record_step(ms(100), 1);
    }
    assert_eq!(g.current_arm_index(), 0);
    assert_eq!(g.take_fresh_switch(), None);
}

#[test]
fn probe_opens_after_the_refresh_interval_and_returns() {
    let mut g = MtpGate::new(two_arms());
    // On the primary arm the cadence is `refresh` (1024 tok). 2 tok/step.
    let probed = run_until_probe(&mut g, 0, 2, ms(50));
    assert_eq!(probed, 1, "the only other arm is arm 1");
    // Probe measures arm 1 SLOWER (25 tok/s vs 40), so we stay on arm 0.
    finish_probe(&mut g, 1, ms(40));
    assert_eq!(g.current_arm_index(), 0);
    assert!(!g.is_probing());
    assert!(
        g.arm_tps_debug(1).is_some(),
        "a probe must set the probed arm's baseline"
    );
}

#[test]
fn switches_when_another_arm_is_clearly_faster_after_dwell() {
    let mut g = MtpGate::new(two_arms());
    // Primary delivers 2 tok / 100ms = 20 tok/s.
    run_until_probe(&mut g, 0, 2, ms(100));
    // Arm 1 probes at 1 tok / 10ms = 100 tok/s — way past any margin.
    finish_probe(&mut g, 1, ms(10));
    // Dwell: at least one more losing evaluation is required.
    for _ in 0..(WINDOW_STEPS * SWITCH_DWELL_WINDOWS * 2) {
        if g.current_arm_index() != 0 {
            break;
        }
        g.record_step(ms(100), 2);
    }
    assert_eq!(
        g.current_arm_index(),
        1,
        "a sustained 5x advantage switches"
    );
    assert_eq!(g.take_fresh_switch(), Some(ArmSwitch { from: 0, to: 1 }));
    assert_eq!(g.take_fresh_switch(), None, "fresh switch is one-shot");
    assert_eq!(g.next_arm_index(), 1);
}

#[test]
fn hysteresis_blocks_within_margin_switches() {
    let mut g = MtpGate::new(two_arms());
    // Primary: 2 tok / 50ms = 40.0 tok/s.
    run_until_probe(&mut g, 0, 2, ms(50));
    // Arm 1 at ~41 tok/s — inside the 5% noise floor (would need > 42).
    finish_probe(&mut g, 1, Duration::from_micros(24_390));
    for _ in 0..(WINDOW_STEPS * 6) {
        if g.current_arm_index() != 0 {
            break;
        }
        g.record_step(ms(50), 2);
    }
    assert_eq!(
        g.current_arm_index(),
        0,
        "a within-margin advantage must not switch arms"
    );
    assert_eq!(g.take_fresh_switch(), None);
}

#[test]
fn recovers_to_the_primary_arm_when_it_becomes_faster_again() {
    let mut g = MtpGate::new(two_arms());
    // Establish primary=20 tok/s, arm1=100 tok/s, and switch to arm 1.
    run_until_probe(&mut g, 0, 2, ms(100));
    finish_probe(&mut g, 1, ms(10));
    for _ in 0..(WINDOW_STEPS * SWITCH_DWELL_WINDOWS * 2) {
        if g.current_arm_index() != 0 {
            break;
        }
        g.record_step(ms(100), 2);
    }
    assert_eq!(g.current_arm_index(), 1);
    let _ = g.take_fresh_switch();

    // Now off the primary the cadence is `reprobe` (256 tok), and the
    // primary has become fast (4 tok / 10ms = 400 tok/s).
    let probed = run_until_probe(&mut g, 1, 1, ms(10));
    assert_eq!(probed, 0);
    finish_probe(&mut g, 4, ms(10));
    for _ in 0..(WINDOW_STEPS * SWITCH_DWELL_WINDOWS * 2) {
        if g.current_arm_index() == 0 {
            break;
        }
        g.record_step(ms(10), 1);
    }
    assert_eq!(
        g.current_arm_index(),
        0,
        "the gate must be able to come back"
    );
    assert_eq!(g.take_fresh_switch(), Some(ArmSwitch { from: 1, to: 0 }));
}

#[test]
fn three_arms_probe_round_robin() {
    let mut g = MtpGate::new(three_arms());
    // Keep every arm looking identical so no switch fires and we observe
    // pure probe scheduling: arm 1, then arm 2, then arm 1 again.
    let mut seen = Vec::new();
    for _ in 0..3 {
        let probed = run_until_probe(&mut g, 0, 2, ms(50));
        seen.push(probed);
        finish_probe(&mut g, 2, ms(50));
        assert_eq!(g.current_arm_index(), 0, "equal arms must not switch");
    }
    assert_eq!(
        seen,
        vec![1, 2, 1],
        "probes must round-robin over the non-current arms"
    );
}

#[test]
fn three_arms_pick_the_best_not_merely_a_better_one() {
    let mut g = MtpGate::new(three_arms());
    // Primary is slow (20 tok/s).
    run_until_probe(&mut g, 0, 2, ms(100));
    // Arm 1 probes at 40 tok/s — better, but not the best.
    finish_probe(&mut g, 2, ms(50));
    // Do not let it switch yet: next probe is arm 2 at 100 tok/s.
    let probed = run_until_probe(&mut g, 0, 2, ms(100));
    if probed == 2 {
        finish_probe(&mut g, 1, ms(10));
    }
    // Drive until it settles.
    for _ in 0..(WINDOW_STEPS * 16) {
        if g.current_arm_index() == 2 {
            break;
        }
        if g.is_probing() {
            // Keep probe windows consistent with each arm's true speed.
            let rate = match g.next_arm_index() {
                1 => (2usize, ms(50)),
                2 => (1usize, ms(10)),
                _ => (2usize, ms(100)),
            };
            g.record_step(rate.1, rate.0);
        } else {
            let rate = match g.current_arm_index() {
                1 => (2usize, ms(50)),
                2 => (1usize, ms(10)),
                _ => (2usize, ms(100)),
            };
            g.record_step(rate.1, rate.0);
        }
    }
    assert_eq!(
        g.current_arm_index(),
        2,
        "with all three measured the gate must land on the fastest"
    );
}

#[test]
fn a_depth_regime_change_marks_baselines_stale_without_changing_arm() {
    let mut g = MtpGate::new(two_arms());
    run_until_probe(&mut g, 0, 2, ms(50));
    finish_probe(&mut g, 1, ms(40));
    let before = g.current_arm_index();
    g.observe_depth(64_000);
    assert_eq!(
        g.current_arm_index(),
        before,
        "a regime change re-measures; it must not itself switch arms"
    );
    assert_eq!(g.take_fresh_switch(), None);
}

#[test]
fn zero_wall_steps_do_not_poison_a_baseline() {
    // A step whose wall rounds to zero (mocked clocks, coarse timers) must
    // not produce an infinite tok/s and pin the gate to that arm forever.
    let mut g = MtpGate::new(two_arms());
    drive(&mut g, WINDOW_STEPS, 1, Duration::ZERO);
    assert!(
        g.arm_tps_debug(0).is_none_or(|t| t.is_finite()),
        "a zero-wall window must not yield a non-finite tok/s"
    );
}

#[test]
fn entry_pin_env_is_explicit_and_fail_closed() {
    assert_eq!(
        parse_entry_pin_tokens(None),
        EntryPinConfig {
            tokens: 8,
            source: "default"
        }
    );
    assert_eq!(parse_entry_pin_tokens(Some("0")).tokens, 0);
    assert_eq!(parse_entry_pin_tokens(Some("12")).tokens, 12);
    assert_eq!(
        parse_entry_pin_tokens(Some("not-a-count")),
        EntryPinConfig {
            tokens: 8,
            source: "invalid-env-default"
        }
    );
    assert_eq!(parse_entry_pin_tokens(Some("256")).tokens, 8);
}

#[test]
fn entry_pin_covers_exactly_the_first_eight_post_think_tokens() {
    assert!(entry_pin_active(8, [(true, false, 0)]));
    assert!(entry_pin_active(8, [(true, false, 7)]));
    assert!(!entry_pin_active(8, [(true, false, 8)]));
    assert!(!entry_pin_active(8, [(false, true, 0)]));
    assert!(!entry_pin_active(0, [(true, false, 0)]));
    assert!(entry_pin_active(8, [(true, false, 20), (true, false, 3)]));
}

#[test]
fn entry_pin_preserves_the_selected_proposers_width() {
    // DFlash's trained B16 contract is γ=15; the native alternate uses one
    // draft. Both override Serial, and neither rewrites the other's width.
    assert_eq!(entry_pin_spec_width(ArmKind::Serial, true, 15), Some(15));
    assert_eq!(entry_pin_spec_width(ArmKind::Serial, true, 1), Some(1));
    assert_eq!(entry_pin_spec_width(ArmKind::Serial, false, 15), None);
    assert_eq!(
        entry_pin_spec_width(
            ArmKind::Spec {
                proposer_arm: PROPOSER_ARM_PRIMARY,
                draft_cap: 0,
                num_drafts: 0,
            },
            true,
            15,
        ),
        None
    );
}

#[test]
fn entry_pin_exit_requires_the_missing_serial_transition_cleanup() {
    assert!(entry_pin_exits_to_serial(ArmKind::Serial, true, None));
    assert!(!entry_pin_exits_to_serial(ArmKind::Serial, true, Some(15)));
    assert!(!entry_pin_exits_to_serial(ArmKind::Serial, false, None));
    assert!(!entry_pin_exits_to_serial(
        ArmKind::Spec {
            proposer_arm: PROPOSER_ARM_PRIMARY,
            draft_cap: 0,
            num_drafts: 0,
        },
        true,
        None,
    ));
}

#[test]
fn entry_counter_saturates_without_reentering_the_pin() {
    assert_eq!(advance_entry_counter(0), 1);
    assert_eq!(advance_entry_counter(7), 8);
    assert_eq!(advance_entry_counter(u8::MAX), u8::MAX);
    assert!(!entry_pin_active(8, [(true, false, u8::MAX)]));
}

#[test]
fn native_mtp_thinks_but_dflash_keeps_its_opt_in() {
    let native = ArmKind::Spec {
        proposer_arm: PROPOSER_ARM_PRIMARY,
        draft_cap: 0,
        num_drafts: 1,
    };
    assert!(arm_allows_thinking(native, false, false, true));
    assert!(!arm_allows_thinking(native, false, false, false));

    let dflash = ArmKind::Spec {
        proposer_arm: PROPOSER_ARM_PRIMARY,
        draft_cap: 0,
        num_drafts: 0,
    };
    assert!(!arm_allows_thinking(dflash, true, false, true));
    assert!(arm_allows_thinking(dflash, true, true, true));

    // In a dual-proposer build arm 1 is the native in-checkpoint head.
    let native_alt = ArmKind::Spec {
        proposer_arm: PROPOSER_ARM_ALT,
        draft_cap: 0,
        num_drafts: 1,
    };
    assert!(arm_allows_thinking(native_alt, true, false, true));
    assert!(arm_allows_thinking(ArmKind::Serial, true, false, false));
}
