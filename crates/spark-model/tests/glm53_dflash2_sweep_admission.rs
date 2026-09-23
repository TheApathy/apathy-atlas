// SPDX-License-Identifier: AGPL-3.0-only
#[path = "../src/model/glm53/dflash2_probe_contract.rs"]
#[allow(dead_code)]
mod contract;
#[path = "../examples/glm53_dflash2_kv_parity/sweep.rs"]
mod sweep;
use contract::{ProbeLayout, ProbeStage};
fn layout() -> ProbeLayout {
    ProbeLayout::new(5, 129 * 16 * 1024 * 2, 65536, 57344, 2168320, 28, 154880).unwrap()
}
#[test]
fn entire_projected_sweep_is_admitted_before_any_context_advance() {
    assert!(sweep::admit(&layout(), &layout(), &[1, 1, 2, 60, 1360], true).is_ok());
    for contexts in [&[1, 1361][..], &[1, 2039, 2047], &[1, 2047, 2]] {
        assert!(sweep::admit(&layout(), &layout(), contexts, true).is_err());
    }
    assert!(sweep::admit(&layout(), &layout(), &[1, 2039, 2047], false).is_ok());
    assert!(sweep::admit(&layout(), &layout(), &[], true).is_err());
    assert!(sweep::admit(&layout(), &layout(), &[0], true).is_err());
}
#[test]
fn actual_layout_not_a_duplicate_constant_controls_admission() {
    let small = ProbeLayout::new(1, 20480, 2, 2, 4, 4, 100).unwrap();
    assert!(sweep::admit(&small, &small, &[2047], true).is_ok());
    assert!(sweep::admit(&layout(), &layout(), &[2047], true).is_err());
    assert!(sweep::admit(&small, &layout(), &[1], true).is_err());
}
#[test]
fn forced_policy_contexts_are_checked_even_when_selected_context_is_one() {
    // About45.75 MiB of normal capture + 2 MiB per projected context: context1 fits,
    // forced full acceptance at9 fits, but first rejection at10 exceeds64MiB.
    let policy_cap =
        ProbeLayout::new(5, 31 * 128 * 1024, 2, 7 * 512 * 1024 * 2, 4, 28, 154880).unwrap();
    assert!(
        policy_cap
            .clone()
            .with_projected_target(1, 512 * 1024)
            .is_ok()
    );
    assert!(
        policy_cap
            .clone()
            .with_projected_target(9, 512 * 1024)
            .is_ok()
    );
    assert!(
        policy_cap
            .clone()
            .with_projected_target(10, 512 * 1024)
            .is_err()
    );
    assert!(sweep::admit(&policy_cap, &policy_cap, &[1], true).is_err());
    assert!(sweep::admit(&policy_cap, &policy_cap, &[1], false).is_ok());
}
