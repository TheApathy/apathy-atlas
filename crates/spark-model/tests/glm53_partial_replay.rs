// SPDX-License-Identifier: AGPL-3.0-only

//! RED: startup-pinned opt-in dispatch only; no arithmetic or GPU claim.

#[path = "../src/model/glm53/partial_replay.rs"]
#[allow(dead_code)]
mod partial_replay;

use partial_replay::{PartialReplaySetting, ReplayPath};
use std::ffi::{OsStr, OsString};

fn setting(value: Option<&str>) -> PartialReplaySetting {
    PartialReplaySetting::parse(value.map(OsStr::new)).unwrap()
}

#[test]
fn absent_and_zero_keep_every_partial_prefix_on_the_scalar_path() {
    for setting in [setting(None), setting(Some("0"))] {
        for total in 2..=8 {
            for committed in 1..total {
                assert_eq!(setting.path(total, committed).unwrap(), ReplayPath::Scalar);
            }
        }
    }
}

#[test]
fn explicit_one_selects_only_multirow_partial_prefixes_never_first_rejection() {
    let setting = setting(Some("1"));
    for total in 2..=8 {
        for committed in 1..total {
            let expected = if committed == 1 {
                ReplayPath::Scalar
            } else {
                ReplayPath::Wide
            };
            assert_eq!(
                setting.path(total, committed).unwrap(),
                expected,
                "total={total} committed={committed}"
            );
        }
    }
}

#[test]
fn full_acceptance_zero_and_out_of_range_extents_are_not_partial_replay_plans() {
    for setting in [setting(None), setting(Some("0")), setting(Some("1"))] {
        for (total, committed) in [
            (0, 0),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 2),
            (8, 0),
            (8, 8),
            (8, 9),
            (9, 1),
            (9, 7),
            (usize::MAX, 1),
            (8, usize::MAX),
            (usize::MAX, usize::MAX),
        ] {
            assert!(
                setting.path(total, committed).is_err(),
                "invalid total={total} committed={committed}"
            );
        }
        for total in 2..=8 {
            assert!(
                setting.path(total, total).is_err(),
                "full branch must stay separate"
            );
        }
    }
}

#[test]
fn selector_is_strict_os_input_and_does_not_borrow_later_configuration_mutation() {
    for value in [
        "", "true", "false", "yes", "on", "off", "01", "2", "-1", " 1", "1\n",
    ] {
        assert!(
            PartialReplaySetting::parse(Some(OsStr::new(value))).is_err(),
            "{value:?}"
        );
    }
    let mut source = OsString::from("0");
    let pinned = PartialReplaySetting::parse(Some(&source)).unwrap();
    source.clear();
    source.push("1");
    assert_eq!(pinned.path(8, 7).unwrap(), ReplayPath::Scalar);
    assert_eq!(
        PartialReplaySetting::parse(Some(&source))
            .unwrap()
            .path(8, 7)
            .unwrap(),
        ReplayPath::Wide
    );
}

#[cfg(unix)]
#[test]
fn non_utf8_selector_is_rejected_instead_of_silently_disabling_the_candidate() {
    use std::os::unix::ffi::OsStrExt;
    assert!(PartialReplaySetting::parse(Some(OsStr::from_bytes(&[0xff]))).is_err());
}
