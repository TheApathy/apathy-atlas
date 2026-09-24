// SPDX-License-Identifier: AGPL-3.0-only

//! Production value-policy RED; no kernel or numerical qualification.

#[path = "../src/layers/ops/glm53_exl3_route_policy.rs"]
mod route_policy;

use route_policy::Glm53Exl3RoutePolicy;
use std::ffi::OsStr;

const ROWS: [u32; 9] = [1, 8, 9, 106, 1023, 1024, 2038, 2047, 2048];
const PAIR_BYTES: usize = 4096 * 4;
const MIB: u64 = 1024 * 1024;

fn policy(private: Option<&str>, prefill: Option<&str>) -> Glm53Exl3RoutePolicy {
    Glm53Exl3RoutePolicy::parse(private.map(OsStr::new), prefill.map(OsStr::new)).unwrap()
}

#[test]
fn absent_and_zero_preserve_unselected_routes_and_baseline_extents() {
    let baseline = policy(None, None);
    for private in [None, Some("0")] {
        for prefill in [None, Some("0")] {
            let got = policy(private, prefill);
            assert_eq!(got, baseline);
            assert!(!got.prefill_enabled());
            assert_eq!(got.scratch_private_bytes(), MIB);
            for rows in ROWS {
                assert!(!got.private_for(rows).unwrap());
                assert_eq!(
                    got.private_bytes(rows).unwrap(),
                    rows.min(8) as usize * 8 * PAIR_BYTES
                );
            }
        }
    }
}

#[test]
fn existing_private_flag_remains_small_row_only_without_new_opt_in() {
    for prefill in [None, Some("0")] {
        let got = policy(Some("1"), prefill);
        assert!(!got.prefill_enabled());
        assert_eq!(got.scratch_private_bytes(), MIB);
        for rows in ROWS {
            assert_eq!(got.private_for(rows).unwrap(), rows <= 8);
            assert_eq!(
                got.private_bytes(rows).unwrap(),
                rows.min(8) as usize * 8 * PAIR_BYTES
            );
        }
    }
}

#[test]
fn explicit_prefill_uses_full_source_pair_extent_through_m2048() {
    let got = policy(Some("1"), Some("1"));
    assert!(got.prefill_enabled());
    assert_eq!(got.scratch_private_bytes(), 256 * MIB);
    for rows in ROWS {
        assert!(got.private_for(rows).unwrap());
        assert_eq!(
            got.private_bytes(rows).unwrap(),
            rows as usize * 8 * PAIR_BYTES
        );
        assert!(got.private_bytes(rows).unwrap() as u64 <= got.scratch_private_bytes());
    }
    assert_eq!(got.private_bytes(106).unwrap(), 13_893_632);
    assert_eq!(got.private_bytes(2038).unwrap(), 267_124_736);
    assert_eq!(got.private_bytes(2048).unwrap(), 268_435_456);
}

#[test]
fn private_prefill_requires_explicit_existing_private_flag() {
    for private in [None, Some("0")] {
        assert!(
            Glm53Exl3RoutePolicy::parse(private.map(OsStr::new), Some(OsStr::new("1"))).is_err()
        );
    }
}

#[test]
fn canonical_flags_are_strict_even_when_the_other_flag_is_off() {
    for bad in ["", " ", "true", "false", "01", "1 ", "+1", "2", "\0", "yes"] {
        for other in [None, Some("0"), Some("1")] {
            assert!(
                Glm53Exl3RoutePolicy::parse(Some(OsStr::new(bad)), other.map(OsStr::new)).is_err(),
                "invalid private={bad:?}, prefill={other:?}"
            );
            assert!(
                Glm53Exl3RoutePolicy::parse(other.map(OsStr::new), Some(OsStr::new(bad))).is_err(),
                "private={other:?}, invalid prefill={bad:?}"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn nonunicode_flags_are_errors_not_absent_or_disabled() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    let bad = OsString::from_vec(vec![b'1', 0xff]);
    for other in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
        assert!(Glm53Exl3RoutePolicy::parse(Some(bad.as_os_str()), other).is_err());
        assert!(Glm53Exl3RoutePolicy::parse(other, Some(bad.as_os_str())).is_err());
    }
    assert!(Glm53Exl3RoutePolicy::parse(Some(&bad), Some(&bad)).is_err());
}

#[test]
fn every_policy_rejects_invalid_geometry_before_selecting_or_sizing() {
    for got in [
        policy(None, None),
        policy(Some("1"), None),
        policy(Some("1"), Some("1")),
    ] {
        for rows in [0, 8193, u32::MAX] {
            assert!(got.private_for(rows).is_err(), "rows={rows}");
            assert!(got.private_bytes(rows).is_err(), "rows={rows}");
        }
    }
}

#[test]
fn selected_prefill_requires_fused_mode_but_legacy_serial_admission_remains() {
    let full = policy(Some("1"), Some("1"));
    for got in [policy(None, None), policy(Some("1"), None), full] {
        for mode in [None, Some(OsStr::new("fused"))] {
            assert!(got.validate_moe_mode(mode).is_ok());
        }
        assert_eq!(
            got.validate_moe_mode(Some(OsStr::new("serial-reference")))
                .is_ok(),
            !got.prefill_enabled()
        );
    }
}

#[test]
fn invalid_moe_modes_do_not_silently_select_fused() {
    for got in [
        policy(None, None),
        policy(Some("1"), None),
        policy(Some("1"), Some("1")),
    ] {
        for bad in ["", "FUSED", "fused ", "serial", "0", "1"] {
            assert!(got.validate_moe_mode(Some(OsStr::new(bad))).is_err());
        }
    }
}

#[cfg(unix)]
#[test]
fn nonunicode_moe_mode_is_rejected_for_every_policy() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    let bad = OsString::from_vec(vec![0xff]);
    for got in [
        policy(None, None),
        policy(Some("1"), None),
        policy(Some("1"), Some("1")),
    ] {
        assert!(got.validate_moe_mode(Some(bad.as_os_str())).is_err());
    }
}

#[test]
fn policy_is_owned_copy_eq_and_independent_of_later_parse_calls() {
    fn require_copy_eq<T: Copy + Eq>() {}
    require_copy_eq::<Glm53Exl3RoutePolicy>();
    let full = {
        let private = String::from("1");
        let prefill = String::from("1");
        policy(Some(&private), Some(&prefill))
    };
    let saved = full;
    let off = policy(None, None);
    let small = policy(Some("1"), None);
    assert_eq!(full, saved);
    assert_ne!(full, off);
    assert_ne!(full, small);
    assert!(!small.private_for(106).unwrap());
    assert!(saved.private_for(106).unwrap());
    assert_eq!(saved.scratch_private_bytes(), 256 * MIB);
}
