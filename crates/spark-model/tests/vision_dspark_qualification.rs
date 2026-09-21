// SPDX-License-Identifier: AGPL-3.0-only

use spark_model::factory::vision_speculation as policy;

use policy::{Qualification, Request};
use std::ffi::OsStr;

fn request() -> Request {
    Request {
        is_vision: true,
        target_only: false,
        dflash_only: true,
        embedded_drafter: true,
        high_speed_swap: false,
        verify_capacity: 6,
        has_adapters: false,
    }
}

#[test]
fn no_opt_in_cannot_admit_speculation() {
    for value in [None, Some(OsStr::new("0"))] {
        assert!(
            !Qualification::parse(value)
                .unwrap()
                .admit(request())
                .unwrap()
        );
    }
}

#[test]
fn opt_in_requires_actual_vision_and_embedded_dspark_only() {
    let enabled = Qualification::parse(Some(OsStr::new("1"))).unwrap();
    assert!(enabled.admit(request()).unwrap());
    for changed in [
        Request {
            is_vision: false,
            ..request()
        },
        Request {
            dflash_only: false,
            ..request()
        },
        Request {
            embedded_drafter: false,
            ..request()
        },
        Request {
            high_speed_swap: true,
            ..request()
        },
        Request {
            verify_capacity: 5,
            ..request()
        },
        Request {
            has_adapters: true,
            ..request()
        },
    ] {
        assert!(enabled.admit(changed).is_err());
    }
    assert!(
        !enabled
            .admit(Request {
                target_only: true,
                ..request()
            })
            .unwrap()
    );
}

#[test]
fn malformed_opt_ins_fail_instead_of_silently_enabling_or_disabling() {
    for value in ["", "true", "false", "01", " 1", "2"] {
        assert!(Qualification::parse(Some(OsStr::new(value))).is_err());
    }
}

#[test]
fn incompatible_state_modes_are_rejected() {
    assert!(policy::INCOMPATIBLE_FLAGS.contains(&"ATLAS_MLA_NO_BATCH"));
    for key in policy::INCOMPATIBLE_FLAGS {
        assert!(policy::validate_runtime_value(key, Some(OsStr::new("1"))).is_err());
        assert!(policy::validate_runtime_value(key, Some(OsStr::new("true"))).is_err());
        policy::validate_runtime_value(key, None).unwrap();
        policy::validate_runtime_value(key, Some(OsStr::new("0"))).unwrap();
    }
}

#[test]
fn captured_context_and_masked_verify_are_required() {
    for key in ["ATLAS_DSPARK_CAPTURE", "ATLAS_DFLASH_MASKED_VERIFY"] {
        policy::validate_required_runtime_value(key, Some(OsStr::new("1"))).unwrap();
        for value in [None, Some(OsStr::new("0")), Some(OsStr::new("true"))] {
            assert!(policy::validate_required_runtime_value(key, value).is_err());
        }
    }
    let source = include_str!("../src/factory/vision_speculation.rs");
    assert!(source.contains("[\"ATLAS_DSPARK_CAPTURE\", \"ATLAS_DFLASH_MASKED_VERIFY\"]"));
}

#[test]
fn factory_common_speculation_switch_does_not_imply_a_second_legacy_proposer() {
    for common in [false, true] {
        assert!(!policy::factory_legacy_speculation(common, false, true));
        assert_eq!(
            policy::factory_legacy_speculation(common, false, false),
            common
        );
        for draft in [false, true] {
            assert!(policy::factory_legacy_speculation(common, true, draft));
        }
    }
    let source = include_str!("../src/factory/build.rs");
    assert!(source.contains("super::vision_speculation::factory_legacy_speculation("));
    let server = include_str!("../../spark-server/src/main_modules/serve_phases/build.rs");
    assert!(server.contains("args.speculative || args.dflash"));
    let admission =
        include_str!("../../spark-server/src/main_modules/serve_phases/vision_dspark.rs");
    assert!(admission.contains("!args.speculative"));
}
