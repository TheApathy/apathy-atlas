// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/qwen3_ssm/qwen4_prefill_gemm/plan.rs"]
mod plan;
use plan::{Mode, Projection, parse_mode};
use std::ffi::OsStr;

#[test]
fn only_documented_modes_parse() {
    assert_eq!(Mode::default(), Mode::Off);
    assert_eq!(parse_mode(None), Ok(Mode::Off));
    for (value, expected) in [
        ("0", Mode::Off),
        ("qkvz", Mode::Qkvz),
        ("out", Mode::Out),
        ("all", Mode::All),
    ] {
        assert_eq!(parse_mode(Some(OsStr::new(value))), Ok(expected));
    }
    for value in ["1", "", "off", "OUT", "All", "all ", " out", "true"] {
        assert!(parse_mode(Some(OsStr::new(value))).is_err());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(parse_mode(Some(OsStr::from_bytes(&[255]))).is_err());
    }
}

#[test]
fn receipt_and_projection_mapping_are_explicit() {
    for (mode, label, qkvz, output) in [
        (Mode::Off, "off", false, false),
        (Mode::Out, "out", false, true),
        (Mode::All, "all", true, true),
    ] {
        assert_eq!(mode.as_str(), label);
        assert_eq!(mode.uses_gemm(Projection::Qkvz), qkvz);
        assert_eq!(mode.uses_gemm(Projection::Output), output);
    }
}

#[test]
fn active_modes_require_ssm_exact_and_reject_strict_check() {
    for mode in [Mode::Off, Mode::Out, Mode::All] {
        for exact in [false, true] {
            for check in [false, true] {
                assert_eq!(
                    mode.admit(exact, check).is_ok(),
                    mode == Mode::Off || (exact && !check)
                );
            }
        }
    }
}

#[test]
fn missing_handle_never_silently_changes_route() {
    assert!(Mode::Off.validate_handle(0).is_ok());
    for mode in [Mode::Out, Mode::All] {
        assert!(mode.validate_handle(0).is_err());
        assert!(mode.validate_handle(1).is_ok());
    }
}
