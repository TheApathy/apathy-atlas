// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/qwen3_ssm/qwen4_prefill_gemm/plan.rs"]
mod plan;

use plan::{Mode, Projection, parse_mode};
use std::ffi::OsStr;

fn qkvz_mode() -> Mode {
    parse_mode(Some(OsStr::new("qkvz"))).expect("explicit QKVZ-only mode is required")
}

#[test]
fn qkvz_only_selects_the_input_projection() {
    let mode = qkvz_mode();
    assert!(mode.uses_gemm(Projection::Qkvz));
    assert!(!mode.uses_gemm(Projection::Output));
    assert_ne!(mode, Mode::Off);
    assert_ne!(mode, Mode::Out);
    assert_ne!(mode, Mode::All);
}

#[test]
fn qkvz_only_has_a_distinct_truthful_selector_label() {
    assert_eq!(qkvz_mode().as_str(), "qkvz");
}

#[test]
fn existing_modes_and_absent_default_are_unchanged() {
    assert_eq!(Mode::default(), Mode::Off);
    assert_eq!(parse_mode(None), Ok(Mode::Off));
    for (value, mode, label, qkvz, output) in [
        ("0", Mode::Off, "off", false, false),
        ("out", Mode::Out, "out", false, true),
        ("all", Mode::All, "all", true, true),
    ] {
        assert_eq!(parse_mode(Some(OsStr::new(value))), Ok(mode));
        assert_eq!(mode.as_str(), label);
        assert_eq!(mode.uses_gemm(Projection::Qkvz), qkvz);
        assert_eq!(mode.uses_gemm(Projection::Output), output);
        for exact in [false, true] {
            for check in [false, true] {
                assert_eq!(
                    mode.admit(exact, check).is_ok(),
                    mode == Mode::Off || (exact && !check)
                );
            }
        }
        assert_eq!(mode.validate_handle(0).is_ok(), mode == Mode::Off);
        assert!(mode.validate_handle(1).is_ok());
    }
}

#[test]
fn qkvz_only_requires_exact_staging_and_rejects_strict_replay_check() {
    let mode = qkvz_mode();
    for exact in [false, true] {
        for check in [false, true] {
            assert_eq!(mode.admit(exact, check).is_ok(), exact && !check);
        }
    }
    assert!(
        mode.admit(false, false)
            .unwrap_err()
            .contains("ATLAS_QWEN4_PREFILL_SSM_EXACT=1")
    );
    assert!(
        mode.admit(true, true)
            .unwrap_err()
            .contains("ATLAS_QWEN4_PREFILL_SSM_CHECK=1")
    );
}

#[test]
fn qkvz_only_requires_the_loaded_kernel_without_fallback() {
    let mode = qkvz_mode();
    assert!(mode.validate_handle(0).is_err());
    assert!(mode.validate_handle(1).is_ok());
    assert!(mode.validate_handle(u64::MAX).is_ok());
}

#[test]
fn aliases_whitespace_and_non_utf8_do_not_select_qkvz() {
    for value in [
        "",
        "1",
        "off",
        "QKVZ",
        "Qkvz",
        " qkvz",
        "qkvz ",
        "qkvz\n",
        "qkvz\0",
        "qkv",
        "qkvz-only",
        "true",
    ] {
        assert!(
            parse_mode(Some(OsStr::new(value))).is_err(),
            "unexpected selector {value:?}"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(parse_mode(Some(OsStr::from_bytes(b"qkvz\xff"))).is_err());
    }
}
