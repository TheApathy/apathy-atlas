// SPDX-License-Identifier: AGPL-3.0-only
//! Pure startup policy and thread-local verification scope; no environment I/O.
#[path = "../src/layers/ops/glm53_native_rows.rs"]
mod scope;
use scope::{
    NativeRowsSetting, glm53_native_rows_active, native_rows_selected, with_glm53_native_rows,
};
use std::ffi::OsStr;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn batching_is_latched_requires_prepared_and_rejects_noncanonical_values() {
    use scope::glm53_native_rows_batched_active;
    let selected =
        NativeRowsSetting::parse_with_batched(Some(OsStr::new("1")), Some(OsStr::new("1")))
            .unwrap();
    assert!(selected.enabled() && selected.batched());
    assert!(selected.validate_exact_mode(None).is_err());
    for prepared in [None, Some(OsStr::new("0"))] {
        assert!(NativeRowsSetting::parse_with_batched(prepared, Some(OsStr::new("1"))).is_err());
    }
    for value in ["", "true", "01", " 1", "1 ", "2", "1\n"] {
        assert!(
            NativeRowsSetting::parse_with_batched(Some(OsStr::new("1")), Some(OsStr::new(value)))
                .is_err()
        );
    }
    for batch in [None, Some(OsStr::new("0"))] {
        assert!(
            !NativeRowsSetting::parse_with_batched(Some(OsStr::new("1")), batch)
                .unwrap()
                .batched()
        );
    }
    assert!(!glm53_native_rows_batched_active());
    with_glm53_native_rows(selected, || {
        assert!(glm53_native_rows_active() && glm53_native_rows_batched_active());
        with_glm53_native_rows(true, || assert!(!glm53_native_rows_batched_active()));
        assert!(glm53_native_rows_batched_active());
        std::thread::spawn(|| assert!(!glm53_native_rows_batched_active()))
            .join()
            .unwrap();
        assert!(
            catch_unwind(AssertUnwindSafe(|| with_glm53_native_rows(
                false,
                || panic!("nested")
            )))
            .is_err()
        );
        assert!(glm53_native_rows_batched_active());
    });
    assert!(!glm53_native_rows_active() && !glm53_native_rows_batched_active());
}

#[test]
fn absent_and_exact_zero_preserve_the_original_path() {
    assert!(!NativeRowsSetting::parse(None).unwrap().enabled());
    assert!(
        !NativeRowsSetting::parse(Some(OsStr::new("0")))
            .unwrap()
            .enabled()
    );
    assert!(
        NativeRowsSetting::parse(Some(OsStr::new("1")))
            .unwrap()
            .enabled()
    );
}

#[test]
fn startup_policy_rejects_noncanonical_values() {
    for value in ["", "true", "false", "01", " 1", "1 ", "2", "-1", "1\n"] {
        assert!(
            NativeRowsSetting::parse(Some(OsStr::new(value))).is_err(),
            "{value:?}"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(NativeRowsSetting::parse(Some(OsStr::from_bytes(&[0xff]))).is_err());
    }
}

#[test]
fn enabled_requires_an_explicit_exact_verifier_without_changing_unselected_admission() {
    let enabled = NativeRowsSetting::parse(Some(OsStr::new("1"))).unwrap();
    assert!(enabled.validate_exact_mode(Some(OsStr::new("1"))).is_ok());
    for value in [
        None,
        Some(OsStr::new("0")),
        Some(OsStr::new("true")),
        Some(OsStr::new("")),
    ] {
        assert!(enabled.validate_exact_mode(value).is_err());
        assert!(
            NativeRowsSetting::parse(None)
                .unwrap()
                .validate_exact_mode(value)
                .is_ok()
        );
    }
}

#[test]
fn nested_true_and_false_scopes_restore_their_actual_parent_setting() {
    assert!(!glm53_native_rows_active());
    with_glm53_native_rows(true, || {
        assert!(glm53_native_rows_active());
        with_glm53_native_rows(false, || assert!(!glm53_native_rows_active()));
        assert!(glm53_native_rows_active());
        with_glm53_native_rows(true, || assert!(glm53_native_rows_active()));
        assert!(glm53_native_rows_active());
    });
    assert!(!glm53_native_rows_active());
}

#[test]
fn panic_and_other_threads_cannot_leak_selected_state() {
    assert!(
        catch_unwind(AssertUnwindSafe(|| with_glm53_native_rows(
            true,
            || panic!("scope")
        )))
        .is_err()
    );
    assert!(!glm53_native_rows_active());
    with_glm53_native_rows(true, || {
        std::thread::spawn(|| assert!(!glm53_native_rows_active()))
            .join()
            .unwrap();
        assert!(glm53_native_rows_active());
    });
}

#[test]
fn only_selected_eager_verifier_native_two_through_eight_rows_change_dispatch() {
    for selected in [false, true] {
        with_glm53_native_rows(selected, || {
            for exact in [false, true] {
                for prefill in [false, true] {
                    for native in [false, true] {
                        for rows in [0, 1, 2, 3, 7, 8, 9, 2048] {
                            assert_eq!(
                                native_rows_selected(exact, prefill, rows, native),
                                selected && exact && !prefill && native && (2..=8).contains(&rows)
                            );
                        }
                    }
                }
            }
        });
    }
}
