// SPDX-License-Identifier: AGPL-3.0-only
#[path = "../src/layers/ops/glm53_exl3_verify_regular.rs"]
mod selection;
use selection::selected;
use std::ffi::OsStr;

#[test]
fn regular_is_explicit_and_only_applies_inside_small_exact_verification() {
    for value in [None, Some(OsStr::new("0")), Some(OsStr::new("1"))] {
        for exact in [false, true] {
            for prefill in [false, true] {
                for rows in 0..=16 {
                    assert_eq!(
                        selected(value, exact, prefill, rows).unwrap(),
                        value == Some(OsStr::new("1"))
                            && exact
                            && !prefill
                            && (2..=8).contains(&rows)
                    );
                }
            }
        }
    }
}

#[test]
fn malformed_configuration_is_rejected_even_outside_selected_scopes() {
    for value in ["", "true", "01", " 1", "1 ", "2", "1\n"] {
        assert!(selected(Some(OsStr::new(value)), false, true, 1).is_err());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        assert!(selected(Some(OsStr::from_bytes(&[0xff])), true, false, 8).is_err());
    }
}

#[test]
fn loader_preserves_plan_and_only_substitutes_the_regular_symbol_after_admission() {
    let source = include_str!("../src/layers/ops/glm53_exl3.rs");
    let body = source
        .split("pub fn load_row_exact(")
        .nth(1)
        .unwrap()
        .split("pub fn launch(")
        .next()
        .unwrap();
    assert!(
        body.find("ensure_row_exact_plan(plan)?").unwrap()
            < body.find("regular_verify::selected(").unwrap()
    );
    assert!(body.contains("glm53_exact_verify_active()"));
    assert!(body.contains("glm53_exact_wide_prefill_active()"));
    assert!(body.contains("plan.symbol()"));
    assert!(body.contains("glm53_exl3_rowexact_k"));
    assert!(!body.contains("Self::load("));
}
