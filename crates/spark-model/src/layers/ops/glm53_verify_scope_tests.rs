// SPDX-License-Identifier: AGPL-3.0-only

// Standalone CPU gate: rustc --edition=2024 --test this_file.rs.
// The module is deliberately absent until the RED gate is recorded.
#[path = "glm53_verify_scope.rs"]
mod scope;

use scope::{glm53_exact_verify_active, with_glm53_exact_verify};

#[test]
fn experimental_arithmetic_requires_an_explicit_canonical_opt_in() {
    use scope::parse_exact_verify_flag;
    assert_eq!(parse_exact_verify_flag(None), Ok(false));
    assert_eq!(parse_exact_verify_flag(Some("0")), Ok(false));
    assert_eq!(parse_exact_verify_flag(Some("1")), Ok(true));
    for value in ["", "true", "false", " 1", "01", "2", "1\n"] {
        assert!(parse_exact_verify_flag(Some(value)).is_err());
    }
}

#[test]
fn verify_scope_is_default_off_and_preserves_return_value() {
    assert!(!glm53_exact_verify_active());
    assert_eq!(with_glm53_exact_verify(|| 37), 37);
    assert!(!glm53_exact_verify_active());
}

#[test]
fn nested_verify_scope_retains_outer_scope() {
    with_glm53_exact_verify(|| {
        assert!(glm53_exact_verify_active());
        with_glm53_exact_verify(|| assert!(glm53_exact_verify_active()));
        assert!(glm53_exact_verify_active());
    });
    assert!(!glm53_exact_verify_active());
}

#[test]
fn panic_unwinds_only_the_scope_it_entered() {
    with_glm53_exact_verify(|| {
        let panic = std::panic::catch_unwind(|| {
            with_glm53_exact_verify(|| panic!("injected verifier failure"));
        });
        assert!(panic.is_err());
        assert!(glm53_exact_verify_active());
    });
    assert!(!glm53_exact_verify_active());
    assert!(
        std::panic::catch_unwind(|| {
            with_glm53_exact_verify(|| panic!("outer verifier failure"));
        })
        .is_err()
    );
    assert!(!glm53_exact_verify_active());
}

#[test]
fn verify_scope_does_not_leak_to_another_thread() {
    with_glm53_exact_verify(|| {
        std::thread::spawn(|| {
            assert!(!glm53_exact_verify_active());
            with_glm53_exact_verify(|| assert!(glm53_exact_verify_active()));
            assert!(!glm53_exact_verify_active());
        })
        .join()
        .unwrap();
        assert!(glm53_exact_verify_active());
    });
    assert!(!glm53_exact_verify_active());
}
