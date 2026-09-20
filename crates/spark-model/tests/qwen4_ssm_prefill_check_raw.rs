// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/qwen3_ssm/qwen4_prefill_check_raw.rs"]
mod raw;
use raw::{Element, compare_finite, validate_check};

#[test]
fn check_requires_exact_without_weakening_unselected_mode() {
    assert!(validate_check(false, false).is_ok());
    assert!(validate_check(true, false).is_ok());
    assert!(validate_check(true, true).is_ok());
    assert!(validate_check(false, true).is_err());
}

#[test]
fn bf16_exact_and_nonfinite_rejection() {
    for bits in [0u16, 0x8000, 0x3f80, 0xbf80, 0x7f7f, 1] {
        assert!(compare_finite(&bits.to_ne_bytes(), &bits.to_ne_bytes(), Element::Bf16).is_ok());
    }
    for bits in [0x7f80u16, 0xff80, 0x7fc0, 0xff81] {
        let bytes = bits.to_ne_bytes();
        assert!(compare_finite(&bytes, &bytes, Element::Bf16).is_err());
        assert!(compare_finite(&[0, 0], &bytes, Element::Bf16).is_err());
    }
    // Numerically equal signed zero is deliberately NOT byte-exact.
    assert!(compare_finite(&0u16.to_ne_bytes(), &0x8000u16.to_ne_bytes(), Element::Bf16).is_err());
}

#[test]
fn fp32_preserves_bits_and_rejects_nan_inf_on_both_sides() {
    for bits in [0u32, 0x80000000, 0x3f800000, 0xbf800000, 0x7f7fffff, 1] {
        assert!(compare_finite(&bits.to_ne_bytes(), &bits.to_ne_bytes(), Element::F32).is_ok());
    }
    for bits in [0x7f800000u32, 0xff800000, 0x7fc00000, 0xff800001] {
        let bytes = bits.to_ne_bytes();
        assert!(compare_finite(&bytes, &bytes, Element::F32).is_err());
        assert!(compare_finite(&0u32.to_ne_bytes(), &bytes, Element::F32).is_err());
    }
}

#[test]
fn hostile_shapes_and_first_mismatch_are_reported() {
    assert!(compare_finite(&[], &[], Element::Bf16).is_err());
    assert!(compare_finite(&[0], &[0], Element::Bf16).is_err());
    assert!(compare_finite(&[0, 0], &[0, 0, 0, 0], Element::Bf16).is_err());
    assert!(compare_finite(&[0, 0], &[0, 0], Element::F32).is_err());
    let error = compare_finite(&[0, 0, 1, 0], &[0, 0, 2, 0], Element::Bf16).unwrap_err();
    assert!(error.contains("element 1, byte 2"));
}
