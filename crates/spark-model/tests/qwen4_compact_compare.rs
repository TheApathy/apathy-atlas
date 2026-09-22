// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/moe/qwen4_compact_compare.rs"]
mod compare;

#[test]
fn finite_bf16_including_signed_zero_subnormals_and_maximum_passes() {
    let bytes: Vec<u8> = [0u16, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert!(compare::finite(&bytes).is_ok());
    assert!(compare::exact(&bytes, &bytes).is_ok());
}

#[test]
fn infinities_and_nan_in_either_operand_fail_even_when_equal() {
    for bits in [0x7f80u16, 0xff80, 0x7f81, 0x7fc0, 0xffff] {
        let bad = bits.to_le_bytes();
        assert!(compare::finite(&bad).is_err());
        assert!(compare::exact(&bad, &bad).is_err());
        assert!(compare::exact(&[0, 0], &bad).is_err());
        assert!(compare::exact(&bad, &[0, 0]).is_err());
    }
}

#[test]
fn byte_mismatch_signed_zero_and_malformed_extents_fail() {
    assert!(compare::finite(&[]).is_err());
    assert!(compare::finite(&[0]).is_err());
    assert!(compare::exact(&[0, 0], &[0, 0, 0, 0]).is_err());
    assert!(compare::exact(&[0, 0], &[0, 0x80]).is_err());
    assert!(compare::exact(&[0, 0, 0, 0], &[0, 0, 1, 0]).is_err());
}

#[test]
fn check_selector_requires_both_dependencies() {
    assert!(compare::admit(false, false, false).is_ok());
    assert!(compare::admit(true, true, true).is_ok());
    for (compact, f8) in [(false, false), (false, true), (true, false)] {
        assert!(compare::admit(true, compact, f8).is_err());
    }
}

#[test]
fn full_extent_bound_is_exact_and_checked() {
    assert_eq!(compare::extent(2048, 2560), Ok(104_857_600));
    assert_eq!(compare::extent(2, 640), Ok(25_600));
    for (rows, n) in [
        (0, 640),
        (1, 640),
        (2049, 640),
        (8192, 2560),
        (2, 641),
        (usize::MAX, 2560),
    ] {
        assert!(compare::extent(rows, n).is_err());
    }
}
