// SPDX-License-Identifier: AGPL-3.0-only
#[allow(dead_code)]
#[path = "../examples/glm53_projection_probe/contract.rs"]
mod contract;
use contract::{compare, validate_bf16};
#[test]
fn finite_exact_extent_is_required_without_padding_or_silent_casts() {
    let x = [0x3f80u16.to_le_bytes(); 8].concat();
    assert!(validate_bf16(&x, 16).is_ok());
    for expected in [0, 15, 18] {
        assert!(validate_bf16(&x, expected).is_err());
    }
    for value in [0x7f80u16, 0xff80, 0x7fc1] {
        assert!(validate_bf16(&value.to_le_bytes(), 2).is_err());
    }
}
#[test]
fn one_bit_difference_cannot_pass_raw_equality() {
    let a = [0x3f80u16.to_le_bytes(); 8].concat();
    let mut b = a.clone();
    b[0] ^= 1;
    assert_eq!(compare(&a, &a).unwrap()["exact"], true);
    let diff = compare(&a, &b).unwrap();
    assert_eq!(diff["exact"], false);
    assert_eq!(diff["different_bytes"], 1);
    assert_eq!(diff["first_byte"], 0);
    assert!(diff["relative_l2"].as_f64().unwrap() > 0.0);
    assert!(compare(&a, &b[..2]).is_err());
}
