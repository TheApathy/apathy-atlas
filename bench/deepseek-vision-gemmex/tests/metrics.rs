// SPDX-License-Identifier: AGPL-3.0-only
#[path = "../src/metrics.rs"]
mod metrics;
use deepseek_vision_p1::contract::check_bf16;
fn uniform(value: f32) -> Vec<u8> {
    let cell = ((value.to_bits() >> 16) as u16).to_le_bytes();
    (0..20 * 5632).flat_map(|_| cell).collect()
}
#[test]
fn fixed_fc1_metrics_identity_and_reference_denominator() {
    let one = uniform(1.0);
    let two = uniform(2.0);
    let identity = metrics::compare(&one, &one).unwrap();
    assert_eq!(identity["cosine"], 1.0);
    assert_eq!(identity["worst_row_cosine"], 1.0);
    assert_eq!(identity["relative_l2"], 0.0);
    assert_eq!(identity["f32_exact_fraction"], 1.0);
    let forward = metrics::compare(&one, &two).unwrap();
    let reverse = metrics::compare(&two, &one).unwrap();
    assert_eq!(forward["relative_l2"], 0.5);
    assert_eq!(reverse["relative_l2"], 1.0);
    assert_eq!(forward["max_abs_error"], 1.0);
}
#[test]
fn metrics_reject_wrong_extent_nonfinite_and_zero_rows() {
    let one = uniform(1.0);
    assert!(metrics::compare(&one[..one.len() - 2], &one).is_err());
    assert!(metrics::compare(&one, &uniform(0.0)).is_err());
    let mut zero_row = one.clone();
    zero_row[..5632 * 2].fill(0);
    assert!(metrics::compare(&zero_row, &one).is_err());
    for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let invalid = uniform(value);
        assert!(check_bf16(&invalid).is_err());
        assert!(metrics::compare(&invalid, &one).is_err());
    }
}
