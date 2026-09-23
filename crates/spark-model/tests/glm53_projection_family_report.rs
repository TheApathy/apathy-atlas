// SPDX-License-Identifier: AGPL-3.0-only
#[allow(dead_code)]
#[path = "../examples/glm53_projection_probe/contract.rs"]
mod contract;
#[path = "../examples/glm53_projection_probe/family_report.rs"]
mod family_report;

#[test]
fn same_family_requires_all_full_split_and_reverse_bytes() {
    let expected = [0x3f80u16.to_le_bytes(); 8].concat();
    let report = family_report::compare(&expected, [&expected; 4]).unwrap();
    assert_eq!(report["gemv_family_exact"], true);
    assert_eq!(report["model_quality_qualified"], false);
    assert_eq!(report["speed_qualified"], false);
    for index in 0..4 {
        let mut changed = expected.clone();
        changed[0] ^= 1;
        let mut arms = [&expected[..]; 4];
        arms[index] = &changed;
        assert_eq!(
            family_report::compare(&expected, arms).unwrap()["gemv_family_exact"],
            false
        );
    }
}

#[test]
fn invalid_extent_or_nonfinite_family_data_is_not_a_receipt() {
    let expected = [0x3f80u16.to_le_bytes(); 8].concat();
    let short = &expected[..2];
    assert!(family_report::compare(&expected, [short; 4]).is_err());
    let nonfinite = [0x7fc0u16.to_le_bytes(); 8].concat();
    assert!(family_report::compare(&expected, [&nonfinite; 4]).is_err());
}
