// SPDX-License-Identifier: AGPL-3.0-only
use serde_json::{Value, json};

#[test]
fn retained_metric_decimals_survive_exact_admission_roundtrip() {
    // Decimal witnesses from the pinned default teacher report. Admission
    // reconstruction compares exact values, not a relaxed numeric tolerance.
    let wire = r#"[0.9999999999999998,1.0000000000000002,
        0.17283950746059418,1.0494204885217145e-7,3.0094284453621966e-8,
        0.000029727769181532315,0.00012052659746673908,
        0.9999985445614205,0.000032824306683748285]"#;
    let original: Value = serde_json::from_str(wire).unwrap();
    let saved = serde_json::to_vec_pretty(&json!({"reference":original.clone()})).unwrap();
    let loaded: Value = serde_json::from_slice(&saved).unwrap();
    assert_eq!(loaded["reference"], original);
}

#[test]
fn adjacent_finite_values_remain_distinct_and_bit_exact() {
    for base in [1.0_f64, 1.0e-7_f64, -0.000032824306683748285_f64] {
        for bits in (base.to_bits() - 8)..=(base.to_bits() + 8) {
            let value = f64::from_bits(bits);
            let raw = serde_json::to_vec(&value).unwrap();
            let parsed: f64 = serde_json::from_slice(&raw).unwrap();
            assert_eq!(
                parsed.to_bits(),
                bits,
                "decimal {}",
                String::from_utf8(raw).unwrap()
            );
        }
    }
}
