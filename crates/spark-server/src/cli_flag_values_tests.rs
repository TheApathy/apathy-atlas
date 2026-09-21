// SPDX-License-Identifier: AGPL-3.0-only

use super::options_for_flag;

#[test]
fn kv_cache_dtype_offers_exactly_the_enum_and_nothing_else_is_enumerated() {
    // Derived, not listed: this is the whole point of pointing at the enum.
    // A variant added to `KvCacheDtype` must appear here without anyone
    // remembering to edit this module.
    let got = options_for_flag("kv-cache-dtype").expect("kv-cache-dtype is enumerated");
    let want: Vec<String> = spark_runtime::kv_cache::KvCacheDtype::ALL
        .iter()
        .map(|d| d.name().to_string())
        .collect();
    assert_eq!(got, want);
    assert!(!got.is_empty(), "an empty picker would be indistinguishable from free text");

    // And the negative arm, so this cannot pass by returning Some(..) always.
    assert!(options_for_flag("max-seq-len").is_none(), "numbers stay free text");
    assert!(options_for_flag("not-a-flag").is_none());
}

#[test]
fn the_long_name_is_taken_without_leading_dashes() {
    // A caller passing "--kv-cache-dtype" gets None, which would silently turn
    // the picker into a text field. Pinned so the convention cannot drift.
    assert!(options_for_flag("--kv-cache-dtype").is_none());
}
