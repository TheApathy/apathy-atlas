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

#[test]
fn validation_refuses_a_bad_enum_and_names_every_problem_not_just_the_first() {
    use clap::Parser as _;
    let ok = crate::cli::ServeArgs::parse_from(["spark", "org/m"]);
    assert!(crate::cli::validate_serve_args(&ok).is_ok(), "clap defaults are valid");

    // One bad value: named, with the accepted set.
    let mut bad = ok.clone();
    bad.scheduling_policy = "nonsense".into();
    let e = crate::cli::validate_serve_args(&bad).expect_err("refused");
    assert!(e.contains("scheduling-policy"), "{e}");
    assert!(e.contains("fifo") && e.contains("slai"), "the set is shown: {e}");

    // TWO bad values: both reported. An operator who fixes one and hits the
    // same wall again learned nothing the first message could not have said.
    let mut worse = bad.clone();
    worse.kv_cache_dtype = "not-a-dtype".into();
    let e2 = crate::cli::validate_serve_args(&worse).expect_err("refused");
    assert!(e2.contains("scheduling-policy"), "{e2}");
    assert!(e2.contains("kv-cache-dtype"), "{e2}");
}

#[test]
fn a_fraction_out_of_range_and_a_stranded_pair_are_both_caught() {
    use clap::Parser as _;
    let ok = crate::cli::ServeArgs::parse_from(["spark", "org/m"]);

    // clap takes 9.0 as a valid f64; it is not a valid FRACTION.
    let mut over = ok.clone();
    over.gpu_memory_utilization = 9.0;
    let e = crate::cli::validate_serve_args(&over).expect_err("refused");
    assert!(e.contains("gpu-memory-utilization") && e.contains("fraction"), "{e}");

    // The PAIR, not the field: num-drafts alone is meaningless.
    let mut stranded = ok.clone();
    stranded.num_drafts = 2;
    stranded.speculative = false;
    stranded.self_speculative = false;
    stranded.ngram_speculative = false;
    stranded.dflash = false;
    let e2 = crate::cli::validate_serve_args(&stranded).expect_err("refused");
    assert!(e2.contains("num-drafts") && e2.contains("speculative"), "{e2}");

    // And the same num-drafts is fine once a method is on — otherwise this
    // check would just forbid the flag.
    let mut fine = stranded.clone();
    fine.speculative = true;
    assert!(crate::cli::validate_serve_args(&fine).is_ok());
}
