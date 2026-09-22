// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn dsv41() -> ModelProfile {
    profile_for_model_type("deepseek_v41").expect("the built-in DeepSeek-V4.1 profile exists")
}

#[test]
fn the_deepseek_v41_profile_carries_the_measured_numbers() {
    let p = dsv41();
    assert_eq!(p.recipe_id, "deepseek/deepseek-v4.1-flash-next");
    assert!(p.run_alone);
    assert_eq!(p.resident_gb, 84.0);
    assert_eq!(p.dspark_extra_gb, 8.0);
    assert_eq!(p.headroom_gb, 16.0);
    assert_eq!(p.env.get("ATLAS_DSV41_PACKED_KEEP").map(String::as_str), Some("124"));
    assert_eq!(p.env.get("ATLAS_DSV41_DSPARK").map(String::as_str), Some("0"));
    // 84 + 16 = 100 GB without DSpark, 108 with it.
    assert_eq!(p.required_bytes(false), 100_000_000_000);
    assert_eq!(p.required_bytes(true), 108_000_000_000);
}

#[test]
fn an_unknown_model_type_has_no_profile() {
    assert!(profile_for_model_type("qwen3_5_moe").is_none());
}

/// Admission: enough memory admits, one byte short refuses — and the refusal names the
/// numbers an operator needs.
#[test]
fn run_alone_admission_refuses_one_byte_short() {
    let p = dsv41();
    let need = p.required_bytes(false);
    admit(&p, need, false).expect("exactly enough must be admitted");
    let err = admit(&p, need - 1, false).expect_err("one byte short must be refused").to_string();
    assert!(err.contains("must run alone"), "{err}");
    assert!(err.contains("100.0 GB free"), "{err}");
    // DSpark raises the bar: what admitted without it is refused with it.
    assert!(admit(&p, need, true).is_err());
}

/// Control: a model that is not run-alone is never refused by this rule.
#[test]
fn a_shared_model_is_not_subject_to_run_alone_admission() {
    let mut p = dsv41();
    p.run_alone = false;
    admit(&p, 0, false).expect("non-run-alone profiles are admitted");
}

#[test]
fn mem_available_parses_the_kernel_format() {
    let meminfo = "MemTotal:       125000000 kB\nMemFree:         1000 kB\nMemAvailable:   97656250 kB\n";
    assert_eq!(parse_mem_available(meminfo).unwrap(), 97_656_250 * 1024);
    assert!(parse_mem_available("MemTotal: 1 kB\n").is_err(), "a missing line is an error, not zero");
}

#[test]
fn a_recipe_without_an_atlas_block_has_no_profile() {
    let yaml = "recipe_version: \"2\"\nmodel: x/y\nruntime: atlas\ncontainer: c\ndefaults:\n  port: 1\n";
    assert!(parse_profile("x/y", yaml).unwrap().is_none());
}
