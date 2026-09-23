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
    assert_eq!(p.resident_gb, 85.0);
    assert_eq!(p.dspark_extra_gb, 8.0);
    assert_eq!(p.headroom_gb, 16.0);
    assert_eq!(p.env.get("ATLAS_DSV41_PACKED_KEEP").map(String::as_str), Some("124"));
    assert_eq!(p.env.get("ATLAS_DSV41_DSPARK").map(String::as_str), Some("0"));
    assert_eq!(p.env.get("ATLAS_DSV41_CHUNK").map(String::as_str), Some("2048"));
    // 85 + 16 = 101 GB without DSpark, 109 with it.
    assert_eq!(p.required_bytes(false), 101_000_000_000);
    assert_eq!(p.required_bytes(true), 109_000_000_000);
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
    assert!(err.contains("101.0 GB free"), "{err}");
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

/// Two built-ins share `qwen3_5`, so the profile must be chosen by the checkpoint's
/// directory. Control: a qwen3_5 checkpoint with no built-in of its own gets NO profile
/// (the type alone is ambiguous) rather than a neighbour's environment.
#[test]
fn qwen3_5_checkpoints_get_their_own_profile_by_directory() {
    let q38 = profile_for(Some("/home/flocka/atlas/qwen38/optimized-qwen"), "qwen3_5").unwrap();
    assert_eq!(q38.recipe_id, "qwen3.8/qwen3.8-27b-optimized-local");
    assert!(q38.env.contains_key("ATLAS_ATTN_PROJ_CUBLASLT"));
    // The TC verify switches ship only together with decode parity, which keeps
    // speculation byte-identical to plain decode.
    for tc in ["ATLAS_FFN_TC", "ATLAS_SSM_PROJ_TC", "ATLAS_LM_HEAD_TC", "ATLAS_DECODE_TC_PARITY"] {
        assert_eq!(q38.env.get(tc).map(String::as_str), Some("1"), "{tc}");
    }
    let aeon = profile_for(Some("/m/AEON-Q36-27B-Full/"), "qwen3_5").unwrap();
    assert_eq!(aeon.recipe_id, "qwen3.6/aeon-q36-27b-full-local");
    assert!(profile_for(Some("/m/Some-Other-27B"), "qwen3_5").is_none());
    let fn_ = profile_for(Some("Qwen3.8-Flash-Next-NVFP4-Offload"), "qwen4_exp").unwrap();
    assert!(fn_.run_alone && fn_.env.contains_key("ATLAS_QWEN4_PREFILL_ATTN_FLASH"));
}
