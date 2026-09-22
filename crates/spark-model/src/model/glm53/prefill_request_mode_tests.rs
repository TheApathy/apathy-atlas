// SPDX-License-Identifier: AGPL-3.0-only

//! Real environment selector tests isolated in child test processes.
use super::{mixed_prefill_rows, requested_wide_prefill};
use std::process::Command;

fn check_case(case: &str, variables: &[(&str, &str)]) {
    let mut child = Command::new(std::env::current_exe().unwrap());
    child.args([
        "--exact",
        "model::glm53::prefill_exl3::request_mode_tests::environment_fixture",
        "--ignored",
        "--nocapture",
    ]);
    for key in [
        "ATLAS_GLM53_LAYER_MAJOR_PREFILL",
        "ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS",
        "ATLAS_GLM53_LAYER_MAJOR_VISION_PREFILL",
        "ATLAS_GLM53_WIDE_PREFILL",
        "ATLAS_GLM53_WIDE_PREFILL_ROWS",
    ] {
        child.env_remove(key);
    }
    child
        .env("ATLAS_GLM53_ROUTE_TEST_CASE", case)
        .envs(variables.iter().copied());
    let output = child.output().unwrap();
    assert!(
        output.status.success(),
        "{case}: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
}

#[test]
fn text_and_images_select_separate_explicit_row_limits() {
    check_case(
        "combined",
        &[
            ("ATLAS_GLM53_LAYER_MAJOR_PREFILL", "1"),
            ("ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS", "2048"),
            ("ATLAS_GLM53_WIDE_PREFILL", "1"),
            ("ATLAS_GLM53_WIDE_PREFILL_ROWS", "8"),
        ],
    );
}

#[test]
fn large_text_optin_does_not_enable_unqualified_large_image_rows() {
    check_case(
        "large_only",
        &[
            ("ATLAS_GLM53_LAYER_MAJOR_PREFILL", "1"),
            ("ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS", "2048"),
        ],
    );
}

#[test]
fn mixed_layer_major_requires_its_own_explicit_opt_in() {
    check_case(
        "vision_large",
        &[
            ("ATLAS_GLM53_LAYER_MAJOR_PREFILL", "1"),
            ("ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS", "2048"),
            ("ATLAS_GLM53_LAYER_MAJOR_VISION_PREFILL", "1"),
        ],
    );
}

#[test]
fn invalid_small_image_rows_still_reject() {
    check_case(
        "invalid_small",
        &[
            ("ATLAS_GLM53_LAYER_MAJOR_PREFILL", "1"),
            ("ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS", "2048"),
            ("ATLAS_GLM53_WIDE_PREFILL", "1"),
            ("ATLAS_GLM53_WIDE_PREFILL_ROWS", "16"),
        ],
    );
}

#[test]
fn serial_and_small_only_recipes_are_unchanged() {
    check_case("serial", &[]);
    check_case(
        "small_only",
        &[
            ("ATLAS_GLM53_WIDE_PREFILL", "1"),
            ("ATLAS_GLM53_WIDE_PREFILL_ROWS", "8"),
        ],
    );
}

#[test]
fn real_request_selects_from_prepared_input_before_capture_or_reset() {
    let source = include_str!("target_prefill_exl3.rs")
        .split_whitespace()
        .collect::<String>();
    let route = source
        .find("requested_wide_prefill(prepared.is_some())")
        .expect("input-aware selector");
    assert!(route < source.find("self.preflight_prefill_capture(").unwrap());
    assert!(route < source.find("self.reset_sequence()").unwrap());
}

#[test]
#[ignore = "isolated helper invoked by the environment tests"]
fn environment_fixture() {
    match std::env::var("ATLAS_GLM53_ROUTE_TEST_CASE")
        .unwrap()
        .as_str()
    {
        "combined" => {
            assert_eq!(
                requested_wide_prefill(false)
                    .unwrap()
                    .unwrap()
                    .layer_major_rows(),
                Some(2048)
            );
            assert_eq!(
                mixed_prefill_rows(requested_wide_prefill(true).unwrap()).unwrap(),
                8
            );
        }
        "large_only" => {
            assert_eq!(
                requested_wide_prefill(false)
                    .unwrap()
                    .unwrap()
                    .layer_major_rows(),
                Some(2048)
            );
            assert_eq!(
                mixed_prefill_rows(requested_wide_prefill(true).unwrap()).unwrap(),
                1
            );
        }
        "vision_large" => {
            let config = requested_wide_prefill(true).unwrap().unwrap();
            assert_eq!(config.layer_major_rows(), Some(2048));
            assert_eq!(mixed_prefill_rows(Some(config)).unwrap(), 2048);
        }
        "invalid_small" => {
            assert!(requested_wide_prefill(false).unwrap().is_some());
            assert!(requested_wide_prefill(true).is_err());
        }
        "serial" => {
            assert!(requested_wide_prefill(false).unwrap().is_none());
            assert!(requested_wide_prefill(true).unwrap().is_none());
        }
        "small_only" => {
            for mixed in [false, true] {
                assert_eq!(
                    mixed_prefill_rows(requested_wide_prefill(mixed).unwrap()).unwrap(),
                    8
                );
            }
        }
        other => panic!("unknown isolated case {other}"),
    }
}
