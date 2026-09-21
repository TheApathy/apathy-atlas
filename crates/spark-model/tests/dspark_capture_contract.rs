// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/weight_loader/deepseek_v4/dspark/capture_plan.rs"]
#[allow(dead_code)] // Environment adapter is exercised by serving, not by mutating test env.
mod capture_plan;

use capture_plan::{
    CAPTURE_LAYERS, capture_layers_plan, parse_capture_layers, validate_capture_layers,
};
use std::ffi::OsStr;

#[test]
fn default_and_explicit_layers_match_the_actual_three_stage_contract() {
    assert_eq!(CAPTURE_LAYERS, [40, 41, 42]);
    assert_eq!(parse_capture_layers(None, 43).unwrap(), CAPTURE_LAYERS);
    for text in ["40,41,42", " 40, 41, 42 "] {
        assert_eq!(
            parse_capture_layers(Some(OsStr::new(text)), 43).unwrap(),
            CAPTURE_LAYERS
        );
    }
}

#[test]
fn malformed_count_syntax_order_and_duplicates_never_silently_drop_entries() {
    for text in [
        "",
        "40",
        "40,41",
        "40,41,42,43",
        "40,x,41,42",
        "40,,41,42",
        "40,41,42,",
        ",40,41,42",
        "42,41,40",
        "40,40,42",
        "39,40,41",
        "+40,41,42",
        "-40,41,42",
        "4 0,41,42",
        "184467440737095516160,41,42",
    ] {
        assert!(
            parse_capture_layers(Some(OsStr::new(text)), 43).is_err(),
            "{text}"
        );
    }
}

#[test]
fn target_depth_and_module_layer_contract_are_validated() {
    for layers in [0, 40, 42] {
        assert!(parse_capture_layers(None, layers).is_err());
    }
    validate_capture_layers(&[40, 41, 42], 43).unwrap();
    for wrong in [vec![], vec![40, 41], vec![42, 41, 40], vec![40, 41, 43]] {
        assert!(validate_capture_layers(&wrong, 44).is_err());
    }
}

#[test]
fn ordinary_run_stays_inert_but_explicit_invalid_settings_fail_closed() {
    assert_eq!(capture_layers_plan(false, false, None, 12).unwrap(), None);
    for (capture, dump) in [(true, false), (false, true), (true, true)] {
        assert_eq!(
            capture_layers_plan(capture, dump, None, 43).unwrap(),
            Some(CAPTURE_LAYERS)
        );
    }
    assert!(capture_layers_plan(false, false, Some(OsStr::new("40,x,42")), 43).is_err());
    assert_eq!(
        capture_layers_plan(false, false, Some(OsStr::new("40,41,42")), 43).unwrap(),
        None
    );
}

#[cfg(unix)]
#[test]
fn non_utf8_layer_setting_is_rejected() {
    use std::os::unix::ffi::OsStrExt;
    assert!(parse_capture_layers(Some(OsStr::from_bytes(b"40,41,\xff")), 43).is_err());
}

#[test]
fn constructor_uses_checked_layers_before_allocating_capture_memory() {
    let source = include_str!("../src/model/impl_a1.rs");
    assert!(!source.contains(".filter_map(|s| s.trim().parse().ok())"));
    let first = source.find("capture_layers_from_env(").unwrap();
    assert!(first < source.find("let buf = gpu.alloc(layers.len()").unwrap());
    let factory = include_str!("../src/factory/build.rs");
    assert!(
        factory.find("capture_layers_from_env(").unwrap()
            < factory.find("loader.load_embedding(").unwrap()
    );
    let server = include_str!("../../spark-server/src/main_modules/serve.rs");
    assert!(
        server.find("capture_layers_from_env(").unwrap()
            < server.find("serve_phases::init_gpu_backend(").unwrap()
    );
}

#[test]
fn loader_and_head_validate_actual_layer_identity_before_installation() {
    let loader = include_str!("../src/weight_loader/deepseek_v4/dspark.rs");
    assert!(loader.contains("target_layer_ids: capture_plan::CAPTURE_LAYERS.to_vec()"));
    let load = loader.split("pub fn load_dspark_drafter(").nth(1).unwrap();
    assert!(
        load.find("capture_plan::validate_capture_layers(").unwrap()
            < load.find("gpu.kernel(").unwrap()
    );
    assert!(load.contains("n_stages == capture_plan::CAPTURE_LAYERS.len()"));
    let head = include_str!("../src/layers/dspark_head.rs");
    let install = head
        .split("pub fn set_capture(")
        .nth(1)
        .unwrap()
        .split("fn captures_at(")
        .next()
        .unwrap();
    assert!(
        install.find("validate_capture_layers(layers,").unwrap()
            < install.find("self.capture_buf = buf").unwrap()
    );
    assert!(install.contains("&self.module.params.target_layer_ids"));
}
