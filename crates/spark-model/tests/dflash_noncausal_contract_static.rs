// SPDX-License-Identifier: AGPL-3.0-only

const FROM_WEIGHTS: &str = include_str!("../src/layers/dflash_head/from_weights.rs");
const DFLASH_CONFIG: &str = include_str!("../src/weight_loader/dflash_loader.rs");
const FACTORY_BUILD: &str = include_str!("../src/factory/build.rs");
const NATIVE_VALIDATION: &str =
    include_str!("../src/weight_loader/dflash_validation/native_flash_next.rs");

#[test]
fn native_v3_and_dflash2_require_an_explicit_noncausal_checkpoint() {
    assert!(NATIVE_VALIDATION.contains("config.is_causal == Some(false)"));
    assert!(NATIVE_VALIDATION.contains("is_causal must be explicitly false"));
}

#[test]
fn explicit_noncausal_semantics_are_not_hidden_behind_process_environment() {
    assert!(DFLASH_CONFIG.contains("pub is_causal: Option<bool>"));
    assert!(!FROM_WEIGHTS.contains("ATLAS_DFLASH_HONOR_IS_CAUSAL"));
    assert!(!FACTORY_BUILD.contains("ATLAS_DFLASH_HONOR_IS_CAUSAL"));
    assert!(!FROM_WEIGHTS.contains("honour_is_causal"));
    assert!(FACTORY_BUILD.contains("ATLAS_EXPERIMENTAL_NATIVE_QWEN4_DFLASH"));
}

#[test]
fn explicit_false_forces_every_layer_noncausal_without_changing_swa() {
    let branch = FROM_WEIGHTS
        .split_once("if weights.config.is_causal == Some(false) {")
        .expect("explicit noncausal branch missing")
        .1
        .split_once("let causal_count = causals.iter().filter(|causal| **causal).count();")
        .expect("per-layer SWA summary missing after noncausal branch")
        .0;
    assert!(branch.contains("causals.iter_mut().for_each(|causal| *causal = false);"));
    assert!(!branch.contains("windows.iter_mut"));
    assert!(!branch.contains("windows.clear"));
}

#[test]
fn omitted_or_true_declarations_keep_the_layer_type_heuristic() {
    let branch = FROM_WEIGHTS
        .split_once("if weights.config.is_causal == Some(false) {")
        .expect("explicit noncausal branch missing")
        .0;
    assert!(branch.contains("causals.push(is_sliding);"));
    assert!(!branch.contains("is_causal == Some(true)"));
    assert!(!branch.contains("is_causal.is_none()"));
}

#[test]
fn summary_reports_the_post_override_causal_vector() {
    assert!(
        FROM_WEIGHTS
            .contains("let causal_count = causals.iter().filter(|causal| **causal).count();")
    );
    assert!(FROM_WEIGHTS.contains("0 => \"noncausal\""));
    assert!(FROM_WEIGHTS.contains("count if count == num_layers => \"causal\""));
    assert!(FROM_WEIGHTS.contains("_ => \"mixed\""));
    assert!(FROM_WEIGHTS.contains("causal_layers={causal_count}/{num_layers}"));
    assert!(FROM_WEIGHTS.contains("mode={causal_mode}"));
    assert!(!FROM_WEIGHTS.contains("sliding_window={sw} causal=true"));
}
