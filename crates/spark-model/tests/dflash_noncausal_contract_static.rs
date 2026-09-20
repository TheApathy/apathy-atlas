// SPDX-License-Identifier: AGPL-3.0-only

const FROM_WEIGHTS: &str = include_str!("../src/layers/dflash_head/from_weights.rs");
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
        .split_once("tracing::info!(\n                    \"DFlash per-layer SWA")
        .expect("per-layer SWA summary missing after noncausal branch")
        .0;
    assert!(branch.contains("causals.iter_mut().for_each(|causal| *causal = false);"));
    assert!(!branch.contains("windows.iter_mut"));
    assert!(!branch.contains("windows.clear"));
}
