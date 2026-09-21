// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/dflash_head/dflash_context_window.rs"]
mod dflash_context_window;

const FROM_WEIGHTS: &str = include_str!("../src/layers/dflash_head/from_weights.rs");
const RING_WINDOW: &str = include_str!("../src/layers/dflash_head/ring_window.rs");

use dflash_context_window::{
    MAX_ABSOLUTE_CONTEXT, NATIVE_FLASH_NEXT_WINDOW, is_native_flash_next, resolve_context_window,
};

fn native_geometry(architecture: &str) -> bool {
    is_native_flash_next(
        &[architecture.to_string()],
        Some("qwen3"),
        2560,
        8704,
        6,
        48,
        20,
        4,
        128,
        248_320,
        &[1, 7, 13, 20, 26, 33, 39, 46],
    )
}

#[test]
fn exact_native_v3_and_dflash2_markers_share_the_strict_contract() {
    assert!(native_geometry("DFlashDraftModel"));
    assert!(native_geometry("DFlash2DraftModel"));
    assert!(!native_geometry("DSparkDraftModel"));
    assert!(!is_native_flash_next(
        &["DFlash2DraftModel".to_string()],
        Some("qwen3"),
        5120,
        17_408,
        6,
        64,
        40,
        8,
        128,
        248_320,
        &[7, 15, 23, 31, 39, 47, 55, 63],
    ));
}

#[test]
fn exact_native_flash_next_requires_the_4096_resident_window() {
    assert_eq!(
        resolve_context_window(Some(4096), MAX_ABSOLUTE_CONTEXT, true).unwrap(),
        NATIVE_FLASH_NEXT_WINDOW
    );
    for requested in [Some(0), Some(1), Some(512), Some(4095), Some(4097), None] {
        assert!(
            resolve_context_window(requested, 1_048_576, true).is_err(),
            "native Flash-Next unexpectedly accepted {requested:?}"
        );
    }
    assert!(resolve_context_window(Some(4096), 4095, true).is_err());
    assert_eq!(
        resolve_context_window(Some(4096), 4096, true).unwrap(),
        4096
    );
    assert!(resolve_context_window(Some(4096), MAX_ABSOLUTE_CONTEXT + 1, true).is_err());
    assert!(resolve_context_window(Some(4096), usize::MAX, true).is_err());
}

#[test]
fn legacy_full_attention_is_exact_and_bounded() {
    assert_eq!(resolve_context_window(None, 2048, false).unwrap(), 2048);
    assert_eq!(
        resolve_context_window(Some(1), 1_048_576, false).unwrap(),
        1
    );
    assert_eq!(
        resolve_context_window(Some(4096), 1_048_576, false).unwrap(),
        4096
    );
    assert!(resolve_context_window(None, 4097, false).is_err());
    assert!(resolve_context_window(Some(0), 4096, false).is_err());
    assert!(resolve_context_window(Some(4097), 1_048_576, false).is_err());
    assert!(resolve_context_window(Some(1), 0, false).is_err());
}

#[test]
fn early_absolute_limit_matches_the_runtime_ring_limit() {
    assert!(RING_WINDOW.contains("pub(crate) const MAX_ABSOLUTE_CONTEXT: usize = 1_048_576;"));
    assert_eq!(MAX_ABSOLUTE_CONTEXT, 1_048_576);
}

#[test]
fn constructor_uses_one_passed_window_without_a_model_layer_env_override() {
    assert!(
        !FROM_WEIGHTS.contains("ATLAS_DFLASH_CTX_WINDOW"),
        "model layer must not reread or advertise a process environment override"
    );
    assert!(FROM_WEIGHTS.contains("resolve_context_window("));
    assert!(FROM_WEIGHTS.contains("window_size,"));
    assert!(FROM_WEIGHTS.contains("max_seq_len,"));
    assert!(FROM_WEIGHTS.contains("ctx_window,"));
    assert!(FROM_WEIGHTS.contains("let n_attn = row_layout.query_rows + ctx_window;"));
    assert!(FROM_WEIGHTS.contains("gpu.alloc(ctx_window * hidden_size * bf16)?"));

    let resolve_at = FROM_WEIGHTS.find("resolve_context_window(").unwrap();
    let first_kernel_at = FROM_WEIGHTS.find("gpu.kernel(").unwrap();
    assert!(
        resolve_at < first_kernel_at,
        "context contract must fail before backend kernel lookup"
    );
}

#[test]
fn checkpoint_sliding_window_remains_a_separate_layer_contract() {
    assert!(FROM_WEIGHTS.contains("let (layer_window_sizes, layer_causal)"));
    assert!(FROM_WEIGHTS.contains("weights.config.sliding_window"));
    assert!(FROM_WEIGHTS.contains("layer_window_sizes,"));
}
