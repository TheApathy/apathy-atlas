// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/qwen4_prefill_moe/plan.rs"]
mod plan;

use plan::{Limits, Plan, parse_selector, validate_conflict};

fn limits() -> Limits {
    Limits {
        capacity: 2048,
        qkv: 2048 * 32768,
        norm: 2048 * 5120,
        hidden: 2048 * 20480,
        residual: 2048 * 20480,
        output: 2048 * 5120,
    }
}

#[test]
fn explicit_selector_is_default_off_and_rejects_typos() {
    assert_eq!(parse_selector(None), Ok(false));
    assert_eq!(parse_selector(Some("0")), Ok(false));
    assert_eq!(parse_selector(Some("1")), Ok(true));
    for bad in ["", "true", "2", "01", " 1", "1\n"] {
        assert!(parse_selector(Some(bad)).is_err(), "{bad:?}");
    }
}

#[test]
fn isolated_experiment_rejects_active_or_malformed_competing_flags() {
    for value in [None, Some("0")] {
        assert!(validate_conflict(value, false).is_ok());
    }
    for bad in ["1", "2", "true", "", " 0"] {
        assert!(validate_conflict(Some(bad), false).is_err());
    }
    assert!(validate_conflict(None, true).is_ok());
    assert!(validate_conflict(Some("1"), true).is_ok());
    for bad in ["0", "2", "true", ""] {
        assert!(validate_conflict(Some(bad), true).is_err());
    }
}

#[test]
fn staging_starts_after_decode_hyper_projection_and_fits_exactly() {
    let p = Plan::new(2048, 0, 2560, 10240, limits()).unwrap();
    assert_eq!(p.row_bytes, 20480);
    assert_eq!(p.core_bytes, 5120);
    assert_eq!(p.staging_offset, 20480);
    assert_eq!(p.input_bytes, 10485760);
    let mut tight = limits();
    tight.qkv = p.staging_offset + p.input_bytes;
    assert!(Plan::new(2048, 0, 2560, 10240, tight).is_ok());
    tight.qkv -= 1;
    assert!(Plan::new(2048, 0, 2560, 10240, tight).is_err());
}

#[test]
fn row_count_and_dense_window_are_bounded_without_overflow() {
    for rows in [0, 1, 2049, usize::MAX] {
        assert!(Plan::new(rows, 0, 2560, 10240, limits()).is_err());
    }
    assert!(Plan::new(2, 2046, 2560, 10240, limits()).is_ok());
    assert!(Plan::new(2, 2047, 2560, 10240, limits()).is_err());
    assert!(Plan::new(2, usize::MAX, 2560, 10240, limits()).is_err());
    assert!(Plan::new(2, 0, usize::MAX, 10240, limits()).is_err());
    assert!(Plan::new(2, 0, 2560, usize::MAX, limits()).is_err());
    assert!(Plan::new(2, 0, 0, 10240, limits()).is_err());
    assert!(Plan::new(2, 0, 2560, 0, limits()).is_err());
}

#[test]
fn every_live_arena_is_checked_before_staging() {
    let base = limits();
    for short in [
        Limits {
            capacity: 1,
            ..base
        },
        Limits {
            norm: 10239,
            ..base
        },
        Limits {
            output: 10239,
            ..base
        },
        Limits {
            hidden: 40959,
            ..base
        },
        Limits {
            residual: 40959,
            ..base
        },
    ] {
        assert!(Plan::new(2, 0, 2560, 10240, short).is_err());
    }
}

#[test]
fn qkv16_packs_before_exact_projection_and_shipping_attention() {
    let route = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");
    let pack = route.find("attn_hyper.pack_saved_tile_or_rows(").unwrap();
    let project = route.find("self.qwen4_k5_project_qkv_exact(").unwrap();
    let shipping = route
        .find("self.attention_forward_preprojected_raw(")
        .unwrap();
    assert!(pack < project && project < shipping);
}

#[test]
fn qkv16_batches_exact_o_projection_after_ordered_attention() {
    let route = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");
    let raw = route
        .find("self.attention_forward_preprojected_raw(")
        .unwrap();
    let retain = route[raw..]
        .find("raw_outputs.offset(local * attn_row_bytes)")
        .unwrap()
        + raw;
    let project = route[retain..]
        .find("ops::w4a16_gemv_batch_logits_exact_with(")
        .unwrap()
        + retain;
    let inject = route[project..]
        .find("attn_hyper.inject_saved_tile_or_rows(")
        .unwrap()
        + project;
    assert!(raw < retain && retain < project && project < inject);
}

#[test]
fn qwen4_batch_core_uses_scaled_yarn_before_ordinary_rope() {
    let core = include_str!("../src/layers/qwen3_attention/prefill/cache_skip.rs");
    let qwen4 = core.find("!self.qwen4_yarn_inv_freq.is_null()").unwrap();
    let scaled = core[qwen4..].find("ops::rope_yarn_scaled(").unwrap();
    let ordinary = core[qwen4..].find("ops::rope(").unwrap();
    assert!(
        scaled < ordinary,
        "Qwen4 scaled YaRN must precede ordinary RoPE"
    );
}
