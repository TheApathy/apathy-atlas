// SPDX-License-Identifier: AGPL-3.0-only

#[path = "../src/layers/qwen4_hyper/prefill_exact_plan.rs"]
mod plan;
use plan::Layout;

fn layout(rows: usize, capacity: usize, bytes: [usize; 5]) -> Result<Layout, &'static str> {
    Layout::new(rows, 2560, 4, 320, capacity, bytes)
}

fn roomy() -> [usize; 5] {
    [1 << 20, 1 << 28, 1 << 28, 1 << 28, 1 << 28]
}

#[test]
fn arbitrary_rows_include_exact_tier_and_singleton_boundaries() {
    for rows in [1, 2, 4, 5, 8, 9, 17, 18, 31, 32, 33, 63, 64, 65, 2048, 8192] {
        let p = layout(rows, 8192, roomy()).unwrap();
        assert_eq!(p.row_bytes, 20480);
        assert_eq!(p.input_bytes, rows * 5120);
        assert_eq!(p.tile, rows.min(32));
        assert_eq!(p.residual_bytes, rows * 20480);
    }
}

#[test]
fn zero_capacity_and_integer_overflow_fail() {
    assert!(layout(0, 8192, roomy()).is_err());
    assert!(layout(33, 32, roomy()).is_err());
    assert!(layout(usize::MAX, usize::MAX, [usize::MAX; 5]).is_err());
    assert!(Layout::new(2, usize::MAX, 4, 320, 2, roomy()).is_err());
    assert!(Layout::new(2, 2560, 4, usize::MAX, 2, roomy()).is_err());
}

#[test]
fn every_scratch_extent_is_checked_before_dispatch() {
    let required = [32 * 324 * 2, 32 * 20480, 33 * 5120, 33 * 20480, 33 * 20480];
    assert!(layout(33, 33, required).is_ok());
    for index in 0..required.len() {
        let mut short = required;
        short[index] -= 1;
        assert!(layout(33, 33, short).is_err(), "extent {index}");
    }
}

#[test]
fn singleton_uses_fixed_decode_injection_offset() {
    let mut bytes = roomy();
    bytes[0] = 2056;
    assert!(layout(1, 1, bytes).is_ok());
    bytes[0] -= 1;
    assert!(layout(1, 1, bytes).is_err());
    // A singleton after full tiles needs that same fixed-offset decode path.
    assert!(layout(33, 33, roomy()).is_ok());
}

#[test]
fn reject_unsupported_projection_and_residual_geometry() {
    for (h, hc, rank) in [
        (0, 4, 320),
        (2560, 1, 320),
        (2560, 4, 0),
        (2561, 4, 320),
        (2560, 4, 321),
    ] {
        assert!(Layout::new(2, h, hc, rank, 2, roomy()).is_err());
    }
}

#[test]
fn mlp_exact_is_opt_in_and_uses_saved_batched_injection() {
    let source = include_str!("../src/layers/qwen4_prefill_moe.rs");
    assert!(source.contains("ATLAS_QWEN4_PREFILL_HC_EXACT"));
    let helper = include_str!("../src/layers/qwen4_prefill_moe/hyper.rs");
    assert!(helper.contains("prepare_prefill_exact("));
    assert!(helper.contains("inject_saved_batched("));
    assert!(!helper.contains("prepare_prefill("));
    assert!(source.contains("if hyper_selected()?"));
}

#[test]
fn serial_attention_consumes_preserved_mixed_rows_not_clobbered_norm_scratch() {
    let source = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");
    let admission = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_admit.rs");
    let validate = admission.find("mlp_hyper.validate_prefill_exact(").unwrap();
    let prepare = admission.find("attn_hyper.prepare_prefill_exact(").unwrap();
    assert!(validate < prepare);
    let setup = source.find("self.prepare_prefill_moe_setup(").unwrap();
    let token_loop = source.find("for local in 0..tile_rows").unwrap();
    assert!(setup < token_loop);
    assert!(source.contains("(residual_row, Some(attn_hyper.saved_inject(residual_row)))"));
}
