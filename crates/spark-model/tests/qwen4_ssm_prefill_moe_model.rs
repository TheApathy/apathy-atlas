// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the opt-in serial-SSM-core/grouped-FFN path.

use std::{fs, path::PathBuf};

fn source(relative: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative))
        .expect("production source must exist")
}

fn ordered(source: &str, markers: &[&str]) {
    let mut remaining = source;
    for marker in markers {
        let index = remaining
            .find(marker)
            .unwrap_or_else(|| panic!("missing {marker}"));
        remaining = &remaining[index + marker.len()..];
    }
}

#[test]
fn opt_in_precedes_legacy_routes_and_returns_directly() {
    let hook = source("src/layers/qwen3_ssm/trait_prefill.rs");
    ordered(
        &hook,
        &[
            "crate::layers::qwen4_prefill_moe::selected()?",
            "return self.prefill_qwen4_moe_batch(",
            "ATLAS_QWEN4_SSM_PREFILL_BATCH",
        ],
    );
    assert!(source("src/layers/qwen3_ssm/mod.rs").contains("mod qwen4_prefill_moe;"));
}

#[test]
fn all_ssm_admission_precedes_first_core_effect() {
    let code = source("src/layers/qwen3_ssm/qwen4_prefill_moe.rs");
    ordered(
        &code,
        &[
            "crate::layers::qwen4_prefill_moe::validate(ctx, rows, seq_len_start)?;",
            "self.qwen4_attn_hyper",
            "self.qwen4_mlp_hyper",
            "attn_hyper.inject.is_some() && mlp_hyper.inject.is_some()",
            "matches!(&self.ffn, FfnComponent::Moe(_))",
            ".downcast_mut::<SsmLayerState>()",
            "for row in 0..rows",
            "attn_hyper.prepare_decode(",
        ],
    );
    assert!(code.contains("hidden.is_null()"));
    assert!(code.contains("residual.is_null()"));
    assert!(code.contains("checked_mul"));
}

#[test]
fn shipping_core_stays_token_ordered_and_ffn_finishes_once() {
    let code = source("src/layers/qwen3_ssm/qwen4_prefill_moe.rs");
    ordered(
        &code,
        &[
            "for row in 0..rows",
            "attn_hyper.prepare_decode(",
            "self.ssm_forward(mixed, ssm_state, ctx, stream, trace)?",
            "attn_hyper.inject_decode(",
            "crate::layers::qwen4_prefill_moe::finish(",
            "crate::model::qwen4_prefill_engagement::engage(",
            "crate::model::qwen4_prefill_engagement::PrefillPath::Ssm",
        ],
    );
    assert_eq!(code.matches("qwen4_prefill_moe::finish(").count(), 1);
    for forbidden in [
        "decode_qwen4_batched_inner(",
        "prepare_batched(",
        "prepare_prefill(",
        "ssm_forward_qwen4_exact_rows(",
        "forward_prefill(",
        "forward_batched(",
        "ATLAS_QWEN4_SSM_PREFILL_ROWWISE",
        "conv1d_update_l2norm_f32_sequence(",
    ] {
        assert!(
            !code.contains(forbidden),
            "unexpected alternate route: {forbidden}"
        );
    }
}
