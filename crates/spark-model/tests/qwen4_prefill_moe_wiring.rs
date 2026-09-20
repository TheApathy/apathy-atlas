// SPDX-License-Identifier: AGPL-3.0-only

const SHARED: &str = include_str!("../src/layers/qwen4_prefill_moe.rs");
const SINGLE: &str = include_str!("../src/model/trait_impl/prefill_a.rs");
const CHUNK: &str = include_str!("../src/model/trait_impl/prefill_b.rs");

#[test]
fn one_token_chunks_remain_on_decode_without_batch_credit() {
    for source in [
        include_str!("../src/layers/qwen3_attention/trait_impl/prefill_inner.rs"),
        include_str!("../src/layers/qwen3_ssm/trait_prefill.rs"),
    ] {
        assert!(source.contains("qwen4_prefill_moe::selected()? && num_tokens > 1"));
    }
    let receipt = include_str!("../src/model/qwen4_prefill_engagement.rs");
    assert!(receipt.contains("if selectors.moe_only && m == 1"));
}

fn ordered(source: &str, markers: &[&str]) {
    let mut rest = source;
    for marker in markers {
        let at = rest
            .find(marker)
            .unwrap_or_else(|| panic!("missing/order: {marker}"));
        rest = &rest[at + marker.len()..];
    }
}

#[test]
fn request_admission_precedes_embedding_and_buffer_mutation() {
    let dispatch = SINGLE
        .split("pub(super) fn prefill_dispatch(")
        .nth(1)
        .unwrap();
    ordered(
        dispatch,
        &[
            "qwen4_prefill_moe::admit_request(",
            "self.buffers.zero_all(",
        ],
    );
    let chunk = CHUNK
        .split("pub(super) fn prefill_chunk_dispatch(")
        .nth(1)
        .unwrap();
    ordered(
        chunk,
        &[
            "qwen4_prefill_moe::admit_request(&self.config, total, 0)?",
            "self.buffers.zero_all(",
            "self.prefill_b_embed_chunk(",
        ],
    );
}

#[test]
fn mlp_preparation_is_serial_then_copied_to_normal_ffn_input() {
    let finish = SHARED.split("pub(crate) fn finish(").nth(1).unwrap();
    ordered(
        finish,
        &[
            "validate_ffn(ffn)?",
            "qkv_output().offset(p.staging_offset)",
            "for row in 0..rows",
            "mlp.prepare_decode(",
            "staging.offset(row * p.core_bytes)",
            "let input = ctx.buffers.norm_output()",
            "copy_d2d_async(staging, input, p.input_bytes, stream)",
            "ffn.forward_prefill(input, rows, ctx, stream)",
            "for row in 0..rows",
            "mlp.inject_decode(",
            "mlp.saved_inject(",
        ],
    );
    assert_eq!(finish.matches("ffn.forward_prefill(").count(), 1);
    for forbidden in [
        "prepare_batched(",
        "prepare_prefill_exact(",
        "inject_saved_batched(",
        "gpu.alloc(",
        "gpu.synchronize(",
    ] {
        assert!(
            !finish.contains(forbidden),
            "unexpected alternate stage: {forbidden}"
        );
    }
}

#[test]
fn all_selected_layers_are_counted_in_single_shot_prefill() {
    let dispatch = SINGLE
        .split("pub(super) fn prefill_dispatch(")
        .nth(1)
        .unwrap();
    ordered(
        dispatch,
        &[
            "qwen4_prefill_engagement::begin(",
            "for (i, layer) in self.layers.iter().enumerate()",
            "if let Some(receipt) = prefill_receipt",
            "receipt.finish()?",
            "// ── 5. Final norm",
        ],
    );
}

#[test]
fn layer_ffn_admission_is_before_each_core_effect() {
    let attn = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");
    let attn_admission =
        include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_admit.rs");
    let ssm = include_str!("../src/layers/qwen3_ssm/qwen4_prefill_moe.rs");
    ordered(
        attn_admission,
        &["validate_ffn(&self.ffn)?", "prepare_prefill_exact("],
    );
    ordered(
        attn,
        &[
            "self.prepare_prefill_moe_setup(",
            "copy_h2d_group_on_stream",
        ],
    );
    ordered(
        ssm,
        &["validate_ffn(&self.ffn)?", "attn_hyper.prepare_decode("],
    );
}
