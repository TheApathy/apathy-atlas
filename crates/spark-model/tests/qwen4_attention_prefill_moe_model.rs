// SPDX-License-Identifier: AGPL-3.0-only

const CANDIDATE: &str =
    include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");
const ADMISSION: &str =
    include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_admit.rs");
const PREFILL: &str = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_inner.rs");
const REGISTRY: &str = include_str!("../src/layers/qwen3_attention/trait_impl.rs");

fn ordered(source: &str, items: &[&str]) {
    let mut rest = source;
    for item in items {
        let at = rest
            .find(item)
            .unwrap_or_else(|| panic!("missing/order: {item}"));
        rest = &rest[at + item.len()..];
    }
}

#[test]
fn opt_in_precedes_unchanged_legacy_dispatch() {
    assert!(REGISTRY.contains("mod prefill_moe_only;"));
    ordered(
        PREFILL,
        &[
            "qwen4_prefill_moe::selected()?",
            "self.prefill_moe_only(",
            "if let Some(attn_hyper) = self.qwen4_attn_hyper.as_ref()",
            "ATLAS_QWEN4_ATTN_PREFILL_BATCH",
            "self.decode_inner(",
        ],
    );
}

#[test]
fn admission_completes_before_first_device_effect() {
    ordered(
        ADMISSION,
        &[
            "qwen4_prefill_moe::validate(ctx, num_tokens, seq_len_start)?",
            "qwen4_prefill_moe::validate_ffn(&self.ffn)?",
            "batched_meta.is_none()",
            "self.qwen4_attn_hyper.as_ref()",
            "self.qwen4_mlp_hyper.as_ref()",
            "attn_hyper.inject.as_ref()",
            "mlp_hyper.inject.as_ref()",
            "matches!(&self.ffn, FfnComponent::Moe(_))",
            "metadata.num_seqs == 1",
            "metadata.block_table",
            "metadata.seq_len",
            "hidden != DevicePtr::NULL",
            "attn_hyper.validate_prefill_exact(",
            "mlp_hyper.validate_prefill_exact(",
            "attn_hyper.prepare_prefill_exact(",
        ],
    );
    ordered(
        CANDIDATE,
        &[
            "self.prepare_prefill_moe_setup(",
            "let packed_inputs = ctx.buffers.norm_output()",
            "copy_d2d_async(",
        ],
    );
}

#[test]
fn serial_core_and_metadata_order_are_preserved() {
    let body = CANDIDATE.split("if !batched_core").nth(1).unwrap();
    ordered(
        body,
        &[
            "for local in 0..tile_rows",
            "let row = tile_start + local",
            "seq_len_start + row + 1",
            "HostToDeviceCopy::new(&seq_len_bytes, metadata.seq_len)",
            "positions: metadata.positions.offset(row * 4)",
            "positions_h: metadata.positions_h.offset(row * 4)",
            "positions_w: metadata.positions_w.offset(row * 4)",
            "slot: metadata.slot.offset(row * 8)",
            "seq_len: metadata.seq_len",
            "attn_hyper.prepare_decode(",
            "self.attention_forward(",
            "seq_len_start + row",
            "attn_hyper.inject_decode(",
            "qwen4_prefill_moe::finish(",
            "PrefillPath::Attention",
        ],
    );
    let core = body.split("qwen4_prefill_moe::finish(").next().unwrap();
    for forbidden in ["mlp_hyper.prepare", "self.ffn.forward", "prepare_batched("] {
        assert!(
            !core.contains(forbidden),
            "unexpected core replacement: {forbidden}"
        );
    }
    assert!(!CANDIDATE.contains("gpu.alloc("));
    assert!(!CANDIDATE.contains("gpu.synchronize("));
}

#[test]
fn candidate_is_bounded_and_licensed() {
    for source in [CANDIDATE, ADMISSION] {
        assert!(source.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
        assert!(source.lines().count() <= 250);
    }
}
