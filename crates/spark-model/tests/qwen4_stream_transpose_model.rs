// SPDX-License-Identifier: AGPL-3.0-only

const POLICY: &str = include_str!("../src/layers/moe/qwen4_prefill_compact.rs");
const STREAM: &str = include_str!("../src/layers/moe/qwen4_stream_t.rs");
const FACTORY: &str = include_str!("../src/factory/qwen4_stream_t.rs");
const OPS: &str = include_str!("../src/layers/moe/qwen4_compact_ops.rs");
const TRAIT: &str = include_str!("../src/layer/transformer_layer.rs");
const ATTN: &str = include_str!("../src/layers/qwen3_attention/trait_impl.rs");
const SSM: &str = include_str!("../src/layers/qwen3_ssm/mod.rs");

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
fn selector_is_strict_and_conflicts_with_replacement_layouts() {
    assert!(POLICY.contains("ATLAS_QWEN4_PREFILL_MOE_STREAM_T"));
    for dependency in ["SELECTOR", "K32_SELECTOR", "ATLAS_QWEN4_PREFILL_MOE_BATCH"] {
        assert!(
            POLICY.contains(dependency),
            "missing dependency {dependency}"
        );
    }
    for conflict in [
        "TRANSPOSED_SELECTOR",
        "ATLAS_UNIFIED_MOE_LAYOUT",
        "ATLAS_HYBRID_MOE_LAYOUT",
        "ATLAS_NVFP4_MOE_WORKLIST",
        "CHECK_SELECTOR",
    ] {
        assert!(POLICY.contains(conflict), "missing conflict {conflict}");
    }
}

#[test]
fn one_complete_shared_scratch_is_sized_and_wired_post_load() {
    assert!(FACTORY.contains("1_415_577_600"));
    assert!(FACTORY.contains("512"));
    assert!(FACTORY.contains("640"));
    assert!(FACTORY.contains("2560"));
    assert!(FACTORY.contains("set_moe_stream_transpose_scratch"));
    assert!(TRAIT.contains("fn set_moe_stream_transpose_scratch("));
    assert!(ATTN.contains("fn set_moe_stream_transpose_scratch("));
    assert!(SSM.contains("fn set_moe_stream_transpose_scratch("));
}

#[test]
fn every_prefill_layer_transposes_gate_up_down_before_compact_gemms() {
    ordered(
        STREAM,
        &["&self.gate_ptrs,", "&self.up_ptrs,", "&self.down_ptrs,"],
    );
    assert_eq!(STREAM.matches("moe_transpose_u8_batched(").count(), 2);
    ordered(
        OPS,
        &[
            "populate_qwen4_stream_t",
            "compact_tables",
            "KernelLaunch::new(ctx.gpu, kernels.plan)",
        ],
    );
}

#[test]
fn streaming_path_keeps_originals_and_decode_dispatch_untouched() {
    assert!(!STREAM.contains("gpu.free("));
    for decode in [
        include_str!("../src/layers/moe/forward.rs"),
        include_str!("../src/layers/moe/forward_k2.rs"),
        include_str!("../src/layers/moe/forward_k3.rs"),
    ] {
        assert!(!decode.contains("STREAM_T"));
        assert!(!decode.contains("stream_t_scratch"));
    }
    assert!(STREAM.lines().count() <= 250);
    assert!(FACTORY.lines().count() <= 250);
}
