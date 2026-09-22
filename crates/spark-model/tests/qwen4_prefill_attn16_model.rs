// SPDX-License-Identifier: AGPL-3.0-only

const ADMISSION: &str = include_str!("../src/layers/qwen4_prefill_moe.rs");
const PLAN: &str = include_str!("../src/layers/qwen4_prefill_moe/attn16.rs");
const QSA: &str = include_str!("../src/layers/qwen4_qsa/prefill_exact.rs");
const ATTN: &str = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_attn16.rs");
const DEVICE: &str =
    include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_attn16_device.rs");
const METADATA: &str =
    include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_attn16_metadata.rs");
const DEVICE_CUDA: &str =
    include_str!("../../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_attn16_device.cu");
const DEVICE_PARITY: &str = include_str!("fixtures/qwen4_attn16_device_parity.cu");
const CORE_ADMISSION: &str =
    include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_admit.rs");
const WIRING: &str = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");
const HC_TILE: &str = include_str!("../src/layers/qwen4_hyper/prefill_tile.rs");
const HC_PARITY: &str = include_str!("fixtures/qwen4_hc16_parity.cu");
const PREFILL_A: &str = include_str!("../src/model/trait_impl/prefill_a.rs");
const PREFILL_B: &str = include_str!("../src/model/trait_impl/prefill_b.rs");

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
fn selector_is_checked_before_disabled_moe_return() {
    ordered(
        ADMISSION,
        &[
            "let selected = selected()?",
            "attn16::admit_request(",
            "if !selected",
        ],
    );
    for required in [
        "moe_selected && exact_hyper && exact_qkv16 && exact_o16",
        "ATLAS_QWEN4_PREFILL_SSM_GEMM",
        "ATLAS_PAGED_DECODE_SPLITK",
    ] {
        assert!(PLAN.contains(required), "missing admission: {required}");
    }
}

#[test]
fn request_surface_rejects_stateful_modes_before_buffer_effects() {
    for source in [PREFILL_A, PREFILL_B] {
        ordered(
            source,
            &[
                "attn16::admit_surface(",
                "qwen4_prefill_moe::admit_request",
                "preflight_qwen4_attn16",
                "zero_all",
            ],
        );
    }
    for required in [
        "!has_vision",
        "sequence_start == 0",
        "chunk_rows == total_rows",
        "!high_speed_swap",
    ] {
        assert!(PLAN.contains(required), "missing surface gate: {required}");
    }
}

#[test]
fn qsa_group4_preserves_the_scalar_transaction_order() {
    ordered(
        QSA,
        &[
            "ops::dense_gemv_batchn(",
            "self.stage_prefill_raw",
            "self.pool_prefill",
            "ops::rms_norm(",
            "self.apply_rope_n(",
            "self.store_prefill_compressed",
        ],
    );
    assert!(!QSA.contains("dense_gemm"));
    assert!(QSA.contains("group_start.is_multiple_of(COMPRESS_RATIO)"));
}

#[test]
fn attention16_uses_ordinary_paged_decode_after_qsa_and_kv() {
    ordered(
        ATTN,
        &[
            "qsa.update_prefill_exact_group4(",
            "mrope_or_scalar(",
            "self.write_kv_cache(",
            "expand_or_copy_metadata(",
            "self.run_paged_decode(",
            "ops::sigmoid_gate_mul_batched(",
            "ENGAGED ATLAS_QWEN4_PREFILL_ATTN_CORE16",
        ],
    );
    for forbidden in [
        "prefill_attention",
        "kgamma",
        "gpu.alloc(",
        "gpu.synchronize(",
        "rope_yarn_scaled",
        "qwen4_yarn_inv_freq",
        "stream == gpu.default_stream()",
        "stream == ctx.gpu.default_stream()",
    ] {
        assert!(!ATTN.contains(forbidden), "unsafe substitute: {forbidden}");
    }
    assert!(ATTN.matches("stream,").count() >= 6);
    assert!(DEVICE.contains("ops::rope_mrope_interleaved("));
    assert!(METADATA.contains("tables.extend_from_slice(block_table)"));
}

#[test]
fn device16_preserves_exact_mrope_and_metadata_contracts() {
    for required in [
        "row_stride_bf16",
        "k_offset_bf16",
        "pair_idx % 3",
        "pow((double)theta, freq_exp_d)",
        "cosf(angle)",
        "sinf(angle)",
        "tile_start + row",
        "tile_start + 16",
    ] {
        assert!(
            DEVICE.contains(required)
                || METADATA.contains(required)
                || DEVICE_CUDA.contains(required),
            "missing exact device contract: {required}"
        );
    }
    for forbidden in ["powf(", "sincosf(", "metadata.seq_len)"] {
        assert!(
            !DEVICE_CUDA.contains(forbidden),
            "drifting device operation: {forbidden}"
        );
    }
}

#[test]
fn full_tiles_fail_closed_and_tails_remain_scalar() {
    ordered(
        WIRING,
        &[
            "if exact_attn16 && (tile_rows == 16 || tile_rows == 32)",
            "exact_qkv.ok_or_else",
            "self.prefill_moe_attn16(",
            "if !batched_core",
            "for local in 0..tile_rows",
            "self.attention_forward_preprojected_raw(",
        ],
    );
}

#[test]
fn hc16_batches_only_complete_f37_tiles_before_scratch_reuse() {
    for required in [
        "ATLAS_QWEN4_PREFILL_ATTN_HC16",
        "core_selected && device_selected",
    ] {
        assert!(PLAN.contains(required), "missing HC16 gate: {required}");
    }
    ordered(
        WIRING,
        &[
            "attn_hyper.preflight_saved_tile(",
            "if exact_qkv16 && (tile_rows == 16 || tile_rows == 32)",
            "attn_hyper.pack_saved_tile_or_rows(",
            "self.qwen4_k5_project_qkv_exact(",
            "ops::w4a16_gemv_batch_logits_exact_with(",
            "attn_hyper.inject_saved_tile_or_rows(",
            "tile_start += tile_rows",
        ],
    );
    for required in [
        "ROWS * self.residual_width() * 2",
        "ROWS * self.hidden_size * 2",
        "spans must be disjoint",
        "self.inject_saved_batched(hidden, packed, residual, ROWS",
    ] {
        assert!(
            HC_TILE.contains(required),
            "missing HC16 contract: {required}"
        );
    }
}

#[test]
fn core32_keeps_qkv_and_hc_in_m16_order_while_widening_the_causal_core() {
    for required in [
        "ATLAS_QWEN4_PREFILL_ATTN_CORE32",
        "core_selected && device_selected && hc_selected",
        "tile_rows == 16 || tile_rows == 32",
        "route_for_rows(admitted_rows as u32)",
        "qwen4_attn32_expand_meta",
        "tile_start + 32",
    ] {
        assert!(
            PLAN.contains(required)
                || WIRING.contains(required)
                || CORE_ADMISSION.contains(required)
                || DEVICE.contains(required)
                || DEVICE_CUDA.contains(required),
            "missing core32 contract: {required}"
        );
    }
    ordered(
        WIRING,
        &[
            "for chunk in (0..tile_rows).step_by(16)",
            "self.qwen4_k5_project_qkv_exact(",
            "self.prefill_moe_attn16(",
            "ops::w4a16_gemv_batch_logits_exact_with(",
            "for chunk in (0..tile_rows).step_by(16)",
            "attn_hyper.inject_saved_tile_or_rows(",
        ],
    );
}

#[test]
fn new_source_files_are_bounded_and_licensed() {
    for source in [
        PLAN,
        QSA,
        ATTN,
        DEVICE,
        METADATA,
        CORE_ADMISSION,
        WIRING,
        DEVICE_CUDA,
        DEVICE_PARITY,
        HC_TILE,
        HC_PARITY,
    ] {
        assert!(source.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
        assert!(source.lines().count() <= 250);
    }
}
