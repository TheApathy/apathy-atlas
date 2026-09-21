// SPDX-License-Identifier: AGPL-3.0-only

const CUDA: &str = include_str!("../../../kernels/gb10/common/qwen4_qsa.cu");
const HOST: &str = include_str!("../src/layers/qwen4_qsa.rs");
const DEVICE: &str = include_str!("../src/layers/qwen4_qsa/device.rs");
const PREFILL: &str = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_inner.rs");
const F8: &str = include_str!("../src/layers/qwen3_attention/trait_impl/prefill_moe_only.rs");

fn kernel(name: &str) -> &str {
    CUDA.split(&format!("void {name}("))
        .nth(1)
        .expect("production kernel")
        .split("extern \"C\"")
        .next()
        .unwrap()
}

fn ordered(source: &str, markers: &[&str]) {
    let mut rest = source;
    for marker in markers {
        let position = rest
            .find(marker)
            .unwrap_or_else(|| panic!("missing/order: {marker}"));
        rest = &rest[position + marker.len()..];
    }
}

#[test]
fn serial_and_batched_staging_use_physical_slots() {
    assert!(CUDA.contains("#include \"qwen4_qsa_positions.cuh\""));
    for name in [
        "qwen4_qsa_stage_pool",
        "qwen4_qsa_store_compressed",
        "qwen4_qsa_stage_prefill_raw",
    ] {
        let body = kernel(name);
        assert!(
            body.contains("qwen4_qsa_positions::physical_slot("),
            "{name}"
        );
        assert!(body.contains("physical.ring_row"), "{name}");
        assert!(!body.contains("position & (COMPRESS_RATIO - 1)"), "{name}");
        assert!(!body.contains("positions[token] &"), "{name}");
    }
}

#[test]
fn device_selection_uses_sequence_length_not_rotary_position() {
    let body = kernel("qwen4_qsa_select_expand_device_meta");
    assert!(body.contains("qwen4_qsa_positions::physical_query(sequence_length, position)"));
    assert!(!body.contains("position = *position_ptr"));
    // Keep the parameter layout stable; removal requires an explicit ABI change.
    assert!(body.contains("const unsigned int* __restrict__ position_ptr,"));
    ordered(
        DEVICE,
        &[
            "self.select_expand_device)",
            ".arg_ptr(meta.positions)",
            ".arg_ptr(meta.seq_len)",
        ],
    );
}

#[test]
fn host_requires_current_physical_query_before_effects() {
    let body = HOST.split("pub fn update_and_select(").nth(1).unwrap();
    ordered(
        body,
        &[
            "valid_physical_query(position, sequence_length)",
            "self.cache(",
            "ops::dense_gemv(",
        ],
    );
}

#[test]
fn optional_batch_fails_before_affected_layer_and_index_effects() {
    ordered(
        PREFILL,
        &[
            "ATLAS_QWEN4_ATTN_PREFILL_BATCH",
            "ATLAS_QWEN4_QSA_PREFILL_GEMM",
            "validate_prefill_index_batch(num_tokens, seq_len_start)",
            "attn_hyper.prepare_prefill_exact(",
        ],
    );
    let body = HOST.split("pub fn update_prefill_index(").nth(1).unwrap();
    ordered(
        body,
        &[
            "validate_prefill_index_batch(num_tokens, seq_len_start)",
            "self.cache(",
            "ops::dense_gemm(",
        ],
    );
    assert!(HOST.contains("prefill_index_batch_is_safe(num_tokens, seq_len_start)"));
    assert!(HOST.contains("ATLAS_QWEN4_QSA_PREFILL_GEMM"));
}

#[test]
fn rotary_streams_and_f8_shipping_sequence_are_retained() {
    assert!(HOST.contains("mod position_contract;"));
    assert!(HOST.contains(
        "self.apply_rope(\n            cache.projected_qk,\n            meta.positions,"
    ));
    ordered(
        F8,
        &[
            "seq_len_start + row + 1",
            "positions: metadata.positions.offset(row * 4)",
            "positions_h: metadata.positions_h.offset(row * 4)",
            "positions_w: metadata.positions_w.offset(row * 4)",
            "slot: metadata.slot.offset(row * 8)",
            "self.attention_forward(",
            "seq_len_start + row",
        ],
    );
    assert!(!HOST.contains("ATLAS_QWEN4_QSA_PREFILL_GEMM_FALLBACK"));
}

#[test]
fn cache_and_stage_launch_abis_remain_unchanged() {
    ordered(
        HOST,
        &[
            "KernelLaunch::new(gpu, self.stage_pool)",
            ".arg_ptr(cache.projected_qk)",
            ".arg_ptr(cache.raw_ring)",
            ".arg_ptr(cache.pooled_key)",
            ".arg_ptr(cache.first_position)",
            ".arg_ptr(meta.slot)",
            ".arg_ptr(meta.positions)",
            ".arg_u32(cache.block_size as u32)",
        ],
    );
    ordered(
        DEVICE,
        &[
            "KernelLaunch::new(gpu, self.stage_pool)",
            ".arg_ptr(cache.projected_qk)",
            ".arg_ptr(cache.raw_ring)",
            ".arg_ptr(cache.pooled_key)",
            ".arg_ptr(cache.first_position)",
            ".arg_ptr(meta.slot)",
            ".arg_ptr(meta.positions)",
            ".arg_u32(cache.block_size as u32)",
        ],
    );
}
