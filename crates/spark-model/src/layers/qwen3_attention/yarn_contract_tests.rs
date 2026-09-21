// SPDX-License-Identifier: AGPL-3.0-only

//! Source-bound coverage for the four dense-Qwen YaRN routing surfaces.

#[test]
fn all_dense_rope_routes_use_the_shared_mrope_dispatch() {
    let paged = include_str!("prefill/paged.rs");
    let cache_skip = include_str!("prefill/cache_skip.rs");
    let decode = include_str!("decode/attention_forward.rs");
    let multi = include_str!("trait_impl/multi_seq/attn.rs");

    for (name, source) in [
        ("paged", paged),
        ("cache-skip", cache_skip),
        ("decode", decode),
        ("multi-seq", multi),
    ] {
        assert!(
            source.contains("self.apply_mrope("),
            "missing YaRN route in {name}"
        );
    }
    assert!(paged.contains("ATLAS_PREFILL_QKNORM_ROPE=1 is incompatible with YaRN"));
    assert!(cache_skip.contains("ATLAS_PREFILL_QKNORM_ROPE=1 is incompatible with YaRN"));
    assert!(multi.contains("ATLAS_ATTN_QKV_MEGA=1 is incompatible with YaRN"));
}

#[test]
fn incompatible_multi_seq_mega_is_rejected_before_gpu_effects() {
    let multi = include_str!("trait_impl/multi_seq/mod.rs");
    let inner = &multi[multi.find("fn decode_multi_seq_inner_impl").unwrap()..];
    let guard = inner
        .find("self.validate_yarn_multi_seq_controls()?;")
        .unwrap();
    assert!(guard < inner.find("MultiSeqCtx::new_with_qkv_base").unwrap());
    assert!(guard < inner.find("ops::rms_norm_residual(").unwrap());
    assert!(guard < inner.find("self.ms_phase_qkv(&c)").unwrap());

    let batched = include_str!("trait_impl/multi_seq/qkv_batched.rs");
    let guard = batched
        .find("self.validate_yarn_multi_seq_controls()?;")
        .unwrap();
    assert!(guard < batched.find("MultiSeqCtx::new(").unwrap());
    assert!(guard < batched.find("ops::rms_norm(").unwrap());
}

#[test]
fn scaled_mrope_uses_a_distinct_kernel_abi() {
    let ops = include_str!("../ops/embeddings.rs");
    let cuda = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/common/rope_mrope_interleaved.cu"
    ));
    assert!(ops.contains("pub fn rope_mrope_interleaved_yarn("));
    assert!(ops.contains(".arg_f32(attention_factor)"));
    assert!(ops.contains(".grid([seq_blocks, total_heads, 1])"));
    assert!(cuda.contains("void rope_forward_mrope_interleaved_yarn("));
    assert!(cuda.contains("const unsigned int seq_block = blockIdx.x"));
    assert!(cuda.contains("const unsigned int head_idx = blockIdx.y"));
    assert!(cuda.contains("inv_freq_interp * ramp + inv_freq_extrap * (1.0f - ramp)"));
    assert!(cuda.contains("cosf(angle) * attention_factor"));
    assert!(!cuda.contains("sin_f"), "unexpected alternate sine path");
    assert!(cuda.contains("sinf(angle) * attention_factor"));
    assert!(cuda.contains("const size_t token_idx"));
    assert!(cuda.contains("(size_t)seq_pos * (size_t)num_q_heads * (size_t)head_dim"));
}
