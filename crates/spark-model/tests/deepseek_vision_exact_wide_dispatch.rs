// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn vision_bf16_verify_head_uses_exact_small_m_weight_streaming_kernel() {
    let init = include_str!("../src/model/impl_a1.rs");
    let types = include_str!("../src/model/types.rs");
    let head = include_str!("../src/model/impl_a3.rs");

    assert!(types.contains("dense_gemv_batchm_kernel: KernelHandle"));
    assert!(init.contains("let dense_gemv_batchm_kernel = crate::layers::try_kernel("));
    assert!(init.contains("\"dense_gemv_bf16_batchm\""));
    assert!(head.contains("self.config.deepseek_vision.is_some()"));
    assert!(head.contains("num_tokens <= ops::DENSE_GEMV_BATCHM_MAX_M"));
    assert!(head.contains("ops::dense_gemv_batchm("));
}

#[test]
fn native_fp8_mla_verify_keeps_every_projection_in_exact_row_order() {
    let source = include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/mla.rs");

    let q_b = source
        .split("let wqb = mla.wq_b_fp8.as_ref().unwrap();")
        .nth(1)
        .expect("native FP8 q_b branch");
    assert!(q_b.starts_with("\n            if verify_exact_gemv"));
    assert!(q_b.contains("ops::w8a16_gemv_batchm_exact("));

    let o_proj = source
        .split("if nv4_ok && let Some(ref woa4)")
        .nth(1)
        .expect("native FP8 O projection fallback");
    assert!(o_proj.contains("if verify_exact_gemv"));
    assert!(o_proj.matches("ops::w8a16_gemv_batchm_exact(").count() >= 2);
}

#[test]
fn dspark_bf16_draft_head_streams_the_weight_once_for_all_rows() {
    let source = include_str!("../src/layers/dspark_head.rs");

    assert!(source.contains("k_lmhead_bf16_batchm: KernelHandle"));
    assert!(source.contains("let bf16_head_batched ="));
    assert!(source.contains("ops::dense_gemv_batchm("));
}

#[test]
fn native_shared_verify_batches_all_three_fp8_projections_exactly() {
    let source = include_str!("../src/layers/moe/native_shared_fp8.rs");
    let verify = source
        .split("pub(super) fn run_native_fp8_shared_verify")
        .nth(1)
        .expect("native shared verify implementation")
        .split("pub(super) fn run_native_fp8_shared_expert")
        .next()
        .unwrap();

    assert_eq!(verify.matches("ops::w8a16_gemv_batchm_exact(").count(), 1);
    assert_eq!(verify.matches("project(").count(), 3);
    assert!(!verify.contains("for [input, gate, up, down] in plan.rows()"));
    assert!(verify.contains("ops::silu_mul("));
}

#[test]
fn compressed_verify_interleaves_pool_updates_with_each_attention_row() {
    let source = include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/mla.rs");

    // Snapshot first, but never advance the shared compressed pool ahead of
    // the row whose attention is about to consume it.
    assert!(source.contains("self.v4_compress_speculate(c.fwd, base, rows, eps, false, stream)?"));
    assert!(
        source.contains("let interleave_comp = self.ms_mla_v4_verify_crosses_comp_boundary(c)")
    );
    assert!(source.contains("self.v4_compress_append("));
    assert!(source.contains("(base + i) as u32"));
    assert!(source.contains("let attn_batched = !interleave_comp"));

    // The compressor uses expert_up_out as temporary storage. Batched Phase A
    // must live after that prefix or an interleaved append would clobber the
    // remaining Q/K rows before they are consumed.
    assert!(source.contains("fn ms_mla_v4_comp_scratch_prefix"));
    assert!(source.contains("let scratch = c.fwd.buffers.expert_up_out().offset(comp_prefix)"));
}

#[test]
fn rows_batch_control_also_preserves_plain_compressor_order() {
    let source = include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/mla.rs");
    let phase_b_fallback = source
        .split("} else {\n            for i in 0..n {")
        .nth(1)
        .expect("rows-batched Phase-B fallback");

    assert!(phase_b_fallback.contains("ms_mla_v4_verify_crosses_comp_boundary(c)"));
    assert!(phase_b_fallback.contains("c.verify_base_pos.unwrap() + i"));
}

#[test]
fn compressed_attention_batches_only_rows_with_one_pool_version() {
    let source = include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/mla.rs");

    assert!(source.contains("fn ms_mla_v4_comp_group_end"));
    assert!(source.contains("while lo < n"));
    assert!(source.contains("let hi = Self::ms_mla_v4_comp_group_end"));
    assert!(source.contains("for i in lo..hi"));
    assert!(source.contains("q_batch.offset(lo * q_dim as usize * bf16)"));
    assert!(source.contains("meta.seq_len.offset(lo * 4)"));
    assert!(source.contains("(hi - lo) as u32"));
}
