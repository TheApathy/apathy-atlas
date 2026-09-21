// SPDX-License-Identifier: AGPL-3.0-only

//! Aggregate, non-timing budget for the isolated V4 prefill fusion components.
//!
//! The model keeps mutually dependent opportunities from being counted twice.
//! Logical bytes and launch counts prioritize a future GB10 run; they are not
//! device traffic, elapsed time, or a throughput result.

use std::fs;
use std::path::PathBuf;

const TOKENS: u64 = 2_410;
const LAYERS: u64 = 43;
const NQ: u64 = 64;
const HEAD_DIM: u64 = 512;
const ROPE: u64 = 64;
const Q_LORA: u64 = 1_024;
const CACHE_DIM: u64 = 576;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required V4 fusion source {relative}: {error}"))
}

#[test]
fn unique_logical_traffic_budget_does_not_double_count_composability() {
    // Current extract/rotate/writeback chains -> the existing direct forward
    // and inverse RoPE component. This is the source-modeled value already
    // frozen in v4_prefill_rope_fused_model.
    let direct_rope_per_layer = 159_175_680u64;

    // Incremental over direct forward RoPE: retain the Q RMS apply operands
    // and avoid the normalized-tail global write plus direct-RoPE reread.
    let qb_norm_direct_per_layer = TOKENS * NQ * HEAD_DIM * 2 + TOKENS * NQ * ROPE * 4;

    // Incremental over direct inverse RoPE: avoid its tail write and both
    // quantize_a_fp8_rows tail rereads (amax pass and FP8 emit pass).
    let inverse_rope_quant_per_layer = TOKENS * NQ * ROPE * 6;

    // W8A8-only q_a path: RMS apply reread + normalized BF16 write +
    // activation-quantizer reread.
    let qa_norm_quant_per_layer = TOKENS * Q_LORA * 6;

    // Cache-only fusion: eliminate the two BF16 scratch rows' write+read.
    let cache_assemble_fp8_per_layer = TOKENS * CACHE_DIM * 2 * 4;

    // Separate host alias opportunity: K->V copies one 512-wide BF16 row,
    // so logical traffic is one read plus one write.
    let k_to_v_alias_per_layer = TOKENS * HEAD_DIM * 2 * 2;

    assert_eq!(qb_norm_direct_per_layer, 197_427_200);
    assert_eq!(inverse_rope_quant_per_layer, 59_228_160);
    assert_eq!(qa_norm_quant_per_layer, 14_807_040);
    assert_eq!(cache_assemble_fp8_per_layer, 11_105_280);
    assert_eq!(k_to_v_alias_per_layer, 4_935_680);

    let unique_per_layer = direct_rope_per_layer
        + qb_norm_direct_per_layer
        + inverse_rope_quant_per_layer
        + qa_norm_quant_per_layer
        + cache_assemble_fp8_per_layer
        + k_to_v_alias_per_layer;
    let unique_pass = unique_per_layer * LAYERS;
    assert_eq!(unique_per_layer, 446_679_040);
    assert_eq!(unique_pass, 19_207_198_720);

    // The strided-K cache component avoids recreating the K RoPE extraction
    // after direct in-place RoPE. That extraction is already absent from the
    // direct-RoPE budget above, so it is a composition prerequisite, not an
    // additive seventh opportunity.
    let strided_k_unlock = TOKENS * ROPE * 4 * LAYERS;
    assert_eq!(strided_k_unlock, 26_529_280);
    assert_ne!(unique_pass, unique_pass + strided_k_unlock);
}

#[test]
fn peak_bandwidth_equivalent_is_not_a_two_thousand_tok_s_claim() {
    let unique_pass_bytes = 19_207_198_720f64;
    let peak_bandwidth_equivalent_s = unique_pass_bytes / 273_000_000_000f64;
    let required_reduction_s = 2.660_f64 - 1.205_f64;

    assert!((peak_bandwidth_equivalent_s - 0.070_356_039_267_399_27).abs() < 1.0e-12);
    assert!((required_reduction_s - 1.455).abs() < 1.0e-12);
    assert!(peak_bandwidth_equivalent_s < required_reduction_s * 0.05);

    // CTA scheduling, cache residency, and overlap can dominate this byte
    // equivalent in either direction. Only the receipted GPU ABBA probes and
    // five fresh end-to-end TTFT samples may promote a throughput result.
    let byte_equivalent_ttft = 2.660 - peak_bandwidth_equivalent_s;
    assert!(TOKENS as f64 / byte_equivalent_ttft < 1_000.0);
}

#[test]
fn aggregate_budget_is_source_locked_and_marks_conditional_paths() {
    let direct = source("kernels/gb10/experiments/v4_prefill_rope_fused.cu");
    let qb = source("kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu");
    let qa = source("kernels/gb10/experiments/v4_prefill_qa_norm_w8a8_quant_fused.cu");
    let inverse = source("kernels/gb10/experiments/v4_prefill_inverse_rope_w8a8_quant_fused.cu");
    let cache = source("kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_cache_kfull_fp8_fused.cu");
    let dispatch = source("crates/spark-model/src/layers/qwen3_attention/prefill/v4_fp8_proj.rs");

    assert!(direct.contains("v4_prefill_rope_fused_forward"));
    assert!(direct.contains("v4_prefill_rope_fused_inverse"));
    assert!(qb.contains("v4_prefill_qb_norm_rope_fused"));
    assert!(qa.contains("v4_prefill_qa_norm_w8a8_quant_fused"));
    assert!(inverse.contains("v4_prefill_inverse_rope_w8a8_quant_fused"));
    assert!(inverse.contains("require diag_this == false"));
    assert!(cache.contains("v4_prefill_cache_kfull_fp8_fused"));

    // q_a and inverse-RoPE quant fusion are useful only when the existing
    // released-BF16, FP8-native W8A8 dispatch is eligible. Cache fusion also
    // presumes the FP8 paged-cache arm. These conditions must be reported with
    // the aggregate rather than presented as universal serving savings.
    assert!(dispatch.contains("ATLAS_V4_ATTN_RELEASE_BF16=1"));
    assert!(dispatch.contains("ATLAS_V4_PROJ_FP8MMA=1"));
    assert!(dispatch.contains("quantize_a_fp8_rows"));
    assert!(
        source("kernels/gb10/common/reshape_and_cache.cu").contains("reshape_and_cache_flash_fp8")
    );

    // q_b diagnostics currently observe normalized Q before forward RoPE, so
    // a future combined q_b dispatch must retain the old chain when that
    // diagnostic observation is enabled.
    let prefill = source("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
    let qb_norm = prefill.find("// q_b_norm:").expect("q_b norm boundary");
    let qb_diag = prefill[qb_norm..]
        .find("if diag_this {")
        .map(|offset| qb_norm + offset)
        .expect("q_b diagnostic boundary");
    let forward_rope = prefill[qb_diag..]
        .find("ops::mla_q_rope_extract_batched(")
        .map(|offset| qb_diag + offset)
        .expect("forward RoPE boundary");
    assert!(qb_norm < qb_diag && qb_diag < forward_rope);
}
