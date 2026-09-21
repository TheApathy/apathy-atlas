// SPDX-License-Identifier: AGPL-3.0-only

//! Production source contracts for the exact DeepSeek-V4 HC-finish/RMS arm.

use std::fs;
use std::path::PathBuf;

const CUDA_WRAPPER: &str = "kernels/gb10/deepseek-v4-flash/nvfp4/v4_hc_pre_finish_rms_fused.cu";
const DISPATCH: &str = "crates/spark-model/src/layers/qwen3_attention/trait_impl/prefill_hc_rms.rs";
const PREFILL: &str = "crates/spark-model/src/layers/qwen3_attention/trait_impl/prefill_inner.rs";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    fs::read_to_string(root().join(path))
        .unwrap_or_else(|error| panic!("required HC/RMS source {path}: {error}"))
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn section<'a>(source: &'a str, name: &str) -> &'a str {
    let begin = format!("// BEGIN {name}");
    let end = format!("// END {name}");
    assert_eq!(source.matches(&begin).count(), 1, "duplicate {begin}");
    assert_eq!(source.matches(&end).count(), 1, "duplicate {end}");
    let start = source.find(&begin).expect("missing begin marker") + begin.len();
    let finish = source[start..].find(&end).expect("missing end marker") + start;
    &source[start..finish]
}

#[test]
fn production_wrapper_and_optional_handle_are_canonical() {
    let wrapper = read(CUDA_WRAPPER);
    assert!(wrapper.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
    assert!(wrapper.contains("../../experiments/v4_hc_pre_finish_rms_fused.cu"));
    assert!(!wrapper.contains("extern \"C\""));

    let types = read("crates/spark-model/src/layers/qwen3_attention/types.rs");
    let init = read("crates/spark-model/src/layers/qwen3_attention/init.rs");
    assert!(types.contains("v4_hc_pre_finish_rms_fused_k"));
    assert!(init.contains("v4_hc_pre_finish_rms_fused_k:"));
    assert!(init.contains(
        "\"v4_hc_pre_finish_rms_fused\",\n                \"v4_hc_pre_finish_rms_fused\""
    ));
}

#[test]
fn strict_gate_is_literal_one_and_exact_qualification_shape() {
    let source = compact(&read(DISPATCH));
    for guard in [
        "std::env::var(\"ATLAS_V4_PREFILL_HC_RMS_FUSED\").as_deref()==Ok(\"1\")",
        "std::env::var(\"ATLAS_PREFILL_MAX_REQUIRE_ARMS\").as_deref()==Ok(\"1\")",
        "ctx.config.model_type==\"deepseek_v4\"",
        "constTARGET_LAYERS:usize=43",
        "ctx.config.num_hidden_layers==TARGET_LAYERS",
        "self.kv_dtype==KvCacheDtype::Fp8",
        "self.norm_vanilla",
        "constTARGET_TOKENS:usize=2_410",
        "constTARGET_HIDDEN:usize=4_096",
        "constTARGET_HC:usize=4",
        "constTARGET_SINKHORN_ITERS:usize=20",
        "num_tokens==TARGET_TOKENS",
        "hidden_size==TARGET_HIDDEN",
        "hc_mult==TARGET_HC",
        "sinkhorn_iters==TARGET_SINKHORN_ITERS",
        "!ctx.graph_capture",
        "!diagnostic",
        "!ctx.profile",
        "batched_meta.is_none()",
        "self.v4_hc_pre_finish_rms_fused_k.0!=0",
        "self.hc_pre_mix_tiled_k.0!=0",
        "hc_norm_eps.is_finite()",
        "hc_norm_eps>0.0",
        "hc_norm_eps==(ctx.config.rms_norm_epsasf32)",
        "(ctx.config.rms_norm_epsasf32).is_finite()",
        "(ctx.config.rms_norm_epsasf32)>0.0",
        "pointers_nonzero",
        "pointers_aligned",
        "canonical_pointers",
        "buffers_disjoint",
        "arena_fits",
    ] {
        assert!(
            source.contains(guard),
            "missing exact HC/RMS guard: {guard}"
        );
    }
    assert!(
        !source.contains("rms_eps==1.0e-6"),
        "trained RMS epsilon must not be pinned to the text checkpoint"
    );
    assert!(
        !source.contains("hc_norm_eps==1.0e-6"),
        "HC normalization epsilon must follow the effective checkpoint value"
    );
    assert!(source.contains("refusingincumbentfallback"));
}

#[test]
fn all_checked_planning_precedes_the_first_hc_mutation() {
    let prefill = read(PREFILL);
    let attn_plan = prefill
        .find("let use_hc_rms_attn =")
        .expect("attention HC/RMS plan");
    let ffn_plan = prefill
        .find("let use_hc_rms_ffn =")
        .expect("FFN HC/RMS plan");
    let expand = prefill.find("ops::hc_expand(").expect("first HC mutation");
    assert!(attn_plan < ffn_plan && ffn_plan < expand);

    let dispatch = compact(&read(DISPATCH));
    let eligibility = dispatch
        .find("letengaged=requested&&")
        .expect("eligibility decision");
    let launch = dispatch
        .find("//BEGINV4HC-finish/RMSmutation")
        .expect("mutation marker");
    assert!(eligibility < launch);
}

#[test]
fn fused_abi_writes_hidden_normed_post_and_comb_after_the_tiled_mix() {
    let dispatch = compact(section(&read(DISPATCH), "V4 HC-finish/RMS mutation"));
    let mix = dispatch
        .find("KernelLaunch::new(ctx.gpu,self.hc_pre_mix_tiled_k)")
        .expect("tiled mix launch");
    let fused = dispatch
        .find("KernelLaunch::new(ctx.gpu,self.v4_hc_pre_finish_rms_fused_k)")
        .expect("fused finish/RMS launch");
    assert!(mix < fused);
    for argument in [
        ".arg_ptr(streams)",
        ".arg_ptr(mix_scratch)",
        ".arg_ptr(site.hc_scale)",
        ".arg_ptr(site.hc_base)",
        ".arg_ptr(norm_weight.weight)",
        ".arg_ptr(hidden_out)",
        ".arg_ptr(normed_out)",
        ".arg_ptr(post_out)",
        ".arg_ptr(comb_out)",
        ".arg_u32(num_tokens)",
        ".arg_u32(hidden_size)",
        ".arg_u32(hc_mult)",
        ".arg_u32(sinkhorn_iters)",
        ".arg_f32(hc_norm_eps)",
        ".arg_f32(hc_eps)",
        ".arg_f32(rms_eps)",
    ] {
        assert!(dispatch.contains(argument), "fused ABI omits {argument}");
    }
}

#[test]
fn incumbent_fallback_and_diagnostic_order_are_preserved_at_both_sites() {
    let source = read(PREFILL);
    assert_eq!(source.matches("if use_hc_rms_").count(), 2);
    assert_eq!(source.matches("ops::hc_pre_tiled(").count(), 2);
    assert_eq!(source.matches("if !use_hc_rms_").count(), 2);

    let first_dispatch = source
        .find("if use_hc_rms_attn")
        .expect("attention dispatch");
    let first_diag = source[first_dispatch..]
        .find("if diag_this {")
        .map(|offset| first_dispatch + offset)
        .expect("attention diagnostic");
    let first_hprof = source[first_diag..]
        .find("hprof!(\"hc0_pre_attn\")")
        .map(|offset| first_diag + offset)
        .expect("attention profiling boundary");
    assert!(first_dispatch < first_diag && first_diag < first_hprof);

    let second_dispatch = source.find("if use_hc_rms_ffn").expect("FFN dispatch");
    let second_diag = source[second_dispatch..]
        .find("if diag_this {")
        .map(|offset| second_dispatch + offset)
        .expect("FFN diagnostic");
    let second_hprof = source[second_diag..]
        .find("hprof!(\"hc1_mid\")")
        .map(|offset| second_diag + offset)
        .expect("FFN profiling boundary");
    assert!(second_dispatch < second_diag && second_diag < second_hprof);
}

#[test]
fn engagement_receipts_follow_successful_launches_for_both_sites() {
    let dispatch = read(DISPATCH);
    let source = section(&dispatch, "V4 HC-finish/RMS mutation");
    let launch = source
        .rfind(".launch(stream)?;")
        .expect("successful fused launch");
    let receipt = source
        .find("V4_PREFILL_MAX_ARM_ENGAGED")
        .expect("engagement receipt");
    assert!(launch < receipt);
    for contract in [
        "V4_HC_RMS_ATTN_ENGAGED_LOGGED",
        "V4_HC_RMS_FFN_ENGAGED_LOGGED",
        "arm=hc_pre_finish_rms_fused",
        "site={}",
        "layer={}",
        "n={}",
    ] {
        assert!(source.contains(contract), "receipt omits {contract}");
    }
}
