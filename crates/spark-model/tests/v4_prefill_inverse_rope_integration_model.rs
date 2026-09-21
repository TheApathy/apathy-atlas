// SPDX-License-Identifier: AGPL-3.0-only

//! Production source contracts for the exact-shape V4 inverse-RoPE arm.

use std::fs;
use std::path::PathBuf;

const KERNEL: &str = "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_rope_fused_inverse.cu";
const PREFILL: &str = "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    fs::read_to_string(root().join(path))
        .unwrap_or_else(|error| panic!("required inverse-RoPE source {path}: {error}"))
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn between<'a>(source: &'a str, begin: &str, end: &str) -> &'a str {
    let start = source.find(begin).expect("missing begin marker") + begin.len();
    let finish = source[start..].find(end).expect("missing end marker") + start;
    &source[start..finish]
}

#[test]
fn production_wrapper_and_optional_handle_are_canonical() {
    let kernel = read(KERNEL);
    assert!(kernel.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
    assert!(kernel.contains("../../experiments/v4_prefill_rope_fused.cu"));
    assert!(!kernel.contains("extern \"C\""));

    let types = read("crates/spark-model/src/layers/qwen3_attention/types.rs");
    let init = read("crates/spark-model/src/layers/qwen3_attention/init.rs");
    assert!(types.contains("v4_prefill_rope_fused_inverse_k"));
    assert!(init.contains("v4_prefill_rope_fused_inverse_k:"));
    assert!(init.contains(
        "\"v4_prefill_rope_fused_inverse\",\n                \"v4_prefill_rope_fused_inverse\""
    ));
    assert!(!types.contains("v4_prefill_rope_fused_forward_k"));
}

#[test]
fn strict_default_off_gate_and_exact_shape_decline_before_launch() {
    let source = read(PREFILL);
    let flat = compact(&source);
    assert!(
        flat.contains(
            "std::env::var(\"ATLAS_V4_PREFILL_INVERSE_ROPE_FUSED\").as_deref()==Ok(\"1\")"
        )
    );
    for guard in [
        "ctx.config.model_type==\"deepseek_v4\"",
        "self.v4_prefill_rope_fused_inverse_k.0!=0",
        "n==2410",
        "nq==64",
        "nkv==1",
        "hd_mla==512",
        "nope==448",
        "rope==64",
        "!ctx.graph_capture",
        "!diag_this",
        "!ctx.profile",
        "rope_mscale.is_finite()",
        "rope_mscale>0.0",
        "inverse_pointers_nonzero",
        "inverse_pointers_aligned",
        "inverse_buffers_disjoint",
    ] {
        assert!(flat.contains(guard), "missing inverse-RoPE guard: {guard}");
    }

    let decision = source
        .find("let use_v4_prefill_inverse_rope_fused =")
        .expect("inverse-RoPE decision");
    let launch = source
        .find("// BEGIN V4 inverse-only RoPE dispatch")
        .expect("inverse-RoPE launch");
    assert!(decision < launch, "eligibility must precede mutation");
}

#[test]
fn candidate_launch_abi_and_incumbent_fallback_are_both_locked() {
    let source = read(PREFILL);
    let dispatch = compact(between(
        &source,
        "// BEGIN V4 inverse-only RoPE dispatch",
        "// END V4 inverse-only RoPE dispatch",
    ));
    assert!(dispatch.contains(
        "KernelLaunch::new(ctx.gpu,self.v4_prefill_rope_fused_inverse_k).grid([n,nq,1]).block([32,1,1]).arg_ptr(attn_out).arg_ptr(meta.positions).arg_ptr(rope_inv_freq).arg_u32(n).arg_u32(nq).arg_u32(0).arg_u32(hd_mla).arg_u32(nope).arg_u32(rope).arg_f32(rope_mscale).launch(stream)?"
    ));

    let fallback = between(
        &source,
        "// BEGIN V4 inverse-only RoPE fallback",
        "// END V4 inverse-only RoPE fallback",
    );
    for incumbent in [
        "ops::mla_q_rope_extract_batched(",
        "self.rope_yarn_interleaved_inv_k",
        "ops::mla_q_rope_writeback_batched(",
    ] {
        assert!(fallback.contains(incumbent), "missing fallback {incumbent}");
    }
}

#[test]
fn retained_bf16_cublaslt_consumer_remains_after_materialized_output() {
    let prefill = read(PREFILL);
    let launch = prefill
        .find("// BEGIN V4 inverse-only RoPE dispatch")
        .expect("inverse launch");
    let diagnostic = prefill[launch..]
        .find("if diag_this {")
        .map(|offset| launch + offset)
        .expect("post-inverse diagnostic");
    let grouped = prefill
        .find("self.v4_grouped_wo_a_prefill(")
        .expect("grouped wo_a consumer");
    assert!(launch < diagnostic && diagnostic < grouped);

    let projection = read("crates/spark-model/src/layers/qwen3_attention/prefill/v4_fp8_proj.rs");
    assert!(projection.contains("RELEASE_BF16=0 + ATLAS_V4_PREFILL_CUBLASLT=1"));
    assert!(projection.contains("try_v4_cublas_prefill_strided("));
}
