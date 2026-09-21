// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the default-off V4 q_b/RoPE + strided-K cache integration.

use std::fs;
use std::path::PathBuf;

const QB_KERNEL: &str = "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu";
const CACHE_KERNEL: &str =
    "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_cache_kfull_fp8_fused.cu";
const PREFILL: &str = "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    fs::read_to_string(root().join(path))
        .unwrap_or_else(|error| panic!("required joint-integration source {path}: {error}"))
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
fn production_modules_and_optional_handles_are_canonical() {
    let qb = read(QB_KERNEL);
    let cache = read(CACHE_KERNEL);
    assert!(qb.contains("void v4_prefill_qb_norm_rope_fused("));
    assert!(cache.contains("void v4_prefill_cache_kfull_fp8_fused("));

    let types = read("crates/spark-model/src/layers/qwen3_attention/types.rs");
    let init = read("crates/spark-model/src/layers/qwen3_attention/init.rs");
    for handle in [
        "v4_prefill_qb_norm_rope_fused_k",
        "v4_prefill_cache_kfull_fp8_fused_k",
    ] {
        assert!(types.contains(handle), "missing layer handle {handle}");
        assert!(
            init.contains(handle),
            "missing handle initialization {handle}"
        );
        let symbol = handle.strip_suffix("_k").expect("kernel handle suffix");
        assert!(init.contains(&format!("\"{symbol}\"")));
    }
}

#[test]
fn one_strict_gate_owns_both_launches_and_every_abi_guard() {
    let source = read(PREFILL);
    let flat = compact(&source);
    assert!(
        flat.contains(
            "std::env::var(\"ATLAS_V4_PREFILL_QB_ROPE_CACHE_FUSED\").as_deref()==Ok(\"1\")"
        )
    );
    for contract in [
        "self.v4_prefill_qb_norm_rope_fused_k.0!=0",
        "self.v4_prefill_cache_kfull_fp8_fused_k.0!=0",
        "n==2410",
        "nq==64",
        "nkv==1",
        "hd_mla==512",
        "nope==448",
        "rope==64",
        "kv_lora==512",
        "mla_cache_dim==576",
        "self.kv_dtype==KvCacheDtype::Fp8",
        "kv_cache.dtype_for_layer(self.attn_layer_idx)==KvCacheDtype::Fp8",
        "cache_dims==(1,576)",
        "kv_cache.block_size()==16",
        "cache_stride==16*576",
        "kv_cache.k_block_stride_bytes_for_layer(self.attn_layer_idx)==16*576",
        "kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx)==16*576",
        "cache_num_blocks!=0",
        "self.fp8_calibration.is_none()",
        "!ctx.graph_capture",
        "!diag_this",
        "!ctx.profile",
        "eps.is_finite()",
        "eps>0.0",
        "rope_mscale.is_finite()",
        "cache_k_scale.is_finite()",
        "cache_k_scale>0.0",
        "cache_v_scale.is_finite()",
        "cache_v_scale>0.0",
        "packed_qk_layout",
        "pointers_nonzero",
        "pointers_aligned",
        "buffers_disjoint",
    ] {
        assert!(flat.contains(contract), "missing joint guard: {contract}");
    }
    assert!(compact(&source).contains(
        "letuse_v4_qb_rope_kernel=use_v4_qb_rope_cache_fused||use_v4_prefill_qb_rope_fused;"
    ));
    assert_eq!(source.matches("if use_v4_qb_rope_cache_fused").count(), 2);
}

#[test]
fn joint_arm_has_no_raw_rope_scratch_and_fallback_keeps_incumbents() {
    let source = read(PREFILL);
    let rope_joint = between(
        &source,
        "// BEGIN V4 joint q_b/RoPE dispatch",
        "// END V4 joint q_b/RoPE dispatch",
    );
    let cache_joint = between(
        &source,
        "// BEGIN V4 joint direct-cache dispatch",
        "// END V4 joint direct-cache dispatch",
    );
    let rope_flat = compact(rope_joint);
    assert!(rope_flat.contains(
        ".grid([n,fused_nq,1]).block([512,1,1]).arg_ptr(q_full).arg_ptr(fused_k_full).arg_ptr(norm_unit_w).arg_ptr(meta.positions).arg_ptr(rope_inv_freq).arg_u32(n).arg_u32(fused_nq).arg_u32(nkv).arg_u32(hd_mla).arg_u32(nope).arg_u32(rope).arg_f32(eps).arg_f32(rope_mscale).launch(stream)?"
    ));
    let cache_flat = compact(cache_joint);
    assert!(cache_flat.contains(
        ".grid([n,1,1]).block([256,1,1]).arg_ptr(kv_latent).arg_ptr(fused_k_full).arg_ptr(cache_k_pool).arg_ptr(cache_v_pool).arg_ptr(meta.slot).arg_u32(n).arg_u32(cache_num_blocks).arg_u32(kv_cache.block_size()asu32).arg_f32(cache_k_scale).arg_f32(cache_v_scale).arg_u64(cache_strideasu64).launch(stream)?"
    ));
    assert!(rope_joint.contains("v4_prefill_qb_norm_rope_fused_k"));
    assert!(!rope_joint.contains("q_rope_tmp"));
    assert!(!rope_joint.contains("mla_q_rope_extract_batched"));
    assert!(cache_joint.contains("v4_prefill_cache_kfull_fp8_fused_k"));
    assert!(!cache_joint.contains("mla_cache_assemble_batched"));
    assert!(!cache_joint.contains("k_rope_tmp"));

    assert!(source.contains("ops::rms_norm("));
    assert!(source.matches("ops::mla_q_rope_extract_batched(").count() >= 4);
    assert!(source.contains("ops::mla_cache_assemble_batched("));
    assert!(source.contains("v4_prefill_kv_alias_enabled()"));

    let rope_dispatch = source
        .find("// BEGIN V4 joint q_b/RoPE dispatch")
        .expect("joint rope dispatch");
    let cache_dispatch = source
        .find("// BEGIN V4 joint direct-cache dispatch")
        .expect("joint cache dispatch");
    let output_reuse = source
        .find("let o_out = ctx.buffers.qkv_output();")
        .expect("post-cache qkv scratch reuse");
    assert!(rope_dispatch < cache_dispatch && cache_dispatch < output_reuse);
}
