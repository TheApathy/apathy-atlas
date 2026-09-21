// SPDX-License-Identifier: AGPL-3.0-only

//! Production-reachability contracts for the exact-shape TC2 warp-0 arm.

use std::fs;
use std::path::PathBuf;

const KERNEL: &str = "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_attn_compressed_tc2_warp0.cu";
const HOST: &str = "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    fs::read_to_string(root().join(path))
        .unwrap_or_else(|error| panic!("required TC2 warp-0 source {path}: {error}"))
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn production_wrapper_and_optional_handle_are_canonical() {
    let kernel = read(KERNEL);
    assert!(kernel.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
    assert!(kernel.contains("../../experiments/v4_prefill_attn_compressed_tc2_warp0.cu"));
    assert!(!kernel.contains("extern \"C\""));

    let types = read("crates/spark-model/src/layers/qwen3_attention/types.rs");
    let init = read("crates/spark-model/src/layers/qwen3_attention/init.rs");
    assert!(types.contains("v4_prefill_attn_compressed_tc2_warp0_k"));
    assert!(init.contains("v4_prefill_attn_compressed_tc2_warp0_k:"));
    assert!(init.contains(
        "\"v4_prefill_attn_compressed_tc2_warp0\",\n                \"v4_prefill_attn_compressed_tc2_warp0\""
    ));
}

#[test]
fn strict_default_off_gate_and_exact_deepseek_shape_are_locked() {
    let source = read(HOST);
    let flat = compact(&source);
    for contract in [
        "std::env::var(\"ATLAS_V4_PREFILL_TC2_WARP0\").as_deref()==Ok(\"1\")",
        "ctx.config.model_type==\"deepseek_v4\"",
        "self.v4_prefill_attn_compressed_tc2_warp0_k.0!=0",
        "v4_prefill_tc2_enabled()",
        "n==2410",
        "nq==64",
        "nkv==1",
        "hd_mla==512",
    ] {
        assert!(flat.contains(contract), "missing warp-0 guard: {contract}");
    }
}

#[test]
fn candidate_is_selected_at_both_tc2_sites_with_incumbent_tc2_fallback() {
    let source = compact(&read(HOST));
    let candidate = "ifuse_v4_prefill_tc2_warp0{self.v4_prefill_attn_compressed_tc2_warp0_k}elseifv4_prefill_tc2_enabled()&&self.prefill_attn_compressed_tc2_k.0!=0{self.prefill_attn_compressed_tc2_k}else{self.prefill_attn_compressed_tc_k}";
    assert_eq!(
        source.matches(candidate).count(),
        2,
        "CSA/HCA and dense attention must share the same candidate -> TC2 -> TC fallback"
    );
    assert_eq!(
        source.matches("KernelLaunch::new(ctx.gpu,attn_k)").count(),
        1
    );
    assert_eq!(
        source
            .matches("KernelLaunch::new(ctx.gpu,tc_dense_k)")
            .count(),
        1
    );
}
