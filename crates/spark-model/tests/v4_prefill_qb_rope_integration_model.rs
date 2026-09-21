// SPDX-License-Identifier: AGPL-3.0-only

//! Production source contracts for the exact-shape V4 Q-B norm + forward-RoPE arm.

use std::fs;
use std::path::PathBuf;

const KERNEL: &str = "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu";
const PREFILL: &str = "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs";

const TOKENS: u64 = 2_410;
const LAYERS: u64 = 43;
const NQ: u64 = 64;
const HEAD_DIM: u64 = 512;
const ROPE_DIM: u64 = 64;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    fs::read_to_string(root().join(path))
        .unwrap_or_else(|error| panic!("required Q-B forward-RoPE source {path}: {error}"))
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
fn strict_default_off_gate_is_independent_of_cache_scale_and_calibration() {
    let source = read(PREFILL);
    let gate = between(
        &source,
        "// BEGIN V4 Q-B forward RoPE eligibility",
        "// END V4 Q-B forward RoPE eligibility",
    );
    let flat = compact(gate);

    assert!(
        compact(&source)
            .contains("std::env::var(\"ATLAS_V4_PREFILL_QB_ROPE_FUSED\").as_deref()==Ok(\"1\")")
    );
    assert!(
        flat.contains("letv4_prefill_qb_rope_fused_requested=v4_prefill_qb_rope_fused_enabled();")
    );
    for guard in [
        "ctx.config.model_type==\"deepseek_v4\"",
        "self.v4_prefill_qb_norm_rope_fused_k.0!=0",
        "self.mla_q_rope_extract_batched_k.0!=0",
        "n==2410",
        "nq==64",
        "nkv==1",
        "hd_mla==512",
        "nope==448",
        "rope==64",
        "!ctx.graph_capture",
        "!diag_this",
        "!ctx.profile",
        "eps.is_finite()",
        "eps>0.0",
        "rope_mscale.is_finite()",
        "qb_rope_pointers_nonzero",
        "qb_rope_pointers_aligned",
        "qb_rope_buffers_disjoint",
        "packed_qk_layout",
    ] {
        assert!(flat.contains(guard), "missing Q-B guard: {guard}");
    }
    for forbidden in [
        "cache_k_scale",
        "cache_v_scale",
        "self.fp8_calibration",
        "KvCacheDtype",
        "cache_pool",
        "cache_stride",
    ] {
        assert!(
            !gate.contains(forbidden),
            "Q-B-only eligibility must not depend on {forbidden}"
        );
    }
}

#[test]
fn decision_precedes_mutation_and_fallback_retains_the_incumbent_chain() {
    let source = read(PREFILL);
    let decision = source
        .find("// BEGIN V4 Q-B forward RoPE eligibility")
        .expect("Q-B decision");
    let q_norm = source
        .find("// BEGIN V4 Q-B norm incumbent fallback")
        .expect("Q-B norm fallback");
    let dispatch = source
        .find("// BEGIN V4 Q-B forward RoPE dispatch")
        .expect("Q-B dispatch");
    assert!(decision < q_norm && q_norm < dispatch);

    let fallback = between(
        &source,
        "// BEGIN V4 forward RoPE incumbent fallback",
        "// END V4 forward RoPE incumbent fallback",
    );
    for incumbent in [
        "ops::mla_q_rope_extract_batched(",
        "ops::rope_yarn(",
        "ops::mla_q_rope_writeback_batched(",
    ] {
        assert!(fallback.contains(incumbent), "missing fallback {incumbent}");
    }
}

#[test]
fn production_dispatch_reuses_the_verified_qb_kernel_without_cache_writes() {
    let kernel = read(KERNEL);
    assert!(kernel.starts_with("// SPDX-License-Identifier: AGPL-3.0-only\n"));
    assert!(kernel.contains("void v4_prefill_qb_norm_rope_fused("));

    let source = read(PREFILL);
    let dispatch = between(
        &source,
        "// BEGIN V4 Q-B forward RoPE dispatch",
        "// END V4 Q-B forward RoPE dispatch",
    );
    let flat = compact(dispatch);
    assert!(flat.contains(
        "KernelLaunch::new(ctx.gpu,self.v4_prefill_qb_norm_rope_fused_k).grid([n,fused_nq,1]).block([512,1,1]).arg_ptr(q_full).arg_ptr(fused_k_full).arg_ptr(norm_unit_w).arg_ptr(meta.positions).arg_ptr(rope_inv_freq).arg_u32(n).arg_u32(fused_nq).arg_u32(nkv).arg_u32(hd_mla).arg_u32(nope).arg_u32(rope).arg_f32(eps).arg_f32(rope_mscale).launch(stream)?"
    ));
    for forbidden in ["cache_k_pool", "cache_v_pool", "meta.slot", "cache_k_scale"] {
        assert!(
            !dispatch.contains(forbidden),
            "Q-B dispatch unexpectedly writes cache via {forbidden}"
        );
    }
    assert!(flat.contains("ifuse_v4_qb_rope_cache_fused"));
    assert!(flat.contains("letk_rope_tmp=q_latent"));
    assert!(flat.contains("ops::mla_q_rope_extract_batched("));
    assert!(flat.contains("Some(k_rope_tmp)"));
}

#[test]
fn exact_non_overlapping_budget_is_not_a_runtime_claim() {
    // Retaining the Q RMS input and rotating its rounded tail before the final
    // store removes the RMS apply reread plus Q-tail round trip. The incumbent
    // cache ABI still needs one K-tail extraction, so no separate K saving is
    // counted here.
    let qb_incremental_bytes = TOKENS * NQ * HEAD_DIM * 2 + TOKENS * NQ * ROPE_DIM * 4;

    assert_eq!(qb_incremental_bytes, 197_427_200);
    assert_eq!(qb_incremental_bytes * LAYERS, 8_489_369_600);

    // Incumbent: q_b norm, Q extract, K extract, shared RoPE, Q writeback,
    // K writeback. Candidate: fused q_b/Q/K plus K-tail re-extraction.
    assert_eq!((6 - 2) * LAYERS, 172);
}
