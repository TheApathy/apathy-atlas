// SPDX-License-Identifier: AGPL-3.0-only

use std::{fs, path::PathBuf};

fn source(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fs::read_to_string(root.join(path)).unwrap_or_default()
}

#[test]
fn private_derivative_reuses_the_original_arithmetic() {
    let wrapper = source("kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact.cu");
    assert!(wrapper.contains("#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1"));
    let original = ["moe_w4a16_orig_i640", "compact_prefill"].join("_");
    assert!(wrapper.contains(&format!("#include \"{original}.cu\"")));
    assert!(wrapper.contains("qwen4_moe_compact_plan"));
    assert!(wrapper.contains("qwen4_moe_compact_gemm"));
    assert!(!wrapper.contains("mma.sync"));
    let header = source(&format!(
        "kernels/gb10/qwen3.8-flash-next/nvfp4/{original}.cuh"
    ));
    assert!(header.contains("#ifdef ATLAS_QWEN4_MOE_COMPACT_CONTRACT"));
    assert!(header.contains("return rows >= 2 && rows <= 2048;"));
    assert!(header.contains("return rows == M_SMALL || rows == M_LARGE;"));
}

#[test]
fn k32_derivative_reuses_ordered_k16_mma_arithmetic() {
    let wrapper = source("kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_k32.cu");
    assert!(wrapper.contains("#define OI640_STEP_K 32"));
    assert!(wrapper.contains("qwen4_moe_compact_gemm_k32"));
    assert!(wrapper.contains("moe_w4a16_orig_i640_compact_prefill.cu"));
    assert!(!wrapper.contains("mma.sync"));

    let gemm = source(
        "kernels/gb10/qwen3.8-flash-next/nvfp4/moe_w4a16_orig_i640_compact_prefill_gemm.cuh",
    );
    assert!(gemm.contains("for (unsigned int k_sub = 0; k_sub < oi640::STEP_K; k_sub += 16)"));
    assert!(gemm.contains("const unsigned int fc0 = k_sub + tid * 2"));
    assert!(gemm.contains("const unsigned int k0 = k_sub + tid * 2"));
    assert!(gemm.contains("S[static_cast<unsigned long long>(gn) * groups + gk / oi640::GROUP_K]"));
}

#[test]
fn transposed_k32_derivative_changes_only_global_weight_addressing() {
    let wrapper = source("kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_moe_compact_t_k32.cu");
    assert!(wrapper.contains("#define OI640_STEP_K 32"));
    assert!(wrapper.contains("#define OI640_TRANSPOSED 1"));
    assert!(wrapper.contains("qwen4_moe_compact_t_gemm_k32"));
    assert!(wrapper.contains("moe_w4a16_orig_i640_compact_prefill.cu"));
    assert!(!wrapper.contains("mma.sync"));

    let gemm = source(
        "kernels/gb10/qwen3.8-flash-next/nvfp4/moe_w4a16_orig_i640_compact_prefill_gemm.cuh",
    );
    assert!(gemm.contains("#if OI640_TRANSPOSED"));
    assert!(gemm.contains("static_cast<unsigned long long>(gk / 2) * N + gn"));
    assert!(gemm.contains("static_cast<unsigned long long>(gk / oi640::GROUP_K) * N + gn"));
    assert!(gemm.contains("for (unsigned int k_sub = 0; k_sub < oi640::STEP_K; k_sub += 16)"));
}

#[test]
fn optional_bundle_is_loaded_only_when_explicitly_selected() {
    let init = source("crates/spark-model/src/layers/moe/init.rs");
    assert!(init.contains("qwen4_prefill_compact::load(gpu, config)?"));
    let admission = source("crates/spark-model/src/layers/moe/qwen4_prefill_admission.rs");
    assert!(admission.contains("qwen4_prefill_compact::validate_handles"));
    let policy = source("crates/spark-model/src/layers/moe/qwen4_prefill_compact.rs");
    assert!(policy.contains("ATLAS_QWEN4_PREFILL_MOE_COMPACT"));
    assert!(policy.contains("ATLAS_QWEN4_PREFILL_MOE_COMPACT_K32"));
    assert!(policy.contains("ATLAS_QWEN4_PREFILL_MOE_COMPACT_T"));
    assert!(policy.contains("qwen4_moe_compact_t_gemm_k32"));
    assert!(policy.contains("ATLAS_UNIFIED_MOE_LAYOUT"));
    assert!(policy.contains("qwen4_moe_compact_gemm_k32"));
    assert!(policy.contains("ATLAS_QWEN4_PREFILL_MOE_BATCH"));
    assert!(policy.contains("pub(crate) fn validate_context("));
    assert!(policy.contains("config.tp_world_size == 1"));
}

#[test]
fn compact_dispatch_preserves_zeroing_and_returns_only_after_status() {
    let routed = source("crates/spark-model/src/layers/moe/forward_prefill_routed.rs");
    let zero = routed
        .find("memset_async(ctx.buffers.expert_down_out()")
        .unwrap();
    let compact = routed
        .find("run_qwen4_compact")
        .expect("missing compact dispatch");
    let legacy = routed.find("if max_m_tiles > 0").unwrap();
    assert!(zero < compact && compact < legacy);
    let ops = source("crates/spark-model/src/layers/moe/qwen4_compact_ops.rs");
    let down = ops
        .find("&self.down_ptrs")
        .expect("missing down projection");
    let status = ops.find("copy_d2h_on_stream").expect("missing status read");
    let checked = ops.find("check_status").expect("missing status guard");
    let receipt = ops
        .find("MOE_PREFILL_COMPACT_K32_ENGAGED")
        .expect("missing K32 success receipt");
    assert!(down < status && status < checked);
    assert!(checked < receipt);
    assert!(ops.contains("ops::silu_mul("));
    assert!(ops.contains("compact_tables"));
    assert!(ops.contains("self.gate_ptrs_t"));
    assert!(ops.contains("self.down_ptrs_t"));
    assert!(!ops.contains(".unwrap_or"));
    assert!(!ops.contains(".alloc("));
}

#[test]
fn preflight_checks_capacity_before_any_launch() {
    let policy = source("crates/spark-model/src/layers/moe/qwen4_prefill_compact.rs");
    for field in [
        "sizes.moe_worklist",
        "sizes.moe_worklist_total",
        "sizes.expert_gate_out",
        "sizes.expert_up_out",
        "sizes.expert_down_out",
    ] {
        assert!(policy.contains(field), "missing {field}");
    }
    let ops = source("crates/spark-model/src/layers/moe/qwen4_compact_ops.rs");
    assert!(ops.find("validate_context").unwrap() < ops.find("KernelLaunch::new").unwrap());
}
