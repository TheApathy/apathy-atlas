// SPDX-License-Identifier: AGPL-3.0-only

const SOURCE: &str = include_str!("../src/layers/moe/qwen4_prefill_admission.rs");
const REGISTRY: &str = include_str!("../src/layers/moe/mod.rs");

#[test]
fn actual_native_fp8_and_nonoriginal_routes_are_rejected() {
    assert!(REGISTRY.contains("mod qwen4_prefill_admission;"));
    for field in [
        "fp8_gate_weight_ptrs",
        "fp8_up_weight_ptrs",
        "fp8_down_weight_ptrs",
        "fp8_shared_expert",
        "gate_ptrs_t",
        "up_ptrs_t",
        "down_ptrs_t",
        "shared_gate_t",
        "shared_up_t",
        "shared_down_t",
        "down_t_scratch_packed",
        "down_t_scratch_scale",
        "weights.router_pre_norm",
        "pre_expert_norm",
        "correction_bias_dev",
        "weights.correction_bias",
    ] {
        assert!(
            SOURCE.contains(&format!("self.{field}.is_none()")),
            "unguarded: {field}"
        );
    }
    assert!(SOURCE.contains("!self.nvfp4_moe_worklist"));
    assert!(SOURCE.contains("!self.gelu_activation"));
}

#[test]
fn original_weights_tables_and_selected_kernels_are_checked() {
    for field in ["gate_ptrs", "up_ptrs", "down_ptrs"] {
        assert!(SOURCE.contains(&format!("table_valid(&self.{field})")));
    }
    for field in [
        "moe_topk_batched",
        "moe_sort_by_expert",
        "moe_grouped_gemm",
        "moe_act_mul",
        "moe_unpermute_reduce",
        "moe_batched_blend",
    ] {
        assert!(SOURCE.contains(&format!("self.{field}.0")));
    }
    assert!(SOURCE.contains("self.weights.experts.len() == 512"));
    assert!(SOURCE.contains("self.weights.experts.iter().all(expert_valid)"));
    assert!(SOURCE.contains("expert_valid(&self.weights.shared_expert)"));
    assert!(SOURCE.contains("self.weights.shared_expert_gate.weight != DevicePtr::NULL"));
}

#[test]
fn router_and_shared_predequant_are_one_explicit_bundle() {
    for field in [
        "gate_fp8",
        "shared_gate_fp8",
        "shared_up_fp8",
        "shared_down_fp8",
    ] {
        assert!(SOURCE.contains(&format!("self.{field}.map(|p| p.0)")));
    }
    assert!(SOURCE.contains("self.fp8_gemm_k.0"));
    assert!(SOURCE.contains("self.w4a16_gemm.0"));
    assert!(SOURCE.contains("projection_bundle_valid("));
    assert!(!SOURCE.contains("gpu.alloc("));
    assert!(!SOURCE.contains("copy_d2h("));
    assert!(SOURCE.lines().count() <= 250);
}
