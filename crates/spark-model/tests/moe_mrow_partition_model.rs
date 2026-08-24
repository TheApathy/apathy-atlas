// SPDX-License-Identifier: AGPL-3.0-only

const KERNEL: &str = include_str!("../../../kernels/gb10/common/moe_shared_expert_fused_t.cu");
const DISPATCH: &str = include_str!("../src/layers/moe/forward_phase.rs");
const MOE: &str = include_str!("../src/layers/moe/mod.rs");
const OPS: &str = include_str!("../src/layers/ops/fp8_moe.rs");
const EXL3: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const EXL3_KERNEL: &str = include_str!("../../../kernels/gb10/common/exl3_gemv.cu");
const BUFFER_SIZES: &str = include_str!("../../spark-runtime/src/buffers/sizes.rs");
const EXL3_DISPATCH: &str = include_str!("../src/layers/moe/exl3_decode.rs");
const W4A16: &str = include_str!("../../../kernels/gb10/common/w4a16_gemv.cu");

#[test]
fn wide_verify_partitions_unique_and_duplicated_experts() {
    for symbol in [
        "moe_expert_gate_up_shared_t_e8m0_m1uv2s4",
        "moe_expert_gate_up_shared_t_e8m0_m6dv2s4",
        "moe_expert_silu_down_shared_t_e8m0_m1uv2s4",
        "moe_expert_silu_down_shared_t_e8m0_m6dv2s4",
        "moe_expert_silu_down_shared_t_e8m0_m2c2v2s4",
        "moe_expert_silu_down_shared_t_e8m0_m4c34v2s4",
        "moe_expert_silu_down_shared_t_e8m0_m6c56v2s4",
    ] {
        assert!(KERNEL.contains(symbol), "missing partition kernel {symbol}");
    }

    assert!(KERNEL.contains("GROUP_UNIQUE"));
    assert!(KERNEL.contains("GROUP_DUPLICATED"));
    assert!(KERNEL.contains("GROUP_COUNT_2"));
    assert!(KERNEL.contains("GROUP_COUNT_3_4"));
    assert!(KERNEL.contains("GROUP_COUNT_5_PLUS"));
    assert!(KERNEL.contains("moe_gate_up_partial_finalize_m_act"));
    assert!(KERNEL.contains("PRECOMPUTED_ACT"));
    assert!(DISPATCH.contains("ATLAS_MOE_MROW_PARTITION"));
    assert!(DISPATCH.contains("splitk_m_t_partition_handles"));
    assert!(MOE.contains("if !(3..=MOE_VERIFY_MAX_ROWS).contains(&num_tokens)"));
}

#[test]
fn deepseek_exl3_has_full_dflash2_verify_width() {
    assert!(EXL3.contains("EXL3_MROW_MAX_ROWS: u32 = 16"));
    for symbol in [
        "exl3_gemv_mrow_fused_gate_up_m16",
        "exl3_gemv_mrow_fused_down_m16",
    ] {
        assert!(
            EXL3_KERNEL.contains(symbol),
            "missing EXL3 MROW=16 kernel {symbol}"
        );
    }
    assert!(BUFFER_SIZES.contains("if !config.dflash_capture_layers.is_empty()"));
    assert!(!BUFFER_SIZES.contains("batch_moe && !config.dflash_capture_layers.is_empty()"));
}

#[test]
fn exl3_shared_expert_batches_exact_gemv_for_k2_and_dflash2() {
    assert!(W4A16.contains("W4A16_GB_V2_ENTRY(16,"));
    assert!(EXL3_DISPATCH.contains("num_tokens <= 8"));
    assert!(EXL3_DISPATCH.contains("num_tokens == 16"));
    assert!(EXL3_DISPATCH.contains("w4a16_gemv_grouped_batchm_v2"));
    assert!(EXL3_DISPATCH.contains("num_tokens * shared_inter"));
}

#[test]
fn both_partition_arms_launch_before_partial_finalize() {
    let gate_up = OPS
        .split_once("pub fn moe_expert_gate_up_shared_t_splitk_m")
        .unwrap()
        .1
        .split_once("pub fn moe_expert_silu_down_shared_t_splitk_m")
        .unwrap()
        .0;
    let down = OPS
        .split_once("pub fn moe_expert_silu_down_shared_t_splitk_m")
        .unwrap()
        .1
        .split_once("/// Split-K fused gate+up GEMV")
        .unwrap()
        .0;

    assert!(down.contains("for (bucket, bucket_mrow) in buckets"));

    for body in [gate_up, down] {
        let primary = body.find("launch_partition_arm").unwrap();
        let secondary = body[primary + 1..].find("launch_partition_arm").unwrap() + primary + 1;
        let finalize = body.find("KernelLaunch::new(gpu, finalize)").unwrap();
        assert!(primary < secondary && secondary < finalize);
    }
}
