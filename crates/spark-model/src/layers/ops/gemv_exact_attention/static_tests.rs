// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::KernelHandle;

use super::{ExactAttentionQkvRoute, W4a16ExactAttentionKernels, exact_attention_qkv_route};

fn kernels(qg: u64, dual_kv: u64) -> W4a16ExactAttentionKernels {
    W4a16ExactAttentionKernels::new(KernelHandle(qg), KernelHandle(dual_kv))
}

fn kernels_with_m4(
    qg_m4: u64,
    dual_kv_m4: u64,
    qg_m17: u64,
    dual_kv_m17: u64,
) -> W4a16ExactAttentionKernels {
    kernels(qg_m17, dual_kv_m17).with_m4(KernelHandle(qg_m4), KernelHandle(dual_kv_m4))
}

#[test]
fn m4_route_requires_both_selected_handles() {
    assert_eq!(
        exact_attention_qkv_route(4, true, true, kernels_with_m4(1, 2, 0, 0)),
        Some(ExactAttentionQkvRoute::ExactM4)
    );
    assert_eq!(
        exact_attention_qkv_route(4, true, true, kernels_with_m4(0, 2, 1, 2)),
        Some(ExactAttentionQkvRoute::SerialK1M4)
    );
    assert_eq!(
        exact_attention_qkv_route(4, true, true, kernels_with_m4(1, 0, 1, 2)),
        Some(ExactAttentionQkvRoute::SerialK1M4)
    );
    for rows in [2, 3] {
        assert_eq!(
            exact_attention_qkv_route(rows, true, true, kernels_with_m4(1, 2, 1, 2)),
            None,
            "existing K{rows} route must remain unchanged"
        );
    }
}

#[test]
fn m17_route_requires_both_selected_handles() {
    for rows in [9, 16, 17] {
        assert_eq!(
            exact_attention_qkv_route(rows, true, true, kernels(1, 2)),
            Some(ExactAttentionQkvRoute::ExactM17)
        );
        assert_eq!(
            exact_attention_qkv_route(rows, true, true, kernels(0, 2)),
            Some(ExactAttentionQkvRoute::SerialK1M17)
        );
        assert_eq!(
            exact_attention_qkv_route(rows, true, true, kernels(1, 0)),
            Some(ExactAttentionQkvRoute::SerialK1M17)
        );
    }
}

#[test]
fn route_is_narrow_to_gated_ordinary_nvfp4() {
    let complete = kernels_with_m4(1, 2, 1, 2);
    // rows 5..=17 are served by the M17 kernel (bit-exact at M=5..=8); only
    // out-of-range rows fall through to the serial fallback.
    assert_eq!(
        exact_attention_qkv_route(8, true, true, complete),
        Some(ExactAttentionQkvRoute::ExactM17)
    );
    assert_eq!(exact_attention_qkv_route(18, true, true, complete), None);
    for rows in [4, 16] {
        assert_eq!(exact_attention_qkv_route(rows, false, true, complete), None);
        assert_eq!(exact_attention_qkv_route(rows, true, false, complete), None);
    }
}

#[test]
fn cuda_source_pins_k1_order_and_output_layouts() {
    let source =
        include_str!("../../../../../../kernels/gb10/common/w4a16_gemv_exact_attention.cu");
    assert!(source.contains("w4a16_gemv_qg_exact_m4"));
    assert!(source.contains("w4a16_gemv_dual_kv_exact_m4"));
    assert!(source.contains("w4a16_gemv_qg_exact_m17"));
    assert!(source.contains("w4a16_gemv_dual_kv_exact_m17"));
    assert!(source.contains("k8 = lane; k8 < K8; k8 += 64u"));
    assert!(source.contains("acc[row] += __bfloat162float(a_lo) * w_lo[b];"));
    assert!(source.contains("acc[row] += __bfloat162float(a_hi) * w_hi[b];"));
    assert!(source.contains("__float2bfloat16(smem[base] + smem[base + 1])"));
    assert!(source.contains("out_idx = h * head_dim + idx"));
    assert!(source.contains("out_idx = q_total + h * head_dim"));
    assert!(source.contains("const unsigned int proj = blockIdx.z"));
}

/// The batched `ATLAS_ATTN_QKV_BATCHED` route in `ms_phase_qkv` is guarded by
/// `else if` behind this route's `Some(..)`, so on gated ordinary-NVFP4
/// attention it is unreachable for every width the exact tiers claim. On
/// Qwen3.8-27B (`attn_output_gate = true`, NVFP4 q/k/v) that covers the whole
/// DFlash verify range: enabling the flag there changes nothing, whether or
/// not the exact kernel symbols loaded. Only n∈18..=32 can reach it.
#[test]
fn gated_nvfp4_never_falls_through_to_the_batched_route() {
    for handles in [
        kernels_with_m4(1, 2, 1, 2), // exact symbols present
        kernels_with_m4(0, 0, 0, 0), // exact symbols absent → serial K1
    ] {
        for rows in 4..=17 {
            assert!(
                exact_attention_qkv_route(rows, true, true, handles).is_some(),
                "rows={rows}: batched QKV must stay unreachable on gated NVFP4"
            );
        }
        for rows in [2, 3, 18, 32] {
            assert_eq!(
                exact_attention_qkv_route(rows, true, true, handles),
                None,
                "rows={rows}: outside the exact tiers the existing routes own dispatch"
            );
        }
    }
}
