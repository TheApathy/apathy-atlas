// SPDX-License-Identifier: AGPL-3.0-only

use spark_runtime::gpu::{DevicePtr, KernelHandle};

use crate::weight_map::QuantizedWeight;

use super::{
    ExactAttentionKernelArg, ExactAttentionM17AStageRoute, ExactAttentionQkvRoute,
    W4a16ExactAttentionKernels, W4a16ExactAttentionM17AStageKernels, dual_kv_launch_plan,
    exact_attention_m17_astage_route, exact_attention_qkv_route, exact_attention_rows_supported,
    qg_launch_plan,
};

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

fn kernels_with_m32(
    qg_m17: u64,
    dual_kv_m17: u64,
    qg_m32: u64,
    dual_kv_m32: u64,
) -> W4a16ExactAttentionKernels {
    kernels(qg_m17, dual_kv_m17).with_m32(KernelHandle(qg_m32), KernelHandle(dual_kv_m32))
}

fn astage_kernels(qg: u64, dual_kv: u64) -> W4a16ExactAttentionM17AStageKernels {
    W4a16ExactAttentionM17AStageKernels::new(KernelHandle(qg), KernelHandle(dual_kv))
}

fn mock_weight(weight: u64, scale: u64, scale2: f32) -> QuantizedWeight {
    QuantizedWeight {
        weight: DevicePtr(weight),
        weight_scale: DevicePtr(scale),
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
    }
}

fn m32_pair_loads_qg_then_dual(source: &str) -> bool {
    let qg = "w4a16_gemv_qg_exact_m32";
    let dual_kv = "w4a16_gemv_dual_kv_exact_m32";
    source.matches(qg).count() == 1
        && source.matches(dual_kv).count() == 1
        && source.find(qg) < source.find(dual_kv)
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
fn m32_route_requires_both_selected_handles_for_every_owned_row() {
    for rows in 18..=32 {
        let complete = kernels_with_m32(0, 0, 3, 4);
        assert_eq!(complete.qg_for_rows(rows).0, 3);
        assert_eq!(complete.dual_kv_for_rows(rows).0, 4);
        assert_eq!(
            exact_attention_qkv_route(rows, true, true, complete),
            Some(ExactAttentionQkvRoute::ExactM32),
            "rows={rows}"
        );
        for incomplete in [
            kernels_with_m32(1, 2, 0, 4),
            kernels_with_m32(1, 2, 3, 0),
            kernels_with_m32(1, 2, 0, 0),
        ] {
            assert_eq!(
                exact_attention_qkv_route(rows, true, true, incomplete),
                Some(ExactAttentionQkvRoute::SerialK1M32),
                "rows={rows}: incomplete M32 pairs must fail closed as one phase"
            );
        }
    }
}

#[test]
fn wrapper_row_admission_matches_all_exact_tier_boundaries() {
    for rows in 0..=36 {
        assert_eq!(
            exact_attention_rows_supported(rows),
            rows == 4 || (5..=32).contains(&rows),
            "rows={rows}"
        );
    }
    assert!(!exact_attention_rows_supported(u32::MAX));
}

#[test]
fn qg_m32_launch_plan_pins_grid_block_and_all_eleven_abi_arguments() {
    let weight = mock_weight(0x2000, 0x3000, 1.25);
    for (rows, out_stride) in [(18, 12_288), (32, 14_336)] {
        let plan = qg_launch_plan(
            DevicePtr(0x1000),
            &weight,
            DevicePtr(0x4000),
            rows,
            12_288,
            5_120,
            24,
            256,
            out_stride,
        )
        .expect("valid M32 QG launch plan");
        assert_eq!(plan.grid, [3_072, 1, 1]);
        assert_eq!(plan.block, [256, 1, 1]);
        assert_eq!(
            plan.args,
            [
                ExactAttentionKernelArg::Buffer(DevicePtr(0x1000)),
                ExactAttentionKernelArg::Buffer(DevicePtr(0x2000)),
                ExactAttentionKernelArg::Buffer(DevicePtr(0x3000)),
                ExactAttentionKernelArg::F32Bits(1.25f32.to_bits()),
                ExactAttentionKernelArg::Buffer(DevicePtr(0x4000)),
                ExactAttentionKernelArg::U32(rows),
                ExactAttentionKernelArg::U32(12_288),
                ExactAttentionKernelArg::U32(5_120),
                ExactAttentionKernelArg::U32(24),
                ExactAttentionKernelArg::U32(256),
                ExactAttentionKernelArg::U32(out_stride),
            ]
        );
    }
}

#[test]
fn dual_kv_m32_launch_plan_pins_grid_block_and_all_thirteen_abi_arguments() {
    let k_weight = mock_weight(0x2000, 0x3000, 1.25);
    let v_weight = mock_weight(0x5000, 0x6000, 2.5);
    for (rows, out_stride) in [(18, 1_024), (32, 14_336)] {
        let plan = dual_kv_launch_plan(
            DevicePtr(0x1000),
            &k_weight,
            DevicePtr(0x4000),
            &v_weight,
            DevicePtr(0x7000),
            rows,
            1_024,
            5_120,
            out_stride,
        )
        .expect("valid M32 dual-KV launch plan");
        assert_eq!(plan.grid, [256, 1, 2]);
        assert_eq!(plan.block, [256, 1, 1]);
        assert_eq!(
            plan.args,
            [
                ExactAttentionKernelArg::Buffer(DevicePtr(0x1000)),
                ExactAttentionKernelArg::Buffer(DevicePtr(0x2000)),
                ExactAttentionKernelArg::Buffer(DevicePtr(0x3000)),
                ExactAttentionKernelArg::F32Bits(1.25f32.to_bits()),
                ExactAttentionKernelArg::Buffer(DevicePtr(0x4000)),
                ExactAttentionKernelArg::Buffer(DevicePtr(0x5000)),
                ExactAttentionKernelArg::Buffer(DevicePtr(0x6000)),
                ExactAttentionKernelArg::F32Bits(2.5f32.to_bits()),
                ExactAttentionKernelArg::Buffer(DevicePtr(0x7000)),
                ExactAttentionKernelArg::U32(rows),
                ExactAttentionKernelArg::U32(1_024),
                ExactAttentionKernelArg::U32(5_120),
                ExactAttentionKernelArg::U32(out_stride),
            ]
        );
    }
}

#[test]
fn m32_handles_do_not_change_other_route_tiers() {
    let complete = kernels_with_m32(1, 2, 3, 4).with_m4(KernelHandle(5), KernelHandle(6));
    assert_eq!(
        exact_attention_qkv_route(4, true, true, complete),
        Some(ExactAttentionQkvRoute::ExactM4)
    );
    for rows in 5..=17 {
        assert_eq!(
            exact_attention_qkv_route(rows, true, true, complete),
            Some(ExactAttentionQkvRoute::ExactM17),
            "rows={rows}"
        );
    }
    for rows in [0, 1, 2, 3, 33, usize::MAX] {
        assert_eq!(exact_attention_qkv_route(rows, true, true, complete), None);
    }
    for rows in [4, 5, 17, 18, 32] {
        assert_eq!(exact_attention_qkv_route(rows, false, true, complete), None);
        assert_eq!(exact_attention_qkv_route(rows, true, false, complete), None);
    }
}

#[test]
fn m17_astage_is_an_atomic_separate_handle_pair() {
    let complete = astage_kernels(3, 4);
    assert!(complete.complete());
    assert_eq!(complete.qg().0, 3);
    assert_eq!(complete.dual_kv().0, 4);
    for incomplete in [astage_kernels(0, 4), astage_kernels(3, 0)] {
        assert!(!incomplete.complete());
    }

    // The proven baseline selector remains structurally independent.
    assert_eq!(kernels(1, 2).qg_for_rows(17).0, 1);
    assert_eq!(kernels(1, 2).dual_kv_for_rows(17).0, 2);
}

#[test]
fn m17_astage_selector_distinguishes_disabled_ineligible_complete_and_missing() {
    let complete = astage_kernels(3, 4);
    assert_eq!(
        exact_attention_m17_astage_route(false, 17, true, true, complete),
        ExactAttentionM17AStageRoute::Disabled
    );
    for (rows, gated, ordinary) in [
        (4, true, true),
        (18, true, true),
        (17, false, true),
        (17, true, false),
    ] {
        assert_eq!(
            exact_attention_m17_astage_route(true, rows, gated, ordinary, complete),
            ExactAttentionM17AStageRoute::Ineligible
        );
    }
    for rows in 5..=17 {
        assert_eq!(
            exact_attention_m17_astage_route(true, rows, true, true, complete),
            ExactAttentionM17AStageRoute::Complete
        );
        assert_eq!(
            exact_attention_m17_astage_route(true, rows, true, true, astage_kernels(0, 4)),
            ExactAttentionM17AStageRoute::Missing
        );
        assert_eq!(
            exact_attention_m17_astage_route(true, rows, true, true, astage_kernels(3, 0)),
            ExactAttentionM17AStageRoute::Missing
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
    assert_eq!(
        exact_attention_qkv_route(18, true, true, complete),
        Some(ExactAttentionQkvRoute::SerialK1M32)
    );
    assert_eq!(exact_attention_qkv_route(33, true, true, complete), None);
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
    assert!(source.contains("w4a16_gemv_qg_exact_m32"));
    assert!(source.contains("w4a16_gemv_dual_kv_exact_m32"));
    assert!(source.contains("w4a16_attention_exact_body<32, true, false>"));
    assert!(source.contains("w4a16_attention_exact_body<32, false, false>"));
    assert!(source.contains("w4a16_gemv_qg_exact_m17_astage"));
    assert!(source.contains("w4a16_gemv_dual_kv_exact_m17_astage"));
    assert!(source.contains("k8 = lane; k8 < K8; k8 += 64u"));
    assert!(source.contains("#define ASTAGE_K8_PER_WAVE 64"));
    assert!(source.contains("tile_k8 = threadIdx.x"));
    assert!(source.contains("wave * ASTAGE_K8_PER_WAVE + lane"));
    assert!(source.contains("__shared__ uint4 s_a[17 * ASTAGE_K8_PER_WAVE]"));
    assert_eq!(17 * 64 * 16, 17_408, "M17 staged activation bytes");
    assert!(source.contains("including invalid tail-output groups"));
    assert!(source.contains("acc[row] += __bfloat162float(a_lo) * w_lo[b];"));
    assert!(source.contains("acc[row] += __bfloat162float(a_hi) * w_hi[b];"));
    assert!(source.contains("__float2bfloat16(smem[base] + smem[base + 1])"));
    assert!(source.contains("out_idx = h * head_dim + idx"));
    assert!(source.contains("out_idx = q_total + h * head_dim"));
    assert!(source.contains("const unsigned int proj = blockIdx.z"));

    let route_source = include_str!("../../qwen3_attention/trait_impl/multi_seq/qkv.rs");
    assert!(route_source.contains("ATLAS_ATTN_QKV_EXACT_M17_ASTAGE"));
    assert!(route_source.contains("exact_attention_m17_astage_route("));
    assert!(route_source.contains("ExactAttentionM17AStageRoute::Missing"));
    assert!(route_source.contains("staged QG/dual-KV kernel pair is incomplete"));
    assert!(route_source.contains("log_m17_astage_engagement_once"));
    assert!(route_source.contains("ExactAttentionQkvRoute::ExactM32"));
    assert!(route_source.contains("ExactAttentionQkvRoute::SerialK1M32"));

    let init_source = include_str!("../../qwen3_attention/init.rs");
    let m32_pair = init_source
        .split(".with_m32(")
        .nth(1)
        .expect("M32 handle pair must be loaded atomically")
        .split("w4a16_exact_qkv_m17_astage_kernels:")
        .next()
        .expect("next field after M32 pair");
    let qg = "w4a16_gemv_qg_exact_m32";
    let dual_kv = "w4a16_gemv_dual_kv_exact_m32";
    assert!(m32_pair_loads_qg_then_dual(m32_pair));

    let swapped = m32_pair
        .replacen(qg, "__M32_QG__", 1)
        .replacen(dual_kv, qg, 1)
        .replacen("__M32_QG__", dual_kv, 1);
    assert!(!m32_pair_loads_qg_then_dual(&swapped));
}

fn cuda_function<'a>(source: &'a str, name: &str) -> &'a str {
    let needle = format!("extern \"C\" __global__ void {name}(");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("missing {name}"));
    let open = start
        + source[start..]
            .find('{')
            .unwrap_or_else(|| panic!("missing body for {name}"));
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated body for {name}");
}

#[test]
fn m32_cuda_siblings_change_only_the_row_tier_from_m17() {
    let source =
        include_str!("../../../../../../kernels/gb10/common/w4a16_gemv_exact_attention.cu");
    for (m17, m32) in [
        ("w4a16_gemv_qg_exact_m17", "w4a16_gemv_qg_exact_m32"),
        (
            "w4a16_gemv_dual_kv_exact_m17",
            "w4a16_gemv_dual_kv_exact_m32",
        ),
    ] {
        let normalized = cuda_function(source, m32)
            .replace(m32, m17)
            .replace("[32 * N_PER_BLOCK * 2]", "[17 * N_PER_BLOCK * 2]")
            .replace("exact_body<32,", "exact_body<17,");
        assert_eq!(normalized, cuda_function(source, m17), "{m32}");
    }
}

/// The batched `ATLAS_ATTN_QKV_BATCHED` route in `ms_phase_qkv` is guarded by
/// `else if` behind this route's `Some(..)`, so on gated ordinary-NVFP4
/// attention it is unreachable for every width the exact tiers claim. On
/// Qwen3.8-27B (`attn_output_gate = true`, NVFP4 q/k/v) that covers the whole
/// DFlash verify range through M32: enabling the flag there changes nothing,
/// whether or not the exact kernel symbols loaded. Only rows above M32 fall
/// outside the exact selectors.
#[test]
fn gated_nvfp4_never_falls_through_to_the_batched_route() {
    for handles in [
        kernels_with_m4(1, 2, 1, 2).with_m32(KernelHandle(3), KernelHandle(4)),
        kernels_with_m4(0, 0, 0, 0), // exact symbols absent → serial K1
    ] {
        for rows in 4..=32 {
            assert!(
                exact_attention_qkv_route(rows, true, true, handles).is_some(),
                "rows={rows}: batched QKV must stay unreachable on gated NVFP4"
            );
        }
        for rows in [2, 3, 33] {
            assert_eq!(
                exact_attention_qkv_route(rows, true, true, handles),
                None,
                "rows={rows}: outside the exact tiers the existing routes own dispatch"
            );
        }
    }
}

#[test]
fn exact_selector_precedes_all_lossy_or_batched_projection_effects() {
    let route_source = include_str!("../../qwen3_attention/trait_impl/multi_seq/qkv.rs");
    let selector = route_source
        .find("let exact_route = ops::exact_attention_qkv_route(")
        .expect("exact selector");
    let exact_dispatch = route_source[selector..]
        .find("ExactAttentionQkvRoute::ExactM32")
        .map(|offset| selector + offset)
        .expect("exact M32 dispatch");
    let serial_dispatch = route_source[selector..]
        .find("ExactAttentionQkvRoute::SerialK1M32")
        .map(|offset| selector + offset)
        .expect("serial M32 dispatch");
    for later_effect in [
        "self.ms_qkv_batch3(c)?",
        "self.ms_qkv_batch2(c)?",
        "self.ms_qkv_batched_m16(c)?",
        "self.ms_qkv_batched_plain(c)?",
    ] {
        let later = route_source[selector..]
            .find(later_effect)
            .map(|offset| selector + offset)
            .unwrap_or_else(|| panic!("missing later route {later_effect}"));
        assert!(exact_dispatch < later, "{later_effect}");
        assert!(serial_dispatch < later, "{later_effect}");
    }
}
