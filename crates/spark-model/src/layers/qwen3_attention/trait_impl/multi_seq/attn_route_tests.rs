// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    AttnGateBatchRoute, AttnOProjM17AStageRoute, ExactAttnOProjDispatch, attn_gate_batch_route,
    attn_o_proj_m17_astage_route, exact_attn_o_proj_dispatch, parse_attn_gate_batched,
    parse_attn_o_proj_m17_astage, should_auto_serialize_paged_split_boundary,
};
use crate::layers::ops::exact_lm_head_route_for_rows;

#[test]
fn ordinary_nvfp4_uses_exact_tiers_at_every_boundary() {
    for rows in [2, 4, 5, 8, 9, 17, 18, 32] {
        assert_eq!(
            exact_attn_o_proj_dispatch(true, exact_lm_head_route_for_rows(rows, true),),
            ExactAttnOProjDispatch::Exact,
            "rows={rows}"
        );
    }
}

#[test]
fn missing_exact_tier_fails_closed_to_k1_rows() {
    for rows in 2..=32 {
        assert_eq!(
            exact_attn_o_proj_dispatch(true, exact_lm_head_route_for_rows(rows, false),),
            ExactAttnOProjDispatch::PerRowK1,
            "rows={rows}"
        );
    }
}

#[test]
fn other_encodings_and_out_of_range_rows_keep_existing_paths() {
    assert_eq!(
        exact_attn_o_proj_dispatch(false, exact_lm_head_route_for_rows(5, true)),
        ExactAttnOProjDispatch::Existing
    );
    for rows in [0, 1, 33, 256] {
        assert_eq!(
            exact_attn_o_proj_dispatch(true, exact_lm_head_route_for_rows(rows, true),),
            ExactAttnOProjDispatch::Existing,
            "rows={rows}"
        );
    }
}

#[test]
fn split_boundary_serialization_is_flat_only() {
    assert!(should_auto_serialize_paged_split_boundary(
        2, false, false, true
    ));
    assert!(!should_auto_serialize_paged_split_boundary(
        2, false, true, true
    ));
    assert!(!should_auto_serialize_paged_split_boundary(
        2, true, false, true
    ));
    assert!(!should_auto_serialize_paged_split_boundary(
        1, false, false, true
    ));
}

#[test]
fn batched_gate_selector_is_strict_and_fail_closed() {
    assert_eq!(
        attn_gate_batch_route(false, true, 17, 6144, 14336, false),
        AttnGateBatchRoute::Disabled
    );
    for rows in [0, 4, 18, 33] {
        assert_eq!(
            attn_gate_batch_route(true, true, rows, 6144, 14336, false),
            AttnGateBatchRoute::Ineligible,
            "rows={rows}"
        );
    }
    assert_eq!(
        attn_gate_batch_route(true, false, 17, 6144, 14336, false),
        AttnGateBatchRoute::Ineligible
    );
    assert_eq!(
        attn_gate_batch_route(true, true, 17, 0, 14336, false),
        AttnGateBatchRoute::Ineligible
    );
    assert_eq!(
        attn_gate_batch_route(true, true, 17, 6144, 6143, false),
        AttnGateBatchRoute::Ineligible
    );
    for rows in 5..=17 {
        assert_eq!(
            attn_gate_batch_route(true, true, rows, 6144, 14336, true),
            AttnGateBatchRoute::Complete,
            "rows={rows}"
        );
        assert_eq!(
            attn_gate_batch_route(true, true, rows, 6144, 14336, false),
            AttnGateBatchRoute::Missing,
            "rows={rows}"
        );
    }
}

#[test]
fn batched_gate_flag_accepts_only_absent_zero_or_one() {
    assert_eq!(parse_attn_gate_batched(None), Ok(false));
    assert_eq!(parse_attn_gate_batched(Some("0")), Ok(false));
    assert_eq!(parse_attn_gate_batched(Some("1")), Ok(true));
    for invalid in ["", "true", "false", "2", "01", " 1"] {
        assert!(
            parse_attn_gate_batched(Some(invalid)).is_err(),
            "{invalid:?}"
        );
    }
}

#[test]
fn flattened_gate_addresses_match_the_scalar_loop() {
    for dim in [1usize, 255, 256, 257, 6144] {
        for gate_stride in [dim, dim + 1, dim + 8192] {
            for rows in [5usize, 8, 9, 17] {
                for flat in 0..rows * dim {
                    let token = flat / dim;
                    let channel = flat % dim;
                    assert_eq!(flat, token * dim + channel);
                    assert_eq!(
                        token * gate_stride + channel,
                        (flat / dim) * gate_stride + flat % dim
                    );
                }
            }
        }
    }
}

#[test]
fn batched_gate_source_preserves_scalar_arithmetic_and_bf16_boundary() {
    let cuda = include_str!("../../../../../../../kernels/gb10/common/residual_add.cu");
    let batched = cuda
        .split("sigmoid_gate_mul_batched(")
        .nth(1)
        .expect("batched kernel")
        .split("// BF16 concatenation")
        .next()
        .expect("batched kernel body");
    for expression in [
        "unsigned int t = i / dim;",
        "unsigned int d = i % dim;",
        "float x = __bfloat162float(input[i]);",
        "float g = __bfloat162float(gate[t * gate_stride + d]);",
        "float sigmoid_g = 1.0f / (1.0f + expf(-g));",
        "output[i] = __float2bfloat16(x * sigmoid_g);",
    ] {
        assert!(batched.contains(expression), "missing {expression}");
    }
    let host = include_str!("attn.rs");
    assert!(host.contains("ENGAGED ATLAS_ATTN_GATE_BATCHED: multi_seq"));
    assert!(host.contains("AttnGateBatchRoute::Missing"));
    assert!(host.contains("ops::sigmoid_gate_mul_batched("));
}

#[test]
fn o_proj_m17_astage_selector_is_atomic_default_off_and_fail_closed() {
    assert_eq!(
        attn_o_proj_m17_astage_route(false, true, true, 17, false, true, true),
        AttnOProjM17AStageRoute::Disabled
    );
    for rows in [0, 1, 8, 18, 32] {
        assert_eq!(
            attn_o_proj_m17_astage_route(true, true, true, rows, true, false, false),
            AttnOProjM17AStageRoute::Ineligible,
            "rows={rows}"
        );
    }
    assert_eq!(
        attn_o_proj_m17_astage_route(true, false, true, 17, true, false, false),
        AttnOProjM17AStageRoute::Ineligible
    );
    assert_eq!(
        attn_o_proj_m17_astage_route(true, true, false, 17, true, false, false),
        AttnOProjM17AStageRoute::Ineligible
    );
    for rows in 9..=17 {
        assert_eq!(
            attn_o_proj_m17_astage_route(true, true, true, rows, true, false, false),
            AttnOProjM17AStageRoute::Complete,
            "rows={rows}"
        );
        assert_eq!(
            attn_o_proj_m17_astage_route(true, true, true, rows, false, false, false),
            AttnOProjM17AStageRoute::Missing,
            "rows={rows}"
        );
        assert_eq!(
            attn_o_proj_m17_astage_route(true, true, true, rows, true, true, false),
            AttnOProjM17AStageRoute::Conflict,
            "rt2 rows={rows}"
        );
        assert_eq!(
            attn_o_proj_m17_astage_route(true, true, true, rows, true, false, true),
            AttnOProjM17AStageRoute::Conflict,
            "serial rows={rows}"
        );
    }
}

#[test]
fn o_proj_m17_astage_flag_accepts_only_absent_zero_or_one() {
    assert_eq!(parse_attn_o_proj_m17_astage(None), Ok(false));
    assert_eq!(parse_attn_o_proj_m17_astage(Some("0")), Ok(false));
    assert_eq!(parse_attn_o_proj_m17_astage(Some("1")), Ok(true));
    for invalid in ["", "true", "false", "2", "01", " 1", "1 "] {
        assert!(
            parse_attn_o_proj_m17_astage(Some(invalid)).is_err(),
            "{invalid:?}"
        );
    }
}

#[test]
fn o_proj_m17_astage_source_preserves_parent_arithmetic_and_barriers() {
    let cuda = include_str!("../../../../../../../kernels/gb10/common/w4a16_gemv.cu");
    let staged = cuda
        .split("w4a16_gemv_batch_logits_exact_m17_astage_body(")
        .nth(1)
        .expect("staged body")
        .split("extern \"C\" __global__ __launch_bounds__(256, 2)")
        .next()
        .expect("staged body end");
    for invariant in [
        "const unsigned int k16 = wave + lane;",
        "const unsigned long long packed8 =",
        "for (int b = 0; b < 8; ++b)",
        "acc[row] += __bfloat162float(a_lo_bf) * w_lo[b];",
        "acc[row] += __bfloat162float(a_hi_bf) * w_hi[b];",
        "acc[row] += __shfl_down_sync(0xFFFFFFFF, acc[row], offset);",
        "__float2bfloat16(smem[base] + smem[base + 1])",
        "slot += BLOCK_SIZE",
        "local_k16 < wave_k16",
    ] {
        assert!(staged.contains(invariant), "missing {invariant}");
    }
    assert_eq!(
        staged.matches("__syncthreads();").count(),
        4,
        "LUT, publish, overwrite and final-reduction barriers must remain"
    );

    let host = include_str!("attn.rs");
    let resolve = host
        .find("let o_proj_m17_astage_route =")
        .expect("candidate preflight");
    let gate = host.find("if self.gated {").expect("gate phase");
    let launch = host
        .find("ops::w4a16_gemv_batch_logits_exact_m17_astage(")
        .expect("candidate launch");
    let receipt = host
        .find("ENGAGED ATLAS_ATTN_O_PROJ_EXACT_M17_ASTAGE")
        .expect("post-success receipt");
    assert!(
        resolve < gate,
        "candidate admission must precede gate mutation"
    );
    assert!(
        launch < receipt,
        "receipt must follow a successful launch call"
    );
}
