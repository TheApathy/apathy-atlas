// SPDX-License-Identifier: AGPL-3.0-only

const CUDA: &str = include_str!("../../../kernels/gb10/common/w4a16_gemv.cu");
const OPS: &str = include_str!("../src/layers/ops/gemv_exact_lm_head.rs");
const INIT: &str = include_str!("../src/layers/qwen3_attention/init.rs");
const ROUTE: &str = include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/attn.rs");
const RAW: &str = include_str!("../examples/w4a16_oproj_m17_astage_microgate.rs");

fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split(start)
        .nth(1)
        .unwrap_or_else(|| panic!("missing start marker {start}"))
        .split(end)
        .next()
        .unwrap_or_else(|| panic!("missing end marker {end}"))
}

#[test]
fn staged_symbol_is_abi_separate_and_resource_bounded() {
    let entry = between(
        CUDA,
        "extern \"C\" __global__ __launch_bounds__(256, 2)\nvoid w4a16_gemv_batch_logits_exact_m17_astage(",
        "// ============================================================\n// W4A16 GEMV — SINGLE-WARP-PER-OUTPUT",
    );
    for argument in [
        "const __nv_bfloat16* __restrict__ A",
        "const unsigned char* __restrict__ B_packed",
        "const unsigned char* __restrict__ B_scale",
        "const float scale2",
        "__nv_bfloat16* __restrict__ C",
        "unsigned int M",
        "unsigned int N",
        "unsigned int K",
    ] {
        assert!(entry.contains(argument), "missing ABI argument {argument}");
    }
    assert!(entry.contains("__shared__ float smem[17 * N_PER_BLOCK * 2];"));
    assert!(entry.contains("__shared__ __align__(16) uint4 s_a[17 * ASTAGE_U4_PER_ROW];"));
    assert_eq!(
        CUDA.matches("w4a16_gemv_batch_logits_exact_m17_astage(")
            .count(),
        1
    );
}

#[test]
fn staged_body_keeps_parent_lane_and_arithmetic_order() {
    let body = between(
        CUDA,
        "w4a16_gemv_batch_logits_exact_m17_astage_body(",
        "extern \"C\" __global__ __launch_bounds__(256, 2)",
    );
    for exact in [
        "const unsigned int k16 = wave + lane;",
        "const unsigned long long packed8 =",
        "for (int b = 0; b < 8; ++b)",
        "w_lo[b] = s_lut[byte_val & 0xF] * scale;",
        "w_hi[b] = s_lut[byte_val >> 4] * scale;",
        "acc[row] += __bfloat162float(a_lo_bf) * w_lo[b];",
        "acc[row] += __bfloat162float(a_hi_bf) * w_hi[b];",
        "acc[row] += __shfl_down_sync(0xFFFFFFFF, acc[row], offset);",
        "__float2bfloat16(smem[base] + smem[base + 1])",
    ] {
        assert!(
            body.contains(exact),
            "missing exact arithmetic fragment {exact}"
        );
    }
    assert_eq!(body.matches("__syncthreads();").count(), 4);
    assert!(body.contains("local_k16 < wave_k16"));
    assert!(body.contains("if (valid && lane < wave_k16)"));
}

#[test]
fn rust_launcher_and_init_use_the_same_ordered_abi() {
    let launch = between(
        OPS,
        "pub fn w4a16_gemv_batch_logits_exact_m17_astage(",
        "        .launch(stream)\n}",
    );
    let ordered = [
        ".arg_ptr(input)",
        ".arg_ptr(weight.weight)",
        ".arg_ptr(weight.weight_scale)",
        ".arg_f32(weight.weight_scale_2)",
        ".arg_ptr(output)",
        ".arg_u32(rows)",
        ".arg_u32(n)",
        ".arg_u32(k)",
    ];
    let mut prior = 0;
    for argument in ordered {
        let position = launch
            .find(argument)
            .unwrap_or_else(|| panic!("missing {argument}"));
        assert!(position >= prior, "ABI order drift at {argument}");
        prior = position;
    }
    assert!(launch.contains("(9..=17).contains(&rows)"));
    assert!(INIT.contains(".with_m17_astage(super::super::try_kernel("));
    assert!(INIT.contains("\"w4a16_gemv_batch_logits_exact_m17_astage\""));
}

#[test]
fn production_selector_is_default_off_strict_atomic_and_post_success() {
    for required in [
        "AttnOProjM17AStageRoute::Disabled",
        "AttnOProjM17AStageRoute::Ineligible",
        "AttnOProjM17AStageRoute::Complete",
        "AttnOProjM17AStageRoute::Missing",
        "AttnOProjM17AStageRoute::Conflict",
        "None | Some(\"0\") => Ok(false)",
        "Some(\"1\") => Ok(true)",
        "ATLAS_ATTN_O_PROJ_EXACT_M17_ASTAGE must be exactly 0 or 1",
    ] {
        assert!(
            ROUTE.contains(required),
            "missing route invariant {required}"
        );
    }
    let preflight = ROUTE.find("let o_proj_m17_astage_route =").unwrap();
    let gate = ROUTE.find("if self.gated {").unwrap();
    let launch = ROUTE
        .find("ops::w4a16_gemv_batch_logits_exact_m17_astage(")
        .unwrap();
    let receipt = ROUTE
        .find("ENGAGED ATLAS_ATTN_O_PROJ_EXACT_M17_ASTAGE")
        .unwrap();
    assert!(preflight < gate);
    assert!(launch < receipt);
    assert!(ROUTE.contains("ops::w4a16_gemv_rt2_enabled()"));
    assert!(ROUTE.contains("serial_o_proj"));
}

#[test]
fn raw_gate_covers_production_tails_pathologies_and_timing_stop() {
    for required in [
        "for rows in 9..=17",
        "n: PROD_N",
        "k: PROD_K",
        "for n in [1, 2, 3, 5_119]",
        "k: 528",
        "InputKind::Cancellation",
        "InputKind::Extreme",
        "0x0000, 0x8000, 0x0001, 0x8001",
        "0x7f80",
        "0xff80",
        "0x7fc1",
        "0xffc1",
        "parent-vs-serial",
        "staged-vs-serial",
        "determinism",
        "leading 4-KiB redzone changed",
        "inactive M17 row or trailing stride guard changed",
        "immutable image changed",
        "ATLAS_M17_OPROJ_ASTAGE_TIMING",
        "TIMING_ROUNDS: usize = 21",
        "modeled_frame_saving >= 0.5",
        "s_med < p_med && s_p90 < p_p90",
        "conservative > 0.0",
    ] {
        assert!(
            RAW.contains(required),
            "missing raw-gate invariant {required}"
        );
    }
}
