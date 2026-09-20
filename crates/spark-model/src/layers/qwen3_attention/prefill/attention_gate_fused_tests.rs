// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    Qwen38PrefillAttnGateRoute, parse_qwen38_prefill_attn_gate, qwen38_prefill_attn_gate_route,
};

fn route(
    requested: bool,
    tokens: u32,
    parent: bool,
    gate: bool,
    fused: bool,
) -> Qwen38PrefillAttnGateRoute {
    qwen38_prefill_attn_gate_route(
        requested, true, true, true, true, tokens, 24, 4, 256, 12_288, parent, gate, fused,
    )
}

#[test]
fn selector_is_exact_qwen_chunk0_only_and_fail_closed() {
    assert_eq!(
        route(false, 8192, false, false, false),
        Qwen38PrefillAttnGateRoute::Disabled
    );
    assert_eq!(
        route(true, 63, true, true, true),
        Qwen38PrefillAttnGateRoute::Ineligible
    );
    for tokens in [64, 65, 127, 128, 129, 2048, 8192] {
        assert_eq!(
            route(true, tokens, true, true, true),
            Qwen38PrefillAttnGateRoute::Complete,
            "tokens={tokens}"
        );
        for present in [
            (false, true, true),
            (true, false, true),
            (true, true, false),
        ] {
            assert_eq!(
                route(true, tokens, present.0, present.1, present.2),
                Qwen38PrefillAttnGateRoute::Missing,
                "tokens={tokens} present={present:?}"
            );
        }
    }
    for incompatible in [
        qwen38_prefill_attn_gate_route(
            true, false, true, true, true, 2048, 24, 4, 256, 12288, true, true, true,
        ),
        qwen38_prefill_attn_gate_route(
            true, true, false, true, true, 2048, 24, 4, 256, 12288, true, true, true,
        ),
        qwen38_prefill_attn_gate_route(
            true, true, true, false, true, 2048, 24, 4, 256, 12288, true, true, true,
        ),
        qwen38_prefill_attn_gate_route(
            true, true, true, true, false, 2048, 24, 4, 256, 12288, true, true, true,
        ),
        qwen38_prefill_attn_gate_route(
            true, true, true, true, true, 2048, 23, 4, 256, 12288, true, true, true,
        ),
        qwen38_prefill_attn_gate_route(
            true, true, true, true, true, 2048, 24, 5, 256, 12288, true, true, true,
        ),
        qwen38_prefill_attn_gate_route(
            true, true, true, true, true, 2048, 24, 4, 128, 12288, true, true, true,
        ),
        qwen38_prefill_attn_gate_route(
            true, true, true, true, true, 2048, 24, 4, 256, 12287, true, true, true,
        ),
    ] {
        assert_eq!(incompatible, Qwen38PrefillAttnGateRoute::Ineligible);
    }
}

#[test]
fn flag_accepts_only_absent_zero_or_one() {
    assert_eq!(parse_qwen38_prefill_attn_gate(None), Ok(false));
    assert_eq!(parse_qwen38_prefill_attn_gate(Some("0")), Ok(false));
    assert_eq!(parse_qwen38_prefill_attn_gate(Some("1")), Ok(true));
    for invalid in ["", "true", "false", "2", "01", " 1"] {
        assert!(parse_qwen38_prefill_attn_gate(Some(invalid)).is_err());
    }
}

#[test]
fn host_selects_before_parent_and_attributes_only_the_fused_launch() {
    let host = include_str!("cache_skip.rs");
    let selector = host.find("let gate_fused_route =").unwrap();
    let qkv_projection = host.find("self.prefill_attention_cache_skip_qkv(").unwrap();
    let cache_write = host.find("self.write_kv_cache(").unwrap();
    let fused = host.find("ops::prefill_attention_64_gate_fused(").unwrap();
    let marker = host
        .find("mark_qwen38_prefill_attn_gate_engaged();")
        .unwrap();
    let parent_attention = host.find("ops::prefill_attention_64(").unwrap();
    let parent_gate = host.find("ops::sigmoid_gate_mul_batched(").unwrap();
    assert!(selector < fused);
    assert!(selector < qkv_projection);
    assert!(selector < cache_write);
    assert!(selector < parent_attention);
    assert!(selector < parent_gate);
    assert!(fused < marker);
    assert!(host.contains("Qwen38PrefillAttnGateRoute::Missing"));
    assert!(host.contains("rejected before cache mutation"));
    assert!(host.contains("self.gated && !gate_fused"));
}

#[test]
fn cuda_shadow_retains_the_attention_bf16_boundary_and_parent_gate_order() {
    let parent = include_str!("../../../../../../kernels/gb10/common/inferspark_prefill.cu");
    let fused =
        include_str!("../../../../../../kernels/gb10/common/inferspark_prefill_gate_fused.cu");
    assert!(parent.contains("ATLAS_PREFILL_64_STORE_PAIR("));
    assert!(fused.contains("#include \"inferspark_prefill.cu\""));
    assert!(fused.contains("inferspark_prefill_64_gate_fused"));

    let round0 = fused.find("attn0_bf16 = __float2bfloat16").unwrap();
    let widen0 = fused.find("x0 = __bfloat162float(attn0_bf16)").unwrap();
    let gate0 = fused.find("g0 = __bfloat162float").unwrap();
    let sigmoid0 = fused
        .find("sigmoid_g0 = 1.0f / (1.0f + expf(-g0))")
        .unwrap();
    let final0 = fused.find("__float2bfloat16(x0 * sigmoid_g0)").unwrap();
    assert!(round0 < widen0 && widen0 < gate0 && gate0 < sigmoid0 && sigmoid0 < final0);
    assert!(!fused.contains("__expf(-g0)"));
    assert!(!fused.contains("x0 * g0"));
}

#[test]
fn launcher_appends_gate_and_stride_after_the_range_aware_parent_abi() {
    let ops = include_str!("../../ops/prefill_attn_main_a.rs");
    let start = ops.find("pub fn prefill_attention_64_gate_fused(").unwrap();
    let body = &ops[start..];
    let ordered = [
        ".arg_ptr(q)",
        ".arg_ptr(k)",
        ".arg_ptr(v)",
        ".arg_ptr(output)",
        ".arg_u32(seq_len)",
        ".arg_u32(0)",
        ".arg_u32(seq_len)",
        ".arg_u32(num_q_heads)",
        ".arg_u32(num_kv_heads)",
        ".arg_u32(head_dim)",
        ".arg_f32(inv_sqrt_d)",
        ".arg_u32(if causal { 1 } else { 0 })",
        ".arg_u32(sliding_window)",
        ".arg_ptr(gate)",
        ".arg_u32(gate_stride)",
    ];
    let mut cursor = 0;
    for argument in ordered {
        let relative = body[cursor..]
            .find(argument)
            .unwrap_or_else(|| panic!("missing or misordered launcher argument {argument}"));
        cursor += relative + argument.len();
    }

    let parent = include_str!("../../../../../../kernels/gb10/common/inferspark_prefill.cu");
    assert_eq!(parent.matches("const unsigned int query_start,").count(), 2);
    assert_eq!(
        parent
            .matches("const unsigned int query_len_total,")
            .count(),
        2
    );
    assert!(parent.contains("const unsigned int q_start = query_start + q_block * BR;"));
    assert!(parent.contains("const unsigned int q_start = query_start + q_block * BR64;"));
}

#[test]
fn bundle_registration_marker_and_mask_arguments_are_pinned() {
    let init = include_str!("../init.rs");
    let manifest = include_str!("../../../../../../kernels/gb10/qwen3.8-27b/nvfp4/KERNEL.toml");
    let module = include_str!("mod.rs");
    let host = include_str!("cache_skip.rs");
    assert!(init.contains("\"inferspark_prefill_gate_fused\","));
    assert!(init.contains("\"inferspark_prefill_64_gate_fused\","));
    assert!(manifest.contains("inferspark_prefill_gate_fused = \"inferspark_prefill_gate_fused\""));
    let marker = "ENGAGED ATLAS_PREFILL_ATTN_GATE_FUSED: cache-skip-br64";
    assert_eq!(module.matches(marker).count(), 1);
    assert!(module.contains("static ENGAGED: std::sync::Once"));

    let fused = host.find("ops::prefill_attention_64_gate_fused(").unwrap();
    let fused_tail = &host[fused..];
    let fused_causal = fused_tail.find("                    true,").unwrap();
    let fused_window = fused_tail
        .find("self.sliding_window.unwrap_or(0),")
        .unwrap();
    assert!(fused_causal < fused_window);

    let parent = host.find("ops::prefill_attention_64(").unwrap();
    let parent_tail = &host[parent..];
    let parent_causal = parent_tail.find("                    true,").unwrap();
    let parent_window = parent_tail
        .find("self.sliding_window.unwrap_or(0),")
        .unwrap();
    assert!(parent_causal < parent_window);
}

#[test]
fn raw_gate_pins_boundaries_canaries_replay_and_strict_timing() {
    let raw = include_str!("../../../../examples/qwen38_prefill_attn_gate_microgate.rs");
    for tokens in [
        "tokens: 63",
        "tokens: 64",
        "tokens: 65",
        "tokens: 127",
        "tokens: 128",
        "tokens: 129",
        "tokens: 2_048",
        "tokens: 8_192",
    ] {
        assert!(raw.contains(tokens), "raw gate is missing {tokens}");
    }
    for window in [
        "sliding_window: 0",
        "sliding_window: 1",
        "sliding_window: 31",
        "sliding_window: 32",
        "sliding_window: 33",
        "sliding_window: 4_096",
    ] {
        assert!(raw.contains(window), "raw gate is missing {window}");
    }
    for contract in [
        "const REDZONE: usize = 4 * 1024",
        "Fixture::Cancellation",
        "Fixture::ExtremeGate",
        "parent.reset_payload_fill(gpu, 0xe7)",
        "candidate.reset_payload_fill(gpu, 0x19)",
        "q.immutable",
        "k.immutable",
        "v.immutable",
        "gate.immutable",
        "median_saving - 3.0 * mad",
        "candidate_median < parent_median && candidate_p90 < parent_p90",
        "ATLAS_PREFILL_ATTN_GATE_MICROGATE_FULL",
        "atlas_kernels::available_targets()",
        "raw qualification requires GB10 SM121 with 48 SMs",
        "SMOKE PASS ONLY — NOT QUALIFIED",
        "FULL PARITY PASS — TIMING NOT RUN",
        "FULL PARITY+TIMING PASS",
    ] {
        assert!(raw.contains(contract), "raw gate lost contract {contract}");
    }
}
