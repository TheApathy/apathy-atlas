// SPDX-License-Identifier: AGPL-3.0-only

#[test]
fn native_shared_fp8_dispatch_precedes_all_legacy_shared_compute() {
    let prefill = include_str!("../src/layers/moe/forward_prefill_phase.rs");
    let shared = prefill
        .split("pub(super) fn run_shared_expert_prefill(")
        .nth(1)
        .unwrap();
    assert!(
        shared
            .find("run_native_fp8_shared_expert_observed(")
            .unwrap()
            < shared.find("run_bf16_shared_expert(").unwrap()
    );
    let decode = include_str!("../src/layers/moe/exl3_decode.rs");
    assert!(
        decode.find("run_native_fp8_shared_expert(").unwrap()
            < decode
                .find("EXL3 decode arm computes the shared expert via w4a16_gemv")
                .unwrap()
    );
    let verify = decode
        .split("pub(super) fn dispatch_exl3_verify(")
        .nth(1)
        .unwrap();
    assert!(!verify.contains("native block-FP8 shared expert does not support wide verify"));
    assert!(
        verify.find("run_native_fp8_shared_verify(").unwrap()
            < verify
                .find("EXL3 verify arm computes the shared expert via w4a16_gemv")
                .unwrap()
    );
}

#[test]
fn native_shared_fp8_preserves_valid_legacy_storage_and_uses_vision_clamped_activation() {
    let assemble = include_str!("../src/weight_loader/deepseek_v4/assemble.rs");
    let shared = assemble
        .split("let shared_expert = ExpertWeight {")
        .nth(1)
        .unwrap();
    assert!(
        shared
            .split("};")
            .next()
            .unwrap()
            .contains("load_expert_proj(")
    );
    assert!(assemble.contains("native_shared_fp8::install("));
    let native = include_str!("../src/layers/moe/native_shared_fp8.rs");
    assert!(native.contains("gpu.kernel(\"moe_silu_mul\", \"moe_silu_mul\")?"));
    assert!(!native.contains("\"silu_mul_noclamp\""));
    assert!(native.contains("ops::w8a16_gemv("));
    assert!(native.contains("ops::w8a16_gemm("));
    assert!(!native.contains("quantize_to_fp8"));
    assert!(!native.contains("quantize_to_nvfp4"));
}

#[test]
fn native_shared_fp8_activation_matches_pinned_vision_clamp_contract() {
    // Official DeepSeek-V4-Flash-Vision-Exp inference/model.py at revision
    // 6821d6ad3681a4b137b066b76094fa82ebd0a380 (SHA256 ae8de79223d53c16
    // edc7fa67af124ddd0212f27bddd3207bdd4cb05b48a0a1a7), lines 655-658/682:
    // shared and routed Expert both receive swiglu_limit; this checkpoint is 10.
    let cuda = include_str!("../../../kernels/gb10/common/moe_silu_mul.cu");
    let clamped = cuda
        .split("void moe_silu_mul(")
        .nth(1)
        .unwrap()
        .split("extern \"C\"")
        .next()
        .unwrap();
    assert!(clamped.contains("const float SWIGLU_LIMIT = 10.0f;"));
    let gate_clamp = clamped.find("g = fminf(g, SWIGLU_LIMIT);").unwrap();
    let up_clamp = clamped
        .find("u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);")
        .unwrap();
    let silu = clamped.find("float sigmoid_g =").unwrap();
    assert!(gate_clamp < silu && up_clamp < silu);
    assert!(clamped.contains("float result = g * sigmoid_g * u;"));
    assert!(!clamped.contains("fmaxf(g,"));
    let loader = include_str!("../src/weight_loader/deepseek_v4/native_shared_fp8.rs");
    assert!(loader.contains("SwiGLU clamp10"));
    assert!(!loader.contains("unclamped SiLU"));
}

#[test]
fn pinned_vision_shared_clamp10_cpu_boundary_vectors() {
    // CPU truth vectors, not a claim of CUDA numerical qualification. The
    // source contract above binds this oracle to the selected production entry.
    for (gate, up, expected_gate, expected_up) in [
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64),
        (9.5, -9.5, 9.5, -9.5),
        (10.0, 10.0, 10.0, 10.0),
        (10.0, -10.0, 10.0, -10.0),
        (10.5, 10.5, 10.0, 10.0),
        (10.5, -10.5, 10.0, -10.0),
        (100.0, 100.0, 10.0, 10.0),
        (100.0, -100.0, 10.0, -10.0),
        (-10.0, 10.0, -10.0, 10.0),
        (-10.5, 10.5, -10.5, 10.0),
        (-100.0, -100.0, -100.0, -10.0),
    ] {
        let actual_gate = gate.min(10.0);
        let actual_up = up.clamp(-10.0, 10.0);
        assert_eq!((actual_gate, actual_up), (expected_gate, expected_up));
        let actual = actual_gate / (1.0 + (-actual_gate).exp()) * actual_up;
        let expected = expected_gate / (1.0 + (-expected_gate).exp()) * expected_up;
        assert_eq!(actual, expected);
        assert!(actual.is_finite());
    }
    let silu = |gate: f64, up: f64| gate / (1.0 + (-gate).exp()) * up;
    assert!(silu(100.0, 100.0) > 100.0 * silu(10.0, 10.0));
    assert_ne!(silu(-100.0, 10.0), silu(-10.0, 10.0));
}
