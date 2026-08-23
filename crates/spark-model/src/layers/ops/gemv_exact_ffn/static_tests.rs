// SPDX-License-Identifier: AGPL-3.0-only

use std::fs;
use std::path::Path;

use super::ExactFfnTier;

fn fused_cuda() -> String {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10/common/w4a16_gemv_fused.cu");
    let mut source = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let m8_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common/w4a16_gemv_exact_f32_m8.cuh");
    source.push_str(
        &fs::read_to_string(&m8_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", m8_path.display())),
    );
    let fused_m17_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common/w4a16_gemv_exact_f32_m17_fused.cuh");
    source.push_str(
        &fs::read_to_string(&fused_m17_path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", fused_m17_path.display())),
    );
    source
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start {start}"));
    let rest = &source[start..];
    let end = rest
        .find(end)
        .unwrap_or_else(|| panic!("missing section end {end}"));
    &rest[..end]
}

#[test]
fn every_route_symbol_is_instantiated_in_fused_cuda() {
    let source = fused_cuda();
    for tier in [
        ExactFfnTier::M4,
        ExactFfnTier::M8,
        ExactFfnTier::M17,
        ExactFfnTier::M32,
    ] {
        assert!(source.contains(tier.dual_symbol()), "missing dual {tier:?}");
        assert!(
            source.contains(tier.silu_input_symbol()),
            "missing SiLU-input {tier:?}"
        );
    }
}

#[test]
fn exact_bodies_pin_k1_k8_association_and_reduction_order() {
    let source = fused_cuda();
    let dual = section(
        &source,
        "w4a16_gemv_dual_exact_body(",
        "w4a16_gemv_silu_input_exact_body(",
    );
    let down = section(
        &source,
        "w4a16_gemv_silu_input_exact_body(",
        "#define DEFINE_W4A16_GEMV_DUAL_EXACT",
    );

    for (name, body) in [("dual", dual), ("down", down)] {
        assert!(
            body.contains("for (unsigned int k8 = lane; k8 < K8; k8 += 64u)"),
            "{name} changed K1 lane ownership or stride"
        );
        assert!(
            body.contains("for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)"),
            "{name} changed the five-step shuffle tree"
        );
        assert!(
            body.contains("smem[base] + smem[base + 1]"),
            "{name} changed the ordered cross-warp add"
        );
        assert!(
            !body.contains("return;"),
            "{name} must keep tail groups in both block barriers"
        );
    }

    assert!(dual.contains("acc[row] += __bfloat162float(a_lo) * w_lo[b];"));
    assert!(dual.contains("acc[row] += __bfloat162float(a_hi) * w_hi[b];"));
    assert!(down.contains("(gf_lo / (1.0f + __expf(-gf_lo)))"));
    assert!(down.contains("acc[row] += a_lo * w_lo[b];"));
    assert!(down.contains("acc[row] += a_hi * w_hi[b];"));
}

#[test]
fn materialized_m8_m17_round_projection_outputs_before_silu() {
    let source = fused_cuda();
    assert!(source.contains("w4a16_gemv_dual_silu_f32_exact_m8"));
    assert!(source.contains("w4a16_gemv_f32_input_exact_m8"));
    assert!(source.contains("w4a16_gemv_dual_silu_f32_exact_m17"));
    assert!(source.contains("w4a16_gemv_f32_input_exact_m17"));
    assert!(source.contains("const __nv_bfloat16 gate_bf16 = gate_out[idx]"));
    assert!(source.contains("const __nv_bfloat16 up_bf16 = up_out[idx]"));
    assert!(source.contains("const float gate = __bfloat162float(gate_bf16)"));
    assert!(source.contains("const float up = __bfloat162float(up_bf16)"));
    assert!(source.contains("(gate / (1.0f + __expf(-gate))) * up"));
    assert!(source.contains("const float* A_row"));
}

#[test]
fn fused_m17_preserves_exact_projection_order_and_bf16_boundary() {
    let source = fused_cuda();
    let fused = section(
        &source,
        "w4a16_dual_exact_materialize_f32_body(",
        "extern \"C\" __global__ void w4a16_gemv_dual_exact_materialize_f32_m17",
    );

    assert!(source.contains("w4a16_gemv_dual_exact_materialize_f32_m17"));
    assert!(fused.contains("for (int proj = 0; proj < 2; ++proj)"));
    assert!(fused.contains("for (unsigned int k8 = lane; k8 < K8; k8 += 64u)"));
    assert!(fused.contains("acc[row] += __bfloat162float(a_lo) * w_lo[b];"));
    assert!(fused.contains("acc[row] += __bfloat162float(a_hi) * w_hi[b];"));
    assert!(fused.contains("for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)"));
    assert!(fused.contains("smem[slot * 2] + smem[slot * 2 + 1]"));
    assert!(fused.contains("rounded_gate[slot] = rounded"));
    assert!(fused.contains("__bfloat162float(rounded_gate[slot])"));
    assert!(fused.contains("(gate / (1.0f + __expf(-gate))) * up"));
}
