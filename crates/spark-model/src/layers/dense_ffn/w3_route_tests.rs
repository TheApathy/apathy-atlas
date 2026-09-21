// SPDX-License-Identifier: AGPL-3.0-only

use super::w3_kgamma_applicable;

#[test]
fn w3_route_requires_small_batched_silu_and_complete_preparation() {
    for rows in 2..=32 {
        assert!(w3_kgamma_applicable(rows, true, true), "rows={rows}");
    }
    for rows in [0, 1, 33, 128, u32::MAX] {
        assert!(!w3_kgamma_applicable(rows, true, true), "rows={rows}");
    }
    assert!(!w3_kgamma_applicable(17, false, true));
    assert!(!w3_kgamma_applicable(17, true, false));
}

#[test]
fn w3_dispatch_is_complete_ordered_and_precedes_w4_routes() {
    let source = include_str!("../dense_ffn.rs");
    let start = source.find("pub fn forward_kgamma(").unwrap();
    let w3_start = source[start..]
        .find("if w3_kgamma_applicable(")
        .map(|offset| start + offset)
        .unwrap();
    let w4_start = source[w3_start..]
        .find("let wide_m128 =")
        .map(|offset| w3_start + offset)
        .unwrap();
    let body = &source[w3_start..w4_start];

    assert_eq!(body.matches("ops::w3a16_gemm_n64_m32(").count(), 3);
    let gate = body.find("ffn_gate_w3_kgamma").unwrap();
    let up = body.find("ffn_up_w3_kgamma").unwrap();
    let silu = body.find("ffn_silu_mul_w3_kgamma").unwrap();
    let down = body.find("ffn_down_w3_kgamma").unwrap();
    let engaged = body.find("ENGAGED W3 FFN K-gamma").unwrap();
    let complete = body.rfind("return Ok(())").unwrap();
    assert!(gate < up && up < silu && silu < down);
    assert!(down < engaged && engaged < complete);
    assert!(body.contains("self.has_w3_gemm()"));
    assert!(body.contains("self.activation == FfnActivation::SiLU"));
    assert!(body.contains(".context(\"W3 K-gamma route selected without transposed weights\")"));
    assert!(!body.contains("&self.weights."));
    assert!(!body.contains("ops::w4a16_"));
}

#[test]
fn w3_preparation_requires_weight_handle_and_silu() {
    let source = include_str!("../dense_ffn.rs");
    let start = source.find("fn has_w3_gemm(&self)").unwrap();
    let end = source[start..]
        .find("/// Whether ANY W3 routing")
        .map(|offset| start + offset)
        .unwrap();
    let body = &source[start..end];
    assert!(body.contains("self.w3_weights_t.is_some()"));
    assert!(body.contains("self.w3a16_gemm_t_m32_n64_k.0 != 0"));
    assert!(body.contains("self.activation == FfnActivation::SiLU"));
}
