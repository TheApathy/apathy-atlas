// SPDX-License-Identifier: AGPL-3.0-only

//! Offline production-routing guard for exact multi-row NVFP4 LM-head GEMV.

const INIT: &str = include_str!("../src/model/impl_a1.rs");
const FORWARD: &str = include_str!("../src/model/impl_a3.rs");
const TYPES: &str = include_str!("../src/model/types.rs");
const WRAPPER: &str = include_str!("../src/layers/ops/gemv_exact_lm_head.rs");
const ROUTE: &str = include_str!("../src/layers/ops/gemv_exact_lm_head/route.rs");

fn braced_body_after<'a>(source: &'a str, needle: &str) -> &'a str {
    let start = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing section: {needle}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing opening brace: {needle}"));
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("missing closing brace: {needle}");
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn startup_loads_every_exact_tier_as_optional_and_reports_policy() {
    for tier in ["M4", "M8", "M17", "M32"] {
        let symbol = format!("ops::ExactLmHeadTier::{tier}.symbol()");
        assert!(INIT.contains(&symbol), "missing optional load for {tier}");
    }
    let init = compact(INIT);
    assert!(init.contains("W4a16ExactLmHeadKernels::new(crate::layers::try_kernel("));
    assert!(INIT.contains("LM_HEAD_EXACT_STARTUP"));
    assert!(INIT.contains("exact_dynamic_m_or_serial_k1"));
    assert!(TYPES.contains("w4a16_exact_lm_head_kernels: ops::W4a16ExactLmHeadKernels"));
}

#[test]
fn every_qualified_nvfp4_batch_uses_exact_or_serial_k1() {
    let body = braced_body_after(FORWARD, "pub(super) fn lm_head_batched(");
    assert!(body.contains("route_for_rows(num_tokens)"));
    assert!(body.contains("w4a16_gemv_batch_logits_exact"));
    assert!(body.contains("ExactLmHeadRoute::SerialK1"));
    assert!(body.contains("for row in 0..num_tokens"));

    for forbidden in [
        "w4a16_gemv_batch2(",
        "w4a16_gemv_batch3_logits(",
        "w4a16_gemm_n64_m32_ldb(",
        "w4a16_gemm(",
    ] {
        assert!(
            !body.contains(forbidden),
            "qualified production body retained lossy route {forbidden}"
        );
    }
}

#[test]
fn serial_fallback_preserves_k1_rows_logical_stride_and_softcap() {
    let body = compact(braced_body_after(FORWARD, "pub(super) fn lm_head_batched("));
    assert!(body.contains("hidden.offset(rowasusize*hasusize*2)"));
    assert!(body.contains("logits.offset(rowasusize*vasusize*2)"));
    assert!(body.contains("self.w4a16_gemv_kernel"));
    assert!(body.contains("lettotal=num_tokens*v;"));
    assert!(body.contains("self.apply_logit_softcap(logits,total,cap,stream)?"));
}

#[test]
fn nvfp4_verify_width_is_physically_bounded_and_every_tier_is_logged() {
    let body = braced_body_after(FORWARD, "pub(super) fn lm_head_batched(");
    assert!(body.contains("(1..=32).contains(&num_tokens)"));
    assert!(body.contains("NVFP4 speculative LM-head rows must be in 1..=32"));
    assert!(FORWARD.contains("LM_HEAD_EXACT_ENGAGEMENT"));
    for tier in ["M4", "M8", "M17", "M32"] {
        assert!(
            FORWARD.contains(&format!("ops::ExactLmHeadTier::{tier} =>")),
            "engagement latch missing {tier}"
        );
    }
}

#[test]
fn pure_route_provenance_is_fail_closed() {
    assert!(ROUTE.contains("2..=4 => Some(ExactLmHeadTier::M4)"));
    assert!(ROUTE.contains("18..=32 => Some(ExactLmHeadTier::M32)"));
    assert!(ROUTE.contains("ExactLmHeadRoute::SerialK1(tier)"));
    assert!(WRAPPER.contains("self.is_present(tier)"));
    assert!(ROUTE.contains("serial_k1_row_major_nvfp4"));
    assert!(ROUTE.contains("exact_dynamic_m_nvfp4"));
}
