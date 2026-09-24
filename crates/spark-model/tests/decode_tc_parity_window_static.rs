// SPDX-License-Identifier: AGPL-3.0-only

//! Offline guard: `ATLAS_DECODE_TC_PARITY` may only engage for verify widths
//! whose tensor-core routes it mirrors, and the verify dispatch must select
//! those routes from the SAME shared window constants the parity gate reads.
//! A literal threshold on either side lets them drift apart silently.

const LAYERS: &str = include_str!("../src/layers/mod.rs");
const SSM_VERIFY: &str = include_str!("../src/layers/qwen3_ssm/trait_decode_batched.rs");
const SSM_DECODE: &str = include_str!("../src/layers/qwen3_ssm/ssm_forward.rs");
const FFN: &str = include_str!("../src/layers/dense_ffn.rs");
const ATTN_FFN: &str = include_str!("../src/layers/qwen3_attention/trait_impl/multi_seq/ffn.rs");
const DRAFTER_LOAD: &str =
    include_str!("../../spark-server/src/main_modules/serve_phases/weights.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

/// The `else if` condition that immediately precedes `marker`.
fn condition_before<'a>(source: &'a str, marker: &str) -> &'a str {
    let at = source.find(marker).unwrap_or_else(|| panic!("missing {marker}"));
    let start = source[..at].rfind("} else if").expect("branch head");
    &source[start..at]
}

#[test]
fn window_constants_are_the_measured_bounds() {
    let layers = compact(LAYERS);
    assert!(layers.contains("pubconstTC_VERIFY_MIN_ROWS:usize=4;"));
    assert!(layers.contains("pubconstTC_VERIFY_MAX_ROWS:usize=32;"));
    assert!(layers.contains("rows>=TC_VERIFY_MIN_ROWS&&rows<=TC_VERIFY_MAX_ROWS"));
    // Engagement needs BOTH the request and an in-window configured width;
    // no configured width (None) fails closed.
    assert!(layers.contains("Some(rows)=>requested&&rows_in_tc_verify_window(rows),None=>false,"));
    assert!(layers.contains("AtomicBool::new(false)"));
}

#[test]
fn verify_tc_routes_are_selected_from_the_shared_window() {
    for marker in [
        "&& super::super::ssm_qkvz_splitk() > 0",
        "&& super::super::ssm_out_splitk() > 0",
    ] {
        let cond = condition_before(SSM_VERIFY, marker);
        assert!(
            cond.contains("rows_in_tc_verify_window(k as usize)"),
            "split-K verify route not bound to the shared window: {cond}"
        );
        assert!(!cond.contains("k > 3") && !cond.contains("k <= 32"));
    }
    assert!(SSM_VERIFY.contains("let try_kgamma = num_tokens >= super::super::TC_VERIFY_MIN_ROWS"));
    assert!(ATTN_FFN.contains("&& n >= crate::layers::TC_VERIFY_MIN_ROWS"));
    assert!(FFN.contains("let wide_m128 = n as usize > crate::layers::TC_VERIFY_MAX_ROWS"));
    assert!(FFN.contains("&& n as usize <= crate::layers::TC_VERIFY_MAX_ROWS"));
}

#[test]
fn every_decode_parity_route_goes_through_the_window_gated_predicate() {
    let ready = compact(SSM_DECODE);
    assert!(ready.contains("super::super::decode_tc_parity_enabled()&&super::super::ssm_proj_tc_enabled()"));
    assert!(compact(FFN).contains("crate::layers::decode_tc_parity_enabled()&&exact_ffn_tc_override()"));
    // The engagement flag is written only by the configure hook, and only
    // after the SSM route check: no silent per-layer fallback to K1.
    assert_eq!(LAYERS.matches("DECODE_TC_PARITY_ENGAGED.store(").count(), 1);
    let layers = compact(LAYERS);
    assert!(layers.contains(
        "letmirrorable=decode_tc_parity_ssm_mirrorable(ssm_proj_tc_enabled(),ssm_qkvz_splitk(),ssm_out_splitk(),tc_nvfp4_m16_enabled(),);"
    ));
    assert!(layers.contains(
        "letengaged=decode_tc_parity_engaged(requested,verify_rows)&&mirrorable;"
    ));
    assert!(layers.contains("!ssm_proj_tc||(qkvz_splits>0&&out_splits>0&&!tc_nvfp4_m16)"));
}

/// Every model load (first boot and each swap) passes through
/// `load_dflash_drafter`, which must configure the gate on both of its exits.
#[test]
fn every_model_load_configures_the_parity_window() {
    let start = DRAFTER_LOAD
        .find("pub(crate) fn load_dflash_drafter(")
        .expect("drafter loader");
    let end = DRAFTER_LOAD[start..]
        .find("\n}\n")
        .map(|offset| start + offset)
        .expect("end of drafter loader");
    let body = compact(&DRAFTER_LOAD[start..end]);
    assert!(body.contains("configure_decode_tc_parity(None);returnOk(None);"));
    assert!(body.contains(
        "configure_decode_tc_parity(Some(spark_model::layers::dflash_head::effective_verify_rows(drafter_config.resolve_draft_count(args.dflash_gamma)?,),));"
    ));
}
