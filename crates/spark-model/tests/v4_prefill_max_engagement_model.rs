// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only source contracts for fail-closed V4 max-prefill arm engagement.

use std::fs;
use std::path::PathBuf;

const HOST: &str = "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source() -> String {
    fs::read_to_string(root().join(HOST)).expect("read V4 prefill source")
}

fn compact(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn section<'a>(text: &'a str, name: &str) -> &'a str {
    let begin = format!("// BEGIN {name}");
    let end = format!("// END {name}");
    assert_eq!(text.matches(&begin).count(), 1, "duplicate {begin}");
    assert_eq!(text.matches(&end).count(), 1, "duplicate {end}");
    let start = text.find(&begin).expect("missing begin marker") + begin.len();
    let finish = text[start..].find(&end).expect("missing end marker") + start;
    &text[start..finish]
}

fn must_error(strict: bool, requested: bool, engaged: bool) -> bool {
    strict && requested && !engaged
}

#[test]
fn strict_mode_is_cached_and_only_literal_one_enables_it() {
    let source = source();
    let gate = compact(section(&source, "V4 max-arm strict gate"));
    for contract in [
        "fnv4_prefill_max_require_arms()->bool",
        "staticON:std::sync::OnceLock<bool>",
        "std::env::var(\"ATLAS_PREFILL_MAX_REQUIRE_ARMS\").as_deref()==Ok(\"1\")",
    ] {
        assert!(gate.contains(contract), "strict gate omits {contract}");
    }
    assert!(!gate.contains("!=Ok(\"0\")"));
}

#[test]
fn fail_closed_truth_table_applies_to_every_requested_arm() {
    assert!(must_error(true, true, false));
    for allowed in [
        must_error(false, true, false),
        must_error(true, false, false),
        must_error(true, true, true),
    ] {
        assert!(!allowed);
    }

    let flat = compact(&source());
    for contract in [
        "ifv4_prefill_max_require_arms()&&requested",
        "require_v4_prefill_max_arm(\"tc2_warp0\",v4_prefill_tc2_warp0_requested,use_v4_prefill_tc2_warp0",
        "require_v4_prefill_max_arm(\"qb_rope_fused\",v4_prefill_qb_rope_fused_requested,use_v4_prefill_qb_rope_fused",
        "require_v4_prefill_max_arm(\"kv_alias\",kv_alias_requested,kv_alias)",
        "require_v4_prefill_max_arm(\"inverse_rope\",inverse_rope_requested,use_v4_prefill_inverse_rope_fused",
        "refusingincumbentfallback",
    ] {
        assert!(flat.contains(contract), "strict contract omits {contract}");
    }
    assert!(flat.contains("use_v4_prefill_tc2_warp0=v4_prefill_tc2_warp0_requested"));
    assert!(flat.contains("&&n==2410&&nq==64&&nkv==1&&hd_mla==512"));
    assert!(flat.contains("use_v4_prefill_inverse_rope_fused=inverse_rope_requested"));
}

#[test]
fn all_four_checks_run_before_their_incumbent_fallbacks() {
    let source = source();
    let tc2_check = source
        .find("// BEGIN V4 max-arm TC2 warp0 contract")
        .expect("TC2 strict check");
    let first_projection = source.find("// ── 1. Q latent").expect("first projection");
    assert!(tc2_check < first_projection);

    let qb_check = source
        .find("// BEGIN V4 max-arm Q-B forward RoPE contract")
        .expect("Q-B forward RoPE strict check");
    let qb_fallback = source
        .find("// BEGIN V4 Q-B norm incumbent fallback")
        .expect("Q-B norm incumbent fallback");
    assert!(qb_check < qb_fallback);

    let alias_check = source
        .find("// BEGIN V4 max-arm K/V alias contract")
        .expect("K/V alias strict check");
    let alias_fallback = source
        .find("if !kv_alias {")
        .expect("K/V alias incumbent fallback");
    assert!(alias_check < alias_fallback);

    let inverse_check = source
        .find("// BEGIN V4 max-arm inverse RoPE contract")
        .expect("inverse RoPE strict check");
    let inverse_fallback = source
        .find("// BEGIN V4 inverse-only RoPE fallback")
        .expect("inverse RoPE incumbent fallback");
    assert!(inverse_check < inverse_fallback);
}

#[test]
fn engagement_receipts_are_process_bounded_and_emitted_at_successful_dispatch() {
    let source = source();
    for (arm, once, expected_mentions) in [
        ("tc2_warp0", "V4_TC2_WARP0_ENGAGED_LOGGED", 3),
        ("qb_rope_fused", "V4_QB_ROPE_ENGAGED_LOGGED", 2),
        ("kv_alias", "V4_KV_ALIAS_ENGAGED_LOGGED", 2),
        ("inverse_rope", "V4_INVERSE_ROPE_ENGAGED_LOGGED", 2),
    ] {
        assert_eq!(
            source.matches(once).count(),
            expected_mentions,
            "{once} must have one global declaration and only its bounded dispatch uses"
        );
        assert!(
            source.contains(&format!("\"{arm}\"")),
            "missing receipt arm name {arm}"
        );
    }
    assert!(
        source.contains("V4_PREFILL_MAX_ARM_ENGAGED arm={} layer={} n={} nq={} nkv={} hd_mla={}")
    );

    let compact = compact(&source);
    assert_eq!(
        compact
            .matches(
                "use_v4_prefill_tc2_warp0{log_v4_prefill_max_arm_engaged(&V4_TC2_WARP0_ENGAGED_LOGGED,\"tc2_warp0\""
            )
            .count(),
        2,
        "both TC2 warp-0 dispatches must emit a bounded receipt"
    );
    assert!(
        compact.contains(
            "log_v4_prefill_max_arm_engaged(&V4_QB_ROPE_ENGAGED_LOGGED,\"qb_rope_fused\""
        )
    );
    assert!(compact.contains(
        "ifkv_alias{log_v4_prefill_max_arm_engaged(&V4_KV_ALIAS_ENGAGED_LOGGED,\"kv_alias\""
    ));
    assert!(compact.contains(
        "log_v4_prefill_max_arm_engaged(&V4_INVERSE_ROPE_ENGAGED_LOGGED,\"inverse_rope\""
    ));
}

#[test]
fn kernel_engagement_receipts_follow_successful_launches() {
    let source = source();

    let qb_start = source
        .find("// BEGIN V4 Q-B forward RoPE dispatch")
        .expect("Q-B forward RoPE dispatch start");
    let qb_end = source[qb_start..]
        .find("// END V4 Q-B forward RoPE dispatch")
        .expect("Q-B forward RoPE dispatch end")
        + qb_start;
    let qb = &source[qb_start..qb_end];
    assert!(
        qb.find(".launch(stream)?;").expect("Q-B launch")
            < qb.find("\"qb_rope_fused\"").expect("Q-B receipt"),
        "Q-B forward RoPE receipt must follow a successful launch"
    );

    let csa_start = source.find("let attn_k =").expect("CSA dispatch start");
    let csa_end = source[csa_start..]
        .find("aprof!(\"4b_attn_kernel\")")
        .expect("CSA dispatch end")
        + csa_start;
    let csa = &source[csa_start..csa_end];
    assert!(
        csa.find(".launch(stream)?;").expect("CSA launch")
            < csa.find("\"tc2_warp0\"").expect("CSA receipt"),
        "CSA TC2 warp-0 receipt must follow a successful launch"
    );

    let dense_start = source
        .find("if !did_csa {")
        .expect("dense attention dispatch start");
    let dense_end = source[dense_start..]
        .find("if stage_syncs {")
        .expect("dense attention dispatch end")
        + dense_start;
    let dense = &source[dense_start..dense_end];
    assert!(
        dense.find(".launch(stream)?;").expect("dense launch")
            < dense.find("\"tc2_warp0\"").expect("dense receipt"),
        "dense TC2 warp-0 receipt must follow a successful launch"
    );

    let inverse_start = source
        .find("if use_v4_prefill_inverse_rope_fused {")
        .expect("inverse RoPE dispatch start");
    let inverse_end = source[inverse_start..]
        .find("// BEGIN V4 inverse-only RoPE fallback")
        .expect("inverse RoPE dispatch end")
        + inverse_start;
    let inverse = &source[inverse_start..inverse_end];
    assert!(
        inverse.find(".launch(stream)?;").expect("inverse launch")
            < inverse.find("\"inverse_rope\"").expect("inverse receipt"),
        "inverse RoPE receipt must follow a successful launch"
    );
}

#[test]
fn strict_contract_does_not_touch_scale_consistent_compressed_pool_write() {
    let source = source();
    let pool = compact(section(
        &source,
        "V4 scale-consistent compressed-pool write",
    ));
    for contract in [
        "self.effective_fp8_scales()",
        "ops::bf16_to_fp8_scaled(",
        "self.v4_comp_pool_filled.store(n_win",
    ] {
        assert!(
            pool.contains(contract),
            "compressed-pool path omits {contract}"
        );
    }
    assert!(!pool.contains("ATLAS_PREFILL_MAX_REQUIRE_ARMS"));
    assert!(!pool.contains("require_v4_prefill_max_arm"));
}
