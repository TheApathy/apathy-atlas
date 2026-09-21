// SPDX-License-Identifier: AGPL-3.0-only

//! Source contract for fail-closed W2A8 max-prefill benchmark engagement.

use std::fs;
use std::path::PathBuf;

const REQUIRE_FLAG: &str = "ATLAS_PREFILL_MAX_REQUIRE_ARMS";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative))
        .unwrap_or_else(|error| panic!("missing {relative}: {error}"))
}

fn compact(value: &str) -> String {
    value.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn model_load_freezes_a_literal_default_off_max_contract() {
    let state = source("crates/spark-model/src/layers/moe/exl3_decode.rs");
    let flat = compact(&state);

    assert!(state.contains("pub(crate) prefill_max_require_arms: bool"));
    assert!(flat.contains(&format!(
        "std::env::var(\"{REQUIRE_FLAG}\").as_deref()==Ok(\"1\")"
    )));
    assert!(state.contains("requires ATLAS_EXL3_PREFILL_W2A8=1"));
    assert!(state.contains("requires the fused GU N128 arm"));
    assert!(state.contains("rejects the native-losing fused GU N256 arm"));
    assert!(state.contains("requires the N256 down arm"));
    assert!(state.contains("fused_blend_requested"));
}

#[test]
fn exact_runtime_request_fails_closed_before_mutation_but_normal_fallback_remains() {
    let dispatch = source("crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs");
    let mutation = dispatch.find("// MUTATION START").expect("mutation marker");
    let before = &dispatch[..mutation];
    let after = &dispatch[mutation..];
    let flat_before = compact(before);

    assert!(before.contains("let require_exact_max"));
    assert!(flat_before.contains("letrequire_exact_max=pf.prefill_max_require_arms;"));
    assert!(before.contains("total_expanded == W2A8_TARGET_TOTAL_EXPANDED"));
    assert!(before.contains("qualification shape drift"));
    assert!(flat_before.contains("ensure!(!require_exact_max,"));
    assert!(before.contains("base W2A8 arm cannot engage"));
    assert!(flat_before.contains("ensure!(plan.use_fused_gu&&!plan.use_fused_gu_n256,"));
    assert!(flat_before.contains("ensure!(plan.use_n256_down,"));
    assert!(before.contains("return Ok(false)"));
    assert!(after.contains("ATLAS_PREFILL_MAX_ARMS_RECEIPT"));
    assert!(after.contains("fused_gu=n128 down=n256"));
    assert!(!after.contains("return Ok(false)"));
}

#[test]
fn requested_fused_tails_fail_closed_and_log_only_after_launch() {
    let tail = source("crates/spark-model/src/layers/moe/forward_prefill_exl3_tail.rs");

    assert!(tail.contains("prefill_max_require_arms"));
    assert!(tail.contains("fused_blend_requested"));
    assert!(tail.contains("requested fused unpermute tail cannot engage"));
    assert!(tail.contains("requested fused blend tail cannot engage"));
    assert!(tail.matches("ATLAS_PREFILL_MAX_ARMS_RECEIPT").count() >= 2);

    for receipt in ["tail=fused_unpermute", "tail=fused_blend"] {
        let log = tail.find(receipt).expect("tail engagement receipt");
        let launch = tail[..log]
            .rfind(".launch(stream)?;")
            .expect("successful launch");
        assert!(
            launch < log,
            "{receipt} must be logged after launch success"
        );
    }
}
