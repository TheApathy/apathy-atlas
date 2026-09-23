// SPDX-License-Identifier: AGPL-3.0-only
//! RED structural gates for actual-runtime ownership; behavioral tests are separate.
use std::fs;
use std::path::PathBuf;

fn source(name: &str) -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/model/glm53")
            .join(name),
    )
    .unwrap_or_default()
}
fn compact(s: &str) -> String {
    s.split_whitespace().collect()
}
fn body<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    text.split_once(start)
        .expect(start)
        .1
        .split_once(end)
        .expect(end)
        .0
}

#[test]
fn real_runtime_owns_and_initializes_the_completed_prefix_tracker() {
    let runtime = compact(&source("dflash2_runtime.rs"));
    assert!(
        runtime.contains("modkv_prefix;"),
        "actual runtime must register helper"
    );
    assert!(
        runtime.contains("kv_prefix:"),
        "cache cursor is runtime-owned, not stack-only"
    );
    assert!(
        runtime.contains("KvPrefix::new("),
        "initialize from loaded layer count and capacity"
    );
    assert!(
        runtime.contains("modkv_prefix_runtime;"),
        "effectful adapter must be registered"
    );
}

#[test]
fn strict_opt_in_precedes_proposal_effects_and_retains_the_original_default_path() {
    let runtime = source("dflash2_runtime.rs");
    let proposal = body(&runtime, "pub fn propose(", "pub fn propose_graph_probe(");
    let selected = proposal
        .find("kv_prefix_enabled()?")
        .expect("strict opt-in selector");
    let cached = proposal
        .find("propose_with_kv_prefix(")
        .expect("real cached path");
    let original = proposal
        .find("self.stage_anchor(")
        .expect("original path retained");
    assert!(selected < cached && cached < original);
    let adapter = source("dflash2_kv_prefix_runtime.rs");
    assert!(adapter.contains("ATLAS_GLM53_DFLASH2_KV_PREFIX"));
    assert!(
        adapter.contains("NotUnicode"),
        "malformed OS env may not silently disable"
    );
    assert!(adapter.contains("parse_kv_prefix_flag("));
}

#[test]
fn graph_probe_rejects_opt_in_before_capture_or_any_anchor_upload() {
    let runtime = source("dflash2_runtime.rs");
    let graph = body(&runtime, "pub fn propose_graph_probe(", "fn stage_anchor(");
    let guard = graph
        .find("kv_prefix_enabled()?")
        .expect("explicit eager-only cache guard");
    assert!(guard < graph.find("self.stage_anchor(").unwrap());
    assert!(guard < graph.find("gpu.begin_capture(").unwrap());
    let adapter = source("dflash2_kv_prefix_runtime.rs");
    assert!(adapter.contains("stream_is_capturing(stream)"));
}

#[test]
fn actual_projection_attention_adapter_consumes_the_tail_plan_without_new_kv_allocations() {
    let adapter = compact(&source("dflash2_kv_prefix_runtime.rs"));
    for needle in [
        "implKvPrefixIofor",
        ".source_row()",
        ".new_rows()",
        ".retained_rows()",
        "Glm53Dflash2AttentionPlan::new(",
        ".attention.execute(",
        ".kv[",
    ] {
        assert!(
            adapter.contains(needle),
            "missing actual tail/cache seam: {needle}"
        );
    }
    assert!(
        !adapter.contains("gpu.alloc("),
        "reuse five already-owned KV pairs"
    );
    assert!(
        !adapter.contains("gpu.memset("),
        "do not clear retained KV between proposals"
    );
}

#[test]
fn success_publishes_after_proposal_readback_and_failures_abort_without_fallback() {
    let adapter = source("dflash2_kv_prefix_runtime.rs");
    let read = adapter
        .find("read_proposal(")
        .expect("actual selector validation retained");
    let finish = adapter.find(".finish(").expect("fenced publication");
    assert!(read < finish);
    assert!(adapter.contains(".abort("));
    assert!(adapter.contains("poison_verify("));
    assert!(
        adapter.contains("impl Drop for"),
        "abandoned proposal must retain fail-closed owner"
    );
    assert!(adapter.contains(".begin("));
    assert!(adapter.contains(".enqueue_layer("));
}

#[test]
fn reset_and_release_drain_cursor_owner_before_invalidating_or_freeing_storage() {
    let runtime = source("dflash2_runtime.rs");
    let reset = body(&runtime, "pub fn reset_context(", "pub fn propose(");
    assert!(reset.contains("reset_kv_prefix("));
    assert!(
        reset.find("reset_kv_prefix(").unwrap() < reset.find("self.context_tokens = 0").unwrap()
    );
    let free = body(&runtime, "pub fn free(", "fn region(");
    assert!(free.contains("drain_kv_prefix("));
    assert!(free.find("drain_kv_prefix(").unwrap() < free.find("gpu.free(").unwrap());
}
