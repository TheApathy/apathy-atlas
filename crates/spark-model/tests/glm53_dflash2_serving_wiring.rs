// SPDX-License-Identifier: AGPL-3.0-only
//! Serving seams are checked here; live text/image policy parity is a GPU gate.
fn read(file: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/model/glm53")
            .join(file),
    )
    .unwrap_or_default()
}
fn compact(text: &str) -> String {
    text.split_whitespace().collect()
}
fn section<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    text.split_once(start)
        .expect(start)
        .1
        .split_once(end)
        .expect(end)
        .0
}

#[test]
fn startup_choice_and_handles_are_pinned_before_drafter_weight_or_arena_allocations() {
    let runtime = read("dflash2_runtime.rs");
    assert!(runtime.contains("dflash2_serving_contract.rs"));
    assert!(runtime.contains("dflash2_serving.rs"));
    assert!(compact(&runtime).contains("serving_projection:CommittedProjection"));
    let load = section(&runtime, "pub fn load(", "pub fn observe_target(");
    let parse = load
        .find("ServingProjection::parse(")
        .expect("startup policy admission");
    let resolve = load
        .find("CommittedProjection::for_serving(")
        .expect("startup kernel binding");
    let weights = load.find("SafetensorsLoader::new().load(").unwrap();
    assert!(parse < resolve && resolve < weights);
    assert!(resolve < load.find("gpu.alloc(").unwrap());
    assert!(load.contains("ATLAS_GLM53_DFLASH2_COMMITTED_PROJECTION"));
    assert!(compact(load).contains("serving_projection,"));
}

#[test]
fn serving_admits_before_any_path_and_preserves_original_full_and_cached_wrappers() {
    let runtime = read("dflash2_runtime.rs");
    let body = section(&runtime, "pub fn propose(", "pub fn propose_graph_probe(");
    let admit = body
        .find("admit_serving(")
        .expect("pinned serving admission");
    let gemv = body
        .find("propose_serving_gemv(")
        .expect("explicit new route");
    let cached = body
        .find("propose_with_kv_prefix(")
        .expect("unchanged original cached wrapper");
    let full = body
        .find("self.stage_anchor(")
        .expect("unchanged original full path");
    assert!(admit < gemv && gemv < cached && cached < full);
    assert!(compact(body).contains("enqueue_proposal(target,anchor,stream,None,None)"));
    assert!(!body.contains("propose_diagnostic("));
    let cached_source = compact(&read("dflash2_kv_prefix_runtime.rs"));
    assert!(cached_source.contains("self.propose_cached_observed(target,anchor,stream,None)"));
}

#[test]
fn gemv_serving_uses_actual_preflight_owner_and_typed_cached_body_without_new_io_owners() {
    let source = read("dflash2_serving.rs");
    assert!(source.contains("admit_current("));
    assert!(source.contains("admit_proposal("));
    let body = source
        .split_once("fn propose_serving_gemv(")
        .expect("actual serving adapter")
        .1;
    let plan = body
        .find("preflight_gemv_projection(")
        .expect("all-layer operands");
    let family = body
        .find("ensure_projection_ready(")
        .expect("readonly family admission");
    let submit = body
        .find("propose_cached_observed_with_projection(")
        .expect("real cached path");
    assert!(plan < submit && family < submit);
    assert!(body.contains("catch_unwind("));
    assert!(body.contains("poison_verify("));
    assert!(compact(body).contains("self.serving_projection"));
    for forbidden in [
        "gpu.alloc(",
        "copy_h2d",
        "copy_d2h",
        "observer",
        "propose_diagnostic(",
        "enqueue_proposal(",
        "set_var(",
        "reset_kv_prefix(",
    ] {
        assert!(
            !source.contains(forbidden),
            "unexpected separate path/owner: {forbidden}"
        );
    }
}

#[test]
fn graph_admission_and_reset_keep_startup_choice_separate_from_completed_prefix_authority() {
    let runtime = read("dflash2_runtime.rs");
    let graph = section(&runtime, "pub fn propose_graph_probe(", "fn stage_anchor(");
    let admit = graph
        .find("admit_graph(")
        .expect("startup choice rejects graph capture");
    assert!(admit < graph.find("self.stage_anchor(").unwrap());
    assert!(admit < graph.find("gpu.begin_capture(").unwrap());
    let reset = section(&runtime, "pub fn reset_context(", "pub fn propose(");
    assert!(
        reset.find("reset_kv_prefix(").unwrap() < reset.find("self.context_tokens = 0").unwrap()
    );
    assert!(!reset.contains("serving_projection ="));
    assert!(!reset.contains("for_serving("));
    let binding = read("dflash2_projection_contract.rs");
    assert!(
        binding.find("prefix.reset(io)?").unwrap() < binding.find("self.family = None").unwrap()
    );
}

#[test]
fn stable_serving_mapping_reuses_existing_kernel_family_not_a_changed_diagnostic_default() {
    let projection = read("dflash2_committed_projection.rs");
    let start = projection
        .find("fn for_serving(")
        .expect("explicit serving resolver");
    let body = &projection[start..];
    assert!(body.contains("ServingProjection::Original"));
    assert!(body.contains("ServingProjection::StableGemv"));
    assert!(body.contains("GemvKernels::load(gpu)"));
    assert!(projection.contains("fn serving_choice("));
    let diagnostic = read("dflash2_probe_runtime.rs");
    assert!(diagnostic.contains("CommittedProjection::for_probe("));
    assert!(!diagnostic.contains("admit_serving("));
    assert!(!diagnostic.contains("admit_current("));
}

#[test]
fn mixed_image_rows_keep_the_shared_committed_capture_path_without_a_blanket_mode_guard() {
    let mixed = read("target_prefill_exl3.rs");
    let vision = read("target_vision_exl3.rs");
    let runtime = read("dflash2_runtime.rs");
    let target = read("target_model_exl3.rs");
    assert!(mixed.contains("WalkInput::ExternalEmbedding"));
    assert!(mixed.contains("verify_inputs_staged("));
    assert!(mixed.contains("observe_committed_wide_rows("));
    assert!(target.contains("runtime.observe_target(self, stream)"));
    assert!(runtime.contains("target.copy_dflash_capture_rows("));
    assert!(runtime.contains("self.weights.fc.weight"));
    assert!(runtime.contains("&self.weights.hidden_norm"));
    for source in [&mixed, &vision] {
        assert!(!source.contains("StableGemv"));
        assert!(!source.contains("ATLAS_GLM53_DFLASH2_COMMITTED_PROJECTION"));
    }
}
