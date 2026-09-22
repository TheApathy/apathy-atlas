// SPDX-License-Identifier: AGPL-3.0-only

//! Source admission/ownership gates, not a numerical or GPU qualification.
//! Kept std-only so root can run these separately from missing-API RED tests.

const TARGET: &str = include_str!("target_model_exl3.rs");
const PREFILL: &str = include_str!("prefill_exl3.rs");
const DRAFT: &str = include_str!("dflash2_runtime.rs");
const CAPTURE: &str = include_str!("target_prefill_capture_exl3.rs");

fn target() -> String {
    [
        TARGET,
        include_str!("target_prefill_exl3.rs"),
        include_str!("target_vision_exl3.rs"),
        include_str!("target_staged_exl3.rs"),
    ]
    .join("\n")
}

fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

fn function(source: &str, name: &str) -> String {
    let source = compact(source);
    let marker = format!("fn{name}(");
    let start = source
        .find(&marker)
        .unwrap_or_else(|| panic!("missing {name}"));
    let start = start + source[start..].find('{').unwrap();
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..=start + offset].to_owned();
                }
            }
            _ => (),
        }
    }
    panic!("unterminated {name}");
}

fn before(source: &str, first: &str, second: &str) {
    let a = source
        .find(first)
        .unwrap_or_else(|| panic!("missing {first}"));
    let b = source
        .find(second)
        .unwrap_or_else(|| panic!("missing {second}"));
    assert!(a < b, "{first} must precede {second}");
}

#[test]
fn whole_prompt_and_actual_draft_capacity_are_admitted_before_reset() {
    let request = function(&target(), "prefill_request");
    before(&request, "context_capacity()", "InputPlan::new(");
    before(&request, "InputPlan::new(", "self.reset_sequence()");
    before(&request, "mixed_prefill_rows(", "self.reset_sequence()");
    let capacity = function(DRAFT, "context_capacity");
    assert!(capacity.contains("MAX_CONTEXT_TOKENS"));
    assert!(!request.contains("letwide_prefill=ifexpected_rows==0{"));
}

#[test]
fn token_and_mixed_rows_share_one_staged_body_with_ordered_image_overwrite() {
    let token = function(&target(), "verify_tokens_staged");
    assert!(token.contains("self.verify_inputs_staged("));
    assert!(token.contains("self.embed_dflash_tokens("));
    let mixed = function(&target(), "prefill_inputs_wide");
    assert!(mixed.contains("self.verify_inputs_staged("));
    assert!(mixed.contains("plan.chunk("));
    assert!(mixed.contains(".enqueue("));
    assert!(mixed.contains(".single_source()"));
    assert!(mixed.contains("WalkInput::ExternalEmbedding("));
    assert!(mixed.contains("self.embed_dflash_tokens("));
    assert!(mixed.contains("plan.overwrite_vision("));
    before(
        &mixed,
        "self.embed_dflash_tokens(",
        "plan.overwrite_vision(",
    );
    assert!(mixed.contains("self.prefill_layer_major_dflash2_with_fill("));
    let staged = function(&target(), "verify_inputs_staged");
    assert!(staged.contains("self.verify_inputs_staged_with_capture("));
    assert!(staged.contains("None"));
    let staged = function(&target(), "verify_inputs_staged_with_capture");
    before(&staged, "fill(", "Glm53Dispatcher::new_exl3(");
    assert!(!staged.contains("self.embed_dflash_tokens("));
    assert!(staged.contains("self.captures"));
}

#[test]
fn mixed_rows_keep_bounded_prefill_scope_commit_and_ordered_capture_observation() {
    let mixed = function(&target(), "prefill_inputs_wide");
    assert!(mixed.contains("with_glm53_exact_wide_prefill("));
    before(
        &mixed,
        "self.commit_accepted(",
        "self.observe_committed_wide_rows(",
    );
    before(
        &mixed,
        "self.preflight_prefill_rows(",
        "self.verify_inputs_staged(",
    );
    assert!(!mixed.contains("with_glm53_layer_major_prefill("));
    let admission = function(PREFILL, "mixed_prefill_rows");
    assert!(admission.contains("MAX_WIDE_ROWS"));
    assert!(admission.contains("layer_major"));
    let text = function(PREFILL, "prefill_tokens_wide");
    assert!(text.contains("self.preflight_prefill_capture("));
    assert!(text.contains("self.prefill_layer_major_dflash2("));
}

#[test]
fn mixed_layer_major_reuses_capture_commit_and_abort_lifecycle() {
    let wrapper = function(CAPTURE, "prefill_layer_major_dflash2");
    assert!(wrapper.contains("self.prefill_layer_major_dflash2_with_fill("));
    let capture = function(CAPTURE, "prefill_layer_major_dflash2_with_fill");
    before(
        &capture,
        "prefill_capture_bank.lock()",
        "self.dflash2.lock()",
    );
    before(
        &capture,
        "runtime.bind_prefill_capture(",
        "self.verify_inputs_staged_with_capture(",
    );
    before(&capture, "fill,", "self.commit_accepted(");
    before(
        &capture,
        "self.commit_accepted(",
        "runtime.observe_prefill_capture(",
    );
    before(
        &capture,
        "runtime.observe_prefill_capture(",
        "owner.complete(",
    );
    assert!(capture.contains("poisoned_stream=Some(stream)"));
    assert!(capture.contains("owner.abort("));
}

#[test]
fn ordinary_eight_row_preflight_does_not_reject_layer_major_capture() {
    let mixed = function(&target(), "prefill_inputs_wide");
    assert!(mixed.contains("if!layer_major{self.preflight_prefill_rows("));
    before(
        &mixed,
        "if!layer_major{self.preflight_prefill_rows(",
        "self.prefill_layer_major_dflash2_with_fill(",
    );
}

#[test]
fn prepared_owner_stays_in_model_and_partial_effects_poison_before_cleanup() {
    assert!(compact(TARGET).contains("prepared_vision:Mutex<PreparedOwner>"));
    let request = function(&target(), "prefill_request");
    assert!(!request.contains("prepared_vision.lock().unwrap().take()"));
    assert!(!request.contains("self.gpu.free("));
    before(&request, ".begin()", "InputPlan::new(");
    before(
        &request,
        "self.arm_prepared_vision(stream)",
        "self.prefill_inputs_wide(",
    );
    assert!(request.contains("effects_started"));
    before(
        &request,
        "poisoned_stream=Some(stream)",
        "self.finish_prepared_vision(",
    );
    assert!(request.contains("result.is_err()"));
}

#[test]
fn quarantine_retry_precedes_reset_prepare_and_any_model_allocation_release() {
    let reset = function(&target(), "reset_sequence");
    before(
        &reset,
        "self.retry_prepared_vision()",
        "runtime.reset_context(",
    );
    before(&reset, "self.retry_prepared_vision()", "self.gpu.memset(");
    let prepare = function(&target(), "prepare_vision_images");
    before(
        &prepare,
        "self.clear_prepared_vision()",
        "ifimages.is_empty()",
    );
    before(&prepare, "self.clear_prepared_vision()", "self.gpu.alloc(");
    let free = function(&target(), "free");
    // The consuming prefix-drain guard returns the same model as `this`.
    before(
        &free,
        "retain_until_drained(self,",
        "this.clear_prepared_vision()",
    );
    before(
        &free,
        "runtime.drain_kv_prefix(",
        "this.clear_prepared_vision()",
    );
    before(&free, "this.clear_prepared_vision()", "letSelf{");
    before(&free, "this.clear_prepared_vision()", "gpu.free(");
}

#[test]
fn failed_shutdown_retains_consumed_owner_before_any_field_teardown() {
    let free = function(&target(), "free");
    before(
        &free,
        "this.clear_prepared_vision()",
        "std::mem::forget(this)",
    );
    before(&free, "std::mem::forget(this)", "letSelf{");
    before(&free, "returnErr(error.context(", "letSelf{");
    assert!(free.contains("resourcesremainunreclaimeduntilprocessteardown"));
}
