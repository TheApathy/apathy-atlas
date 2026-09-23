// SPDX-License-Identifier: AGPL-3.0-only
//! Narrow source/order gates for the GPU-only wiring; behavioral contracts live
//! in glm53_prefill_capture{,_owner,_taps}.rs. These are not numeric proof.
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model/glm53")
}
fn read(name: &str) -> String {
    std::fs::read_to_string(root().join(name)).unwrap_or_default()
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
    let mut depth = 0;
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
fn target() -> String {
    [
        "target_model_exl3.rs",
        "target_prefill_exl3.rs",
        "target_prefill_capture_exl3.rs",
        "target_staged_exl3.rs",
        "target_vision_exl3.rs",
    ]
    .map(read)
    .join("\n")
}
fn draft() -> String {
    ["dflash2_runtime.rs", "dflash2_prefill_capture.rs"]
        .map(read)
        .join("\n")
}

#[test]
fn owner_and_helpers_are_registered_and_the_target_owns_one_lazy_bank() {
    let modules = compact(&read("mod.rs"));
    for name in [
        "prefill_capture_plan",
        "prefill_capture_ingest",
        "prefill_capture_owner",
    ] {
        assert!(
            modules.contains(&format!("mod{name};")),
            "unregistered {name}"
        );
    }
    let target = compact(&read("target_model_exl3.rs"));
    assert!(target.contains("prefill_capture_bank:Mutex<CaptureBankOwner>"));
    assert!(target.contains("prefill_capture_bank:Mutex::new(CaptureBankOwner::new())"));
}

#[test]
fn admission_precedes_reset_and_layermajor_draft_dispatch_is_not_a_bare_guard_removal() {
    let request = function(&target(), "prefill_request");
    before(
        &request,
        "self.preflight_prefill_capture(",
        "self.reset_sequence()",
    );
    let admission = function(&target(), "preflight_prefill_capture");
    assert!(admission.contains("PrefillCapturePlan::new("));
    assert!(admission.contains("context_capacity()"));
    assert!(admission.contains("self.preflight_capture_workspace("));
    let workspace = function(&target(), "preflight_capture_workspace");
    assert!(workspace.contains("self.wide_workspace_bytes") && workspace.contains(".checked_add("));
    let prefill = function(&read("prefill_exl3.rs"), "prefill_tokens_wide");
    assert!(prefill.contains("self.prefill_layer_major_dflash2("));
    assert!(prefill.contains("config.layer_major") && prefill.contains("self.has_dflash2()"));
    assert!(prefill.contains("self.observe_committed_wide_rows(")); // Original K8 path retained.
}

#[test]
fn original_staged_api_borrows_no_capture_and_only_new_path_supplies_a_receipt() {
    let source = read("target_staged_exl3.rs");
    let wrapper = function(&source, "verify_inputs_staged");
    assert!(wrapper.contains("self.verify_inputs_staged_with_capture("));
    assert!(wrapper.contains("None"));
    let body = function(&source, "verify_inputs_staged_with_capture");
    before(&body, "fill(", "Glm53Dispatcher::new_exl3(");
    assert!(body.contains("dispatch_with_prefill_capture("));
    let legacy = compact(&read("capture_slots.rs"));
    assert!(
        legacy.contains("GLM53_CAPTURE_MAX_ROWS:u32=8"),
        "fixed legacy capture ABI must stay eight rows"
    );
}

#[test]
fn actual_capture_events_use_mean_inside_the_success_only_enqueue_helper() {
    let dispatcher = read("dispatch.rs");
    assert!(
        compact(&dispatcher).contains("#[path=\"dispatch_prefill_capture.rs\"]modprefill_capture;")
    );
    let route = function(
        &read("dispatch_prefill_capture.rs"),
        "dispatch_with_prefill_capture",
    );
    assert!(route.contains("CaptureWidenedMhc{layer,slot}"));
    assert!(route.contains("glm53_layer_major_prefill_active()"));
    before(&route, ".enqueue_tap(", ".mean(");
    assert!(route.contains("self.bound.widened_hc"));
    assert!(route.contains("self.dispatch("));
    let existing = function(&dispatcher, "dispatch");
    assert!(existing.contains("self.captures.contracted_slot_rows("));
}

#[test]
fn commit_ingest_and_completion_are_ordered_and_failures_poison_both_owners() {
    let capture = function(&target(), "prefill_layer_major_dflash2_with_fill");
    before(&capture, ".begin(", "verify_inputs_staged_with_capture(");
    before(
        &capture,
        "verify_inputs_staged_with_capture(",
        "self.commit_accepted(",
    );
    before(
        &capture,
        "self.commit_accepted(",
        ".observe_prefill_capture(",
    );
    before(&capture, ".observe_prefill_capture(", ".complete(");
    assert!(capture.contains("with_glm53_layer_major_prefill("));
    assert!(capture.contains("poisoned_stream=Some(stream)"));
    assert!(capture.contains(".abort("));
    assert!(!capture.contains("observe_committed_wide_rows(")); // No duplicate ingestion.
}

#[test]
fn runtime_reuses_fc_norm_staging_and_advances_only_from_successful_receipt() {
    let source = draft();
    let observe = function(&source, "observe_prefill_capture");
    before(&observe, ".ingest(", "self.context_tokens=");
    let adapter = function(&source, "project_norm");
    assert!(adapter.contains("dense("));
    assert!(adapter.contains("weights.fc.weight"));
    assert!(adapter.contains("ops::rms_norm("));
    assert!(adapter.contains("weights.hidden_norm"));
    assert!(!adapter.contains("context_tokens="));
    assert!(!source.contains("capture_input=gpu.alloc(2048"));
}

#[test]
fn reset_and_consuming_free_fence_capture_ownership_before_other_state_teardown() {
    let source = target();
    let reset = function(&source, "reset_sequence");
    before(
        &reset,
        "self.release_prefill_capture_bank()",
        "runtime.reset_context(",
    );
    let free = function(&source, "free");
    before(&free, "this.release_prefill_capture_bank()", "letSelf{");
    before(&free, "std::mem::forget(this)", "letSelf{");
    let release = function(&source, "release_prefill_capture_bank");
    assert!(release.contains("prefill_capture_bank") && release.contains(".release("));
}
