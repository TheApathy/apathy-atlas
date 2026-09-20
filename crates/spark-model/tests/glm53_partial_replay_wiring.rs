// SPDX-License-Identifier: AGPL-3.0-only

//! RED integration seams only; the real-state/numerical GPU gate is separate.

use std::{fs, path::PathBuf};

fn source(name: &str) -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/model/glm53")
            .join(name),
    )
    .unwrap_or_default()
}

fn compact(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

fn body<'a>(text: &'a str, name: &str) -> &'a str {
    let start = text
        .find(&format!("fn{name}("))
        .unwrap_or_else(|| panic!("missing production method {name}"));
    let open = start + text[start..].find('{').expect("method body");
    let mut depth = 0usize;
    for (at, byte) in text.as_bytes().iter().enumerate().skip(open) {
        if *byte == b'{' {
            depth += 1;
        }
        if *byte == b'}' {
            depth -= 1;
            if depth == 0 {
                return &text[start..=at];
            }
        }
    }
    panic!("unclosed production method {name}");
}

fn before(text: &str, first: &str, second: &str) {
    let a = text
        .find(first)
        .unwrap_or_else(|| panic!("missing {first}"));
    let b = text
        .find(second)
        .unwrap_or_else(|| panic!("missing {second}"));
    assert!(a < b, "{first} must precede {second}");
}

#[test]
fn private_setting_is_pinned_at_startup_before_allocations_not_in_replay() {
    let modules = compact(&source("mod.rs"));
    assert!(modules.contains("modpartial_replay;"));
    assert!(modules.contains("modordered_capture;"));
    let target = compact(&source("target_model_exl3.rs"));
    assert!(target.contains("partial_replay:PartialReplaySetting"));
    assert!(target.contains("#[path=\"target_partial_replay.rs\"]modtarget_partial_replay;"));
    let constructor = body(&target, "new_inner");
    assert!(
        constructor.contains("std::env::var_os(\"ATLAS_GLM53_PARTIAL_WIDE_REPLAY\").as_deref()")
    );
    before(
        constructor,
        "PartialReplaySetting::parse(",
        "build_moe_pointer_tables(",
    );
    let policy = compact(&source("dsa_policy_execution.rs"));
    assert!(!body(&policy, "commit_prefix").contains("std::env::"));
    assert!(!source("target_partial_replay.rs").contains("std::env::"));
}

#[test]
fn partial_dispatch_is_validated_before_effects_and_latches_initial_exact_verify_mode() {
    let policy = compact(&source("dsa_policy_execution.rs"));
    assert!(policy.contains("exact_verify:bool"));
    let stage = body(&policy, "stage");
    assert!(
        stage.contains("exact_verify,"),
        "store the same initially admitted mode"
    );
    let commit = body(&policy, "commit_prefix");
    before(commit, ".partial_replay.path(", ".snapshot.restore(");
    before(commit, ".partial_replay.path(", ".snapshot.begin_commit(");
    assert!(commit.contains("ReplayPath::Wide"));
    assert!(commit.contains("ReplayPath::Scalar"));
    assert!(
        commit.contains("&request.inputs()[..rows]"),
        "never replay the rejected tail"
    );
    assert!(
        commit.contains("self.exact_verify"),
        "wide replay must not silently choose another mode"
    );
    assert!(commit.contains(".replay_partial_wide("));
    assert_eq!(
        commit.matches("self.model.walk(token,self.stream)").count(),
        1,
        "the disabled/row1 scalar path must remain present"
    );
    assert_eq!(
        commit
            .matches(".observe_target_rows(self.model,rowsasu32,self.stream)")
            .count(),
        1,
        "ordinary full acceptance must retain its existing capture path"
    );
}

#[test]
fn full_commit_projects_ordered_only_in_the_latched_exact_mode() {
    let policy = compact(&source("dsa_policy_execution.rs"));
    let commit = body(&policy, "commit_prefix");
    let full = commit
        .split("timing.measure(Phase::FullCommitBody,")
        .nth(1)
        .expect("full commit phase")
        .split("timing.measure(Phase::PartialReplay,")
        .next()
        .unwrap();
    assert!(full.contains("ifself.exact_verify{"));
    let exact = full.split("ifself.exact_verify{").nth(1).unwrap();
    let (ordered, general) = exact.split_once("}else{").expect("retain general mode");
    assert!(ordered.contains(".observe_target_rows_ordered(self.model,rowsasu32,self.stream)?"));
    assert!(!ordered.contains(".observe_target_rows("));
    assert!(general.contains(".observe_target_rows(self.model,rowsasu32,self.stream)?"));
    assert!(!general.contains(".observe_target_rows_ordered("));
    before(full, ".commit_accepted(", ".synchronize(");
    before(full, ".synchronize(", "ifself.exact_verify{");
    before(
        full,
        "position=u32::try_from(request.start()+rows)?",
        "ifself.exact_verify{",
    );
    assert!(!full.contains("std::env::"));
    assert!(commit.contains("ifletErr(error)=committed{self.poison();"));
    before(commit, "ifletErr(error)=committed", ".snapshot.finish(");
}

#[test]
fn wide_replay_uses_existing_staged_core_commit_and_ordered_capture_without_nested_transaction() {
    let helper = compact(&source("target_partial_replay.rs"));
    let replay = body(&helper, "replay_partial_wide");
    assert!(replay.contains("exact_verify:bool"));
    assert!(replay.contains("with_glm53_exact_verify("));
    assert!(replay.contains(".verify_tokens_staged("));
    assert!(replay.contains(".observe_target_rows_ordered("));
    before(replay, ".verify_tokens_staged(", ".commit_accepted(");
    before(replay, ".commit_accepted(", ".synchronize(");
    before(replay, ".synchronize(", ".observe_target_rows_ordered(");
    for forbidden in [
        "bind_dsa_snapshot",
        "DsaVerifyPlan",
        "VerifySnapshot",
        "PolicyTarget::stage",
        "verify_dflash2_transaction",
        "verify_dflash2_with_policy",
        "with_exact_wide_prefill",
        "with_glm53_layer_major_prefill",
        "apply_penalties",
        "argmax_",
    ] {
        assert!(
            !replay.contains(forbidden),
            "wide replay introduces {forbidden}"
        );
    }
}

#[test]
fn ordered_runtime_adapter_projects_m1_rows_then_fences_before_cursor_publication() {
    let runtime = compact(&source("dflash2_runtime.rs"));
    assert!(runtime.contains("#[path=\"dflash2_ordered_capture.rs\"]modordered_capture;"));
    let adapter = compact(&source("dflash2_ordered_capture.rs"));
    let observe = body(&adapter, "observe_target_rows_ordered");
    assert!(observe.contains("OrderedCapturePlan::new("));
    assert!(observe.contains("self.ensure_capture_kv_ready()?"));
    before(observe, "OrderedCapturePlan::new(", ".execute(");
    before(observe, ".execute(", "self.context_tokens=end");
    assert!(!observe.contains("self.context_tokens+="));
    let project = body(&adapter, "project_row");
    assert!(
        project.contains(
            ".project_capture_rows(self.target,capture_row,1,destination_position,stream)"
        ),
        "ordered path must call the actual shared FC/RMS function with M=1"
    );
    assert!(!project.contains("context_tokens="));
    assert!(!project.contains("context_tokens+="));
    assert!(body(&adapter, "fence").contains(".synchronize(stream)"));
    assert!(
        !adapter.contains("dense("),
        "do not duplicate projection arithmetic in the adapter"
    );
    assert!(!adapter.contains("ops::rms_norm("));
}

#[test]
fn existing_capture_routes_share_one_unchanged_fc_rms_projection_block() {
    let runtime = compact(&source("dflash2_runtime.rs"));
    assert!(
        body(&runtime, "observe_target")
            .contains("self.observe_target_rows_inner(target,0,1,stream)")
    );
    assert!(
        body(&runtime, "observe_target_rows")
            .contains("self.observe_target_rows_inner(target,0,rows,stream)")
    );
    let old_inner = body(&runtime, "observe_target_rows_inner");
    assert!(old_inner.contains(
        "self.project_capture_rows(target,first_capture_row,rows,self.context_tokens,stream)"
    ));
    before(
        old_inner,
        ".project_capture_rows(",
        "self.context_tokens+=rows",
    );
    let project = body(&runtime, "project_capture_rows");
    assert!(project.contains(
        "target.copy_dflash_capture_rows(first_capture_row,rows,self.capture_input,stream)"
    ));
    assert_eq!(project.matches("dense(").count(), 1);
    assert_eq!(project.matches("ops::rms_norm(").count(), 1);
    assert!(project.contains("self.weights.fc.weight"));
    assert!(project.contains("rows,HIDDEN,5*HIDDEN,stream"));
    assert!(project.contains("&self.weights.hidden_norm"));
    assert!(project.contains("rows,HIDDEN,1.0e-5,stream"));
    assert!(project.contains("destination_position"));
    assert!(!project.contains(".synchronize("));
    assert!(!project.contains("context_tokens="));
    assert!(!project.contains("context_tokens+="));
}

#[test]
fn existing_failure_owner_and_public_policy_remain_authoritative() {
    let policy = compact(&source("dsa_policy_execution.rs"));
    let commit = body(&policy, "commit_prefix");
    before(commit, "self.poison()", ".snapshot.fail_commit(");
    before(commit, ".snapshot.finish(", "self.closed=true");
    assert!(policy.contains("if!self.closed{self.model.poison_verify(self.stream);}"));
    let transaction = source("verify_policy_transaction.rs");
    for forbidden in [
        "PartialReplaySetting",
        "ReplayPath",
        "OrderedCapture",
        "partial_replay",
    ] {
        assert!(
            !transaction.contains(forbidden),
            "private replay leaked into public policy: {forbidden}"
        );
    }
}
