// SPDX-License-Identifier: AGPL-3.0-only

//! RED production seams; callback behavior is covered by the companion tests.
//! These guards are not GPU numerical, completion, or performance evidence.

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

fn section<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
    text.split_once(start)
        .expect("existing start seam")
        .1
        .split_once(end)
        .expect("existing end seam")
        .0
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
fn setting_is_registered_and_pinned_before_model_allocation_not_per_proposal() {
    let modules = compact(&source("mod.rs"));
    assert!(modules.contains("modphase_timing;"));
    assert!(modules.contains("modphase_timing_wrappers;"));
    let target = compact(&source("target_model_exl3.rs"));
    assert!(target.contains("phase_timing:TimingSetting"));
    let constructor = section(&target, "fnnew_inner(", "fnpropose_dflash2(");
    assert!(constructor.contains("std::env::var_os(\"ATLAS_GLM53_PHASE_TIMING\")"));
    assert!(constructor.contains(".as_deref()"));
    before(
        constructor,
        "TimingSetting::parse(",
        "build_moe_pointer_tables(",
    );
    let proposal = section(&target, "fnpropose_dflash2(", "fnclaim_sequence(");
    assert!(
        !proposal.contains("std::env::"),
        "startup choice cannot drift during requests"
    );
}

#[test]
fn served_proposal_is_measured_without_changing_runtime_path_or_completion() {
    let target = compact(&source("target_model_exl3.rs"));
    let proposal = section(&target, "fnpropose_dflash2(", "fnclaim_sequence(");
    assert!(proposal.contains("Phase::Proposal"));
    assert!(proposal.contains("self.phase_timing"));
    assert_eq!(
        proposal
            .matches("runtime.propose(self,anchor,stream)")
            .count(),
        1
    );
    for forbidden in [
        ".synchronize(",
        ".copy_d2h",
        ".copy_h2d",
        ".copy_d2d",
        "propose_diagnostic",
        "propose_graph",
        "cudaEvent",
    ] {
        assert!(
            !proposal.contains(forbidden),
            "new proposal timing effect: {forbidden}"
        );
    }
}

#[test]
fn actual_policy_transaction_uses_forwarding_wrappers_and_one_outer_wall_scope() {
    let execution = compact(&source("dsa_verify_execution.rs"));
    let served = section(
        &execution,
        "fnverify_dflash2_with_policy(",
        "fnbind_dsa_snapshot(",
    );
    for needle in [
        "Phase::VerifyTotal",
        "Phase::StageTotal",
        "TimedLogits::new(",
        "TimedPolicy::new(",
        "TimedCommit::new(",
        "PolicyTarget::stage(",
    ] {
        assert!(
            served.contains(needle),
            "missing served timing seam: {needle}"
        );
    }
    assert_eq!(served.matches("run_verify_policy_transaction(").count(), 1);
    before(served, "consume_policy_request(", "Phase::VerifyTotal");
    before(served, "Phase::StageTotal", "PolicyTarget::stage(");
    for forbidden in [
        ".synchronize(",
        ".copy_d2h",
        ".copy_h2d",
        "ATLAS_GLM53_WIDE_TIMING",
        "with_exact_wide_prefill",
        "apply_penalties",
        "argmax_",
    ] {
        assert!(
            !served.contains(forbidden),
            "timing cannot add work or replace policy: {forbidden}"
        );
    }
}

#[test]
fn internal_save_stage_restore_replay_and_finish_have_distinct_existing_boundaries() {
    let policy = compact(&source("dsa_policy_execution.rs"));
    for phase in [
        "SnapshotSaveEnqueue",
        "WideStage",
        "PartialRestore",
        "FullCommitBody",
        "PartialReplay",
        "FinishFence",
        "FailureDrain",
    ] {
        assert!(
            policy.contains(&format!("Phase::{phase}")),
            "missing internal phase {phase}"
        );
    }
    before(&policy, "Phase::SnapshotSaveEnqueue", ".snapshot.save(");
    before(&policy, "Phase::WideStage", "model.verify_tokens_staged(");
    before(&policy, "Phase::PartialRestore", "self.snapshot.restore(");
    before(&policy, "Phase::FinishFence", "self.snapshot.finish(");
    // Preserve the shipping effects; timers wrap these operations, never add a
    // fence to make an enqueue bucket look like separately completed GPU time.
    for (needle, count) in [
        (".snapshot.save(", 1),
        (".snapshot.restore(", 2),
        (".snapshot.finish(", 1),
        (".snapshot.fail_commit(", 1),
        (".synchronize(", 1),
        (".observe_target_rows(", 1),
        ("self.model.walk(token,self.stream)", 1),
    ] {
        assert_eq!(
            policy.matches(needle).count(),
            count,
            "changed existing effect {needle}"
        );
    }
    assert!(!policy.contains("ATLAS_GLM53_WIDE_TIMING"));
    assert!(
        policy.contains("if!self.closed{self.model.poison_verify(self.stream);}"),
        "existing target Drop remains the abandonment/poison authority"
    );
}

#[test]
fn timing_receipts_are_explicit_diagnostics_with_correlation_and_inclusive_labels() {
    let files = [
        source("phase_timing.rs"),
        source("phase_timing_wrappers.rs"),
        source("target_model_exl3.rs"),
        source("dsa_verify_execution.rs"),
        source("dsa_policy_execution.rs"),
    ]
    .join("\n");
    assert!(files.contains("atlas.glm53.phase_timing.v1"));
    for field in [
        "host_wall_only",
        "inclusive",
        "performance_eligible",
        "may_include_prior_capture",
        "generation",
        "readback_bytes",
        "accepted",
        "stream",
    ] {
        assert!(files.contains(field), "missing receipt contract {field}");
    }
    // Receipt acquisition itself does not sample the clock (behavior test),
    // and logging belongs after measured callbacks, never in wrapper Drop.
    let wrappers = source("phase_timing_wrappers.rs");
    assert!(!wrappers.is_empty());
    for forbidden in [
        "tracing::",
        "eprintln!",
        "println!",
        "impl Drop",
        "catch_unwind",
    ] {
        assert!(
            !wrappers.contains(forbidden),
            "wrapper changes callback boundary: {forbidden}"
        );
    }
}

#[test]
fn helpers_are_cpu_only_and_cannot_add_gpu_copies_events_or_cleanup() {
    for name in ["phase_timing.rs", "phase_timing_wrappers.rs"] {
        let helper = source(name);
        assert!(!helper.is_empty(), "missing {name}");
        for forbidden in [
            "GpuBackend",
            "DevicePtr",
            "KernelLaunch",
            "cudaEvent",
            ".synchronize(",
            ".copy_d2h",
            ".copy_h2d",
            ".copy_d2d",
            "unsafe {",
            "catch_unwind",
            "ATLAS_GLM53_WIDE_TIMING",
        ] {
            assert!(!helper.contains(forbidden), "{name} introduces {forbidden}");
        }
    }
}

#[test]
fn public_transaction_traits_and_selection_remain_independent_of_instrumentation() {
    let transaction = source("verify_policy_transaction.rs");
    assert!(!transaction.is_empty());
    for forbidden in [
        "phase_timing",
        "Instant",
        "Clock",
        "TimingSetting",
        "cudaEvent",
    ] {
        assert!(
            !transaction.contains(forbidden),
            "public transaction changed for {forbidden}"
        );
    }
    let compact = compact(&transaction);
    for signature in [
        "pubtraitLogitsIo{",
        "pubtraitVerifyPolicy{",
        "pubtraitVerifyCommitIo{",
        "fncopy_logits(&mutself,destination:&mut[u8])->Result<usize>;",
        "fncommit_prefix(&mutself,request:&VerifyRequest,rows:usize)->Result<usize>;",
        "fnabort_staged(&mutself)->Result<()>;",
        "fnpoison(&mutself);",
    ] {
        // The source has documentation inside trait bodies, so check signatures
        // rather than constraining the comments or the complete body spelling.
        assert!(
            compact.contains(signature),
            "public policy ABI drift: {signature}"
        );
    }
}
