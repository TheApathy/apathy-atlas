// SPDX-License-Identifier: AGPL-3.0-only

//! Executable diagnostic wiring RED, not numerical or GPU lifetime proof.
//! Real equal-input raw-state/output/draft comparisons remain a separate gate.

use std::{fs, path::PathBuf};

fn source(name: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)).unwrap_or_default()
}

fn example(name: &str) -> String {
    source(&format!("examples/glm53_partial_replay_parity/{name}"))
}

fn compact(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .flat_map(str::chars)
        .filter(|c| !c.is_whitespace())
        .collect()
}

fn item<'a>(text: &'a str, needle: &str) -> &'a str {
    let start = text
        .find(needle)
        .unwrap_or_else(|| panic!("missing {needle}"));
    let open = start + text[start..].find('{').expect("item body");
    let mut depth = 0usize;
    for (at, byte) in text.as_bytes().iter().enumerate().skip(open) {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &text[start..=at];
                }
            }
            _ => {}
        }
    }
    panic!("unclosed {needle}");
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

fn all_example_sources() -> String {
    let mut text = source("examples/glm53_partial_replay_parity.rs");
    for file in ["session.rs", "snapshots.rs", "case.rs", "artifacts.rs"] {
        text.push_str(&example(file));
    }
    compact(&text)
}

fn call_arguments<'a>(text: &'a str, name: &str) -> Vec<&'a str> {
    let marker = format!("{name}(");
    text.match_indices(&marker)
        .map(|(start, _)| {
            let begin = start + marker.len();
            let mut depth = 1usize;
            for (at, byte) in text.as_bytes().iter().enumerate().skip(begin) {
                match byte {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            return &text[begin..at];
                        }
                    }
                    _ => {}
                }
            }
            panic!("unclosed call {name}");
        })
        .collect()
}

#[test]
fn executable_is_feature_gated_and_requires_explicit_candidate_admission() {
    let cargo = compact(&source("Cargo.toml"));
    let entry = cargo
        .split("[[example]]")
        .find(|entry| entry.contains("name=\"glm53_partial_replay_parity\""))
        .expect("missing executable registration");
    assert!(entry.contains("required-features=[\"cuda\",\"gpu-examples\"]"));
    let main = compact(&source("examples/glm53_partial_replay_parity.rs"));
    for child in [
        "session.rs",
        "snapshots.rs",
        "case.rs",
        "forced_policy.rs",
        "compare.rs",
        "artifacts.rs",
    ] {
        assert!(
            main.contains(child),
            "missing actual executable child {child}"
        );
    }
    assert!(main.contains("run_case("));
    assert!(main.contains("ATLAS_GLM53_PARTIAL_WIDE_REPLAY"));
    assert!(
        main.contains("ATLAS_GLM53_DFLASH2_KV_PREFIX"),
        "admit prefix0 because these state regions exclude drafter KV cache"
    );
    assert!(
        main.contains("ensure!("),
        "required candidate mode must be checked, not set"
    );
    assert!(
        !main.contains("set_var("),
        "do not change model mode behind the caller"
    );
    for argument in ["positions", "rows", "corpus", "target", "draft", "output"] {
        assert!(
            main.contains(argument),
            "missing explicit probe argument {argument}"
        );
    }
}

#[test]
fn session_installs_one_drafter_inside_the_whole_model_owner_envelope() {
    let session = compact(&example("session.rs"));
    let owner = item(&session, "structSession");
    for field in [
        "model:ManuallyDrop<Glm53Exl3Model>",
        "seq:Option<SequenceState>",
        "stream:u64",
        "logits:DevicePtr",
    ] {
        assert!(owner.contains(field), "missing real session owner {field}");
    }
    assert_eq!(session.matches(".install_dflash2(").count(), 1);
    before(&session, "ManuallyDrop::new(", ".install_dflash2(");
    assert!(!all_example_sources().contains("Glm53Dflash2Runtime::load("));
    assert!(session.contains(".alloc_sequence("));
    let drop = item(&session, "implDropforSession");
    for forbidden in [
        ".gpu(",
        ".synchronize(",
        ".free(",
        ".free_sequence(",
        "ManuallyDrop::take(",
    ] {
        assert!(
            !drop.contains(forbidden),
            "abandoned Drop must retain owners: {forbidden}"
        );
    }
    let close = item(&session, "fnclose(");
    before(close, "catch_unwind(", ".synchronize(");
    assert!(close.contains("mem::forget(self)"));
    before(close, ".synchronize(", "ManuallyDrop::take(");
    before(close, "ManuallyDrop::take(", ".free(");
    assert!(
        close.contains("Err("),
        "completion error/panic cannot fall through to teardown"
    );
}

#[test]
fn snapshots_enumerate_all_model_regions_and_use_bounded_owned_readback_chunks() {
    let snapshots = compact(&example("snapshots.rs"));
    assert!(snapshots.contains(".state_probe("));
    assert!(snapshots.contains(".regions()?"));
    assert!(snapshots.contains(".region_bytes("));
    assert!(snapshots.contains(".read("));
    assert!(
        snapshots.contains(".checked_add("),
        "aggregate raw allocation must be checked"
    );
    let numbers = snapshots.replace('_', "");
    assert!(
        numbers.contains("65534"),
        "logical chunk leaves room for the aligned cover"
    );
    assert!(numbers.contains("256*1024*1024") || numbers.contains("268435456"));
    assert!(
        snapshots.contains("BTreeMap"),
        "retain complete named region maps"
    );
    let captures = item(&snapshots, "fncapture_rows(");
    assert!(captures.contains("StateProbeRegion::Capture"));
    assert!(captures.contains(".region_bytes("));
    assert!(captures.contains(".read("));
    assert!(
        captures.contains("for"),
        "capture every fresh row/tap, not only the final row"
    );
    assert!(
        !snapshots.contains(".filter("),
        "do not discard inconvenient regions"
    );
    assert!(!snapshots.contains(".filter_map("));
}

#[test]
fn case_uses_actual_policy_binding_transaction_and_host_publication() {
    let case = compact(&example("case.rs"));
    let run = item(&case, "fnrun_case(");
    assert!(run.contains("ForcedPrefixPolicy::new("));
    assert!(run.contains(".bind_verify_policy_request("));
    assert!(run.contains(".decode_verify_with_policy("));
    assert!(run.contains("VerifyOutcome::Committed("));
    before(
        run,
        ".bind_verify_policy_request(",
        ".decode_verify_with_policy(",
    );
    before(run, ".decode_verify_with_policy(", ".publish(");
    assert!(run.contains(".accepted_drafts()"));
    assert!(run.contains(".publication_matches(&emitted,published.terminal())"));
    assert!(run.contains("&&publication_exact"));
    before(
        run,
        "output.join(\"publication.json\")",
        "letcandidate_captures=",
    );
    before(
        run,
        "output.join(\"frames.json\")",
        "let(candidate_next,candidate_drafts)=continuation(",
    );
    assert!(run.contains(".seq_len"));
    assert!(run.contains(".kv_valid_tokens"));
    assert!(run.contains(".tokens"));
    assert!(run.contains("capture_rows("));
    assert!(run.contains("natural_acceptance\":false"));
    assert!(run.contains("forced"));
}

#[test]
fn scalar_capture_precedes_scratch_reuse_and_candidate_keeps_chronological_rows() {
    let case = compact(&example("case.rs"));
    let run = item(&case, "fnrun_case(");
    assert!(
        run.contains(".decode("),
        "ordinary baseline must execute actual scalar decode"
    );
    before(run, ".decode(", "capture_rows(");
    assert!(
        call_arguments(run, "capture_rows").iter().any(|args| args
            .trim_end_matches(',')
            .rsplit(',')
            .next()
            == Some("1")),
        "scalar baseline must read the fresh row immediately"
    );
    assert!(
        run.matches("capture_rows(").count() >= 2,
        "read both ordinary and candidate captures"
    );
    assert!(
        all_example_sources().contains(".prefill("),
        "both arms need a fresh prefix"
    );
    assert!(
        !run.contains(".logits_buffer_ptr()"),
        "old wide scratch row is not the next-token oracle"
    );
    assert!(!run.contains("(rows-1)*154"));
}

#[test]
fn complete_state_capture_next_logits_and_actual_drafts_are_compared_without_new_oracle() {
    let case = compact(&example("case.rs"));
    let run = item(&case, "fnrun_case(");
    let all = all_example_sources();
    assert!(
        run.matches("record_comparison(").count() == 4,
        "persist all four whole-map gates through the diagnostic I/O boundary"
    );
    let artifacts = compact(&example("artifacts.rs"));
    assert!(artifacts.contains("compare_snapshots(reference,candidate)?"));
    before(
        run,
        "letstate=record_comparison(",
        "let(candidate_next,candidate_drafts)=continuation(",
    );
    before(
        run,
        "letcapture=record_comparison(",
        "let(candidate_next,candidate_drafts)=continuation(",
    );
    for gate in ["before", "state", "capture", "next", "draft"] {
        assert!(run.contains(gate), "case report omits explicit {gate} gate");
    }
    assert!(
        all.contains(".run_mtp_propose_multi("),
        "exercise the installed real proposal route"
    );
    assert!(
        all.contains("StateProbeRegion::Logits"),
        "next-decode logits use typed model readback"
    );
    assert!(!all.contains("propose_installed_diagnostic"));
    assert!(!all.contains("propose_diagnostic("));
    assert!(!run.contains("different_bytes<=") && !run.contains("relative_l2<"));
    let comparator = compact(&example("compare.rs"));
    assert!(comparator.contains(".keys().eq("));
    assert!(comparator.contains("a.len()==b.len()"));
    assert!(comparator.contains("different_bytes==0"));
}

#[test]
fn failing_case_keeps_its_directory_and_error_without_erasing_completed_gates() {
    let main = compact(&source("examples/glm53_partial_replay_parity.rs"));
    assert!(main.contains("create_dir(&case_output)?"));
    assert!(main.contains("case_output.join(\"error.json\")"));
    before(&main, "create_dir(&case_output)?", "case::run_case(");
    assert!(!all_example_sources().contains("remove_file("));
    assert!(!all_example_sources().contains("remove_dir"));
}

#[test]
fn example_has_no_raw_device_readback_or_another_unretained_readback_owner() {
    let all = all_example_sources();
    assert!(!all.is_empty(), "missing executable source");
    for forbidden in [
        "copy_d2h",
        "cuMemcpyDtoH",
        "cuMemcpyDtoHAsync",
        "OwnedReadback::new(",
        "implProbeIo",
        "implReadbackIo",
        "copy_policy_logits(",
        "copy_dflash_capture_row",
    ] {
        assert!(
            !all.contains(forbidden),
            "example bypasses model-owned typed authority: {forbidden}"
        );
    }
}
