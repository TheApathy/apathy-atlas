// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the standalone EXL3 W2A8 H128 emitter parity probe.

use std::fs;
use std::path::{Path, PathBuf};

const HARNESS_RELATIVE: &str = "kernels/gb10/experiments/exl3_w2a8_h128_emit_probe.cu";
const BUILD_RELATIVE: &str = "scripts/check-exl3-prefill-w2a8-emitter-probe-build.sh";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn required_source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative)).unwrap_or_else(|error| {
        panic!("required emitter probe source {relative} is missing: {error}")
    })
}

fn marked_section<'a>(source: &'a str, name: &str) -> &'a str {
    marked_section_with_prefix(source, name, "//")
}

fn marked_shell_section<'a>(source: &'a str, name: &str) -> &'a str {
    marked_section_with_prefix(source, name, "#")
}

fn marked_section_with_prefix<'a>(source: &'a str, name: &str, prefix: &str) -> &'a str {
    let begin = format!("{prefix} BEGIN {name}");
    let end = format!("{prefix} END {name}");
    assert_eq!(
        source.matches(&begin).count(),
        1,
        "{name} must have exactly one begin marker"
    );
    assert_eq!(
        source.matches(&end).count(),
        1,
        "{name} must have exactly one end marker"
    );
    let start = source.find(&begin).unwrap() + begin.len();
    let finish = source[start..].find(&end).unwrap() + start;
    assert!(start < finish, "{name} markers are reversed");
    &source[start..finish]
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn assert_exactly_once(source: &str, needle: &str) {
    assert_eq!(
        source.matches(needle).count(),
        1,
        "expected exactly one `{needle}` in contract section"
    );
}

fn assert_before(source: &str, first: &str, second: &str) {
    let first_at = source
        .find(first)
        .unwrap_or_else(|| panic!("missing ordered contract `{first}`"));
    let second_at = source
        .find(second)
        .unwrap_or_else(|| panic!("missing ordered contract `{second}`"));
    assert!(first_at < second_at, "`{first}` must precede `{second}`");
}

#[test]
fn dedicated_harness_has_exact_candidate_and_reference_kernel_contract() {
    let harness = required_source(HARNESS_RELATIVE);
    let symbols = marked_section(&harness, "emitter probe kernel contract");

    for candidate in [
        "exl3_w2a8_h128_pre_dual_emit_h4096",
        "exl3_w2a8_h128_post_silu_pre_emit_h2048",
    ] {
        assert_exactly_once(symbols, candidate);
    }
    for incumbent in [
        "exl3_h128_pre_dual_rows_h4096",
        "exl3_h128_post_silu_pre_rows",
        "per_token_group_quant_fp8",
    ] {
        assert_exactly_once(symbols, incumbent);
    }
    assert!(!symbols.contains("try_kernel"));
    assert!(!symbols.contains("fallback"));
}

#[test]
fn parity_contract_is_byte_exact_for_all_a8_and_scale_outputs() {
    let harness = required_source(HARNESS_RELATIVE);
    let parity = marked_section(&harness, "emitter byte parity contract");
    let compact = compact(parity);

    for field in [
        "pre_gate_fp8_mismatches",
        "pre_gate_scale_mismatches",
        "pre_up_fp8_mismatches",
        "pre_up_scale_mismatches",
        "post_down_fp8_mismatches",
        "post_down_scale_mismatches",
    ] {
        assert_exactly_once(parity, field);
    }
    assert!(
        compact.contains("sizeof(float)"),
        "scale comparison must bind float bytes"
    );
    assert!(
        compact.matches("memcmp(").count() >= 6 || compact.matches("byte_mismatches(").count() >= 6,
        "all six FP8/scale products require byte comparisons"
    );
    assert!(!parity.contains("epsilon"));
    assert!(!parity.contains("tolerance"));
    assert!(!parity.contains("fabs"));
}

#[test]
fn routing_boundary_and_output_ownership_are_adversarial() {
    let harness = required_source(HARNESS_RELATIVE);
    let mapping = marked_section(&harness, "emitter routing contract");
    let compact = compact(mapping);

    for required in [
        "sorted_token_ids",
        "sorted_expert_ids",
        "identity_token_ids",
        "gate_suh_tab",
        "up_suh_tab",
        "gate_svh_tab",
        "up_svh_tab",
        "down_suh_tab",
    ] {
        assert!(
            mapping.contains(required),
            "missing routing input `{required}`"
        );
    }
    for rows in ["1", "31", "32", "33", "63", "64", "65", "129"] {
        assert!(
            compact.contains(&format!("{rows},")) || compact.contains(&format!(",{rows}")),
            "missing row boundary {rows}"
        );
    }
    assert!(mapping.contains("DevicePtr(0)") || mapping.contains("nullptr"));
    assert!(mapping.contains("nonidentity"));
    assert!(mapping.contains("duplicate"));
}

#[test]
fn every_geometry_guard_has_an_independent_poison_canary() {
    let harness = required_source(HARNESS_RELATIVE);
    let geometry = marked_section(&harness, "emitter malformed geometry contract");

    for stage in ["pre", "post"] {
        for axis in [
            "block-x", "block-y", "block-z", "grid-x", "grid-y", "grid-z",
        ] {
            assert_exactly_once(geometry, &format!("{stage}-{axis}"));
        }
        assert_exactly_once(geometry, &format!("{stage}-rows"));
    }
    assert_exactly_once(geometry, "pre-k");
    assert_exactly_once(geometry, "post-n");
    assert!(
        geometry.contains("nullptr"),
        "one invalid launch must prove guards precede dereference"
    );
    assert!(geometry.contains("cudaGetLastError"));
    assert!(geometry.contains("cudaDeviceSynchronize"));
    assert!(geometry.contains("canary=clean"));
}

#[test]
fn dual_poison_contract_covers_data_scales_inactive_rows_and_guards() {
    let harness = required_source(HARNESS_RELATIVE);
    let poison = marked_section(&harness, "emitter poison contract");

    assert_exactly_once(poison, "0xa5");
    assert_exactly_once(poison, "0x5a");
    for region in [
        "prefix_guard",
        "suffix_guard",
        "inactive_rows",
        "active_fp8",
        "active_scales",
        "legacy_bf16",
    ] {
        assert!(poison.contains(region), "missing poison region `{region}`");
    }
    assert!(poison.contains("poison_independent"));
    assert!(poison.contains("all_zero_group"));
    assert!(poison.contains("signed_zero"));
    assert!(poison.contains("bf16_boundary"));
}

#[test]
fn runtime_output_is_bounded_and_has_one_terminal_build_bound_result() {
    let harness = required_source(HARNESS_RELATIVE);
    let output = marked_section(&harness, "emitter bounded output contract");

    for key in [
        "build_id=",
        "device_uuid=",
        "driver=",
        "runtime=",
        "input_hash=",
        "routing_hash=",
        "tables_hash=",
        "pre_cases=",
        "post_cases=",
        "geometry_cases=",
        "mismatches=",
        "guards=",
    ] {
        assert_exactly_once(output, key);
    }
    assert_exactly_once(output, "result=PASS");
    assert_before(output, "build_id=", "result=PASS");
    assert!(
        !output.contains("result=%s"),
        "terminal result must not be an unbounded free-form field"
    );
}

#[test]
fn build_receipt_binds_every_input_and_runner_fail_closes_on_output() {
    let build = required_source(BUILD_RELATIVE);
    let inputs = marked_shell_section(&build, "emitter probe immutable inputs");
    for input in [
        "exl3_w2a8_h128_emit_probe.cu",
        "exl3_w2a8_h128_emit.cu",
        "exl3_gemv.cu",
        "per_token_group_quant_fp8.cu",
        "check-exl3-prefill-w2a8-emitter-probe-build.sh",
    ] {
        assert_exactly_once(inputs, input);
    }

    let receipt = marked_shell_section(&build, "emitter probe receipt contract");
    for field in [
        "git_commit=",
        "git_status_sha256=",
        "source_sha256",
        "compile_command=",
        "binary_sha256",
        "cubin_sha256",
        "build_id=",
    ] {
        assert!(receipt.contains(field), "receipt omits `{field}`");
    }
    assert_before(receipt, "source_sha256", "binary_sha256");

    let runner = marked_shell_section(&build, "emitter probe runner verification");
    for check in [
        "actual_receipt_sha256",
        "actual_binary_sha256",
        "expected_build_id",
        "mismatches=0",
        "guards=clean",
        "result=PASS",
    ] {
        assert!(
            runner.contains(check),
            "runner omits fail-closed check `{check}`"
        );
    }
    assert!(runner.contains("exit 2") || runner.contains("exit 1"));

    let receipt_write = build
        .find("receipt_file=")
        .expect("receipt publication assignment");
    let revalidation = build
        .find("build input changed during W2A8 emitter probe compilation")
        .expect("post-build input revalidation");
    assert!(
        revalidation < receipt_write,
        "inputs must be revalidated before receipt publication"
    );
}

#[test]
fn harness_and_build_stay_experimental_and_out_of_model_registry() {
    let harness = Path::new(HARNESS_RELATIVE)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    let build = Path::new(BUILD_RELATIVE)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    assert!(workspace().join(HARNESS_RELATIVE).is_file());
    assert!(workspace().join(BUILD_RELATIVE).is_file());
    assert!(
        !workspace()
            .join("kernels/gb10/common")
            .join(harness)
            .exists()
    );
    assert!(
        !workspace()
            .join("kernels/gb10/deepseek-v4-flash/nvfp4")
            .join(harness)
            .exists()
    );
    assert!(!workspace().join("scripts").join(harness).exists());
    assert_eq!(build, "check-exl3-prefill-w2a8-emitter-probe-build.sh");
}
