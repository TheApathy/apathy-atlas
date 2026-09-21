// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the immutable fused-N128 double-buffer W2A8 A/B probe.

use std::fs;
use std::path::PathBuf;

const HARNESS: &str = "kernels/gb10/experiments/exl3_w2a8_composed_n128_n256_probe.cu";
const BUILDER: &str =
    "scripts/check-exl3-prefill-w2a8-fused-gu-n128-double-buffer-ab-probe-build.sh";
const N128: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative)).unwrap_or_else(|error| {
        panic!("missing fused-N128 double-buffer A/B input {relative}: {error}")
    })
}

fn marked<'a>(source: &'a str, name: &str, prefix: &str) -> &'a str {
    let begin = format!("{prefix} BEGIN {name}");
    let end = format!("{prefix} END {name}");
    assert_eq!(
        source.matches(&begin).count(),
        1,
        "missing/duplicate {begin}"
    );
    assert_eq!(source.matches(&end).count(), 1, "missing/duplicate {end}");
    let start = source.find(&begin).unwrap() + begin.len();
    let finish = source[start..].find(&end).unwrap() + start;
    &source[start..finish]
}

#[test]
fn candidate_is_one_default_off_boolean_switch_in_only_the_fused_n128_component() {
    let component = source(N128);
    for contract in [
        "#ifndef W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE",
        "#define W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE 0",
        "static_assert(W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE == 0 ||",
        "#if W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE",
        "BEGIN double-buffer K64 async stage",
        "BEGIN double-buffer K128 schedule",
    ] {
        assert!(component.contains(contract), "component omits `{contract}`");
    }
}

#[test]
fn production_shape_probe_covers_exact_outputs_hashes_redzones_and_bad_routes() {
    let harness = source(HARNESS);
    for contract in [
        "constexpr unsigned kTokens = 2410",
        "constexpr unsigned kRows = 14460",
        "constexpr unsigned kExperts = 256",
        "constexpr unsigned kHidden = 4096",
        "constexpr unsigned kIntermediate = 2048",
        "constexpr std::size_t kGuardBytes = 256",
        "checkpoint-like-synthetic-v1",
        "intermediate_fp8=exact intermediate_scale=exact",
        "raw_bf16=exact final_bf16=exact production_alias=exact",
        "input_hash=%016llx",
        "output_hash=%016llx",
        "malformed_cases=%u all_offsets=validated",
    ] {
        assert!(harness.contains(contract), "harness omits `{contract}`");
    }

    let malformed = marked(&harness, "guards inputs and malformed no-write", "//");
    assert!(malformed.matches("bad_offsets[0] = 1").count() >= 2);
    assert!(
        malformed
            .matches("bad_offsets[kExperts - 1] = static_cast<int>(kRows + 1)")
            .count()
            >= 2
    );
    for contract in [
        "assert_all_byte(f.candidate.down_fp8, kPoisonA",
        "assert_all_byte(f.candidate.down_scale, kPoisonA",
        "assert_all_byte(f.candidate.raw_down, kPoisonB",
        "f.verify_immutable_inputs()",
        "f.guards_clean()",
        "return 16",
    ] {
        assert!(
            malformed.contains(contract),
            "malformed probe omits `{contract}`"
        );
    }
}

#[test]
fn builder_freezes_packed_e4m3_and_changes_only_the_double_buffer_selector() {
    let build = source(BUILDER);
    let threshold = marked(&build, "fused N128 double buffer A/B threshold", "#");
    assert!(threshold.contains("min_speedup >= 1.01"));
    assert!(threshold.contains("%.17g"));
    assert!(threshold.contains("NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS"));

    let variants = marked(&build, "fused N128 double buffer two binary build", "#");
    for proof in [
        "-DW2A8_PACKED_E4M3_CANDIDATE=1",
        "-DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0",
        "#define W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE %s",
        "<four production wrappers> <generated fused A/B wrapper>",
        "build_variant incumbent 0",
        "build_variant candidate 1",
        "only_ab_compile_factor=W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE",
        "-DW2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP=0",
        "compile_command_sha256",
        "binary_sha256",
        "cubin_sha256",
        "resource_sha256",
        "sass_sha256",
        "LDGSTS",
        "LDGDEPBAR",
        "DEPBAR\\.LE SB0, 0x0",
        "QMMA\\.16832\\.F32\\.E4M3\\.E4M3",
        "different_cubins == 1",
        "changed_n128_sass=PASS",
        "unchanged_n256_sass=PASS",
    ] {
        assert!(variants.contains(proof), "build evidence omits `{proof}`");
    }
}

#[test]
fn generated_runner_is_fail_closed_and_owns_the_only_speed_threshold() {
    let build = source(BUILDER);
    let runner = marked(&build, "fused N128 double buffer generated A/B runner", "#");
    for proof in [
        "expected_receipt_sha256",
        "artifact_sha256",
        "timeout",
        "ulimit -f",
        "order=(incumbent candidate candidate incumbent)",
        "candidate_ms",
        "input_hash",
        "output_hash",
        "guards=clean inputs=immutable",
        "malformed_cases=16 all_offsets=validated",
        "intermediate_fp8=exact intermediate_scale=exact",
        "raw_bf16=exact final_bf16=exact production_alias=exact",
        "variant_outputs=exact",
        "double_buffer_speedup",
        "speedup < threshold",
        "enforcement=parity-only",
        "preserve_failure",
        "failed-run-",
        "result=PASS",
    ] {
        assert!(runner.contains(proof), "runner omits `{proof}`");
    }
    assert!(runner.contains("stdout_bytes -gt 8192"));
    assert!(runner.contains("stderr_bytes -ne 0"));
    assert!(runner.contains("stdout_lines -ne 13"));
    assert!(runner.contains("timed=whole-chain"));
    assert!(!runner.contains("nvidia-smi"));
}
