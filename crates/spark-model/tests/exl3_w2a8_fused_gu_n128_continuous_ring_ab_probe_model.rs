// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the immutable fused-N128 continuous-ring W2A8 A/B probe.

use std::fs;
use std::path::PathBuf;

const HARNESS: &str = "kernels/gb10/experiments/exl3_w2a8_composed_n128_n256_probe.cu";
const BUILDER: &str =
    "scripts/check-exl3-prefill-w2a8-fused-gu-n128-continuous-ring-ab-probe-build.sh";
const N128: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative)).unwrap_or_else(|error| {
        panic!("missing fused-N128 continuous-ring A/B input {relative}: {error}")
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
fn ring_is_one_default_off_boolean_switch_dependent_on_the_prior_double_buffer() {
    let component = source(N128);
    for contract in [
        "#ifndef W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE",
        "#define W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE 0",
        "static_assert(W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE == 0 ||",
        "static_assert(W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE == 1,",
        "static_assert(W2A8_PACKED_E4M3_CANDIDATE == 1,",
        "BEGIN continuous K64 ring initial publication",
        "BEGIN continuous K64 ring schedule",
    ] {
        assert!(component.contains(contract), "component omits `{contract}`");
    }
}

#[test]
fn builder_generates_receipt_bound_ring_arms_outside_the_production_wrapper() {
    let build = source(BUILDER);
    let wrapper = marked(&build, "generated experiment-only fused N128 wrapper", "#");
    for proof in [
        "fused_experiment_wrapper=",
        "GENERATED_FUSED_N128_WRAPPER",
        "#define W2A8_FIXED_N 2048",
        "#define W2A8_FIXED_K 4096",
        "#define W2A8_KERNEL_NAME exl3_w2a8_fused_gu_down_emit_n128",
        "W2A8_PACKED_E4M3_CANDIDATE must be receipt-bound",
        "W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE must be receipt-bound",
        "#if W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE != 1",
        "W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE must remain fixed at 1",
        "W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE must be receipt-bound",
        "#include \"$fused_component\"",
        "generated_wrapper_sha256",
    ] {
        assert!(
            wrapper.contains(proof),
            "experiment wrapper omits `{proof}`"
        );
    }
    assert!(!wrapper.contains("#define W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE"));
    assert!(!wrapper.contains("#define W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE"));
    assert!(!wrapper.contains("$selector"));

    let variants = marked(&build, "continuous ring two binary build", "#");
    assert!(variants.contains("\"$fused_experiment_wrapper\""));
    assert!(!variants.contains("\"$fused_wrapper\""));
    assert!(!variants.contains("-DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=$selector"));
    assert!(!variants.contains("-DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=\"$selector\""));
    for forbidden in [
        "-DW2A8_PACKED_E4M3_CANDIDATE=0",
        "-DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=0",
        "-DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=0",
        "-DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=1",
    ] {
        assert!(!variants.contains(forbidden), "build admits `{forbidden}`");
    }
    assert_eq!(
        variants
            .matches("-DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=$selector")
            .count(),
        1
    );
    assert_eq!(
        variants
            .matches("-DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=\"$selector\"")
            .count(),
        1
    );
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
fn builder_freezes_prior_promotions_and_changes_only_the_ring_selector() {
    let build = source(BUILDER);
    let threshold = marked(&build, "continuous ring A/B threshold", "#");
    assert!(threshold.contains("min_speedup >= 1.01"));
    assert!(threshold.contains("%.17g"));
    assert!(threshold.contains("NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS"));

    let inputs = marked(&build, "continuous ring A/B immutable inputs", "#");
    for dependency in [
        "exl3_w2a8_composed_n128_n256_probe.cu",
        "exl3_w2a8_fused_gu_down_emit_n128.cu",
        "exl3_w2a8_grouped_prefill_n256.cu",
        "exl3_w2a8_fused_gu_n128_continuous_ring_model.rs",
        "check-exl3-prefill-w2a8-fused-gu-n128-continuous-ring-sass.sh",
        "exl3_w2a8_n256_down_double_buffer_model.rs",
        "KERNEL.toml",
        "check-exl3-prefill-w2a8-fused-gu-n128-continuous-ring-ab-probe-build.sh",
    ] {
        assert!(
            inputs.contains(dependency),
            "immutable inputs omit `{dependency}`"
        );
    }

    let variants = marked(&build, "continuous ring two binary build", "#");
    for proof in [
        "-DW2A8_PACKED_E4M3_CANDIDATE=1",
        "-DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=1",
        "-DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1",
        "-DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0",
        "-DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=$selector",
        "build_variant incumbent 0",
        "build_variant candidate 1",
        "only_ab_compile_factor=W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE",
        "variant incumbent compile_define=-DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=0",
        "variant candidate compile_define=-DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=1",
        "fixed_fused_gu_n128_double_buffer=1",
        "fixed_n256_down_double_buffer=1",
        "runtime_device_identity=measured_all_four_runs",
        "-DW2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP=0",
        "compile_command_sha256",
        "binary_sha256",
        "cubin_sha256",
        "resource_sha256",
        "sass_sha256",
        "different_cubins == 1",
        "changed_n128_sass=PASS",
        "unchanged_n256_sass=PASS",
    ] {
        assert!(variants.contains(proof), "build evidence omits `{proof}`");
    }
}

#[test]
fn generated_runner_is_fail_closed_and_binds_the_actual_runtime_device() {
    let build = source(BUILDER);
    let runner = marked(&build, "continuous ring generated A/B runner", "#");
    for proof in [
        "expected_receipt_sha256",
        "runtime_device_identity=measured_all_four_runs",
        "artifact_sha256",
        "timeout",
        "ulimit -f",
        "order=(incumbent candidate candidate incumbent)",
        "candidate_ms",
        "device_uuid=",
        "driver=",
        "runtime=",
        "input_hash",
        "output_hash",
        "guards=clean inputs=immutable",
        "malformed_cases=16 all_offsets=validated",
        "intermediate_fp8=exact intermediate_scale=exact",
        "raw_bf16=exact final_bf16=exact production_alias=exact",
        "variant_outputs=exact",
        "continuous_ring_speedup",
        "speedup < threshold",
        "ring_selectors=incumbent:0,candidate:1",
        "printf '%s\\n' \"${device_lines[0]}\"",
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
    for selector in [0, 1] {
        let adjacent_defines = concat!(
            "-DW2A8_PACKED_E4M3_CANDIDATE=1 ",
            "-DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=1 ",
            "-DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 ",
            "-DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 ",
            "-DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE={}"
        )
        .replace("{}", &selector.to_string());
        assert!(
            runner.contains(&adjacent_defines),
            "runner receipt preflight must accept adjacent fixed defines for ring{selector}"
        );
    }
    assert!(!runner.contains("double_buffer_selectors="));
    assert!(!runner.contains("nvidia-smi"));
}
