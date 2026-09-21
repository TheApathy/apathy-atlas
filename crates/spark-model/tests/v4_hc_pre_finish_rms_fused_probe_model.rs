// SPDX-License-Identifier: AGPL-3.0-only

//! Offline source and arithmetic contracts for the isolated V4 HC promotion probe.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_hc_pre_finish_rms_fused_probe.cu";
const BUILD: &str = "scripts/check-v4-hc-pre-finish-rms-fused-probe-build.sh";
const DOC: &str = "docs/REPRODUCE-PREFILL.md";
const KERNEL_DOC: &str = "docs/kernels/04-elementwise-norm-cache.md";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required HC probe input {relative} is missing: {error}"))
}

fn section<'a>(text: &'a str, name: &str, prefix: &str) -> &'a str {
    let begin = format!("{prefix} BEGIN {name}");
    let end = format!("{prefix} END {name}");
    assert_eq!(text.matches(&begin).count(), 1, "duplicate {begin}");
    assert_eq!(text.matches(&end).count(), 1, "duplicate {end}");
    let start = text.find(&begin).unwrap() + begin.len();
    let finish = text[start..].find(&end).unwrap() + start;
    assert!(start < finish, "reversed markers for {name}");
    &text[start..finish]
}

fn assert_once(text: &str, needle: &str) {
    assert_eq!(text.matches(needle).count(), 1, "expected one `{needle}`");
}

#[test]
fn production_sizes_and_launch_reduction_are_exact() {
    const TOKENS: u64 = 2_410;
    const HIDDEN: u64 = 4_096;
    const HC: u64 = 4;
    const LAYERS: u64 = 43;
    let normed_bytes = TOKENS * HIDDEN * 2;
    let post_bytes = TOKENS * HC * 4;
    let comb_bytes = TOKENS * HC * HC * 4;
    assert_eq!(normed_bytes, 19_742_720);
    assert_eq!(post_bytes, 38_560);
    assert_eq!(comb_bytes, 154_240);
    // Materialized hidden remains observable; only its RMS reread disappears.
    assert_eq!(normed_bytes * 2 * LAYERS, 1_697_873_920);
    assert_eq!(2 * LAYERS, 86);
}

#[test]
fn probe_invokes_exact_incumbent_and_candidate_shapes() {
    let probe = source(PROBE);
    let kernels = section(&probe, "HC probe kernel contract", "//");
    for input in [
        "../deepseek-v4-flash/nvfp4/hyper_connection.cu",
        "../common/rms_norm_vanilla.cu",
        "v4_hc_pre_finish_rms_fused.cu",
        "hc_pre_finish<<<dim3(kTokens, kFinishShards, 1), dim3(256, 1, 1)",
        "rms_norm_vanilla<<<dim3(kTokens, 1, 1), dim3(kRmsThreads, 1, 1)",
        "v4_hc_pre_finish_rms_fused<<<dim3(kTokens, 1, 1), dim3(kRmsThreads, 1, 1)",
    ] {
        assert!(kernels.contains(input), "kernel contract omits `{input}`");
    }
    for exact in [
        "kTokens = 2410",
        "kHidden = 4096",
        "kHc = 4",
        "kFinishShards = 16",
        "kRmsThreads = 1024",
    ] {
        assert!(probe.contains(exact), "shape contract omits `{exact}`");
    }
}

#[test]
fn deterministic_cases_and_dual_poison_cover_every_output_byte() {
    let probe = source(PROBE);
    let cases = section(&probe, "HC probe deterministic cases", "//");
    for field in [
        "kParityCases = 6",
        "sinkhorn_iters",
        "hc_norm_eps",
        "hc_eps",
        "rms_eps",
        "stream_salt",
        "mix_salt",
        "weight_salt",
        "signed-zero",
        "1.0e-20f",
    ] {
        assert!(cases.contains(field), "case model omits `{field}`");
    }

    let parity = section(&probe, "HC probe exact byte parity", "//");
    for field in [
        "hidden_mismatches",
        "normed_mismatches",
        "post_mismatches",
        "comb_mismatches",
        "memcmp",
        "kPoisonA",
        "kPoisonB",
        "poison_a",
        "poison_b",
        "guards_clean",
    ] {
        assert!(parity.contains(field), "byte parity omits `{field}`");
    }
    assert!(!parity.contains("tolerance"));
    assert!(!parity.contains("epsilon"));
}

#[test]
fn malformed_geometry_is_a_positive_no_write_canary() {
    let probe = source(PROBE);
    let malformed = section(&probe, "HC probe malformed geometry", "//");
    for case in [
        "block-x",
        "block-y",
        "block-z",
        "grid-x-small",
        "grid-x-large",
        "grid-y",
        "grid-z",
        "wrong-hidden",
        "wrong-hc",
        "nullptr",
    ] {
        assert_once(malformed, case);
    }
    for contract in [
        "kGeometryCases = 10",
        "expect_unchanged",
        "cudaGetLastError",
        "cudaDeviceSynchronize",
        "prefix_guard",
        "suffix_guard",
    ] {
        assert!(
            malformed.contains(contract),
            "geometry contract omits `{contract}`"
        );
    }
}

#[test]
fn timing_and_success_output_are_bounded_and_fail_closed() {
    let probe = source(PROBE);
    let timing = section(&probe, "HC probe ABBA timing contract", "//");
    for field in [
        "ABBA",
        "cudaEventRecord",
        "cudaEventElapsedTime",
        "baseline_ms",
        "candidate_ms",
        "speedup",
        "kTimingSamples",
    ] {
        assert!(timing.contains(field), "timing contract omits `{field}`");
    }
    let output = section(&probe, "HC probe bounded output contract", "//");
    for key in [
        "build_id=",
        "device_uuid=",
        "driver=",
        "runtime=",
        "input_hash=",
        "mix_hash=",
        "weight_hash=",
        "threshold min_speedup=",
        "baseline_ms=",
        "normed_mismatches=0",
        "post_mismatches=0",
        "comb_mismatches=0",
        "result=PASS",
    ] {
        assert_once(output, key);
    }
}

#[test]
fn receipt_binds_transitive_sources_tools_commands_and_artifacts() {
    let build = source(BUILD);
    let threshold_at = build
        .find("canonicalize_binary64_threshold")
        .expect("binary64 threshold admission");
    let tools_at = build.find("repo_root=").expect("build tool discovery");
    let output_at = build
        .find("V4_HC_PROBE_OUTPUT_DIR")
        .expect("output directory handling");
    assert!(threshold_at < tools_at && threshold_at < output_at);
    for contract in [
        "if [[ $# -ne 1 ]]",
        "ctypes.CDLL(None)",
        "libc.strtod",
        "ctypes.c_double",
        "format(value, '.17g')",
        "value.hex()",
        "min_speedup_text=$min_speedup_text",
        "min_speedup_hex=$min_speedup_hex",
        "V4_HC_PROBE_MIN_SPEEDUP_TEXT",
        "V4_HC_PROBE_MIN_SPEEDUP_HEX",
    ] {
        assert!(
            build.contains(contract),
            "threshold contract omits `{contract}`"
        );
    }
    assert!(1.00000001_f64 > 1.0);
    assert_eq!("1.00000000000000001".parse::<f64>().unwrap(), 1.0);
    let inputs = section(&build, "HC probe immutable inputs", "#");
    for path in [
        "v4_hc_pre_finish_rms_fused_probe.cu",
        "v4_hc_pre_finish_rms_fused.cu",
        "deepseek-v4-flash/nvfp4/v4_hc_pre_finish_rms_fused.cu",
        "deepseek-v4-flash/nvfp4/hyper_connection.cu",
        "common/rms_norm_vanilla.cu",
        "check-v4-hc-pre-finish-rms-fused-probe-build.sh",
    ] {
        assert!(inputs.contains(path), "immutable inputs omit `{path}`");
    }
    for identity in [
        "nvcc_binary_hash",
        "cuobjdump_binary_hash",
        "host_cxx_binary_hash",
        "git_commit",
        "git_status_hash",
        "dependency_hashes",
    ] {
        assert!(
            inputs.contains(identity),
            "immutable inputs omit `{identity}`"
        );
    }

    let receipt = section(&build, "HC probe receipt contract", "#");
    for field in [
        "build_id=",
        "min_speedup_text=",
        "min_speedup_hex=",
        "baseline_kernel=rms_norm_vanilla",
        "normalization_formula=x*rms*weight",
        "compile_command=",
        "compile_command_sha256=",
        "resource_command=",
        "extract_command=",
        "binary_sha256",
        "cubin_sha256",
    ] {
        assert!(receipt.contains(field), "receipt omits `{field}`");
    }

    let runner = section(&build, "HC probe runner verification", "#");
    for field in [
        "if [[ $# -ne 0 ]]",
        "actual_receipt_sha256",
        "actual_binary_sha256",
        "actual_cubin_sha256",
        "expected_build_id",
        "tokens=2410 hidden=4096 hc=4 parity_cases=6 poison_cases=2 geometry_cases=10 hidden_bytes=19742720 normed_bytes=19742720",
        "hidden_mismatches=0 normed_mismatches=0 post_mismatches=0 comb_mismatches=0 guards=clean poison_a=clean poison_b=clean",
        "threshold min_speedup=",
        "expected_min_speedup_text",
        "\"$binary\" >\"$output\"",
        "reject-valid-binary64-extra-arg",
        "1.00000001",
        "result=PASS",
    ] {
        assert!(runner.contains(field), "runner omits `{field}`");
    }
    assert!(build.contains("usage: $0 <min_speedup>"));
    assert!(build.contains("invalid explicit numeric threshold"));
}

#[test]
fn kernel_doc_names_vanilla_semantics_and_strict_production_status() {
    let doc = source(KERNEL_DOC);
    let fusion = doc
        .lines()
        .find(|line| line.contains("hc_pre_finish") && line.contains("rms_norm_vanilla"))
        .expect("HC finish/RMS fusion row must name the DeepSeek-V4 baseline");
    assert!(fusion.contains("`x * rms * weight`"));
    assert!(fusion.contains("strict default-off production arm"));
}

#[test]
fn compiled_probe_uses_only_the_receipt_bound_binary64_threshold() {
    let probe = source(PROBE);
    for contract in [
        "#ifndef V4_HC_PROBE_MIN_SPEEDUP_TEXT",
        "#ifndef V4_HC_PROBE_MIN_SPEEDUP_HEX",
        "constexpr double kMinSpeedup = V4_HC_PROBE_MIN_SPEEDUP_HEX",
        "static_assert(kMinSpeedup > 1.0 && kMinSpeedup <= 100.0",
        "if (argc != 1)",
        "validate_compiled_threshold",
        "V4_HC_PROBE_MIN_SPEEDUP_TEXT",
        "minimum = kMinSpeedup",
    ] {
        assert!(
            probe.contains(contract),
            "compiled threshold omits `{contract}`"
        );
    }
    assert!(!probe.contains("argv[1]"));
}

#[test]
fn hc_runbook_registers_threshold_at_build_and_runs_without_arguments() {
    let doc = source(DOC);
    let hc = doc
        .split("The `hc_pre_finish` + vanilla RMSNorm production arm")
        .nth(1)
        .expect("HC promotion section")
        .split("The dedicated emitter build")
        .next()
        .expect("bounded HC promotion section");
    assert!(hc.contains("check-v4-hc-pre-finish-rms-fused-probe-build.sh 1.01"));
    assert!(hc.contains("/tmp/atlas-v4-hc-fused-probe/run-v4-hc-fused-probe.sh\n"));
    assert!(hc.contains("threshold's canonical decimal text and exact binary64 value"));
    assert!(hc.contains("generated runner rejects arguments"));
    assert!(!hc.contains("run-v4-hc-fused-probe.sh 1.01"));
}
