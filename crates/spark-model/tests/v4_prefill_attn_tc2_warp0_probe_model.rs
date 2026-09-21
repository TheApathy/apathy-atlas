// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contracts for the isolated V4 TC2 warp-0 promotion probe.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_prefill_attn_tc2_warp0_probe.cu";
const BUILD: &str = "scripts/check-v4-prefill-attn-tc2-warp0-probe-build.sh";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required TC2 warp-0 probe input {relative}: {error}"))
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

fn threshold_is_valid(value: &str) -> bool {
    if value.is_empty()
        || value.starts_with(char::is_whitespace)
        || value.ends_with(char::is_whitespace)
        || value.starts_with('+')
        || value.starts_with('-')
    {
        return false;
    }
    value
        .parse::<f64>()
        .is_ok_and(|number| number.is_finite() && number > 1.0 && number <= 100.0)
}

#[test]
fn probe_compiles_the_exact_incumbent_and_isolated_candidate() {
    let probe = source(PROBE);
    let kernels = section(&probe, "TC2 warp0 probe kernel contract", "//");
    for contract in [
        "../deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu",
        "v4_prefill_attn_compressed_tc2_warp0.cu",
        "#define prefill_attn_compressed_tc2 v4_probe_incumbent_tc2",
        "v4_probe_incumbent_tc2<<<grid, block>>>",
        "v4_prefill_attn_compressed_tc2_warp0<<<grid, block>>>",
        "dim3 block(128, 1, 1)",
    ] {
        assert!(
            kernels.contains(contract),
            "kernel contract omits `{contract}`"
        );
    }
    assert!(!probe.contains("tolerance"));
    assert!(!probe.contains("epsilon parity"));
    for contract in [
        "V4_TC2_WARP0_PROBE_MIN_SPEEDUP",
        "V4_TC2_WARP0_PROBE_MIN_SPEEDUP_TEXT",
        "constexpr double kMinSpeedup",
    ] {
        assert!(probe.contains(contract), "probe omits `{contract}`");
    }
}

#[test]
fn parity_matrix_covers_alias_tail_raw_compressed_sliding_and_sinks() {
    let probe = source(PROBE);
    let cases = section(&probe, "TC2 warp0 deterministic cases", "//");
    for contract in [
        "kParityCases = 9",
        "raw_alias",
        "compressed_alias",
        "seq_len",
        "n_comp",
        "ratio",
        "sliding_window",
        "with_sinks",
        "{31, 7, 4, 0, true, true, true",
        "{33, 8, 4, 16, false, false, true",
        "{47, 0, 4, 17, true, false, false",
        "{65, 16, 4, 32, false, true, true",
        "{129, 1, 128, 128, true, true, true",
        "{2410, 602, 4, 128, true, true, true",
        "{2410, 18, 128, 128, true, true, true",
        "{2410, 0, 1, 128, true, true, true, true",
        "{2410, 0, 1, 128, true, true, false, true",
    ] {
        assert!(cases.contains(contract), "case matrix omits `{contract}`");
    }
    assert!(cases.contains("partial 16-row tails"));
    assert!(cases.contains("raw/compressed/sliding boundaries"));
    assert!(cases.contains("K=V=Kc=Vc"));
}

#[test]
fn dual_poison_compares_every_output_byte_and_all_payloads_and_guards() {
    let probe = source(PROBE);
    let parity = section(&probe, "TC2 warp0 exact byte parity", "//");
    for contract in [
        "kPoisonCases = 2",
        "kPoisonA",
        "kPoisonB",
        "memcmp",
        "mismatch_bytes",
        "prefix_guard",
        "suffix_guard",
        "reset_inputs",
        "inputs_immutable",
        "all_guards_clean",
        "poison_a",
        "poison_b",
    ] {
        assert!(
            parity.contains(contract),
            "parity contract omits `{contract}`"
        );
    }
    assert!(!parity.contains("cosine"));
    assert!(!parity.contains("allclose"));
}

#[test]
fn malformed_abi_cases_are_positive_no_write_canaries() {
    let probe = source(PROBE);
    let malformed = section(&probe, "TC2 warp0 malformed ABI no-write", "//");
    for case in [
        "null-q",
        "null-k",
        "null-v",
        "null-kc",
        "null-vc",
        "null-o",
        "misaligned-q",
        "misaligned-k",
        "misaligned-v",
        "misaligned-kc",
        "misaligned-vc",
        "misaligned-sinks",
        "misaligned-o",
        "o-alias-q",
        "o-alias-k",
        "o-alias-v",
        "o-alias-kc",
        "o-alias-vc",
        "o-alias-sinks",
        "o-overlap-q-offset",
        "o-overlap-k-offset",
        "scale-nan",
        "scale-pos-inf",
        "scale-neg-inf",
        "scale-zero",
        "scale-negative",
        "head-dim",
        "kv-heads-zero",
        "head-ratio",
        "ratio-zero",
        "block-x",
        "block-y",
        "block-z",
        "grid-x",
        "grid-y",
        "grid-z",
    ] {
        assert!(
            malformed.contains(case),
            "missing malformed canary `{case}`"
        );
    }
    for contract in [
        "all_payload",
        "inputs_immutable",
        "all_guards_clean",
        "malformed_cases",
        "cudaDeviceSynchronize",
    ] {
        assert!(
            malformed.contains(contract),
            "missing malformed proof `{contract}`"
        );
    }
}

#[test]
fn both_production_shapes_are_separate_abba_with_build_bound_threshold() {
    let probe = source(PROBE);
    let timing = section(&probe, "TC2 warp0 ABBA timing contract", "//");
    for contract in [
        "kTimingTokens = 2410",
        "kTimingHeads = 64",
        "kTimingNComp = 602",
        "kTimingRatio = 4",
        "kTimingWindow = 128",
        "kDenseNComp = 0",
        "kDenseRatio = 1",
        "K=V=Kc=Vc",
        "ABBA",
        "cudaEventRecord",
        "cudaEventElapsedTime",
        "kTimingSamples",
        "baseline_ms",
        "candidate_ms",
        "speedup",
        "csa",
        "dense",
    ] {
        assert!(
            timing.contains(contract),
            "timing contract omits `{contract}`"
        );
    }
    let parse = probe.find("constexpr double kMinSpeedup").unwrap();
    let first_cuda = probe.find("cudaDriverGetVersion").unwrap();
    assert!(
        parse < first_cuda,
        "threshold must be parsed before CUDA initialization"
    );
}

#[test]
fn output_is_bounded_and_machine_checkable() {
    let probe = source(PROBE);
    let output = section(&probe, "TC2 warp0 bounded output contract", "//");
    for key in [
        "build_id=",
        "device_uuid=",
        "driver=",
        "runtime=",
        "input_hash=",
        "threshold min_speedup=",
        "csa_baseline_ms=",
        "csa_candidate_ms=",
        "dense_baseline_ms=",
        "dense_candidate_ms=",
        "malformed_cases=",
        "inputs=immutable",
        "output_mismatches=0",
        "input_guards=clean",
        "output_guards=clean",
        "poison_a=clean",
        "poison_b=clean",
        "result=PASS",
    ] {
        assert_once(output, key);
    }
}

#[test]
fn receipt_binds_sources_tools_commands_binary_and_cubins() {
    let build = source(BUILD);
    let threshold = build.find("canonicalize_binary64_threshold").unwrap();
    let repository = build.find("repo_root=").unwrap();
    assert!(
        threshold < repository,
        "threshold admission must precede repository and tool access"
    );
    let inputs = section(&build, "TC2 warp0 probe immutable inputs", "#");
    for path in [
        "v4_prefill_attn_tc2_warp0_probe.cu",
        "experiments/v4_prefill_attn_compressed_tc2_warp0.cu",
        "deepseek-v4-flash/nvfp4/v4_prefill_attn_compressed_tc2_warp0.cu",
        "deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu",
        "qwen3_attention/prefill/cache_skip_v4.rs",
        "qwen3_attention/types.rs",
        "qwen3_attention/init.rs",
        "check-v4-prefill-attn-tc2-warp0-probe-build.sh",
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
        "status_paths",
    ] {
        assert!(
            inputs.contains(identity),
            "immutable inputs omit `{identity}`"
        );
    }

    let receipt = section(&build, "TC2 warp0 probe receipt contract", "#");
    let threshold_field = receipt.find("min_speedup=$min_speedup").unwrap();
    let build_id = receipt.find("build_id=").unwrap();
    assert!(
        threshold_field < build_id,
        "threshold must contribute to build_id"
    );
    for field in [
        "build_id=",
        "min_speedup=",
        "compile_command=",
        "compile_command_sha256=",
        "resource_command=",
        "resource_command_sha256=",
        "extract_command=",
        "extract_command_sha256=",
        "binary_sha256",
        "cubin_sha256",
    ] {
        assert!(receipt.contains(field), "receipt omits `{field}`");
    }
}

#[test]
fn runner_revalidates_bound_threshold_and_strict_bounded_output() {
    let build = source(BUILD);
    let runner = section(&build, "TC2 warp0 probe runner verification", "#");
    for field in [
        "actual_receipt_sha256",
        "actual_binary_sha256",
        "actual_cubin_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "parity_cases=9 poison_cases=2 malformed_cases=36",
        "output_mismatches=0 inputs=immutable input_guards=clean output_guards=clean poison_a=clean poison_b=clean",
        "threshold min_speedup=$expected_min_speedup",
        "csa_baseline_ms=[0-9]+\\.[0-9]{6}",
        "dense_baseline_ms=[0-9]+\\.[0-9]{6}",
        "result=PASS",
    ] {
        assert!(runner.contains(field), "runner omits `{field}`");
    }
    assert!(runner.contains("[[ $# -eq 0 ]]"));
    assert!(runner.contains("\"$binary\" >\"$output\""));
    assert!(!runner.contains("\"$binary\" \"$1\""));
    assert!(build.contains("printf \"%.17g"));
    assert!(build.contains("runner_sha256="));
    for rejection in [
        "' 1.01'",
        "'1.01 '",
        "0x1.1p1",
        "nan",
        "inf",
        "-inf",
        "1.00000000000000001",
        "100.1",
    ] {
        assert!(build.contains(rejection), "missing rejection `{rejection}`");
    }
}

#[test]
fn binary64_threshold_oracle_preserves_small_real_wins() {
    for good in ["1.00000001", "1.0001", "1.5", "2", "100", "1e1"] {
        assert!(threshold_is_valid(good), "rejected {good:?}");
    }
    for bad in [
        "",
        " 1.01",
        "1.01 ",
        "0x1.1p1",
        "nan",
        "inf",
        "-inf",
        "-1",
        "0",
        "1",
        "1.00000000000000001",
        "100.1",
    ] {
        assert!(!threshold_is_valid(bad), "accepted {bad:?}");
    }
}
