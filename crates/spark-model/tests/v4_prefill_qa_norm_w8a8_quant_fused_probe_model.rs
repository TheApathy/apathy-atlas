// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contracts for the isolated V4 q_a norm/W8A8 promotion probe.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_prefill_qa_norm_w8a8_quant_fused_probe.cu";
const BUILD: &str = "scripts/check-v4-prefill-qa-norm-w8a8-quant-fused-probe-build.sh";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required q_a probe input {relative} is missing: {error}"))
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

fn threshold_valid(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut index = 0;
    let mut digits = 0;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
        digits += 1;
    }
    if index < bytes.len() && bytes[index] == b'.' {
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return false;
    }
    if index < bytes.len() && matches!(bytes[index], b'e' | b'E') {
        index += 1;
        if index < bytes.len() && matches!(bytes[index], b'+' | b'-') {
            index += 1;
        }
        let exponent_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if exponent_start == index {
            return false;
        }
    }
    if index != bytes.len() {
        return false;
    }
    value
        .parse::<f64>()
        .is_ok_and(|number| number.is_finite() && number > 1.0 && number <= 100.0)
}

#[test]
fn probe_invokes_the_exact_local_incumbent_and_candidate() {
    let probe = source(PROBE);
    let contract = section(&probe, "q_a probe kernel contract", "//");
    for needle in [
        "../common/rms_norm_vanilla.cu",
        "../common/w8a8_gemm_pipelined.cu",
        "v4_prefill_qa_norm_w8a8_quant_fused.cu",
        "rms_norm_vanilla<<<dim3(tokens, 1, 1), dim3(kRmsThreads, 1, 1)",
        "quantize_a_fp8_rows<<<dim3(tokens, 1, 1), dim3(kQuantThreads, 1, 1)",
        "v4_prefill_qa_norm_w8a8_quant_fused<<<dim3(tokens, 1, 1), dim3(kRmsThreads, 1, 1)",
    ] {
        assert!(
            contract.contains(needle),
            "kernel contract omits `{needle}`"
        );
    }
    for exact in [
        "kHidden = 1024",
        "kRmsThreads = 1024",
        "kQuantThreads = 256",
        "kParityCases = 4",
        "kPoisonCases = 2",
        "kGeometryCases = 15",
    ] {
        assert!(probe.contains(exact), "shape contract omits `{exact}`");
    }
}

#[test]
fn reachable_source_selects_vanilla_and_keeps_the_bf16_seam() {
    let init = source("crates/spark-model/src/layers/qwen3_attention/init.rs");
    let prefill = source("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
    let vanilla = init
        .find("gpu.kernel(\"rms_norm_vanilla\", \"rms_norm_vanilla\")")
        .expect("reachable V4 weighted norm must select vanilla");
    let selector = init[..vanilla]
        .rfind("crate::ships_vanilla_norm_weights(config)")
        .expect("vanilla selector");
    assert!(selector < vanilla);
    let q_a = prefill
        .find("&mla.q_a_norm")
        .expect("q_a norm call remains reachable");
    let wq_b = prefill[q_a..]
        .find("&mla.wq_b")
        .expect("wq_b follows q_a norm")
        + q_a;
    assert!(q_a < wq_b);

    let candidate = source("kernels/gb10/experiments/v4_prefill_qa_norm_w8a8_quant_fused.cu");
    assert!(candidate.contains("__float2bfloat16(x0 * rms * w0)"));
    assert!(candidate.contains("y0_bits = __bfloat16_as_ushort(normed[q0])"));
    assert_eq!(candidate.matches("normed[q0]").count(), 1);
    assert!(!candidate.contains("1.0f + w0"));
}

#[test]
fn representative_shapes_cases_and_dual_poison_cover_all_output_bytes() {
    let probe = source(PROBE);
    let cases = section(&probe, "q_a probe deterministic cases", "//");
    for needle in [
        "{1, kZeros",
        "{7, kVaried",
        "{128, kExtremes",
        "{2410, kVaried",
        "signed-zero",
        "max-finite",
        "weight_salt",
        "rms_eps",
    ] {
        assert!(cases.contains(needle), "case model omits `{needle}`");
    }
    let parity = section(&probe, "q_a probe exact byte parity", "//");
    for needle in [
        "fp8_mismatches",
        "scale_mismatches",
        "memcmp",
        "kPoisonA",
        "kPoisonB",
        "poison_a",
        "poison_b",
        "guards_clean",
        "input.guards_clean",
        "weight.guards_clean",
    ] {
        assert!(parity.contains(needle), "parity contract omits `{needle}`");
    }
    assert!(!parity.contains("tolerance"));
    assert!(!parity.contains("epsilon"));
}

#[test]
fn malformed_abi_cases_are_positive_no_write_canaries() {
    let malformed = section(&source(PROBE), "q_a probe malformed ABI", "//").to_owned();
    for case in [
        "block-x",
        "block-y",
        "block-z",
        "grid-x-small",
        "grid-x-large",
        "grid-y",
        "grid-z",
        "zero-tokens",
        "wrong-hidden",
        "eps-zero",
        "eps-nan",
        "null-input",
        "null-weight",
        "null-output",
        "null-scale",
    ] {
        assert_once(&malformed, case);
    }
    for needle in [
        "kGeometryCases = 15",
        "expect_unchanged",
        "cudaGetLastError",
        "cudaDeviceSynchronize",
        "prefix_guard",
        "suffix_guard",
    ] {
        assert!(malformed.contains(needle), "malformed ABI omits `{needle}`");
    }
}

#[test]
fn resets_are_outside_cuda_events_and_timing_is_abba() {
    let probe = source(PROBE);
    let timing = section(&probe, "q_a probe ABBA timing contract", "//");
    for needle in [
        "input.upload(host_input)",
        "weight.upload(host_weight)",
        "normed.fill",
        "output.fill",
        "cudaEventRecord(begin)",
        "ABBA",
        "kTimingSamples",
        "baseline_ms",
        "candidate_ms",
        "speedup",
    ] {
        assert!(timing.contains(needle), "timing contract omits `{needle}`");
    }
    let upload = timing.find("input.upload(host_input)").unwrap();
    let record = timing.find("cuda_ok(cudaEventRecord(begin)").unwrap();
    assert!(
        upload < record,
        "timing input reset must be outside the CUDA event"
    );

    let run = probe.find("static int run(").expect("run function");
    let threshold = probe[run..].find("threshold(argc, argv)").unwrap() + run;
    let first_cuda = probe[run..].find("cudaDriverGetVersion").unwrap() + run;
    assert!(
        threshold < first_cuda,
        "threshold parsing must precede every CUDA call"
    );
}

#[test]
fn success_output_is_bounded_and_exact() {
    let output = section(&source(PROBE), "q_a probe bounded output contract", "//").to_owned();
    for key in [
        "build_id=",
        "device_uuid=",
        "driver=",
        "runtime=",
        "input_hash=",
        "weight_hash=",
        "shapes=1,7,128,2410",
        "threshold min_speedup=",
        "baseline_ms=",
        "fp8_mismatches=0",
        "scale_mismatches=0",
        "result=PASS",
    ] {
        assert_once(&output, key);
    }
    assert!(output.contains("fp8_bytes=2467840 scale_bytes=9640 normed_bytes=4935680"));
}

#[test]
fn receipt_binds_sources_tools_commands_binary_and_cubins() {
    let build = source(BUILD);
    let inputs = section(&build, "q_a probe immutable inputs", "#");
    for path in [
        "v4_prefill_qa_norm_w8a8_quant_fused_probe.cu",
        "v4_prefill_qa_norm_w8a8_quant_fused.cu",
        "common/rms_norm_vanilla.cu",
        "common/w8a8_gemm_pipelined.cu",
        "check-v4-prefill-qa-norm-w8a8-quant-fused-probe-build.sh",
    ] {
        assert_once(inputs, path);
    }
    for identity in [
        "dependency_hashes",
        "nvcc_binary_hash",
        "cuobjdump_binary_hash",
        "host_cxx_binary_hash",
        "git_commit",
        "git_status_hash",
    ] {
        assert!(
            inputs.contains(identity),
            "immutable inputs omit `{identity}`"
        );
    }

    let receipt = section(&build, "q_a probe receipt contract", "#");
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

    let runner = section(&build, "q_a probe runner verification", "#");
    for field in [
        "actual_receipt_sha256",
        "actual_binary_sha256",
        "actual_cubin_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "grep -Fxc \"min_speedup=$expected_min_speedup\"",
        "[[ $# -eq 0 ]]",
        "\"$binary\" \"$expected_min_speedup\"",
        "shapes=1,7,128,2410",
        "fp8_mismatches=0 scale_mismatches=0 guards=clean poison_a=clean poison_b=clean",
        "threshold min_speedup=$expected_min_speedup",
        "result=PASS",
    ] {
        assert!(runner.contains(field), "runner omits `{field}`");
    }
}

#[test]
fn threshold_grammar_and_build_environment_fail_closed() {
    let build = source(BUILD);
    let threshold = build
        .find("if [[ $# -ne 1 ]]")
        .expect("build-time threshold admission");
    let repo = build.find("repo_root=").expect("first build work");
    assert!(
        threshold < repo,
        "threshold must be admitted before build work"
    );
    for bad in [
        "", " 1.01", "1.01 ", "junk", "nan", "inf", "0x1.1p1", "-1", "0", "1", "100.1",
    ] {
        assert!(!threshold_valid(bad), "accepted threshold {bad:?}");
    }
    for good in ["1.0001", "1.01", "2", "1e1", "100"] {
        assert!(threshold_valid(good), "rejected threshold {good:?}");
    }
    for rejection in [
        "usage_extra 1.01 extra",
        "speed_junk junk",
        "speed_nan nan",
        "speed_inf inf",
        "speed_whitespace ' 1.01'",
        "speed_hex 0x1.1p1",
        "speed_negative -1",
        "speed_zero 0",
        "speed_no_win 1",
        "speed_lax 101",
    ] {
        assert!(build.contains(rejection), "missing rejection `{rejection}`");
    }
    for contract in [
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing unreceipted nvcc flags",
        "-arch=sm_121a",
        "--fmad=false",
        "usage: $0 <min_speedup>",
        "printf \"%.17g\"",
        "echo \"min_speedup=$min_speedup\"",
    ] {
        assert!(
            build.contains(contract),
            "build contract omits `{contract}`"
        );
    }
}
