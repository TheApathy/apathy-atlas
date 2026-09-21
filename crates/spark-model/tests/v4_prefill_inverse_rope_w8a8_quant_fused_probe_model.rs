// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contracts for the isolated V4 inverse-RoPE/W8A8 promotion probe.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_prefill_inverse_rope_w8a8_quant_fused_probe.cu";
const BUILD: &str = "scripts/check-v4-prefill-inverse-rope-w8a8-quant-fused-probe-build.sh";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(root().join(relative)).unwrap_or_else(|error| {
        panic!("required inverse-RoPE/W8A8 probe input {relative} is missing: {error}")
    })
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
    index == bytes.len()
        && value
            .parse::<f64>()
            .is_ok_and(|number| number.is_finite() && number > 1.0 && number <= 100.0)
}

#[test]
fn probe_launches_the_settled_real_incumbents_and_candidate() {
    let probe = source(PROBE);
    let contract = section(&probe, "V4 inverse quant probe kernel contract", "//");
    for needle in [
        "v4_prefill_rope_fused.cu",
        "../common/w8a8_gemm_pipelined.cu",
        "v4_prefill_inverse_rope_w8a8_quant_fused.cu",
        "v4_prefill_rope_fused_inverse<<<dim3(tokens, kNq, 1), dim3(kRopePairs, 1, 1)",
        "quantize_a_fp8_rows<<<dim3(tokens, 1, 1), dim3(kQuantThreads, 1, 1)",
        "v4_prefill_inverse_rope_w8a8_quant_fused<<<dim3(tokens, 1, 1), dim3(kQuantThreads, 1, 1)",
    ] {
        assert!(
            contract.contains(needle),
            "kernel contract omits `{needle}`"
        );
    }
    for exact in [
        "kProductionTokens = 2410",
        "kNq = 64",
        "kHeadDim = 512",
        "kNopeDim = 448",
        "kRopeDim = 64",
        "kRowWidth = 32768",
        "kRopePairs = 32",
        "kQuantThreads = 256",
    ] {
        assert!(probe.contains(exact), "shape contract omits `{exact}`");
    }

    let direct = source("kernels/gb10/experiments/v4_prefill_rope_fused.cu");
    assert!(direct.contains("void v4_prefill_rope_fused_inverse("));
    let quantizer = source("kernels/gb10/common/w8a8_gemm_pipelined.cu");
    assert!(quantizer.contains("void quantize_a_fp8_rows("));
    let quantize = quantizer
        .split("void quantize_a_fp8_rows(")
        .nth(1)
        .expect("real activation quantizer");
    assert_eq!(quantize.matches("row[k]").count(), 2);
}

#[test]
fn exact_parity_covers_shapes_poisons_scale_bits_and_immutable_input() {
    let probe = source(PROBE);
    let cases = section(&probe, "V4 inverse quant deterministic cases", "//");
    for needle in [
        "{1, 3, 10000.0f, 1.0f}",
        "{7, 11, 160000.0f, 0.70710677f}",
        "{128, 29, 87500.0f, 1.125f}",
        "{2410, 47, 160000.0f, 0.875f}",
        "distinct positions",
        "distinct frequencies",
    ] {
        assert!(cases.contains(needle), "case model omits `{needle}`");
    }
    let parity = section(&probe, "V4 inverse quant exact byte parity", "//");
    for needle in [
        "std::memcmp",
        "fp8_mismatches",
        "scale_mismatches",
        "candidate_input_unchanged",
        "candidate_input_before",
        "candidate_input_after",
        "kPoisonA",
        "kPoisonB",
        "guards_clean",
        "poison_a",
        "poison_b",
        "full FP8 and FP32 scale-bit memcmp",
    ] {
        assert!(parity.contains(needle), "parity contract omits `{needle}`");
    }
    assert!(!parity.contains("tolerance"));
}

#[test]
fn every_candidate_guard_has_a_positive_no_write_canary() {
    let malformed = section(&source(PROBE), "V4 inverse quant malformed ABI", "//").to_owned();
    for case in [
        "null-input",
        "null-positions",
        "null-inv-freq",
        "null-output",
        "null-scale",
        "input-output-alias",
        "zero-tokens",
        "wrong-q-heads",
        "wrong-head-dim",
        "wrong-nope-dim",
        "wrong-rotary-dim",
        "wrong-row-width",
        "wrong-quant-block",
        "mscale-nan",
        "mscale-inf",
        "block-x",
        "block-y",
        "block-z",
        "grid-x-small",
        "grid-x-large",
        "grid-y",
        "grid-z",
    ] {
        assert_once(&malformed, case);
    }
    for needle in [
        "kMalformedCases = 22",
        "input_before == input_after",
        "fp8_before == fp8_after",
        "scale_before == scale_after",
        "cudaGetLastError",
        "cudaDeviceSynchronize",
        "guards_clean",
    ] {
        assert!(malformed.contains(needle), "malformed ABI omits `{needle}`");
    }
}

#[test]
fn production_timing_is_warm_abba_with_all_resets_outside_events() {
    let probe = source(PROBE);
    let timing = section(&probe, "V4 inverse quant ABBA timing", "//");
    for needle in [
        "baseline, candidate, candidate, baseline",
        "cudaMemcpyAsync",
        "cudaMemsetAsync",
        "cudaEventRecord(begin)",
        "cudaEventRecord(end)",
        "cudaEventElapsedTime",
        "kTimingRounds = 8",
        "warmup",
    ] {
        assert!(timing.contains(needle), "timing contract omits `{needle}`");
    }
    let reset = timing.find("reset();").unwrap();
    let begin = timing.find("cuda_ok(cudaEventRecord(begin)").unwrap();
    assert!(reset < begin, "D2D/memset reset must precede the event");
    let run = probe.find("static int run(").unwrap();
    let parse = probe[run..].find("parse_threshold(argc, argv)").unwrap() + run;
    let cuda = probe[run..].find("cudaDriverGetVersion").unwrap() + run;
    assert!(parse < cuda, "threshold admission must precede CUDA");
}

#[test]
fn diagnostics_and_w8a8_eligibility_are_mandatory_dispatch_boundaries() {
    let probe = source(PROBE);
    let applicability = section(&probe, "V4 inverse quant applicability", "//");
    for needle in [
        "decide w8a8_inplace eligibility before launch",
        "require diag_this == false",
        "materialized inverse-rotated BF16 attn_out",
        "bypass the candidate",
    ] {
        assert!(
            applicability.contains(needle),
            "applicability omits `{needle}`"
        );
    }

    let grouped = source("crates/spark-model/src/layers/qwen3_attention/prefill/v4_fp8_proj.rs");
    let arm = grouped
        .split("let w8a8_inplace = fp8.is_some()")
        .nth(1)
        .expect("W8A8-inplace eligibility");
    for needle in [
        "v4_proj_fp8mma_enabled()",
        "self.w8a8_gemm_pipelined_ld_k.0 != 0",
        "self.quantize_a_fp8_rows_k.0 != 0",
        "ctx.buffers.fp8_act_bytes() >= (n as usize) * (input_width as usize)",
    ] {
        assert!(arm.contains(needle), "eligibility omits `{needle}`");
    }

    let prefill = source("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
    let inverse = prefill.find("self.rope_yarn_interleaved_inv_k").unwrap();
    let diagnostics = prefill[inverse..].find("if diag_this {").unwrap() + inverse;
    let wo_a = prefill.find("self.v4_grouped_wo_a_prefill(").unwrap();
    assert!(inverse < diagnostics && diagnostics < wo_a);
    assert!(prefill[diagnostics..wo_a].contains("attn_out"));
}

#[test]
fn receipt_binds_sources_tools_commands_binary_cubins_and_threshold() {
    let build = source(BUILD);
    let inputs = section(&build, "V4 inverse quant probe immutable inputs", "#");
    for path in [
        "v4_prefill_inverse_rope_w8a8_quant_fused_probe.cu",
        "v4_prefill_inverse_rope_w8a8_quant_fused.cu",
        "v4_prefill_rope_fused.cu",
        "common/w8a8_gemm_pipelined.cu",
        "check-v4-prefill-inverse-rope-w8a8-quant-fused-probe-build.sh",
    ] {
        assert!(inputs.contains(path), "immutable inputs omit `{path}`");
    }
    for identity in [
        "dependency_hashes",
        "nvcc_binary_sha256",
        "cuobjdump_binary_sha256",
        "host_cxx_binary_sha256",
        "git_commit",
        "git_status_sha256",
    ] {
        assert!(
            inputs.contains(identity),
            "immutable inputs omit `{identity}`"
        );
    }
    let receipt = section(&build, "V4 inverse quant probe receipt", "#");
    for field in [
        "min_speedup=",
        "build_id=",
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
    let runner = section(&build, "V4 inverse quant runner verification", "#");
    for field in [
        "expected_receipt_sha256",
        "expected_binary_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "actual_cubin_sha256",
        "if [[ $# -ne 0 ]]",
        "\"$binary\" \"$expected_min_speedup\"",
    ] {
        assert!(runner.contains(field), "runner omits `{field}`");
    }
}

#[test]
fn threshold_build_environment_and_runner_output_fail_closed() {
    let build = source(BUILD);
    let admission = build.find("if [[ $# -ne 1 ]]").unwrap();
    let build_work = build.find("repo_root=").unwrap();
    assert!(admission < build_work);
    for bad in [
        "", " 1.01", "1.01 ", "junk", "nan", "inf", "0x1.1p1", "-1", "0", "1", "100.1",
    ] {
        assert!(!threshold_valid(bad), "accepted threshold {bad:?}");
    }
    for good in ["1.0001", "1.01", "2", "1e1", "100"] {
        assert!(threshold_valid(good), "rejected threshold {good:?}");
    }
    for contract in [
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing unreceipted nvcc flags",
        "-arch=sm_121a",
        "--fmad=false",
        "max_output_bytes=4096",
        "max_output_lines=8",
        "candidate_input_unchanged=yes",
        "fp8_mismatches=0 scale_mismatches=0",
        "malformed_cases=22 no_write=22",
        "result=PASS",
    ] {
        assert!(build.contains(contract), "build/runner omits `{contract}`");
    }
}

#[test]
fn probe_is_not_reachable_from_registry_or_serving() {
    for relative in [
        "crates/atlas-kernels/build.rs",
        "kernels/gb10/deepseek-v4-flash/MODEL.toml",
        "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
    ] {
        let production = source(relative);
        assert!(!production.contains("v4_prefill_inverse_rope_w8a8_quant_fused_probe"));
        assert!(!production.contains("V4_INV_QUANT_PROBE"));
    }
}
