// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contracts for the isolated V4 q_a norm-to-real-W8A8 pipeline probe.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_prefill_qa_norm_w8a8_pipeline_probe.cu";
const BUILD: &str = "scripts/check-v4-prefill-qa-norm-w8a8-pipeline-probe-build.sh";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required q_a pipeline input {relative} missing: {error}"))
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

fn threshold_valid(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut at = 0;
    let mut digits = 0;
    while at < bytes.len() && bytes[at].is_ascii_digit() {
        at += 1;
        digits += 1;
    }
    if at < bytes.len() && bytes[at] == b'.' {
        at += 1;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return false;
    }
    if at < bytes.len() && matches!(bytes[at], b'e' | b'E') {
        at += 1;
        if at < bytes.len() && matches!(bytes[at], b'+' | b'-') {
            at += 1;
        }
        let exponent = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        if at == exponent {
            return false;
        }
    }
    at == bytes.len()
        && value
            .parse::<f64>()
            .is_ok_and(|number| number.is_finite() && number > 1.0 && number <= 100.0)
}

#[test]
fn exact_production_shape_and_real_pipeline_are_source_locked() {
    let probe = source(PROBE);
    let contract = section(&probe, "q_a W8A8 pipeline kernel contract", "//");
    for needle in [
        "../common/rms_norm_vanilla.cu",
        "../common/w8a8_gemm_pipelined.cu",
        "v4_prefill_qa_norm_w8a8_quant_fused.cu",
        "rms_norm_vanilla<<<dim3(m, 1, 1), dim3(kRmsThreads, 1, 1)",
        "quantize_a_fp8_rows<<<dim3(m, 1, 1), dim3(kQuantThreads, 1, 1)",
        "w8a8_gemm_pipelined<<<dim3(kN / 32, div_up(m, 128), 1), dim3(256, 1, 1)",
        "v4_prefill_qa_norm_w8a8_quant_fused<<<dim3(m, 1, 1), dim3(kRmsThreads, 1, 1)",
    ] {
        assert!(contract.contains(needle), "pipeline omits `{needle}`");
    }
    for exact in [
        "kProductionM = 2410",
        "kN = 32768",
        "kK = 1024",
        "kRmsThreads = 1024",
        "kQuantThreads = 256",
        "kParityCases = 3",
        "kPoisonCases = 2",
    ] {
        assert!(probe.contains(exact), "shape contract omits `{exact}`");
    }
}

#[test]
fn dispatch_contract_is_only_released_bf16_and_eligible_wq_b() {
    let projection = source("crates/spark-model/src/layers/qwen3_attention/prefill/v4_fp8_proj.rs");
    let packed = projection
        .split("pub(super) fn v4_project_prefill")
        .nth(1)
        .expect("V4 packed projection");
    let bf16 = packed.find("if !dense.weight.is_null()").unwrap();
    let validate = packed.find("let weight = validate_fp8(").unwrap();
    let w8a8 = packed.find("self.try_w8a8_project_prefill").unwrap();
    assert!(bf16 < validate && validate < w8a8);
    let eligibility = projection
        .split("pub(super) fn try_w8a8_project_prefill")
        .nth(1)
        .unwrap();
    for guard in [
        "!v4_proj_fp8mma_enabled()",
        "self.w8a8_gemm_pipelined_k.0 == 0",
        "self.quantize_a_fp8_rows_k.0 == 0",
        "ctx.buffers.fp8_act_bytes() < (m as usize) * (k as usize)",
        "weight.scale_format != WeightQuantFormat::Fp8BlockScaled",
        "weight.n != n",
        "weight.k != k",
        "!n.is_multiple_of(FP8_BLOCK)",
        "!k.is_multiple_of(FP8_BLOCK)",
    ] {
        assert!(eligibility.contains(guard), "eligibility omits `{guard}`");
    }
    let prefill = source("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
    assert!(prefill.contains("&mla.wq_b"));
    assert!(prefill.contains("nq * hd_mla"));
    assert!(prefill.contains("q_lora"));
    assert_eq!(64_u64 * 512, 32_768);
}

#[test]
fn deterministic_inputs_cover_small_and_full_shapes() {
    let inputs = section(
        &source(PROBE),
        "q_a W8A8 pipeline deterministic inputs",
        "//",
    )
    .to_owned();
    for needle in [
        "{1, kZeros",
        "{7, kExtremes",
        "{kProductionM, kVaried",
        "signed-zero",
        "max-finite",
        "E4M3",
        "block_scale",
        "weight_hash",
        "scale_hash",
    ] {
        assert!(inputs.contains(needle), "input model omits `{needle}`");
    }
}

#[test]
fn exact_a8_scale_and_full_bf16_output_bytes_are_required() {
    let parity = section(&source(PROBE), "q_a W8A8 pipeline exact parity", "//").to_owned();
    for needle in [
        "kPoisonA",
        "kPoisonB",
        "a8_mismatches",
        "row_scale_mismatches",
        "gemm_mismatches",
        "memcmp",
        "incumbent.gemm.download",
        "candidate.gemm.download",
        "input.guards_clean",
        "weight_fp8.guards_clean",
        "block_scale.guards_clean",
        "poison_a",
        "poison_b",
        "guards_clean",
    ] {
        assert!(parity.contains(needle), "parity omits `{needle}`");
    }
    assert!(!parity.contains("tolerance"));
    assert!(!parity.contains("cosine"));
}

#[test]
fn authorized_memory_bound_and_full_output_extent_are_exact() {
    const M: u64 = 2_410;
    const N: u64 = 32_768;
    const K: u64 = 1_024;
    let output = M * N * 2;
    assert_eq!(output, 157_941_760);
    let total = M * K * 2
        + K * 2
        + N * K
        + (N / 128) * (K / 128) * 4
        + M * K * 2
        + M * K
        + M * 4
        + output
        + M * K
        + M * 4
        + output;
    assert_eq!(total, 364_274_512);
    assert_eq!(total as f64 / 1_048_576.0, 347.399_246_215_820_3);
}

#[test]
fn production_timing_resets_every_input_and_output_before_events_and_is_abba() {
    let timing = section(&source(PROBE), "q_a W8A8 pipeline ABBA timing", "//").to_owned();
    for needle in [
        "kWarmupRounds = 2",
        "kTimingSamples = 4",
        "input.upload(host_input)",
        "norm_weight.upload(host_norm_weight)",
        "weight_fp8.upload(host_weight_fp8)",
        "block_scale.upload(host_block_scale)",
        "outputs.fill",
        "cudaEventRecord(begin)",
        "baseline, candidate, candidate, baseline",
        "baseline_ms",
        "candidate_ms",
        "speedup",
    ] {
        assert!(timing.contains(needle), "timing omits `{needle}`");
    }
    let reset = timing.find("input.upload(host_input)").unwrap();
    let event = timing.find("cuda_ok(cudaEventRecord(begin)").unwrap();
    assert!(reset < event);
}

#[test]
fn threshold_is_build_bound_canonical_and_precedes_all_build_work() {
    let build = source(BUILD);
    let admission = build.find("if [[ $# -ne 1 ]]").unwrap();
    let work = build.find("repo_root=").unwrap();
    assert!(admission < work);
    for needle in [
        "usage: $0 <min_speedup>",
        "invalid explicit numeric threshold",
        "printf \"%.17g\"",
        "echo \"min_speedup=$min_speedup\"",
        "expected_min_speedup",
        "grep -Fxc \"min_speedup=$expected_min_speedup\"",
        "[[ $# -eq 0 ]]",
        "\"$binary\" \"$expected_min_speedup\"",
    ] {
        assert!(
            build.contains(needle),
            "threshold contract omits `{needle}`"
        );
    }
    for bad in [
        "", " 1.01", "1.01 ", "junk", "nan", "inf", "0x1.1p1", "-1", "0", "1", "100.1",
    ] {
        assert!(!threshold_valid(bad), "accepted threshold {bad:?}");
    }
    for good in ["1.0001", "1.01", "2", "1e1", "100"] {
        assert!(threshold_valid(good), "rejected threshold {good:?}");
    }
}

#[test]
fn receipt_binds_sources_tools_commands_artifacts_and_exact_resources() {
    let build = source(BUILD);
    let inputs = section(&build, "q_a W8A8 pipeline immutable inputs", "#");
    for path in [
        "v4_prefill_qa_norm_w8a8_pipeline_probe.cu",
        "v4_prefill_qa_norm_w8a8_quant_fused.cu",
        "common/rms_norm_vanilla.cu",
        "common/w8a8_gemm_pipelined.cu",
        "prefill/v4_fp8_proj.rs",
        "prefill/cache_skip_v4.rs",
        "check-v4-prefill-qa-norm-w8a8-pipeline-probe-build.sh",
    ] {
        assert_eq!(inputs.matches(path).count(), 1, "immutable input {path}");
    }
    for identity in [
        "dependency_hashes",
        "git_commit",
        "git_status_hash",
        "nvcc_binary_hash",
        "cuobjdump_binary_hash",
        "host_cxx_binary_hash",
    ] {
        assert!(
            inputs.contains(identity),
            "immutable inputs omit `{identity}`"
        );
    }
    let receipt = section(&build, "q_a W8A8 pipeline receipt contract", "#");
    for field in [
        "min_speedup=",
        "build_id=",
        "compile_command=",
        "compile_command_sha256=",
        "resource_command=",
        "resource_command_sha256=",
        "resources_sha256=",
        "extract_command=",
        "extract_command_sha256=",
        "binary_sha256",
        "cubin_sha256",
    ] {
        assert!(receipt.contains(field), "receipt omits `{field}`");
    }
    for resource in [
        "rms_norm_vanilla",
        "quantize_a_fp8_rows",
        "v4_prefill_qa_norm_w8a8_quant_fused",
        "w8a8_gemm_pipelined",
        "expected_resource",
        "STACK:0",
        "LOCAL:0",
    ] {
        assert!(build.contains(resource), "resource gate omits `{resource}`");
    }
}

#[test]
fn generated_runner_is_bounded_and_requires_exact_full_parity() {
    let build = source(BUILD);
    let runner = section(&build, "q_a W8A8 pipeline runner verification", "#");
    for needle in [
        "actual_receipt_sha256",
        "actual_binary_sha256",
        "actual_cubin_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "max_output_bytes=4096",
        "max_output_lines=8",
        "shape m=2410 n=32768 k=1024",
        "a8_mismatches=0 row_scale_mismatches=0 gemm_mismatches=0",
        "result=PASS",
    ] {
        assert!(runner.contains(needle), "runner omits `{needle}`");
    }
    for injected in ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"] {
        assert!(build.contains(injected));
    }
}
