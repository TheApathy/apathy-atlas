// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contracts for the isolated V4 prefill cache-fusion promotion probe.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_prefill_cache_assemble_fp8_fused_probe.cu";
const BUILD: &str = "scripts/check-v4-prefill-cache-assemble-fp8-fused-probe-build.sh";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required cache-fusion probe input {relative}: {error}"))
}

fn compact(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn audit_byte(
    written: bool,
    baseline: u8,
    candidate: u8,
    baseline_poison: u8,
    candidate_poison: u8,
) -> bool {
    if written {
        baseline == candidate
    } else {
        baseline == baseline_poison && candidate == candidate_poison
    }
}

fn section<'a>(text: &'a str, name: &str, prefix: &str) -> &'a str {
    let begin = format!("{prefix} BEGIN {name}");
    let end = format!("{prefix} END {name}");
    assert_eq!(text.matches(&begin).count(), 1, "duplicate {begin}");
    assert_eq!(text.matches(&end).count(), 1, "duplicate {end}");
    let start = text.find(&begin).unwrap() + begin.len();
    let finish = text[start..].find(&end).unwrap() + start;
    assert!(start < finish, "reversed section {name}");
    &text[start..finish]
}

fn threshold_valid(value: &str) -> bool {
    if value.is_empty()
        || value.starts_with(char::is_whitespace)
        || value.ends_with(char::is_whitespace)
    {
        return false;
    }
    value
        .parse::<f64>()
        .is_ok_and(|number| number.is_finite() && number > 1.0 && number <= 100.0)
}

#[test]
fn exact_incumbent_chain_candidate_and_production_shape_are_frozen() {
    let probe = source(PROBE);
    let kernels = section(&probe, "cache fusion probe kernel contract", "//");
    for contract in [
        "../deepseek-v4-flash/nvfp4/mla_absorbed.cu",
        "../common/reshape_and_cache.cu",
        "v4_prefill_cache_assemble_fp8_fused.cu",
        "mla_cache_assemble_batched<<<dim3(kTokens, 1, 1), dim3(kCacheDim, 1, 1)",
        "reshape_and_cache_flash_fp8<<<dim3(kTokens, 1, 1), dim3(kThreads, 1, 1)",
        "v4_prefill_cache_assemble_fp8_fused<<<dim3(kTokens, 1, 1), dim3(kThreads, 1, 1)",
    ] {
        assert!(
            kernels.contains(contract),
            "missing kernel contract: {contract}"
        );
    }
    for exact in [
        "kTokens = 2410",
        "kKvLora = 512",
        "kRope = 64",
        "kCacheDim = 576",
        "kBlockSize = 16",
        "kThreads = 256",
        "kNumBlocks = 151",
        "kCacheStride = kBlockSize * kCacheDim",
    ] {
        assert!(probe.contains(exact), "missing shape contract: {exact}");
    }
}

#[test]
fn parity_uses_opposite_swapped_poisons_and_audits_written_extents() {
    let probe = source(PROBE);
    let cases = section(&probe, "cache fusion deterministic parity", "//");
    let flat = compact(cases);
    for contract in [
        "kParityCases = 3",
        "kPoisonCases = 2",
        "kPoisonA",
        "kPoisonB",
        "kScale",
        "vScale",
        "(token * 37u) % kTokens",
        "slot = -1",
        "unique valid nonmonotonic slots",
        "validate_parity_slots",
        "parity slot exceeds cache pool",
        "parity slot collision",
        "negative_slots",
        "valid_slots",
        "audit_cache_bytes",
        "written_bytes",
        "untouched_bytes",
        "baseline untouched cache byte changed",
        "candidate untouched cache byte changed",
        "k_mismatches",
        "v_mismatches",
        "scratch_guards_clean",
        "cache_guards_clean",
    ] {
        assert!(
            cases.contains(contract),
            "missing parity contract: {contract}"
        );
    }
    for relationship in [
        "baseline_poison=poison_pass==0?kPoisonA:kPoisonB",
        "candidate_poison=poison_pass==0?kPoisonB:kPoisonA",
        "state.baseline.fill(baseline_poison)",
        "state.candidate.fill(candidate_poison)",
        "if(written_slots[slot]!=0)",
        "baseline[index]!=candidate[index]",
        "baseline[index]!=baseline_poison",
        "candidate[index]!=candidate_poison",
        "expected_written=valid_slots*kCacheDim",
        "expected_untouched=(written_slots.size()-valid_slots)*kCacheDim",
    ] {
        assert!(
            flat.contains(relationship),
            "missing poison/write relationship: {relationship}"
        );
    }
    assert!(!flat.contains("state.baseline.fill(value);state.candidate.fill(value)"));
    assert!(!flat.contains("memcmp(baseline_k.data(),candidate_k.data(),baseline_k.size())"));
    assert!(!cases.contains("duplicate slot"));
    assert!(!cases.contains("out-of-pool parity"));

    // An intended byte left unwritten by both arms cannot pass either opposite
    // poison direction, while an unowned byte is checked against each arm's
    // own poison rather than compared across arms.
    for (baseline_poison, candidate_poison) in [(0x5a, 0xc3), (0xc3, 0x5a)] {
        assert_ne!(baseline_poison, candidate_poison);
        assert!(audit_byte(
            true,
            0x12,
            0x12,
            baseline_poison,
            candidate_poison
        ));
        assert!(!audit_byte(
            true,
            baseline_poison,
            candidate_poison,
            baseline_poison,
            candidate_poison
        ));
        assert!(audit_byte(
            false,
            baseline_poison,
            candidate_poison,
            baseline_poison,
            candidate_poison
        ));
        assert!(!audit_byte(
            false,
            baseline_poison ^ 1,
            candidate_poison,
            baseline_poison,
            candidate_poison
        ));
    }
}

#[test]
fn malformed_candidate_cases_are_positive_full_cache_no_write_canaries() {
    let probe = source(PROBE);
    let malformed = section(&probe, "cache fusion malformed ABI", "//");
    for label in [
        "null-latent",
        "null-rope",
        "null-k-cache",
        "null-v-cache",
        "null-slots",
        "aliased-cache",
        "wrong-tokens",
        "zero-blocks",
        "wrong-block-size",
        "wrong-cache-stride",
        "bad-k-scale",
        "bad-v-scale",
        "block-x",
        "block-y",
        "block-z",
        "grid-x-small",
        "grid-x-large",
        "grid-y",
        "grid-z",
        "unaligned-latent",
        "unaligned-rope",
        "unaligned-k-cache",
        "unaligned-v-cache",
        "unaligned-slots",
        "out-of-pool-slot",
    ] {
        assert_eq!(
            malformed.matches(label).count(),
            1,
            "malformed case {label}"
        );
    }
    for contract in [
        "kMalformedCases = 25",
        "expect_unchanged",
        "full K cache changed",
        "full V cache changed",
        "cudaGetLastError",
        "cudaDeviceSynchronize",
        "guards_clean",
    ] {
        assert!(
            malformed.contains(contract),
            "missing malformed contract: {contract}"
        );
    }
}

#[test]
fn inputs_are_reset_outside_events_and_timing_order_is_strict_abba() {
    let probe = source(PROBE);
    let timing = section(&probe, "cache fusion ABBA timing", "//");
    for contract in [
        "kWarmupRounds = 2",
        "kAbbaRounds = 6",
        "reset_inputs",
        "reset_outputs",
        "cudaEventRecord",
        "cudaEventElapsedTime",
        "baseline, candidate, candidate, baseline",
        "baseline_ms",
        "candidate_ms",
        "speedup",
    ] {
        assert!(
            timing.contains(contract),
            "missing timing contract: {contract}"
        );
    }
    let flat = compact(timing);
    assert!(flat.contains("reset();check(cudaEventRecord(start)"));
    assert!(
        flat.contains("timed(true);timed(false);timed(false);timed(true)"),
        "ABBA launch sequence drifted"
    );
}

#[test]
fn strict_threshold_precedes_every_cuda_call() {
    let probe = source(PROBE);
    let parse = probe.find("parse_speedup_threshold").unwrap();
    let cuda = probe.find("cudaDriverGetVersion").unwrap();
    assert!(parse < cuda);
    for contract in [
        "argc != 2",
        "invalid explicit numeric threshold",
        "std::isfinite",
        "min_speedup <= 1.0",
        "min_speedup > 100.0",
    ] {
        assert!(
            probe.contains(contract),
            "missing threshold contract: {contract}"
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
fn receipt_binds_all_transitive_sources_tools_commands_and_artifacts() {
    let build = source(BUILD);
    let inputs = section(&build, "cache fusion probe immutable inputs", "#");
    for path in [
        "v4_prefill_cache_assemble_fp8_fused_probe.cu",
        "v4_prefill_cache_assemble_fp8_fused.cu",
        "deepseek-v4-flash/nvfp4/mla_absorbed.cu",
        "common/reshape_and_cache.cu",
        "check-v4-prefill-cache-assemble-fp8-fused-probe-build.sh",
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
            "missing immutable identity {identity}"
        );
    }
    let receipt = section(&build, "cache fusion probe receipt contract", "#");
    for field in [
        "build_id=",
        "source_sha256",
        "compile_command=",
        "compile_command_sha256=",
        "resource_command=",
        "resource_command_sha256=",
        "extract_command=",
        "extract_command_sha256=",
        "binary_sha256",
        "cubin_sha256",
    ] {
        assert!(receipt.contains(field), "receipt missing {field}");
    }
}

#[test]
fn build_and_generated_runner_are_fail_closed_and_bounded() {
    let build = source(BUILD);
    let runner = section(&build, "cache fusion probe runner verification", "#");
    for contract in [
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing existing V4_CACHE_FP8_PROBE_OUTPUT_DIR",
        "invalid explicit numeric threshold",
        "printf \"%.17g\"",
        "changed during V4 cache-fusion probe compilation",
        "actual_receipt_sha256",
        "actual_binary_sha256",
        "actual_cubin_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "max_output_bytes=4096",
        "max_output_lines=8",
        "parity cases=3 poison_cases=2 k_mismatches=0 v_mismatches=0",
        "malformed_cases=25 unchanged=25",
        "result=PASS",
    ] {
        assert!(build.contains(contract), "build/runner missing {contract}");
    }
    for rejection in [
        "junk junk",
        "nan nan",
        "inf inf",
        "whitespace ' 1.01'",
        "hexadecimal 0x1.1p1",
        "no_win 1",
        "lax 100.1",
    ] {
        assert!(build.contains(rejection), "missing rejection {rejection}");
    }
    assert!(runner.contains("stdout_lines -ne $max_output_lines"));
}

#[test]
fn probe_remains_outside_production_reachability() {
    for relative in [
        "crates/atlas-kernels/build.rs",
        "kernels/gb10/deepseek-v4-flash/MODEL.toml",
        "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
    ] {
        let production = source(relative);
        assert!(!production.contains("v4_prefill_cache_assemble_fp8_fused_probe"));
        assert!(!production.contains("V4_CACHE_FP8_PROBE"));
    }
}
