// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contracts for the isolated strided-K-full cache promotion probe.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_prefill_cache_kfull_fp8_fused_probe.cu";
const BUILD: &str = "scripts/check-v4-prefill-cache-kfull-fp8-fused-probe-build.sh";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required K-full probe input {relative}: {error}"))
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
fn baseline_starts_from_k_full_and_freezes_all_three_real_kernels() {
    let probe = read(PROBE);
    let kernels = section(&probe, "K-full cache probe kernel contract", "//");
    for contract in [
        "../deepseek-v4-flash/nvfp4/mla_absorbed.cu",
        "../common/reshape_and_cache.cu",
        "v4_prefill_cache_kfull_fp8_fused.cu",
        "mla_q_rope_extract_batched<<<dim3(kExtractCtas, 1, 1), dim3(kThreads, 1, 1)",
        "mla_cache_assemble_batched<<<dim3(kTokens, 1, 1), dim3(kCacheDim, 1, 1)",
        "reshape_and_cache_flash_fp8<<<dim3(kTokens, 1, 1), dim3(kThreads, 1, 1)",
        "v4_prefill_cache_kfull_fp8_fused<<<dim3(kTokens, 1, 1), dim3(kThreads, 1, 1)",
    ] {
        assert!(
            kernels.contains(contract),
            "missing kernel contract {contract}"
        );
    }
    for shape in [
        "kTokens = 2410",
        "kKvLora = 512",
        "kKHeadDim = 512",
        "kKNope = 448",
        "kRope = 64",
        "kCacheDim = 576",
        "kBlockSize = 16",
        "kNumBlocks = 151",
        "kExtractCtas = 603",
    ] {
        assert!(probe.contains(shape), "missing shape {shape}");
    }
    let flat = compact(kernels);
    assert!(flat.contains("k_full,rope_scratch,kTokens,1,kKHeadDim,kKNope,kRope,kKHeadDim"));
}

#[test]
fn exact_incremental_extraction_cost_is_source_locked() {
    const TOKENS: u64 = 2_410;
    const ROPE: u64 = 64;
    let bytes = TOKENS * ROPE * 2 * 2;
    let ctas = (TOKENS * ROPE).div_ceil(256);
    assert_eq!(bytes, 616_960);
    assert_eq!(ctas, 603);
    assert_eq!(bytes * 43, 26_529_280);
    assert_eq!(ctas * 43, 25_929);
    let probe = read(PROBE);
    assert!(probe.contains("incremental_bytes=616960"));
    assert!(probe.contains("incremental_ctas=603"));

    let host_ops = compact(&read("crates/spark-model/src/layers/ops/prefill_attn_a.rs"));
    for contract in [
        "constWIDE:u32=256",
        "letwide_grid=div_ceil(total,WIDE)",
        "lettotal=num_tokens*nq*rope",
        "let(grid,block)=rope_copy_launch_dims(total)",
        ".grid([grid,1,1]).block([block,1,1])",
    ] {
        assert!(
            host_ops.contains(contract),
            "host extractor launch drifted: {contract}"
        );
    }

    let flow_source =
        read("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
    let flow = compact(&flow_source);
    assert!(flow.contains("k_out,k_rope_tmp,n,nkv,hd_mla,nope,rope,nkv*hd_mla,stream"));
    let cache_call = flow_source
        .split("ops::mla_cache_assemble_batched(")
        .nth(1)
        .unwrap()
        .split(")?;")
        .next()
        .unwrap();
    for argument in [
        "kv_latent",
        "k_rope_tmp",
        "k_cache_assembled",
        "v_cache_assembled",
    ] {
        assert!(
            cache_call.contains(argument),
            "host cache flow lost {argument}"
        );
    }
}

#[test]
fn parity_has_opposite_swapped_poisons_and_exact_written_extent_audits() {
    let probe = read(PROBE);
    let parity = section(&probe, "K-full cache deterministic parity", "//");
    let flat = compact(parity);
    for contract in [
        "kParityCases = 3",
        "kPoisonCases = 2",
        "kPoisonA",
        "kPoisonB",
        "kScale",
        "vScale",
        "(token * 37u) % kTokens",
        "slot = -1",
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
        "rope_scratch",
        "assembly_scratch_guards_clean",
        "cache_guards_clean",
    ] {
        assert!(
            parity.contains(contract),
            "missing parity contract {contract}"
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
            "missing poison/write relationship {relationship}"
        );
    }
    assert!(!flat.contains("state.baseline.fill(value);state.candidate.fill(value)"));
    assert!(!flat.contains("memcmp(baseline_k.data(),candidate_k.data(),baseline_k.size())"));
    assert!(!parity.contains("out-of-pool parity"));

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
            baseline_poison,
            candidate_poison ^ 1,
            baseline_poison,
            candidate_poison
        ));
    }
}

#[test]
fn malformed_candidate_cases_are_full_cache_no_write_canaries() {
    let probe = read(PROBE);
    let malformed = section(&probe, "K-full cache malformed ABI", "//");
    for label in [
        "null-latent",
        "null-k-full",
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
        "unaligned-k-full",
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
        assert!(malformed.contains(contract), "missing malformed {contract}");
    }
}

#[test]
fn reset_is_outside_events_and_warmup_and_measured_order_are_abba() {
    let probe = read(PROBE);
    let timing = section(&probe, "K-full cache ABBA timing", "//");
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
        assert!(timing.contains(contract), "missing timing {contract}");
    }
    let flat = compact(timing);
    assert!(flat.contains("reset();check(cudaEventRecord(start)"));
    assert!(flat.contains("timed(true);timed(false);timed(false);timed(true)"));
    assert!(flat.contains("baseline_first=timed(true)"));
    assert!(flat.contains("candidate_first=timed(false)"));
    assert!(flat.contains("candidate_second=timed(false)"));
    assert!(flat.contains("baseline_second=timed(true)"));
}

#[test]
fn strict_build_bound_threshold_precedes_cuda() {
    let probe = read(PROBE);
    assert!(
        probe.find("parse_speedup_threshold").unwrap()
            < probe.find("cudaDriverGetVersion").unwrap()
    );
    for contract in [
        "argc != 2",
        "invalid explicit numeric threshold",
        "std::isfinite",
        "min_speedup <= 1.0",
        "min_speedup > 100.0",
    ] {
        assert!(probe.contains(contract), "missing threshold {contract}");
    }
    for bad in [
        "", " 1.01", "1.01 ", "junk", "nan", "inf", "0x1.1p1", "-1", "0", "1", "100.1",
    ] {
        assert!(!threshold_valid(bad), "accepted {bad:?}");
    }
}

#[test]
fn receipt_binds_sources_tools_commands_binary_cubins_and_runner_output() {
    let build = read(BUILD);
    let inputs = section(&build, "K-full cache probe immutable inputs", "#");
    for path in [
        "v4_prefill_cache_kfull_fp8_fused_probe.cu",
        "v4_prefill_cache_kfull_fp8_fused.cu",
        "deepseek-v4-flash/nvfp4/mla_absorbed.cu",
        "common/reshape_and_cache.cu",
        "spark-model/src/layers/ops/prefill_attn_a.rs",
        "spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
        "check-v4-prefill-cache-kfull-fp8-fused-probe-build.sh",
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
        assert!(inputs.contains(identity), "missing identity {identity}");
    }
    let receipt = section(&build, "K-full cache probe receipt contract", "#");
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
    let runner = section(&build, "K-full cache probe runner verification", "#");
    for field in [
        "actual_receipt_sha256",
        "actual_binary_sha256",
        "actual_cubin_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "max_output_bytes=4096",
        "max_output_lines=8",
        "parity cases=3 poison_cases=2 k_mismatches=0 v_mismatches=0",
        "malformed_cases=25 unchanged=25",
        "incremental_bytes=616960 incremental_ctas=603",
        "result=PASS",
    ] {
        assert!(runner.contains(field), "runner missing {field}");
    }
}

#[test]
fn build_is_fail_closed_and_probe_unreachable() {
    let build = read(BUILD);
    for contract in [
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing existing V4_CACHE_KFULL_PROBE_OUTPUT_DIR",
        "invalid explicit numeric threshold",
        "printf \"%.17g\"",
        "changed during V4 K-full cache probe compilation",
        "junk junk",
        "nan nan",
        "inf inf",
        "whitespace ' 1.01'",
        "hexadecimal 0x1.1p1",
        "no_win 1",
        "lax 100.1",
    ] {
        assert!(build.contains(contract), "build missing {contract}");
    }
    for production in [
        "crates/atlas-kernels/build.rs",
        "kernels/gb10/deepseek-v4-flash/MODEL.toml",
        "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
    ] {
        let source = read(production);
        assert!(!source.contains("v4_prefill_cache_kfull_fp8_fused_probe"));
        assert!(!source.contains("V4_CACHE_KFULL_PROBE"));
    }
}
