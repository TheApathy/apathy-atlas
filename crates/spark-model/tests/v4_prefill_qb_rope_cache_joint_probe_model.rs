// SPDX-License-Identifier: AGPL-3.0-only

//! Offline contracts for the isolated composed V4 Q-B/RoPE/cache probe.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_prefill_qb_rope_cache_joint_probe.cu";
const BUILD: &str = "scripts/check-v4-prefill-qb-rope-cache-joint-probe-build.sh";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required joint probe input {relative}: {error}"))
}

fn compact(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
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

#[test]
fn directly_reuses_both_candidates_and_every_real_incumbent_kernel() {
    let probe = read(PROBE);
    let kernels = section(&probe, "joint kernel contract", "//");
    for include in [
        "../common/rms_norm.cu",
        "../deepseek-v4-flash/nvfp4/mla_absorbed.cu",
        "../common/rope.cu",
        "../common/reshape_and_cache.cu",
        "../deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu",
        "../deepseek-v4-flash/nvfp4/v4_prefill_cache_kfull_fp8_fused.cu",
    ] {
        assert!(
            kernels.contains(include),
            "missing direct include {include}"
        );
    }
    assert_eq!(
        probe.matches("__global__").count(),
        0,
        "local CUDA oracle added"
    );
    for shape in [
        "kTokens = 2410",
        "kNq = 64",
        "kNkv = 1",
        "kHeadDim = 512",
        "kNope = 448",
        "kRope = 64",
        "kCacheDim = 576",
        "kBlockSize = 16",
        "kNumBlocks = 151",
        "kQExtractCtas = 38560",
        "kKExtractCtas = 603",
    ] {
        assert!(probe.contains(shape), "missing exact shape {shape}");
    }
}

#[test]
fn whole_baseline_and_candidate_chains_freeze_launch_abis() {
    let probe = read(PROBE);
    let launches = compact(section(&probe, "joint launch ABIs", "//"));
    let mut cursor = 0;
    for launch in [
        "rms_norm<<<dim3(kTokens*kNq,1,1),dim3(kNormThreads,1,1)",
        "mla_q_rope_extract_batched<<<dim3(kQExtractCtas,1,1),dim3(kCopyThreads,1,1)",
        "mla_q_rope_extract_batched<<<dim3(kKExtractCtas,1,1),dim3(kCopyThreads,1,1)",
        "rope_forward_yarn_interleaved<<<dim3(kNq+kNkv,kRopeSeqCtas,1),dim3(kRopeThreads,1,1)",
        "mla_q_rope_writeback_batched<<<dim3(kQExtractCtas,1,1),dim3(kCopyThreads,1,1)",
        "mla_q_rope_writeback_batched<<<dim3(kKExtractCtas,1,1),dim3(kCopyThreads,1,1)",
        "mla_cache_assemble_batched<<<dim3(kTokens,1,1),dim3(kCacheDim,1,1)",
        "reshape_and_cache_flash_fp8<<<dim3(kTokens,1,1),dim3(kCopyThreads,1,1)",
    ] {
        let offset = launches[cursor..]
            .find(launch)
            .unwrap_or_else(|| panic!("missing or misordered incumbent launch {launch}"));
        cursor += offset + launch.len();
    }
    for launch in [
        "v4_prefill_qb_norm_rope_fused<<<dim3(kTokens,kNq,1),dim3(kNormThreads,1,1)",
        "v4_prefill_cache_kfull_fp8_fused<<<dim3(kTokens,1,1),dim3(kCopyThreads,1,1)",
    ] {
        assert!(launches.contains(launch), "missing candidate ABI {launch}");
    }
    assert!(launches.contains("launch_abi_count=2"));
}

#[test]
fn parity_covers_yarn_slots_poisons_full_outputs_and_no_scratch() {
    let probe = read(PROBE);
    let parity = section(&probe, "joint deterministic parity", "//");
    let flat = compact(parity);
    for contract in [
        "kParityCases = 2",
        "main",
        "compressor-yarn",
        "kPoisonCases = 2",
        "(token * 37u) % kTokens",
        "slot = -1",
        "kScale",
        "vScale",
        "q_mismatches",
        "k_mismatches",
        "cache_k_mismatches",
        "cache_v_mismatches",
        "full Q",
        "full K",
        "valid cache bytes",
        "holes and tails",
        "immutable_inputs_clean",
        "all_guards_clean",
        "candidate_no_scratch",
        "allocations_disjoint",
    ] {
        assert!(
            parity.contains(contract),
            "missing parity contract {contract}"
        );
    }
    for relation in [
        "baseline_poison=poison_pass==0?kPoisonA:kPoisonB",
        "candidate_poison=poison_pass==0?kPoisonB:kPoisonA",
        "baseline[index]!=candidate[index]",
        "baseline[index]!=baseline_poison",
        "candidate[index]!=candidate_poison",
        "slot>=kTokens",
    ] {
        assert!(flat.contains(relation), "missing exact audit {relation}");
    }
    assert!(!flat.contains("tolerance"));
}

#[test]
fn abba_times_complete_chains_with_resets_outside_events() {
    let timing = section(&read(PROBE), "joint ABBA timing", "//").to_owned();
    let flat = compact(&timing);
    for contract in [
        "baseline, candidate, candidate, baseline",
        "kAbbaRounds = 6",
        "reset_baseline(); check(cudaEventRecord(start)",
        "launch_baseline_chain(); check(cudaEventRecord(finish)",
        "reset_candidate(); check(cudaEventRecord(start)",
        "launch_candidate_chain(); check(cudaEventRecord(finish)",
        "cudaEventElapsedTime",
    ] {
        assert!(
            flat.contains(&compact(contract)),
            "timing contract missing {contract}"
        );
    }
}

#[test]
fn build_is_sm121a_binary64_provenance_locked_and_runtime_is_bounded() {
    let build = read(BUILD);
    for contract in [
        "-arch=sm_121a",
        "-std=c++17",
        "-O3",
        "--fmad=false",
        "ctypes.CDLL(None)",
        "strtod",
        "format(value, '.17g')",
        "min_speedup=$min_speedup",
        "max_output_bytes=4096",
        "max_output_lines=9",
        "if [[ $# -ne 0 ]]",
        "binary_sha256",
        "cubin_sha256",
        "compile_command_sha256",
        "resource_command_sha256",
        "extract_command_sha256",
        "runner_sha256",
    ] {
        assert!(
            build.contains(contract),
            "build contract missing {contract}"
        );
    }
    for dependency in [
        PROBE,
        "kernels/gb10/common/rms_norm.cu",
        "kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu",
        "kernels/gb10/common/rope.cu",
        "kernels/gb10/common/reshape_and_cache.cu",
        "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu",
        "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_cache_kfull_fp8_fused.cu",
        "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
        "crates/spark-model/src/layers/qwen3_attention/init.rs",
    ] {
        assert!(
            build.contains(dependency),
            "unbound dependency {dependency}"
        );
    }
    for tool in [
        "nvcc_binary_sha256",
        "cuobjdump_binary_sha256",
        "host_cxx_binary_sha256",
    ] {
        assert!(build.contains(tool), "unbound tool {tool}");
    }
    assert!(build.contains("invalid thresholds reject before CUDA"));
}

#[test]
fn package_stays_isolated_from_production_reachability() {
    let probe_name = "v4_prefill_qb_rope_cache_joint_probe";
    for production in [
        "crates/spark-model/src/layers/qwen3_attention/init.rs",
        "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
        "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu",
        "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_cache_kfull_fp8_fused.cu",
    ] {
        assert!(!read(production).contains(probe_name));
    }
}
