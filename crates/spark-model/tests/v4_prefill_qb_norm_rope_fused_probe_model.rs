// SPDX-License-Identifier: AGPL-3.0-only

//! Offline source contracts for the isolated V4 Q-B RMSNorm + RoPE CUDA probe.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_prefill_qb_norm_rope_fused_probe.cu";
const BUILD: &str = "scripts/check-v4-prefill-qb-norm-rope-fused-probe-build.sh";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(root().join(relative))
        .unwrap_or_else(|error| panic!("required Q-B probe input {relative} is missing: {error}"))
}

fn compact(text: &str) -> String {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .collect()
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

fn admitted_f32(value: &str) -> Option<f32> {
    let rounded = value.parse::<f32>().ok()?;
    (rounded.is_finite() && rounded > 1.0 && rounded <= 100.0).then_some(rounded)
}

#[test]
fn local_oracle_and_candidate_freeze_exact_geometries() {
    let probe = source(PROBE);
    let kernel_contract = section(&probe, "V4 Q-B probe kernel contract", "//");
    assert!(probe.contains("v4_prefill_qb_norm_rope_fused.cu"));
    for contract in [
        "probe_incumbent_qb_rms",
        "probe_incumbent_direct_rope",
        "v4_prefill_qb_norm_rope_fused<<<",
    ] {
        assert!(
            kernel_contract.contains(contract),
            "kernel contract omits `{contract}`"
        );
    }
    let flat = compact(kernel_contract);
    for geometry in [
        "dim3(tokens*kNq,1,1),dim3(kThreads,1,1)",
        "dim3(tokens,kNq+kNkv,1),dim3(kRopePairs,1,1)",
        "dim3(tokens,kNq,1),dim3(kThreads,1,1)",
    ] {
        assert!(
            flat.contains(geometry),
            "kernel geometry omits `{geometry}`"
        );
    }
    for exact in [
        "kProductionTokens = 2410",
        "kNq = 64",
        "kNkv = 1",
        "kHeadDim = 512",
        "kNopeDim = 448",
        "kRopeDim = 64",
        "kThreads = 512",
        "kTokenCases[] = {1, 7, 128, kProductionTokens}",
    ] {
        assert!(probe.contains(exact), "shape contract omits `{exact}`");
    }
}

#[test]
fn oracle_locks_reduction_rounding_rope_order_and_zero_weight() {
    let probe = source(PROBE);
    let oracle = section(&probe, "V4 Q-B exact local oracle", "//");
    for contract in [
        "sum_sq += x0 * x0 + x1 * x1",
        "__shfl_xor_sync(0xFFFFFFFF, value, offset)",
        "warp_sums[32]",
        "rsqrtf(warp_sums[0] / static_cast<float>(kHeadDim) + eps)",
        "x0 * rms * (1.0f + w0)",
        "x1 * rms * (1.0f + w1)",
        "__float2bfloat16",
        "static_cast<float>(positions[token]) * inv_freq[pair]",
        "x0 * cos_value - x1 * sin_value",
        "x1 * cos_value + x0 * sin_value",
    ] {
        assert!(oracle.contains(contract), "oracle omits `{contract}`");
    }
    let parity = section(&probe, "V4 Q-B exact byte parity", "//");
    for contract in [
        "kTokenCases",
        "memcmp",
        "q_mismatch_bytes",
        "k_mismatch_bytes",
        "full Q and K",
        "zero_weight",
        "distinct positions",
        "distinct frequencies",
        "prefix_guard",
        "suffix_guard",
    ] {
        assert!(parity.contains(contract), "parity omits `{contract}`");
    }
    assert!(!parity.contains("tolerance"));

    let rms = source("kernels/gb10/common/rms_norm.cu");
    for contract in [
        "sum_sq += v0 * v0 + v1 * v1",
        "sum_sq = warp_reduce_sum(sum_sq)",
        "float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps)",
        "xv0 * rms * (1.0f + wv0)",
        "xv1 * rms * (1.0f + wv1)",
    ] {
        assert!(rms.contains(contract), "production RMS omits `{contract}`");
    }
    let rope = source("kernels/gb10/common/rope.cu");
    let interleaved = rope
        .split("void rope_forward_yarn_interleaved(")
        .nth(1)
        .expect("production interleaved RoPE");
    for contract in [
        "const unsigned int d0 = 2 * pair_idx",
        "const unsigned int d1 = 2 * pair_idx + 1",
        "float y0 = x0 * cos_val - x1 * sin_val",
        "float y1 = x1 * cos_val + x0 * sin_val",
    ] {
        assert!(
            interleaved.contains(contract),
            "production RoPE omits `{contract}`"
        );
    }
    let prefill = source("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
    let projection_at = prefill.find("\"V4 wq_b\"").expect("wq_b projection");
    let norm_at = prefill.find("// q_b_norm:").expect("q_b normalization");
    let q_extract_at = prefill[norm_at..]
        .find("ops::mla_q_rope_extract_batched(")
        .expect("Q RoPE extraction")
        + norm_at;
    let rope_at = prefill[q_extract_at..]
        .find("ops::rope_yarn(")
        .expect("direct Q/K RoPE dispatch")
        + q_extract_at;
    assert!(projection_at < norm_at && norm_at < q_extract_at && q_extract_at < rope_at);
    let norm_dispatch = compact(&prefill[norm_at..q_extract_at]);
    for contract in [
        "ops::rms_norm(ctx.gpu,self.rms_norm_k,q_full,",
        "weight:ctx.buffers.norm_unit_w()",
        "q_full,n*nq,hd_mla,eps,stream",
    ] {
        assert!(
            norm_dispatch.contains(contract),
            "production q_b normalization dispatch omits `{contract}`"
        );
    }
    let rope_end = prefill[rope_at..]
        .find("ops::mla_q_rope_writeback_batched(")
        .expect("Q RoPE writeback")
        + rope_at;
    let rope_dispatch = &prefill[rope_at..rope_end];
    let mut cursor = 0;
    for contract in [
        "ops::rope_yarn(",
        "self.rope_yarn_interleaved_k",
        "q_rope_tmp",
        "k_rope_tmp",
        "meta.positions",
        "nq",
        "nkv",
        "mla.main_inv_freq",
        "mla.yarn_inv_freq",
    ] {
        cursor += rope_dispatch[cursor..]
            .find(contract)
            .unwrap_or_else(|| panic!("production direct RoPE dispatch omits `{contract}`"));
    }
    let buffers = source("crates/spark-runtime/src/buffers.rs");
    let allocate_at = buffers
        .find("let norm_unit_w = gpu.alloc(sizes.norm_unit_w)?;")
        .expect("unit RMS weight allocation");
    let zero_at = buffers
        .find("gpu.memset(norm_unit_w, 0, sizes.norm_unit_w)?;")
        .expect("unit RMS weight zeroing");
    assert!(allocate_at < zero_at);
    assert!(buffers[zero_at..].contains("norm_unit_w,"));
}

#[test]
fn every_candidate_guard_has_an_executable_no_write_canary() {
    let probe = source(PROBE);
    let malformed = section(&probe, "V4 Q-B malformed ABI", "//");
    for label in [
        "null-q",
        "null-k",
        "null-weight",
        "null-positions",
        "null-inv-freq",
        "q-k-alias",
        "zero-tokens",
        "q-heads",
        "kv-heads",
        "head-dim",
        "nope-dim",
        "rotary-dim",
        "eps-zero",
        "eps-nan",
        "eps-inf",
        "mscale-nan",
        "block-x",
        "block-y",
        "block-z",
        "grid-x",
        "grid-y",
        "grid-z",
    ] {
        assert_eq!(
            malformed.matches(label).count(),
            1,
            "malformed ABI case `{label}` is absent or duplicated"
        );
    }
    for contract in [
        "kMalformedCases = 22",
        "before == after",
        "cudaGetLastError",
        "cudaDeviceSynchronize",
        "guards_clean",
    ] {
        assert!(
            malformed.contains(contract),
            "malformed ABI omits `{contract}`"
        );
    }
}

#[test]
fn timing_resets_inputs_before_events_and_orders_abba() {
    let probe = source(PROBE);
    let timing = section(&probe, "V4 Q-B ABBA timing", "//");
    for contract in [
        "baseline, candidate, candidate, baseline",
        "setup();",
        "cudaEventRecord(start)",
        "cudaEventRecord(finish)",
        "cudaEventElapsedTime",
        "timing reset Q",
        "timing reset K",
    ] {
        assert!(timing.contains(contract), "timing omits `{contract}`");
    }
    assert!(probe.contains("kAbbaRounds = 6"));
    let flat = compact(timing);
    assert!(flat.contains("setup();cuda_ok(cudaEventRecord(start)"));
    assert!(flat.contains(
        "event_time(start,finish,baseline_reset,baseline);candidate_ms+=event_time(start,finish,candidate_reset,candidate);candidate_ms+=event_time(start,finish,candidate_reset,candidate);baseline_ms+=event_time(start,finish,baseline_reset,baseline)"
    ));
}

#[test]
fn threshold_grammar_is_strict_and_precedes_cuda() {
    let probe = source(PROBE);
    let parser = probe.find("parse_speedup_threshold").unwrap();
    let first_cuda = probe.find("cudaGetDevice").unwrap();
    assert!(parser < first_cuda);
    for contract in [
        "argc != 2",
        "invalid explicit numeric threshold",
        "std::isfinite",
        "min_speedup <= 1.0f",
        "min_speedup > 100.0f",
    ] {
        assert!(probe.contains(contract), "threshold omits `{contract}`");
    }

    let build = source(BUILD);
    for contract in [
        "canonicalize_f32_threshold",
        "libc.strtof.argtypes",
        "libc.strtof.restype = ctypes.c_float",
        "format(rounded, \".9g\")",
        "1.0 < rounded <= 100.0",
        "min_speedup=$(canonicalize_f32_threshold",
    ] {
        assert!(build.contains(contract), "f32 admission omits `{contract}`");
    }
    assert_eq!(admitted_f32("1.00000001"), None);
    assert!(admitted_f32("1.0001").is_some_and(|value| value > 1.0));
    assert_eq!(admitted_f32("100.000001"), Some(100.0));
    for rejection in [
        "reject_threshold usage",
        "reject_threshold usage_extra 1.01 extra",
        "reject_threshold junk junk",
        "reject_threshold nan nan",
        "reject_threshold inf inf",
        "reject_threshold whitespace ' 1.01'",
        "reject_threshold hexadecimal 0x1.1p1",
        "reject_threshold negative -1",
        "reject_threshold no_win 1",
        "reject_threshold rounded_to_one 1.00000001",
        "reject_threshold lax 100.1",
    ] {
        assert!(build.contains(rejection), "missing rejection `{rejection}`");
    }
}

#[test]
fn receipt_binds_sources_tools_commands_binary_cubins_and_build_id() {
    let build = source(BUILD);
    let inputs = section(&build, "V4 Q-B probe immutable inputs", "#");
    for contract in [
        "v4_prefill_qb_norm_rope_fused_probe.cu",
        "v4_prefill_qb_norm_rope_fused.cu",
        "common/rms_norm.cu",
        "common/rope.cu",
        "qwen3_attention/prefill/cache_skip_v4.rs",
        "spark-runtime/src/buffers.rs",
        "check-v4-prefill-qb-norm-rope-fused-probe-build.sh",
        "dependency_hashes",
        "nvcc_binary_sha256",
        "cuobjdump_binary_sha256",
        "host_cxx_binary_sha256",
        "git_commit",
        "git_status_sha256",
    ] {
        assert!(
            inputs.contains(contract),
            "immutable inputs omit `{contract}`"
        );
    }
    let receipt = section(&build, "V4 Q-B probe receipt contract", "#");
    for field in [
        "build_id=",
        "compile_command=",
        "compile_command_sha256=",
        "resource_command=",
        "resource_command_sha256=",
        "extract_command=",
        "extract_command_sha256=",
        "binary_sha256",
        "cubin_sha256",
        "python_path=",
        "python_version=",
        "python_binary_sha256=",
    ] {
        assert!(receipt.contains(field), "receipt omits `{field}`");
    }
    let runner = section(&build, "V4 Q-B probe runner verification", "#");
    for field in [
        "expected_receipt_sha256",
        "expected_binary_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "actual_cubin_sha256",
    ] {
        assert!(runner.contains(field), "runner omits `{field}`");
    }
}

#[test]
fn runner_output_is_bounded_and_fail_closed() {
    let build = source(BUILD);
    for contract in [
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing existing V4_QB_NORM_ROPE_PROBE_OUTPUT_DIR",
        "changed during V4 Q-B probe compilation",
        "max_output_bytes=4096",
        "max_output_lines=8",
        "q_mismatch_bytes=0 k_mismatch_bytes=0",
        "malformed_cases=22 no_write=22 prefix=clean suffix=clean",
        "result=PASS",
    ] {
        assert!(build.contains(contract), "build/runner omits `{contract}`");
    }
}

#[test]
fn probe_stays_outside_serving_and_registry() {
    for relative in [
        "crates/atlas-kernels/build.rs",
        "kernels/gb10/deepseek-v4-flash/MODEL.toml",
        "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
    ] {
        let production = source(relative);
        assert!(
            !production.contains("v4_prefill_qb_norm_rope_fused_probe"),
            "{relative}"
        );
        assert!(!production.contains("V4_QB_NORM_ROPE_PROBE"), "{relative}");
    }
}
