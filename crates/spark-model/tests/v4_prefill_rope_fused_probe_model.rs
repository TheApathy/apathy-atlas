// SPDX-License-Identifier: AGPL-3.0-only

//! CPU/source contracts for the isolated V4 fused-RoPE promotion probe.

use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    fs::read_to_string(root().join(relative)).unwrap_or_else(|error| panic!("{relative}: {error}"))
}

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn threshold_is_valid(value: &str) -> bool {
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
fn exact_incumbent_chain_and_production_geometry_are_frozen() {
    let source = read("kernels/gb10/experiments/v4_prefill_rope_fused_probe.cu");
    let flat = compact(&source);
    for contract in [
        "../deepseek-v4-flash/nvfp4/mla_absorbed.cu",
        "../common/rope.cu",
        "v4_prefill_rope_fused.cu",
        "mla_q_rope_extract_batched<<<",
        "rope_forward_yarn_interleaved<<<",
        "rope_forward_yarn_interleaved_inv<<<",
        "mla_q_rope_writeback_batched<<<",
        "v4_prefill_rope_fused_forward<<<",
        "v4_prefill_rope_fused_inverse<<<",
    ] {
        assert!(
            source.contains(contract),
            "missing chain contract: {contract}"
        );
    }
    for contract in [
        "kTokens=2410",
        "kNq=64",
        "kNkv=1",
        "kHeadDim=512",
        "kNopeDim=448",
        "kRopeDim=64",
        "dim3(kTokens,kNq+kNkv,1)",
        "dim3(kTokens,kNq,1)",
        "dim3(32,1,1)",
        "dim3(128,1,1)",
        "dim3(256,1,1)",
    ] {
        assert!(
            flat.contains(contract),
            "missing geometry contract: {contract}"
        );
    }
}

#[test]
fn candidate_rejects_alias_and_nonfinite_mscale_before_access() {
    let candidate = read("kernels/gb10/experiments/v4_prefill_rope_fused.cu");
    let flat = compact(&candidate);
    assert!(flat.contains("Q==K"));
    assert_eq!(flat.matches("!isfinite(mscale)").count(), 2);
    let forward_guard = flat.find("Q==K").expect("forward alias guard");
    let first_forward_access = flat
        .find("constunsignedinttoken=blockIdx.x")
        .expect("first forward pointer mapping");
    assert!(forward_guard < first_forward_access);
}

#[test]
fn sass_gate_refreezes_guarded_component_topology() {
    let gate = read("scripts/check-v4-prefill-rope-fused-sass.sh");
    assert!(gate.contains("check_function v4_prefill_rope_fused_forward 338"));
    assert!(gate.contains("check_function v4_prefill_rope_fused_inverse 328"));
    assert!(gate.contains("guard_exit_line"));
    assert!(gate.contains("first_global_load_line"));
    assert!(gate.contains("guard_before_global=1"));
}

#[test]
fn multiple_tables_exact_bytes_and_exhaustive_poison_guards_are_required() {
    let source = read("kernels/gb10/experiments/v4_prefill_rope_fused_probe.cu");
    for contract in [
        "kParityCases = 3",
        "memcmp",
        "forward_mismatches",
        "inverse_mismatches",
        "position_salt",
        "frequency_salt",
        "mscale",
        "BEGIN V4 RoPE forward poison guard",
        "BEGIN V4 RoPE inverse poison guard",
        "kPoisonCases = 38",
        "cudaMemset",
        "guards_unchanged",
        "run_forward_guard",
        "run_inverse_guard",
        "GuardedBf16Buffer",
        "kRedzoneBytes = 256",
        "payload pointer preserves cudaMalloc's 256-byte alignment",
        "redzone_buffers",
    ] {
        assert!(
            source.contains(contract),
            "missing parity/guard contract: {contract}"
        );
    }
}

#[test]
fn every_candidate_pointer_shape_block_and_grid_guard_has_a_no_write_case() {
    let source = read("kernels/gb10/experiments/v4_prefill_rope_fused_probe.cu");
    for case in [
        "forward-null-q",
        "forward-null-k",
        "forward-null-positions",
        "forward-null-inv-freq",
        "forward-q-k-alias",
        "forward-mscale-nan",
        "forward-mscale-pos-inf",
        "forward-mscale-neg-inf",
        "forward-zero-tokens",
        "forward-q-heads",
        "forward-kv-heads",
        "forward-head-dim",
        "forward-nope-dim",
        "forward-rotary-dim",
        "forward-block-x",
        "forward-block-y",
        "forward-block-z",
        "forward-grid-x",
        "forward-grid-y",
        "forward-grid-z",
        "inverse-null-q",
        "inverse-null-positions",
        "inverse-null-inv-freq",
        "inverse-mscale-nan",
        "inverse-mscale-pos-inf",
        "inverse-mscale-neg-inf",
        "inverse-zero-tokens",
        "inverse-q-heads",
        "inverse-kv-heads",
        "inverse-head-dim",
        "inverse-nope-dim",
        "inverse-rotary-dim",
        "inverse-block-x",
        "inverse-block-y",
        "inverse-block-z",
        "inverse-grid-x",
        "inverse-grid-y",
        "inverse-grid-z",
    ] {
        assert!(source.contains(case), "missing guard canary: {case}");
    }

    let flat = compact(&source);
    for contract in [
        "run_forward_guard(nullptr,k_candidate.get(),d_positions.get(),d_frequencies.get()",
        "run_forward_guard(q_candidate.get(),nullptr,d_positions.get(),d_frequencies.get()",
        "run_forward_guard(q_candidate.get(),k_candidate.get(),nullptr,d_frequencies.get()",
        "run_forward_guard(q_candidate.get(),k_candidate.get(),d_positions.get(),nullptr",
        "run_forward_guard(q_candidate.get(),q_candidate.get(),d_positions.get(),d_frequencies.get()",
        "run_inverse_guard(nullptr,d_positions.get(),d_frequencies.get()",
        "run_inverse_guard(inverse_candidate.get(),nullptr,d_frequencies.get()",
        "run_inverse_guard(inverse_candidate.get(),d_positions.get(),nullptr",
        "dim3(16,2,1)",
        "dim3(16,1,2)",
        "dim3(kTokens+1,kNq+kNkv,1)",
        "dim3(kTokens,kNq+kNkv+1,1)",
        "dim3(kTokens,kNq+kNkv,2)",
        "dim3(kTokens+1,kNq,1)",
        "dim3(kTokens,kNq+1,1)",
        "dim3(kTokens,kNq,2)",
        "poison_payload(0xA5)",
        "poison_payload(0x5A)",
        "all_poison(actual,0xA5)",
        "all_poison(actual_k,0x5A)",
    ] {
        assert!(
            flat.contains(contract),
            "missing executable canary contract: {contract}"
        );
    }
}

#[test]
fn one_strict_threshold_precedes_cuda_and_abba_uses_events() {
    let source = read("kernels/gb10/experiments/v4_prefill_rope_fused_probe.cu");
    let parse = source.find("parse_speedup_threshold").unwrap();
    let first_cuda = source.find("cudaGetDevice").unwrap();
    assert!(parse < first_cuda);
    for contract in [
        "argc != 2",
        "invalid explicit numeric threshold",
        "std::isfinite",
        "std::strtod",
        "min_speedup <= 1.0",
        "min_speedup > 100.0",
        "BEGIN V4 RoPE ABBA timing",
        "cudaEventRecord",
        "cudaEventElapsedTime",
        "baseline, candidate, candidate, baseline",
        "kAbbaRounds",
        "time_abba(baseline_reset, baseline, candidate_reset, candidate)",
    ] {
        assert!(
            source.contains(contract),
            "missing timing contract: {contract}"
        );
    }
    let flat = compact(&source);
    assert!(flat.contains("setup();check(cudaEventRecord(start)"));
    assert!(source.matches("timing reset Q").count() >= 2);
    assert!(source.matches("timing reset K").count() >= 2);
    assert!(source.matches("timing reset inverse").count() >= 2);
}

#[test]
fn cpu_threshold_oracle_rejects_nonfinite_nonwinning_and_ambiguous_values() {
    for bad in [
        "", " 1.1", "1.1 ", "junk", "nan", "inf", "-inf", "-1", "0", "1", "100.1", "1.1junk",
    ] {
        assert!(!threshold_is_valid(bad), "accepted {bad:?}");
    }
    for good in ["1.00000001", "1.0001", "1.05", "2", "100", "1e1"] {
        assert!(threshold_is_valid(good), "rejected {good:?}");
    }
}

#[test]
fn receipt_binds_transitive_inputs_tools_commands_cubins_and_build_id() {
    let script = read("scripts/check-v4-prefill-rope-fused-probe-build.sh");
    for contract in [
        "# BEGIN V4 RoPE probe immutable inputs",
        "v4_prefill_rope_fused_probe.cu",
        "v4_prefill_rope_fused.cu",
        "kernels/gb10/common/rope.cu",
        "kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu",
        "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
        "crates/spark-model/src/layers/ops/embeddings.rs",
        "nvcc_binary_sha256",
        "cuobjdump_binary_sha256",
        "host_cxx_binary_sha256",
        "git_status_sha256",
        "compile_command_template=",
        "compile_command=",
        "binary_sha256",
        "cubin_sha256",
        "build_id=",
        "source_sha256",
        "# BEGIN V4 RoPE probe runner verification",
        "expected_receipt_sha256",
        "expected_binary_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "printf \"%.17g\"",
    ] {
        assert!(
            script.contains(contract),
            "missing receipt contract: {contract}"
        );
    }
}

#[test]
fn production_forward_inverse_chain_and_wrapper_arguments_are_source_locked() {
    let prefill = read("crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs");
    let forward_extract = prefill
        .find("ops::mla_q_rope_extract_batched(")
        .expect("production forward extract");
    let forward_rope = prefill[forward_extract..]
        .find("ops::rope_yarn(")
        .expect("production forward RoPE")
        + forward_extract;
    let forward_writeback = prefill[forward_rope..]
        .find("ops::mla_q_rope_writeback_batched(")
        .expect("production forward writeback")
        + forward_rope;
    assert!(forward_extract < forward_rope && forward_rope < forward_writeback);
    let forward = compact(&prefill[forward_rope..forward_writeback]);
    for contract in [
        "self.rope_yarn_interleaved_k",
        "q_rope_tmp,k_rope_tmp,meta.positions,n,nq,nkv,rope,rope",
        "mla.main_inv_freq",
        "mla.yarn_inv_freq",
    ] {
        assert!(
            forward.contains(contract),
            "forward host chain omits `{contract}`"
        );
    }

    let inverse_start = prefill
        .find("// DeepSeek-V4 eq.26: de-rotate")
        .expect("production inverse chain");
    let inverse = compact(&prefill[inverse_start..]);
    for contract in [
        "ops::mla_q_rope_extract_batched(",
        "self.rope_yarn_interleaved_inv_k",
        "o_rope_tmp,o_rope_tmp,meta.positions,n,nq,0,rope,rope",
        "mla.main_inv_freq",
        "mla.yarn_inv_freq",
        "ops::mla_q_rope_writeback_batched(",
    ] {
        assert!(
            inverse.contains(contract),
            "inverse host chain omits `{contract}`"
        );
    }

    let wrapper = compact(&read("crates/spark-model/src/layers/ops/embeddings.rs"));
    for contract in [
        "letpos_per_block=(128/half_rot).max(1)",
        ".grid([num_q_heads+num_kv_heads,seq_blocks,1])",
        ".block([128,1,1])",
        ".arg_ptr(q).arg_ptr(k).arg_ptr(positions)",
        ".arg_u32(seq_len).arg_u32(num_q_heads).arg_u32(num_kv_heads)",
        ".arg_u32(head_dim).arg_u32(rotary_dim).arg_ptr(inv_freq).arg_f32(theta)",
    ] {
        assert!(
            wrapper.contains(contract),
            "RoPE wrapper omits `{contract}`"
        );
    }
}

#[test]
fn build_and_runner_are_fail_closed_and_output_is_bounded() {
    let script = read("scripts/check-v4-prefill-rope-fused-probe-build.sh");
    for contract in [
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing existing V4_ROPE_PROBE_OUTPUT_DIR",
        "invalid explicit numeric threshold",
        "changed during V4 RoPE probe compilation",
        "max_output_bytes=4096",
        "max_output_lines=8",
        "forward_mismatches=0 inverse_mismatches=0",
        "poison_cases=38 unchanged=38 redzone_buffers=6 prefix=clean suffix=clean",
        "result=PASS",
    ] {
        assert!(
            script.contains(contract),
            "missing fail-closed contract: {contract}"
        );
    }
}

#[test]
fn experiment_remains_outside_serving_and_registry() {
    for relative in [
        "crates/atlas-kernels/build.rs",
        "kernels/gb10/deepseek-v4-flash/MODEL.toml",
        "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs",
    ] {
        let source = read(relative);
        assert!(
            !source.contains("v4_prefill_rope_fused_probe"),
            "{relative}"
        );
        assert!(!source.contains("V4_ROPE_PROBE"), "{relative}");
    }
}
