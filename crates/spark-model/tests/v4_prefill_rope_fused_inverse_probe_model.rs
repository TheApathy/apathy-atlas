// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the inverse-only V4 RoPE promotion runner.

use std::fs;
use std::path::PathBuf;

const PROBE: &str = "kernels/gb10/experiments/v4_prefill_rope_fused_inverse_probe.cu";
const BUILD: &str = "scripts/check-v4-prefill-rope-fused-inverse-probe-build.sh";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    fs::read_to_string(root().join(path))
        .unwrap_or_else(|error| panic!("required inverse-only probe source {path}: {error}"))
}

#[test]
fn probe_is_inverse_only_with_exact_parity_and_abba_admission() {
    let source = read(PROBE);
    for contract in [
        "#define main v4_combined_rope_probe_main_unreachable",
        "v4_prefill_rope_fused_probe.cu",
        "incumbent_inverse(",
        "candidate_inverse(",
        "kParityCases",
        "mismatch_bytes(",
        "time_abba(",
        "baseline_ms / candidate_ms",
        "timing.speedup < min_speedup",
        "input_unchanged",
        "table_unchanged",
        "inverse-mscale-zero",
        "inverse-mscale-negative",
        "result=PASS",
    ] {
        assert!(source.contains(contract), "probe omits {contract}");
    }
    let main = source
        .split("int main(int argc, char** argv)")
        .nth(1)
        .expect("inverse-only main");
    assert!(!main.contains("incumbent_forward("));
    assert!(!main.contains("candidate_forward("));
}

#[test]
fn immutable_builder_binds_production_dispatch_tools_and_artifacts() {
    let source = read(BUILD);
    for contract in [
        "set -euo pipefail",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "refusing existing V4_INVERSE_ROPE_PROBE_OUTPUT_DIR",
        "v4_prefill_rope_fused_inverse_probe.cu",
        "v4_prefill_rope_fused_inverse.cu",
        "v4_prefill_rope_fused.cu",
        "v4_prefill_rope_fused_probe.cu",
        "cache_skip_v4.rs",
        "types.rs",
        "init.rs",
        "git_status_sha256",
        "nvcc_binary_sha256",
        "cuobjdump_binary_sha256",
        "host_cxx_binary_sha256",
        "compile_command_template=",
        "binary_sha256",
        "cubin_sha256",
        "expected_receipt_sha256",
        "expected_binary_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "runner_sha256",
        "max_output_bytes=4096",
        "max_output_lines=8",
    ] {
        assert!(source.contains(contract), "builder omits {contract}");
    }
}

#[test]
fn builder_rejects_mutable_or_ambiguous_inputs() {
    let source = read(BUILD);
    for contract in [
        "invalid explicit numeric threshold",
        "changed during inverse-only RoPE probe compilation",
        "repository identity changed during inverse-only RoPE probe compilation",
        "usage: $0 <min_speedup>",
        "usage: $0",
        "receipt identity mismatch",
        "binary identity mismatch",
        "embedded build ID mismatch",
        "cubin identity mismatch",
    ] {
        assert!(source.contains(contract), "missing rejection {contract}");
    }
    assert!(source.contains("grep -Fx \"$expected_build_id\" < <(strings \"$binary\") >/dev/null"));
    assert!(!source.contains("strings \"$binary\" | grep -q"));
}
