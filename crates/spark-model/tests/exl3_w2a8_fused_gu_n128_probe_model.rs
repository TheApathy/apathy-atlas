// SPDX-License-Identifier: AGPL-3.0-only

//! CPU and source contracts for the isolated fused GU-to-down-A8 N128 probe.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const HARNESS_RELATIVE: &str =
    "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128_probe.cu";
const COMPONENT_RELATIVE: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu";
const BUILD_RELATIVE: &str = "scripts/check-exl3-prefill-w2a8-fused-gu-n128-probe-build.sh";
const PROBE_STEM: &str = "exl3_w2a8_fused_gu_down_emit_n128_probe";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn required_source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative)).unwrap_or_else(|error| {
        panic!("required fused GU probe input {relative} is missing: {error}")
    })
}

fn marked_section<'a>(source: &'a str, name: &str, prefix: &str) -> &'a str {
    let begin = format!("{prefix} BEGIN {name}");
    let end = format!("{prefix} END {name}");
    assert_eq!(source.matches(&begin).count(), 1, "duplicate {begin}");
    assert_eq!(source.matches(&end).count(), 1, "duplicate {end}");
    let start = source.find(&begin).unwrap() + begin.len();
    let finish = source[start..].find(&end).unwrap() + start;
    assert!(start < finish, "reversed markers for {name}");
    &source[start..finish]
}

fn assert_once(source: &str, needle: &str) {
    assert_eq!(
        source.matches(needle).count(),
        1,
        "expected exactly one `{needle}`"
    );
}

fn scan_for_probe_reachability(path: &Path) {
    for entry in fs::read_dir(path).expect("production tree") {
        let path = entry.expect("production entry").path();
        if path.is_dir() {
            scan_for_probe_reachability(&path);
        } else if matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("rs" | "toml" | "cu" | "cuh")
        ) {
            let source = fs::read_to_string(&path).expect("UTF-8 production source");
            assert!(!source.contains(PROBE_STEM), "{}", path.display());
        }
    }
}

fn fixed_route_valid(offsets: &[i32], total_rows: u32) -> bool {
    offsets.len() == 257
        && total_rows > 0
        && i32::try_from(total_rows).is_ok()
        && offsets.first() == Some(&0)
        && offsets.last() == Some(&(total_rows as i32))
        && offsets.windows(2).all(|pair| {
            pair[0] >= 0
                && pair[0] <= pair[1]
                && u32::try_from(pair[1]).is_ok_and(|end| end <= total_rows)
        })
}

#[test]
fn exact_down_a8_layout_matches_production_expanded_rows() {
    const ROWS: usize = 2_410 * 6;
    const K: usize = 2_048;
    const GROUP_K: usize = 128;
    let fp8_bytes = ROWS.checked_mul(K).unwrap();
    let scale_bytes = ROWS
        .checked_mul(K / GROUP_K)
        .unwrap()
        .checked_mul(size_of::<f32>())
        .unwrap();
    assert_eq!(ROWS, 14_460);
    assert_eq!(fp8_bytes, 29_614_080);
    assert_eq!(scale_bytes, 925_440);
    assert_eq!(fp8_bytes + scale_bytes, 30_539_520);
    assert_eq!(fp8_bytes % 16, 0);
    assert_eq!(scale_bytes % 16, 0);
}

#[test]
fn fixed_route_model_rejects_late_corruption_and_257_experts() {
    let valid: Vec<i32> = (0..=256).collect();
    assert!(fixed_route_valid(&valid, 256));

    let mut late_inversion = valid.clone();
    late_inversion[248] = late_inversion[247] - 1;
    assert!(!fixed_route_valid(&late_inversion, 256));

    let mut late_out_of_range = valid.clone();
    late_out_of_range[252] = 257;
    assert!(!fixed_route_valid(&late_out_of_range, 256));

    let too_many_experts: Vec<i32> = (0..=257).collect();
    assert!(!fixed_route_valid(&too_many_experts, 257));
}

#[test]
fn harness_requires_exact_bytes_boundaries_and_empty_experts() {
    let harness = required_source(HARNESS_RELATIVE);
    let parity = marked_section(&harness, "fused GU exact byte contract", "//");
    for field in [
        "down_fp8_bytes",
        "down_scale_bytes",
        "fp8_mismatches",
        "scale_mismatches",
        "memcmp",
    ] {
        assert!(parity.contains(field), "missing exact-byte field `{field}`");
    }
    assert!(!parity.contains("epsilon"));
    assert!(!parity.contains("tolerance"));

    let routes = marked_section(&harness, "fused GU routing cases", "//");
    for rows in ["0", "1", "63", "64", "65", "127", "128", "129"] {
        assert!(routes.contains(rows), "missing row boundary {rows}");
    }
    for contract in [
        "empty_expert",
        "expert_offsets",
        "sorted_expert_ids",
        "duplicate",
        "out_of_range",
    ] {
        assert!(
            routes.contains(contract),
            "missing routing case `{contract}`"
        );
    }
}

#[test]
fn full_row_parity_is_synthetic_balanced_and_uses_nondegenerate_signs() {
    let harness = required_source(HARNESS_RELATIVE);
    assert!(
        harness.contains("kMaxExperts = 256"),
        "production probe must instantiate all 256 routed experts"
    );
    let parity = marked_section(&harness, "fused GU exact byte contract", "//");
    for contract in [
        "kProductionRows",
        "balanced",
        "synthetic",
        "unrepresentative",
        "parity_rows",
        "down_fp8_bytes",
        "down_scale_bytes",
    ] {
        assert!(
            parity.contains(contract),
            "full-row parity omits `{contract}`"
        );
    }

    let routes = marked_section(&harness, "fused GU routing cases", "//");
    for contract in ["balanced_offsets", "kMaxExperts", "kProductionRows"] {
        assert!(
            routes.contains(contract),
            "balanced production routing omits `{contract}`"
        );
    }

    let signs = marked_section(&harness, "fused GU nondegenerate signs", "//");
    for contract in ["table_salt", "expert", "chunk", "nondegenerate"] {
        assert!(signs.contains(contract), "sign model omits `{contract}`");
    }
    for table in ["gate_svh", "up_svh", "down_suh"] {
        assert!(signs.contains(table), "missing independent table `{table}`");
    }
}

#[test]
fn harness_guards_every_geometry_with_independent_dual_poison() {
    let harness = required_source(HARNESS_RELATIVE);
    let guards = marked_section(&harness, "fused GU guard and poison contract", "//");
    for axis in [
        "block-x", "block-y", "block-z", "grid-x", "grid-y", "grid-z",
    ] {
        assert_once(guards, axis);
    }
    for contract in [
        "wrong-n",
        "wrong-k",
        "wrong-bits",
        "nullptr",
        "oversized-positive-offset",
        "nonmonotonic-offset",
        "late-nonmonotonic-offset",
        "late-out-of-range-offset",
        "too-many-experts",
        "prefix-gap-offset",
        "final-gap-offset",
        "rows_capacity + 1",
        "prefix_guard",
        "suffix_guard",
        "inactive_rows",
        "0xa5",
        "0x5a",
        "poison_independent",
        "for (unsigned char poison : {kPoisonA, kPoisonB})",
        "cudaGetLastError",
        "cudaDeviceSynchronize",
    ] {
        assert!(guards.contains(contract), "missing guard `{contract}`");
    }
}

#[test]
fn component_validates_the_complete_fixed_expert_route_before_writing() {
    let component = required_source(COMPONENT_RELATIVE);
    let guards = marked_section(&component, "fused exact-shape guards", "//");
    for contract in [
        "num_experts != W2F_EXPERTS",
        "routing_index < W2F_EXPERTS",
        "expert_offsets[routing_index]",
        "expert_offsets[routing_index + 1]",
        "__ballot_sync",
        "routing_invalid",
    ] {
        assert!(
            guards.contains(contract),
            "route preflight omits `{contract}`"
        );
    }
    assert!(
        guards.find("num_experts != W2F_EXPERTS").unwrap()
            < guards.find("expert_offsets[routing_index]").unwrap(),
        "fixed expert guard must precede every offsets access"
    );
    assert!(
        guards
            .find("if (__ballot_sync(0xffffffffu, routing_invalid) != 0) return")
            .unwrap()
            < guards.find("gate_trellis_tab[expert]").unwrap(),
        "global route validation must finish before table/output work"
    );
}

#[test]
fn parity_swaps_opposite_poisons_and_proves_every_active_byte_written() {
    let harness = required_source(HARNESS_RELATIVE);
    let parity = marked_section(&harness, "fused GU exact byte contract", "//");
    for contract in [
        "PoisonPair",
        "{kPoisonA, kPoisonB}",
        "{kPoisonB, kPoisonA}",
        "reference_poison",
        "candidate_poison",
        "verify_active_stable",
        "reference_fp8_first",
        "candidate_fp8_first",
        "reference_scale_first",
        "candidate_scale_first",
    ] {
        assert!(
            parity.contains(contract),
            "active-write oracle omits `{contract}`"
        );
    }
    assert!(parity.contains("down_fp8_bytes"));
    assert!(parity.contains("down_scale_bytes"));
}

#[test]
fn harness_uses_abba_timing_and_one_terminal_result() {
    let harness = required_source(HARNESS_RELATIVE);
    let timing = marked_section(&harness, "fused GU ABBA timing contract", "//");
    for contract in [
        "ABBA",
        "cudaEventRecord",
        "cudaEventElapsedTime",
        "baseline_ms",
        "candidate_ms",
        "speedup",
    ] {
        assert!(
            timing.contains(contract),
            "missing timing field `{contract}`"
        );
    }
    assert!(timing.contains("warmup"));

    let output = marked_section(&harness, "fused GU bounded output contract", "//");
    for key in [
        "build_id=",
        "device_uuid=",
        "driver=",
        "runtime=",
        "input_hash=",
        "routing_hash=",
        "tables_hash=",
        "fp8_mismatches=",
        "scale_mismatches=",
        "guards=",
        "poison_a=",
        "poison_b=",
        "abba_samples=",
    ] {
        assert_once(output, key);
    }
    assert_once(output, "result=PASS");
}

#[test]
fn receipt_binds_all_inputs_tools_commands_and_runner_revalidates() {
    let build = required_source(BUILD_RELATIVE);
    let inputs = marked_section(&build, "fused GU probe immutable inputs", "#");
    for input in [
        "exl3_w2a8_fused_gu_down_emit_n128_probe.cu",
        "exl3_w2a8_fused_gu_down_emit_n128.cu",
        "exl3_w2a8_grouped_prefill_n128.cu",
        "exl3_w2a8_h128_emit.cu",
        "kernels/gb10/common/exl3_gemv.cu",
        "kernels/gb10/common/moe_batched_blend.cuh",
        "check-exl3-prefill-w2a8-fused-gu-n128-probe-build.sh",
    ] {
        assert_once(inputs, input);
    }
    for field in [
        "nvcc_binary_hash",
        "cuobjdump_binary_hash",
        "host_cxx_binary_hash",
        "git_commit",
        "git_status_hash",
        "dependency_hashes",
    ] {
        assert!(
            inputs.contains(field),
            "missing immutable identity `{field}`"
        );
    }

    let receipt = marked_section(&build, "fused GU probe receipt contract", "#");
    for field in [
        "git_commit=",
        "git_status_sha256=",
        "source_sha256",
        "compile_command=",
        "compile_command_sha256=",
        "binary_sha256",
        "cubin_sha256",
        "build_id=",
        "min_end_to_end_speedup=",
    ] {
        assert!(receipt.contains(field), "receipt omits `{field}`");
    }
    assert!(
        receipt
            .find("echo \"min_end_to_end_speedup=$min_speedup\"")
            .unwrap()
            < receipt.find("build_id=$(sha256sum \"$manifest\"").unwrap(),
        "threshold must enter the manifest before the build id is derived"
    );
    assert!(receipt.contains("W2A8_FUSED_GU_PROBE_MIN_SPEEDUP=${min_speedup}"));
    assert!(!receipt.contains("W2A8_FUSED_GU_PROBE_MIN_SPEEDUP=${min_speedup}f"));
    assert!(receipt.contains("W2A8_FUSED_GU_PROBE_MIN_SPEEDUP_TEXT=\\\"$min_speedup\\\""));
    assert!(build.contains("printf \"%.17g\""));

    let harness = required_source(HARNESS_RELATIVE);
    assert!(harness.contains("constexpr double kMinSpeedup"));
    assert!(harness.contains("kMinSpeedup > 1.0 && kMinSpeedup <= 100.0"));
    assert!(!harness.contains("constexpr float kMinSpeedup"));

    let runner = marked_section(&build, "fused GU probe runner verification", "#");
    for field in [
        "actual_receipt_sha256",
        "actual_binary_sha256",
        "actual_cubin_sha256",
        "expected_build_id",
        "expected_min_speedup",
        "fp8_mismatches=0",
        "scale_mismatches=0",
        "guards=clean",
        "poison_a=clean",
        "poison_b=clean",
        "active_extents=written",
        "rows=14460 fp8_bytes=29614080 scale_bytes=925440 experts=256 parity_rows=14460 boundary_cases=8 empty_expert_cases=1 geometry_cases=20",
        "routing=synthetic_balanced representative=0",
        "threshold min_speedup=$expected_min_speedup",
        "result=PASS",
    ] {
        assert!(runner.contains(field), "runner omits `{field}`");
    }
    assert!(runner.contains("\"$binary\" >\"$probe_tmp/stdout\""));
    assert!(!runner.contains("\"$binary\" \"$@\""));
    assert!(build.contains("NVCC_PREPEND_FLAGS"));
    assert!(build.contains("NVCC_APPEND_FLAGS"));
    assert!(build.contains("refusing existing W2A8_FUSED_GU_PROBE_OUTPUT_DIR"));
    assert!(build.contains("invalid canonical decimal threshold"));
    assert!(build.contains("usage: $0 <min_end_to_end_speedup>"));
    assert!(build.contains("W2A8_FUSED_GU_PROBE_MIN_SPEEDUP"));
    assert!(build.contains("if [[ $# -ne 0 ]]"));
    assert!(!build.contains("<min_cosine>"));
    assert!(!build.contains("<max_abs_error>"));
    for rejection in [
        "^(0|[1-9][0-9]*)([.][0-9]*[1-9])?$",
        "min_speedup > 1.0",
        "min_speedup <= 100.0",
    ] {
        assert!(build.contains(rejection), "missing rejection `{rejection}`");
    }
}

#[test]
fn build_rejects_noncanonical_thresholds_before_cuda_tool_discovery() {
    let script = workspace().join(BUILD_RELATIVE);
    for invalid in [
        " 1.0001",
        "0x1.001",
        "nan",
        "inf",
        "1",
        "100.0001",
        "01.0001",
        "1.00010",
        "1e1",
        "+2",
        ".5",
        "1.00000000000000001",
    ] {
        let output = Command::new("bash")
            .arg(&script)
            .arg(invalid)
            .env("CUDA_ROOT_PATH", "/definitely/not/a/cuda/toolchain")
            .output()
            .expect("threshold rejection script execution");
        assert_eq!(output.status.code(), Some(2), "accepted `{invalid}`");
        let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
        assert_eq!(stderr.trim(), "invalid canonical decimal threshold");
        assert!(!stderr.contains("missing build tool"));
    }
}

#[test]
fn build_admits_a_binary64_threshold_that_float_would_round_to_one() {
    let script = workspace().join(BUILD_RELATIVE);
    let output = Command::new("bash")
        .arg(&script)
        .arg("1.00000001")
        .env("CUDA_ROOT_PATH", "/definitely/not/a/cuda/toolchain")
        .output()
        .expect("binary64 threshold admission script execution");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
    assert!(stderr.starts_with("missing build tool:"));
    assert!(!stderr.contains("invalid canonical decimal threshold"));

    let canonical = 1.00000001_f64;
    assert!(canonical > 1.0);
    assert_eq!(1.00000001_f32, 1.0_f32);
}

#[test]
fn promotion_probe_has_no_serving_or_registry_reachability() {
    let root = workspace();
    scan_for_probe_reachability(&root.join("crates/spark-model/src"));
    scan_for_probe_reachability(&root.join("crates/atlas-kernels"));
    scan_for_probe_reachability(&root.join("kernels/gb10/common"));
    scan_for_probe_reachability(&root.join("kernels/gb10/deepseek-v4-flash"));
}
