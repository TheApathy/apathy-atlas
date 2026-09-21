// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the immutable packed-E4M3 composed W2A8 A/B probe.

use std::fs;
use std::path::PathBuf;

const HARNESS: &str = "kernels/gb10/experiments/exl3_w2a8_composed_n128_n256_probe.cu";
const BUILDER: &str = "scripts/check-exl3-prefill-w2a8-packed-e4m3-ab-probe-build.sh";
const N128: &str = "kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu";
const N256: &str = "kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative))
        .unwrap_or_else(|error| panic!("missing packed-E4M3 A/B input {relative}: {error}"))
}

fn marked<'a>(source: &'a str, name: &str, prefix: &str) -> &'a str {
    let begin = format!("{prefix} BEGIN {name}");
    let end = format!("{prefix} END {name}");
    assert_eq!(
        source.matches(&begin).count(),
        1,
        "missing/duplicate {begin}"
    );
    assert_eq!(source.matches(&end).count(), 1, "missing/duplicate {end}");
    let start = source.find(&begin).unwrap() + begin.len();
    let finish = source[start..].find(&end).unwrap() + start;
    &source[start..finish]
}

#[test]
fn candidate_is_one_boolean_compile_switch_over_exact_half2_scale() {
    for path in [N128, N256] {
        let candidate = source(path);
        for contract in [
            "#ifndef W2A8_PACKED_E4M3_CANDIDATE",
            "static_assert(W2A8_PACKED_E4M3_CANDIDATE == 0 ||",
            "#if W2A8_PACKED_E4M3_CANDIDATE",
            "0x4c004c00u",
            "__hmul2(pair, scale.values)",
            "cvt.rn.satfinite.e4m3x2.f16x2",
            "cvt.rn.satfinite.e4m3x2.f32",
        ] {
            assert!(candidate.contains(contract), "{path} omits `{contract}`");
        }
        assert_eq!(
            candidate.matches("#if W2A8_PACKED_E4M3_CANDIDATE").count(),
            1
        );
    }
}

#[test]
fn harness_has_a_reproducible_checkpoint_like_route_and_runtime_hashes() {
    let harness = source(HARNESS);
    let routes = marked(&harness, "production and adversarial routes", "//");
    for contract in [
        "checkpoint-like-synthetic-v1",
        "counts[0] = 512",
        "counts[1] = 384",
        "counts[2] = 256",
        "counts[3] = 192",
        "counts[10] = 0",
        "counts[11] = 0",
    ] {
        assert!(
            routes.contains(contract),
            "route contract omits `{contract}`"
        );
    }
    assert!(harness.contains("run_parity(\n        fixture, \"checkpoint-like-synthetic-v1\""));
    let timing = marked(&harness, "whole-chain ABBA timing", "//");
    assert!(timing.contains("f.load_route(\"checkpoint-like-synthetic-v1\")"));

    let input_hash = marked(&harness, "runtime input content hash", "//");
    for input in [
        "input",
        "gate_trellis",
        "up_trellis",
        "down_trellis",
        "gate_suh",
        "up_suh",
        "gate_svh",
        "up_svh",
        "down_suh",
        "down_svh",
        "offsets",
        "sorted_tokens",
        "sorted_experts",
    ] {
        assert!(input_hash.contains(input), "input hash omits `{input}`");
    }
    assert!(harness.contains("input_hash=%016llx"));
    assert!(harness.contains("variant=%s packed_e4m3=%d"));
    assert!(harness.contains("W2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP"));
    assert!(harness.contains("enforcement=%s"));
}

#[test]
fn checkpoint_like_histogram_is_exact_n2410_top6_and_nonuniform() {
    const EXPERTS: usize = 256;
    const ROWS: usize = 2_410 * 6;
    let mut counts = [0usize; EXPERTS];
    counts[0..10].copy_from_slice(&[512, 384, 256, 192, 129, 128, 127, 65, 64, 63]);
    counts[10] = 0;
    counts[11] = 0;
    for (expert, count) in counts.iter_mut().enumerate().skip(12) {
        *count = if expert < 108 { 52 } else { 51 };
    }
    assert_eq!(counts.iter().sum::<usize>(), ROWS);
    assert_eq!(counts.iter().filter(|&&count| count == 0).count(), 2);
    assert_eq!(counts.iter().copied().max(), Some(512));
    assert!(counts.iter().copied().min().unwrap() < counts.iter().sum::<usize>() / EXPERTS);
}

#[test]
fn builder_freezes_two_variants_and_all_compile_and_sass_evidence() {
    let build = source(BUILDER);
    let threshold = marked(&build, "packed E4M3 A/B threshold", "#");
    assert!(threshold.contains("min_speedup >= 1.01"));
    assert!(threshold.contains("%.17g"));
    assert!(threshold.contains("NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS"));

    let inputs = marked(&build, "packed E4M3 A/B immutable inputs", "#");
    for dependency in [
        "exl3_w2a8_composed_n128_n256_probe.cu",
        "exl3_w2a8_fused_gu_down_emit_n128.cu",
        "exl3_w2a8_grouped_prefill_n256.cu",
        "exl3_w2a8_h128_emit.cu",
        "exl3_w2a8_grouped_prefill_k2_gu.cu",
        "exl3_w2a8_grouped_prefill_k2_down.cu",
        "exl3_w2a8_grouped_prefill_n256_k2_down.cu",
        "KERNEL.toml",
        "check-exl3-prefill-w2a8-packed-e4m3-ab-probe-build.sh",
    ] {
        assert!(
            inputs.contains(dependency),
            "immutable inputs omit `{dependency}`"
        );
    }

    let variants = marked(&build, "packed E4M3 two binary build", "#");
    for flag in [
        "-DW2A8_PACKED_E4M3_CANDIDATE=0",
        "-DW2A8_PACKED_E4M3_CANDIDATE=1",
        "-DW2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP=0",
    ] {
        assert!(variants.contains(flag), "build omits `{flag}`");
    }
    for proof in [
        "incumbent",
        "candidate",
        "compile_command_sha256",
        "binary_sha256",
        "cubin_sha256",
        "resource_sha256",
        "sass_sha256",
        "F2FP\\.SATFINITE\\.E4M3\\.F16",
        "F2FP\\.SATFINITE\\.E4M3\\.F32",
        "STACK",
        "LOCAL",
    ] {
        assert!(
            variants.contains(proof),
            "two-binary evidence omits `{proof}`"
        );
    }
}

#[test]
fn generated_runner_is_fail_closed_and_compares_abba_composed_timings() {
    let build = source(BUILDER);
    let runner = marked(&build, "packed E4M3 generated A/B runner", "#");
    for proof in [
        "expected_receipt_sha256",
        "artifact_sha256",
        "timeout",
        "ulimit -f",
        "order=(incumbent candidate candidate incumbent)",
        "candidate_ms",
        "input_hash",
        "output_hash",
        "guards=clean inputs=immutable",
        "intermediate_fp8=exact intermediate_scale=exact",
        "raw_bf16=exact final_bf16=exact production_alias=exact",
        "variant_outputs=exact",
        "direct_packed_speedup",
        "speedup < threshold",
        "enforcement=parity-only",
        "preserve_failure",
        "failed-run-",
        "result=PASS",
    ] {
        assert!(runner.contains(proof), "runner omits `{proof}`");
    }
    assert!(runner.contains("stdout_bytes -gt 8192"));
    assert!(runner.contains("stderr_bytes -ne 0"));
    assert!(runner.contains("stdout_lines -ne 13"));
    assert!(runner.contains("timed=whole-chain"));
    assert!(!runner.contains("nvidia-smi"));
}
