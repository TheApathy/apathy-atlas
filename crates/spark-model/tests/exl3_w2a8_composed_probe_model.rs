// SPDX-License-Identifier: AGPL-3.0-only

//! Source contracts for the immutable production-shape composed W2A8 probe.

use std::fs;
use std::path::PathBuf;

const HARNESS: &str = "kernels/gb10/experiments/exl3_w2a8_composed_n128_n256_probe.cu";
const BUILDER: &str = "scripts/check-exl3-prefill-w2a8-composed-probe-build.sh";

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace().join(relative))
        .unwrap_or_else(|error| panic!("missing composed-probe input {relative}: {error}"))
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
fn harness_binds_exact_production_shape_and_both_complete_chains() {
    let harness = source(HARNESS);
    for exact in [
        "kTokens = 2410",
        "kTopK = 6",
        "kRows = 14460",
        "kExperts = 256",
        "kHidden = 4096",
        "kIntermediate = 2048",
    ] {
        assert!(harness.contains(exact), "missing `{exact}`");
    }

    let baseline = marked(&harness, "five-launch incumbent", "//");
    for symbol in [
        "exl3_w2a8_h128_pre_dual_emit_h4096",
        "exl3_w2a8_grouped_prefill_k2_gu",
        "exl3_w2a8_h128_post_silu_pre_emit_h2048",
        "exl3_w2a8_grouped_prefill_k2_down",
    ] {
        assert!(baseline.contains(symbol), "incumbent omits `{symbol}`");
    }
    assert_eq!(baseline.matches("<<<").count(), 5);
    assert!(baseline.contains("dim3(kExperts * (kIntermediate / 64))"));
    assert!(baseline.contains("dim3(kExperts * (kHidden / 64))"));

    let candidate = marked(&harness, "three-launch candidate", "//");
    for symbol in [
        "exl3_w2a8_h128_pre_dual_emit_h4096",
        "exl3_w2a8_fused_gu_down_emit_n128",
        "exl3_w2a8_grouped_prefill_n256_k2_down",
    ] {
        assert!(candidate.contains(symbol), "candidate omits `{symbol}`");
    }
    assert_eq!(candidate.matches("<<<").count(), 3);
    assert!(candidate.contains("dim3(kExperts * (kIntermediate / 128))"));
    assert!(candidate.contains("dim3(kExperts * (kHidden / 256))"));
    assert!(candidate.contains("kRows, kIntermediate, kHidden"));
}

#[test]
fn harness_proves_every_numeric_boundary_and_memory_invariant() {
    let harness = source(HARNESS);
    let parity = marked(&harness, "composed exact-byte parity", "//");
    for boundary in [
        "baseline.down_fp8",
        "candidate.down_fp8",
        "baseline.down_scale",
        "candidate.down_scale",
        "baseline.raw_down",
        "candidate.raw_down",
        "baseline.final_down",
        "candidate.final_down",
    ] {
        assert!(parity.contains(boundary), "parity omits `{boundary}`");
    }
    assert!(parity.contains("kPoisonA") && parity.contains("kPoisonB"));
    assert!(parity.contains("compare_exact"));

    let routes = marked(&harness, "production and adversarial routes", "//");
    for route in [
        "balanced",
        "empty-expert",
        "skewed",
        "m64-boundaries",
        "checkpoint-like-synthetic-v1",
    ] {
        assert!(routes.contains(route), "route set omits `{route}`");
    }
    assert!(routes.contains("kExperts + 1"));
    assert!(routes.contains("offsets.back() != static_cast<int>(kRows)"));

    let safety = marked(&harness, "guards inputs and malformed no-write", "//");
    for proof in [
        "guards_clean",
        "verify_immutable_inputs",
        "assert_all_byte",
        "wrong-fused-block",
        "wrong-fused-grid",
        "wrong-fused-N",
        "wrong-fused-K",
        "wrong-fused-bits",
        "wrong-fused-mode",
        "wrong-fused-offsets",
        "wrong-fused-late-offsets",
        "wrong-down-block",
        "wrong-down-grid",
        "wrong-down-N",
        "wrong-down-K",
        "wrong-down-bits",
        "wrong-down-mode",
        "wrong-down-offsets",
        "wrong-down-late-offsets",
    ] {
        assert!(safety.contains(proof), "safety contract omits `{proof}`");
    }
    assert!(safety.contains("bad_offsets[kExperts - 1]"));
    assert!(safety.contains("return 16;"));

    for pass in [
        "run_parity(fixture, \"m64-boundaries\", kPoisonA, kPoisonB)",
        "run_parity(fixture, \"m64-boundaries\", kPoisonB, kPoisonA)",
    ] {
        assert!(harness.contains(pass), "missing boundary pass `{pass}`");
    }
}

#[test]
fn composed_probe_executes_the_production_arena_alias_schedule() {
    let harness = source(HARNESS);
    let alias = marked(&harness, "production arena alias parity", "//");
    for proof in [
        "struct ProductionArena",
        "launch_baseline_alias",
        "launch_candidate_alias",
        "arena.expert_gate.data",
        "arena.expert_up.data",
        "arena.expert_down.data",
        "baseline-alias-raw",
        "baseline-alias-final",
        "candidate-alias-raw",
        "candidate-alias-final",
    ] {
        assert!(alias.contains(proof), "alias proof omits `{proof}`");
    }
    assert!(harness.contains("mix_hash(run_alias_parity(fixture))"));
}

#[test]
fn production_shape_route_histograms_have_all_257_offsets() {
    const EXPERTS: usize = 256;
    const ROWS: usize = 14_460;
    let offsets = |counts: [usize; EXPERTS]| {
        let mut offsets = Vec::with_capacity(EXPERTS + 1);
        offsets.push(0);
        for count in counts {
            offsets.push(offsets.last().copied().unwrap() + count);
        }
        assert_eq!(offsets.len(), EXPERTS + 1);
        assert_eq!(offsets[EXPERTS], ROWS);
        assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        offsets
    };

    let balanced = std::array::from_fn(|expert| if expert < 124 { 57 } else { 56 });
    let empty = std::array::from_fn(|expert| {
        if expert == 0 {
            0
        } else if expert <= 180 {
            57
        } else {
            56
        }
    });
    let skewed = std::array::from_fn(|expert| match expert {
        0 => 512,
        1 => 0,
        2..=233 => 55,
        _ => 54,
    });
    let boundaries = std::array::from_fn(|expert| match expert {
        0 => 63,
        1 => 64,
        2 => 65,
        3 => 127,
        4 => 128,
        5 => 129,
        6..=139 => 56,
        _ => 55,
    });
    assert_eq!(offsets(balanced)[1], 57);
    assert_eq!(offsets(empty)[1], 0);
    assert_eq!(offsets(skewed)[1], 512);
    assert_eq!(offsets(skewed)[2], 512);
    assert_eq!(&offsets(boundaries)[1..=6], &[63, 127, 192, 319, 447, 576]);
}

#[test]
fn timing_and_receipt_are_precommitted_and_artifact_complete() {
    let harness = source(HARNESS);
    let timing = marked(&harness, "whole-chain ABBA timing", "//");
    assert!(timing.contains("baseline_ms0"));
    assert!(timing.contains("candidate_ms0"));
    assert!(timing.contains("candidate_ms1"));
    assert!(timing.contains("baseline_ms1"));
    assert!(timing.contains("kMinSpeedup"));
    assert!(timing.contains("W2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP"));
    assert!(timing.contains("W2A8_COMPOSED_PROBE_MIN_SPEEDUP_TEXT"));
    assert!(timing.contains("f.load_route(\"checkpoint-like-synthetic-v1\")"));
    assert!(!timing.contains("argv"));

    let build = source(BUILDER);
    let inputs = marked(&build, "composed probe immutable inputs", "#");
    for dependency in [
        "exl3_w2a8_composed_n128_n256_probe.cu",
        "exl3_w2a8_fused_gu_down_emit_n128.cu",
        "exl3_w2a8_grouped_prefill_n256.cu",
        "exl3_w2a8_grouped_prefill.cu",
        "exl3_w2a8_h128_emit.cu",
        "forward_prefill_exl3_w2a8.rs",
        "forward_prefill.rs",
        "forward_prefill_exl3.rs",
        "forward_prefill_exl3_tail.rs",
        "forward_prefill_phase.rs",
        "exl3_decode.rs",
        "check-exl3-prefill-w2a8-composed-probe-build.sh",
    ] {
        assert!(inputs.contains(dependency), "receipt omits `{dependency}`");
    }
    let receipt = marked(&build, "composed probe receipt contract", "#");
    for proof in [
        "build_id",
        "git_status_shape_sha256",
        "binary_sha256",
        "cubin_sha256",
        "compile_command_sha256",
        "resource_command_sha256",
    ] {
        assert!(receipt.contains(proof), "receipt omits `{proof}`");
    }
    let threshold = marked(&build, "composed probe threshold contract", "#");
    for proof in ["%.17g", "NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS"] {
        assert!(threshold.contains(proof), "threshold omits `{proof}`");
    }
    let runner = marked(&build, "composed probe runner verification", "#");
    assert!(runner.contains("if [[ $# -ne 0 ]]"));
    assert!(runner.contains("actual_receipt_sha256"));
    assert!(runner.contains("actual_binary_sha256"));
    assert!(runner.contains("expected_min_speedup"));
    assert!(runner.contains("enforcement=enforced"));
    assert!(runner.contains("variant=incumbent packed_e4m3=0"));
    assert!(runner.contains("intermediate_fp8=exact"));
    assert!(runner.contains("raw_bf16=exact final_bf16=exact production_alias=exact"));
    assert!(runner.contains(
        "routes=balanced,empty-expert,skewed,m64-boundaries,checkpoint-like-synthetic-v1"
    ));
    assert!(runner.contains("timing_route=checkpoint-like-synthetic-v1"));
    assert!(runner.contains("offsets=257 poison_passes=8"));
    assert!(runner.contains("input_hash="));
    assert!(runner.contains("malformed_cases=16"));
    assert!(runner.contains("runner_hash=$(sha256sum"));
    assert!(build.contains("runner_sha256=$runner_hash"));
}
