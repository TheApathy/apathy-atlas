// SPDX-License-Identifier: AGPL-3.0-only
//! Source seams supplement actual behavior tests; never evidence of GPU parity.
use std::path::Path;

fn source(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("missing actual source {}", path.display()))
        .split_whitespace()
        .collect()
}

fn body<'a>(source: &'a str, signature: &str) -> &'a str {
    let tail = source
        .split_once(signature)
        .expect("missing actual function")
        .1;
    let start = tail.find('{').expect("missing function body");
    let mut depth = 0usize;
    for (index, ch) in tail[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &tail[start + 1..start + index];
                }
            }
            _ => (),
        }
    }
    panic!("unterminated actual function");
}

#[test]
fn new_binary_reuses_geometry_abi_protocol_weight_reader_and_metrics_without_old_admission() {
    let main = source("src/bin/deepseek_vision_block8.rs");
    for module in [
        "contract.rs",
        "protocol.rs",
        "abi.rs",
        "weight.rs",
        "metrics.rs",
    ] {
        assert!(
            main.contains(&format!("#[path=\"../{module}\"]")),
            "not reusing {module}"
        );
    }
    assert!(main.contains("modgemm_geometry;"));
    assert!(main.contains("../block8/contract.rs"));
    assert!(main.contains("../block8/admission.rs"));
    assert!(!main.contains("#[path=\"../admission.rs\"]"));
    assert!(!main.contains("#[path=\"../runner.rs\"]"));
    let c = source("src/block8/contract.rs");
    assert!(c.contains("gemm_geometry::{"));
    assert!(c.contains("vision.blocks.8.mlp.w1.weight"));
}

#[test]
fn all_admission_and_inputs_complete_before_cuda_open_and_native_replay_precedes_candidates() {
    let main = source("src/bin/deepseek_vision_block8.rs");
    let admitted = main
        .find("admission::load(")
        .expect("reviewed admission load missing");
    let inputs = main
        .find("runner::Inputs::load(")
        .expect("owned host input load missing");
    let open = main
        .find("Driver::open(")
        .expect("actual CUDA owner missing");
    assert!(admitted < inputs && inputs < open);
    let runner = source("src/block8/runner.rs");
    let run = body(&runner, "pubfnrun(");
    let control = run
        .find("native_control(")
        .expect("real native replay gate missing");
    let gemmex = run.find("run_gemmex_modes(").expect("GemmEx modes missing");
    let lt = run.find("run_lt_modes(").expect("Lt modes missing");
    assert!(control < gemmex && control < lt);
    assert!(runner.contains("protocol::execute_mode("));
    assert!(runner.contains("lt_plan::execute("));
    assert!(run.contains("contract::owned_device_bytes()?"));
    assert!(run.contains("contract::OWNED_DEVICE_CAP"));
    assert!(runner.contains("deepseek_vision_linear"));
    assert!(runner.contains("check_native_control("));
    assert!(runner.contains("contract::NATIVE_SHA"));
    assert!(runner.contains("d.guards()"));
    assert!(runner.contains("d.sync()"));
    assert!(runner.contains("d.read("));
    assert!(runner.contains("compare_repeats("));
}

#[test]
fn fresh_block_eight_admission_reconstructs_actual_capture_and_selected_weight() {
    let a = source("src/block8/admission.rs");
    for seam in [
        "validate_capture(",
        "contract::MANIFEST_SHA",
        "contract::STAGES_SHA",
        "contract::INPUT_SHA",
        "contract::NATIVE_SHA",
        "weight::selected(",
        "io::verify_receipt(",
        "current==admission",
        "saved==selected",
    ] {
        assert!(a.contains(seam), "missing actual admission seam {seam}");
    }
    assert!(a.contains("block-08-norm2.bf16"));
    assert!(a.contains("block-08-fc1.bf16"));
    for path in [
        "src/bin/deepseek_vision_block8.rs",
        "src/block8/admission.rs",
        "src/block8/runner.rs",
        "src/block8/report.rs",
    ] {
        let s = source(path);
        for forbidden in [
            "validate_references(",
            ".reference_sha256()",
            "reference_hash(",
            "full_reference_payload",
            "default_reference_payload",
        ] {
            assert!(
                !s.contains(forbidden),
                "{path} reused authoritative block0 seam {forbidden}"
            );
        }
    }
}

#[test]
fn lt_real_adapter_applies_and_reads_back_the_tested_policy_without_autotuning() {
    let lt = source("src/block8/lt.rs");
    assert!(
        lt.contains("implLtIofor"),
        "production must consume tested Lt protocol"
    );
    assert!(lt.contains("cublasLtMatmulAlgoConfigGetAttribute"));
    assert!(lt.contains("size_written==size_of::<u32>()"));
    assert!(lt.contains("preference_mask()"));
    assert!(lt.contains("pref_set"));
    assert!(lt.contains("heuristic"));
    assert!(lt.contains("matmul"));
    assert!(!lt.contains("std::env"));
    assert!(
        !lt.contains("AlgoConfigSetAttribute"),
        "no algorithm tuning in this slice"
    );
    let main = source("src/bin/deepseek_vision_block8.rs");
    assert!(main.contains("driver.guards()"));
    assert!(main.contains("driver.close()"));
    assert!(main.contains("full_encoder_qualified"));
    assert!(main.contains("performance_qualified"));
}
