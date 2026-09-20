// SPDX-License-Identifier: AGPL-3.0-only
//! Actual launch/runner seams supplement CPU plans; no GPU numerical claim.
use std::{fs, path::Path};
fn source(path: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|error| panic!("missing required batchM seam {path}: {error}"))
}
fn compact(text: &str) -> String {
    text.split_whitespace().collect()
}

#[test]
fn explicit_cli_branch_preserves_all_existing_default_operators() {
    let text = source("examples/glm53_projection_probe.rs");
    assert!(text.contains("\"--batchm\""));
    assert!(text.contains("batchm_probe::run("));
    assert!(
        text.find("batchm_probe::run(").unwrap()
            < text
                .find("let mut session = session::Session::new()")
                .unwrap()
    );
    for operator in [
        "Projection::Cublas(ReductionPolicy::Baseline)",
        "Projection::Cublas(ReductionPolicy::ComputeTypeOnly)",
        "Projection::TensorCore",
        "Projection::Gemv(GemvMode::Sequential)",
        "Projection::Gemv(GemvMode::Gather)",
        "Projection::Gemv(GemvMode::Batch2)",
    ] {
        assert!(
            text.contains(operator),
            "old default operator removed: {operator}"
        );
    }
    assert!(text.contains("--operator-timing"));
}

#[test]
fn resolves_once_and_uses_exact_metadata_free_eight_argument_abi() {
    let raw = source("examples/glm53_projection_probe/batchm.rs");
    let text = compact(&raw);
    for lookup in [
        "gpu.kernel(\"gemv\",\"dense_gemv_bf16\")?",
        "gpu.kernel(\"dense_gemv_bf16_batch2\",\"dense_gemv_bf16_batch2\")?",
        "gpu.kernel(\"dense_gemv_bf16_batchm\",\"dense_gemv_bf16_batchm\")?",
    ] {
        assert!(text.contains(lookup), "missing registered symbol {lookup}");
    }
    assert!(text.contains(".grid([OUTPUTS.div_ceil(4),1,1]).block([256,1,1])"));
    assert!(text.contains(".arg_ptr(DevicePtr(launch.input.ptr)).arg_ptr(DevicePtr(launch.weight.ptr)).arg_ptr(DevicePtr(launch.output.ptr)).arg_u32(launch.rows).arg_u32(OUTPUTS).arg_u32(HIDDEN).arg_u32(HIDDEN).arg_u32(OUTPUTS)"));
    assert!(text.contains("validate_kernel_rows(launch.rows)?"));
    assert!(text.contains("ops::dense_gemv("));
    assert!(text.contains("ops::dense_gemv_batch2("));
    let launch = raw
        .split("pub fn launch(")
        .nth(1)
        .expect("missing launch boundary");
    for forbidden in [
        "gpu.kernel(",
        "gpu.alloc",
        "copy_h2d",
        "copy_d2h",
        "cublas",
        "dense_gemm_tc",
        "lora_bgmv",
        "std::fs",
    ] {
        assert!(!launch.contains(forbidden), "launch contains {forbidden}");
    }
}

#[test]
fn independent_schedules_poison_then_read_and_save_raw_before_comparison() {
    let text = source("examples/glm53_projection_probe/batchm_probe.rs");
    let schedule = text
        .split("fn run_schedule(")
        .nth(1)
        .expect("missing independent schedule")
        .split("\nfn ")
        .next()
        .unwrap();
    assert!(schedule.find("poison_output(").unwrap() < schedule.find("execute(").unwrap());
    assert!(schedule.find("execute(").unwrap() < schedule.find("read_output(").unwrap());
    assert!(schedule.find("read_output(").unwrap() < schedule.find("artifacts::write(").unwrap());
    assert!(text.find("artifacts::write(").unwrap() < text.find("contract::compare(").unwrap());
    for needle in [
        "PairReference",
        "BatchMDirect",
        "BatchMPartitioned",
        "INPUT_DERIVATION",
        "check_guards(",
        "catch_unwind",
        "result.json",
        "source_sha256",
        "manifest_sha256",
    ] {
        assert!(text.contains(needle), "missing real-run boundary {needle}");
    }
    assert!(
        compact(&text).contains("[\"exact\"]==true"),
        "qualification must require raw exactness, not an L2 threshold"
    );
}

#[test]
fn guarded_large_io_has_owned_storage_and_failed_fence_quarantine() {
    let text = compact(&source("examples/glm53_projection_probe/batchm_session.rs"));
    for needle in [
        "ManuallyDrop",
        "hosts:Vec<Vec<u8>>",
        "readback:Vec<u8>",
        "poison:Vec<u8>",
        "catch_unwind",
        "synchronize(self.stream)",
        "check_guards",
        "forget",
    ] {
        assert!(text.contains(needle), "missing owned session seam {needle}");
    }
    assert!(text.contains("[0xc0,0x7f]"), "missing BF16 NaN poison");
    assert!(text.contains("GUARD_BYTES"), "missing allocation guards");
}
