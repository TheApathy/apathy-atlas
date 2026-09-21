// SPDX-License-Identifier: AGPL-3.0-only

use std::{fs, path::PathBuf};

fn source(name: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(name)).unwrap()
}

fn ordered(code: &str, markers: &[&str]) {
    let mut tail = code;
    for marker in markers {
        let at = tail
            .find(marker)
            .unwrap_or_else(|| panic!("missing {marker}"));
        tail = &tail[at + marker.len()..];
    }
}

#[test]
fn check_admission_precedes_disabled_return() {
    let code = source("src/layers/qwen3_ssm/qwen4_prefill_exact.rs");
    assert!(code.contains("ATLAS_QWEN4_PREFILL_SSM_CHECK"));
    ordered(
        &code,
        &[
            "pub(crate) fn admit_request(",
            "validate_check(",
            "if !exact {",
        ],
    );
}

#[test]
fn snapshot_precedes_candidate_and_check_follows_injection() {
    let code = source("src/layers/qwen3_ssm/qwen4_prefill_exact_forward.rs");
    ordered(
        &code,
        &[
            "self.preflight_qwen4_prefill_exact(",
            "Snapshot::capture(",
            "attn.prepare_prefill_exact(",
            "attn.inject_saved_batched(",
            "check.verify(",
        ],
    );
}

#[test]
fn restore_then_scalar_replay_then_exact_state_comparison() {
    let code = source("src/layers/qwen3_ssm/qwen4_prefill_check.rs");
    ordered(
        &code,
        &[
            "let candidate = Self::capture(",
            "copy_h2d_group_on_stream(",
            "drop(self);",
            "for row in 0..rows",
            "attn.prepare_decode(",
            "layer.ssm_forward(",
            "attn.inject_decode(",
            "candidate.compare_device(",
        ],
    );
    for marker in [
        "candidate.hidden",
        "candidate.h",
        "candidate.conv",
        "saved_inject",
        "timing_eligible=false",
        "copy_d2h_on_stream(",
        "compare_finite(",
    ] {
        assert!(code.contains(marker), "missing {marker}");
    }
    for forbidden in [
        "gpu.alloc(",
        "prepare_prefill_exact(",
        "std::fs",
        "unwrap()",
        "expect(",
    ] {
        assert!(!code.contains(forbidden), "unexpected {forbidden}");
    }
}
