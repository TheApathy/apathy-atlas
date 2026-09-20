// SPDX-License-Identifier: AGPL-3.0-only

use std::{fs, path::PathBuf};

fn source(name: &str) -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src/layers/moe")
            .join(name),
    )
    .unwrap_or_default()
}

#[test]
fn diagnostic_admission_precedes_the_compact_off_return() {
    let policy = source("qwen4_prefill_compact.rs");
    assert!(policy.contains("ATLAS_QWEN4_PREFILL_MOE_COMPACT_CHECK"));
    let admit = &policy[policy.find("pub(crate) fn admit_request").unwrap()..];
    assert!(admit.find("check_selected()?").unwrap() < admit.find("if !selected()?").unwrap());
}

#[test]
fn candidate_capture_is_finite_then_cleared_before_shipping_replay() {
    let all = source("qwen4_compact_check.rs");
    let check = &all[all.find("fn check_compact_projection(").unwrap()..];
    let capture = check
        .find("capture(ctx")
        .expect("missing candidate capture");
    let zero = check
        .find("memset_async")
        .expect("missing independent initialization");
    let parent = check
        .find("ops::moe_w4a16_grouped_gemm_ptrtable(")
        .expect("missing shipping replay");
    let compare = check
        .find("compare_device(ctx")
        .expect("missing full compare");
    assert!(capture < zero && zero < parent && parent < compare);
    assert!(check.contains("self.moe_grouped_gemm"));
    assert!(all.contains("BF16_CHUNK_BYTES"));
    assert!(!check.contains(".alloc("));
    assert!(!check.contains(".unwrap_or"));
}

#[test]
fn shipping_replay_is_ordered_around_the_identical_activation() {
    let ops = source("qwen4_compact_ops.rs");
    let gate = ops
        .find("check_compact_gate_up")
        .expect("missing gate/up check");
    let silu = ops.find("ops::silu_mul(").unwrap();
    let down = ops.find("check_compact_down").expect("missing down check");
    assert!(gate < silu && silu < down);
    assert!(ops.find("check_planned_after_stream").unwrap() < gate);
    assert!(ops.rfind("abi::check_status").unwrap() < down);
}
