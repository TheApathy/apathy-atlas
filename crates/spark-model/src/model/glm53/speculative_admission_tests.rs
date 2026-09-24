// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const RECORD: &str = include_str!("../../../../../bench/glm53-spec/evidence.json");

#[test]
fn compiled_evidence_matches_the_recorded_json_field_for_field() {
    let json: serde_json::Value = serde_json::from_str(RECORD).unwrap();
    let object = json.as_object().unwrap();
    let e = &GLM53_SPECULATIVE_EVIDENCE;
    let expected = serde_json::json!({
        "schema": e.schema,
        "scope": e.scope,
        "identity_binary_sha256": e.identity_binary_sha256,
        "state_binary_sha256": e.state_binary_sha256,
        "identity_prompts": e.identity_prompts,
        "identity_trials_equal": e.identity_trials_equal,
        "identity_trials_differ": e.identity_trials_differ,
        "perturbed_prompt_control_differs": e.perturbed_prompt_control_differs,
        "state_prefix_positions_equal": e.state_prefix_positions_equal,
        "state_full_positions_equal": e.state_full_positions_equal,
        "state_gate_passed": e.state_gate_passed,
        "state_layers": e.state_layers,
        "control_legacy_restage_fails_state": e.control_legacy_restage_fails_state,
        "control_legacy_restage_fails_identity": e.control_legacy_restage_fails_identity,
        "control_extra_row_fails_state": e.control_extra_row_fails_state,
        "control_extra_row_fails_identity": e.control_extra_row_fails_identity,
        "kernel_harness_bit_identical": e.kernel_harness_bit_identical,
        "chunked_prefill_covered": e.chunked_prefill_covered,
        "spec_source_fnv1a64": e.spec_source_fnv1a64,
    });
    assert_eq!(object.len(), expected.as_object().unwrap().len(), "field set drift");
    assert_eq!(&json, &expected);
}

#[test]
fn recorded_evidence_admits() {
    GLM53_SPECULATIVE_EVIDENCE.validate().unwrap();
}

#[test]
fn every_failure_mode_refuses() {
    let base = GLM53_SPECULATIVE_EVIDENCE;
    let cases: [fn(&mut Glm53SpeculativeEvidence); 10] = [
        |e| e.identity_trials_differ = 1,
        |e| e.identity_trials_equal = 24,
        |e| e.identity_prompts = &["short"],
        |e| e.state_gate_passed = false,
        |e| e.state_prefix_positions_equal = 99,
        |e| e.state_full_positions_equal = 49,
        |e| e.control_legacy_restage_fails_state = false,
        |e| e.control_extra_row_fails_identity = false,
        |e| e.perturbed_prompt_control_differs = false,
        |e| e.kernel_harness_bit_identical = false,
    ];
    for (index, mutate) in cases.iter().enumerate() {
        let mut e = base.clone();
        mutate(&mut e);
        assert!(e.validate().is_err(), "mutation {index} was admitted");
    }
}

/// The certified sources, in `SPEC_SOURCES` order.
const SOURCES: [&str; 5] = [
    include_str!("verify_policy_transaction.rs"),
    include_str!("prefix_commit.rs"),
    include_str!("dsa_policy_execution.rs"),
    include_str!("../../../../../kernels/gb10/glm5.3-flash/exl3/glm53_exl3_rowexact.cuh"),
    include_str!("../../../../spark-server/src/scheduler/glm53_policy_driver.rs"),
];

fn fnv1a64(parts: &[&[u8]]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for &byte in *part {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[test]
fn certified_sources_are_unchanged_since_the_evidence_was_measured() {
    let mut parts: Vec<&[u8]> = Vec::new();
    for (path, text) in SPEC_SOURCES.iter().zip(SOURCES) {
        parts.extend([path.as_bytes(), &b"\0"[..], text.as_bytes(), &b"\0"[..]]);
    }
    let hash = format!("{:016x}", fnv1a64(&parts));
    assert_eq!(
        hash, GLM53_SPECULATIVE_EVIDENCE.spec_source_fnv1a64,
        "speculation-path sources changed: re-run the gate windows and regenerate \
         bench/glm53-spec/evidence.json (make_evidence.py) before admitting"
    );
}
