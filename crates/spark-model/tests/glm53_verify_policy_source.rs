// SPDX-License-Identifier: AGPL-3.0-only

//! Production seam REDs. These do not qualify the future server policy adapter.

const TARGET: &str = include_str!("../src/model/glm53/target_model_exl3.rs");
const MODEL: &str = include_str!("../src/model/glm53/model_trait_exl3.rs");
const TRAIT: &str = include_str!("../src/traits/model.rs");
const TRANSACTION: &str = include_str!("../src/model/glm53/dsa_verify_execution.rs");

#[test]
fn ordinary_logits_copy_cannot_return_success_after_silently_copying_only_one_row() {
    let start = TARGET.find("pub(super) fn copy_logits(").unwrap();
    let body = TARGET[start..].split("\n    fn workspace(").next().unwrap();
    assert!(
        !body.contains("&mut destination[..bytes]"),
        "a K-row destination must not receive only row zero"
    );
    assert!(body.contains("destination.len()"));
}

#[test]
fn ordinary_host_argmax_uses_the_same_owned_readback_boundary() {
    for name in ["pub fn argmax_host(", "pub fn argmax_rows_host("] {
        let start = TARGET.find(name).unwrap();
        let body = TARGET[start..].split("\n    }").next().unwrap();
        assert!(
            !body.contains("self.gpu.copy_d2h("),
            "local Vec can outlive neither a failed fence nor this frame"
        );
        assert!(body.contains("self.copy_logits("));
    }
}

#[test]
fn glm_exposes_policy_before_persistent_commit_instead_of_after_the_raw_transaction() {
    assert!(TRAIT.contains("fn decode_verify_with_policy("));
    assert!(MODEL.contains("fn decode_verify_with_policy("));
    assert!(TRANSACTION.contains("run_verify_policy_transaction("));
}

#[test]
fn policy_transaction_is_integrated_not_just_an_unregistered_test_helper() {
    let source = include_str!("../src/model/glm53/mod.rs");
    assert!(source.contains("mod verify_policy_transaction;"));
    // The scheduler-side half of this check (glm53_policy_driver.rs must
    // publish from a committed receipt) is not in this tree: GLM's DFlash2
    // policy driver was not ported onto this engine's scheduler, and serving
    // refuses GLM speculation (serve_load.rs) so no publication path exists.
}

#[test]
fn legacy_model_verifier_cannot_publish_all_inputs_after_partial_raw_commit() {
    let start = MODEL.find("    fn decode_verify(\n").unwrap();
    let body = MODEL[start..]
        .split("\n    fn decode_verify_graphed(")
        .next()
        .unwrap();
    assert!(!body.contains("verify_dflash2_scheduler("));
    assert!(!body.contains("record_verify_inputs("));
    assert!(
        body.contains("bail!("),
        "legacy callers must use the policy-before-commit entry"
    );
}
