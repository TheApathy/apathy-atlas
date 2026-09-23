// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

const EVIDENCE_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench/glm53_admission/evidence.json"
));

fn passing() -> Glm53GateMetrics {
    Glm53GateMetrics {
        prompts: 4,
        positions: 4092,
        argmax_agreement_excl_ties: 0.95,
        mean_kl: 0.02,
        top1_delta_abs: Some(0.002),
        nll_delta_rel: Some(0.005),
    }
}

fn evidence_with(control: Glm53GateMetrics) -> Glm53TargetOnlyEvidence {
    Glm53TargetOnlyEvidence {
        build_commit: "test",
        binary_sha256: "test",
        reference: "test",
        prompts_sha256: "test",
        prefill: passing(),
        decode: passing(),
        control,
    }
}

#[test]
fn each_bound_rejects_on_its_own() {
    assert!(passing().within_bounds());
    let cases = [
        Glm53GateMetrics {
            prompts: MIN_PROMPTS - 1,
            ..passing()
        },
        Glm53GateMetrics {
            positions: 0,
            ..passing()
        },
        Glm53GateMetrics {
            argmax_agreement_excl_ties: 0.8999,
            ..passing()
        },
        Glm53GateMetrics {
            mean_kl: 0.0501,
            ..passing()
        },
        Glm53GateMetrics {
            top1_delta_abs: Some(0.0101),
            ..passing()
        },
        Glm53GateMetrics {
            nll_delta_rel: Some(0.0201),
            ..passing()
        },
    ];
    for case in cases {
        assert!(!case.within_bounds(), "{case:?} must be rejected");
    }
}

#[test]
fn no_evidence_is_closed() {
    let admission = Glm53TargetOnlyAdmission { evidence: None };
    assert!(admission.validate().is_err());
}

#[test]
fn a_control_that_passes_voids_the_evidence() {
    let admission = Glm53TargetOnlyAdmission {
        evidence: Some(evidence_with(passing())),
    };
    let refusal = format!("{:#}", admission.validate().unwrap_err());
    assert!(refusal.contains("cannot fail"), "{refusal}");

    let failing_control = Glm53GateMetrics {
        mean_kl: 1.0,
        ..passing()
    };
    let admission = Glm53TargetOnlyAdmission {
        evidence: Some(evidence_with(failing_control)),
    };
    admission.validate().unwrap();
}

#[test]
fn prefill_without_true_tokens_is_not_evidence() {
    let mut evidence = evidence_with(Glm53GateMetrics {
        mean_kl: 1.0,
        ..passing()
    });
    evidence.prefill.nll_delta_rel = None;
    let admission = Glm53TargetOnlyAdmission {
        evidence: Some(evidence),
    };
    assert!(admission.validate().is_err());
    // Decode has no true tokens and is judged on agreement and KL alone.
    let mut evidence = evidence_with(Glm53GateMetrics {
        mean_kl: 1.0,
        ..passing()
    });
    evidence.decode.top1_delta_abs = None;
    evidence.decode.nll_delta_rel = None;
    Glm53TargetOnlyAdmission {
        evidence: Some(evidence),
    }
    .validate()
    .unwrap();
}

#[test]
fn out_of_bounds_decode_voids_the_evidence() {
    let mut evidence = evidence_with(Glm53GateMetrics {
        mean_kl: 1.0,
        ..passing()
    });
    evidence.decode.argmax_agreement_excl_ties = 0.5;
    let admission = Glm53TargetOnlyAdmission {
        evidence: Some(evidence),
    };
    assert!(admission.validate().is_err());
}

#[test]
fn negative_control_selector_is_strict() {
    assert!(!negative_control_from(None).unwrap());
    assert!(negative_control_from(Some("skip-kda-commit")).unwrap());
    assert!(negative_control_from(Some("1")).is_err());
    assert!(negative_control_from(Some("")).is_err());
}

fn metrics_from(value: &serde_json::Value) -> Glm53GateMetrics {
    Glm53GateMetrics {
        prompts: value["prompts"].as_u64().unwrap() as u32,
        positions: value["positions"].as_u64().unwrap() as u32,
        argmax_agreement_excl_ties: value["argmax_agreement_excl_ties"].as_f64().unwrap(),
        mean_kl: value["mean_kl"].as_f64().unwrap(),
        top1_delta_abs: value["top1_delta_abs"].as_f64(),
        nll_delta_rel: value["nll_delta_rel"].as_f64(),
    }
}

/// The admission constant and the committed gate record must be the same
/// claim: neither can be edited without the other.
#[test]
fn recorded_evidence_matches_the_committed_gate_record() {
    let record: serde_json::Value = serde_json::from_str(EVIDENCE_JSON).unwrap();
    let status = record["status"].as_str().unwrap();
    match GLM53_EXL3_TARGET_ONLY_EVIDENCE {
        None => assert_eq!(
            status, "pending",
            "record claims {status} but admission is closed"
        ),
        Some(evidence) => {
            assert_eq!(status, "pass");
            assert_eq!(
                record["build_commit"].as_str().unwrap(),
                evidence.build_commit
            );
            assert_eq!(
                record["binary_sha256"].as_str().unwrap(),
                evidence.binary_sha256
            );
            assert_eq!(record["reference"].as_str().unwrap(), evidence.reference);
            assert_eq!(
                record["prompts_sha256"].as_str().unwrap(),
                evidence.prompts_sha256
            );
            assert_eq!(metrics_from(&record["prefill"]), evidence.prefill);
            assert_eq!(metrics_from(&record["decode"]), evidence.decode);
            assert_eq!(metrics_from(&record["control"]), evidence.control);
            Glm53TargetOnlyAdmission::current().validate().unwrap();
        }
    }
}
