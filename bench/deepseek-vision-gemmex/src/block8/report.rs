// SPDX-License-Identifier: AGPL-3.0-only
//! Numerical comparison only. Captured native FC1 is not an official teacher.
use anyhow::{Result, ensure};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug)]
pub enum CandidateMode {
    GemmExDefault,
    GemmExFull,
    LtBaseline,
    LtComputeTypeOnly,
}
impl CandidateMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::GemmExDefault => "gemmex-default",
            Self::GemmExFull => "gemmex-full",
            Self::LtBaseline => "lt-baseline",
            Self::LtComputeTypeOnly => "lt-compute-type-only",
        }
    }
}

pub fn check_native_control(actual: &[u8], captured: &[u8]) -> Result<()> {
    ensure!(
        actual.len() == 20 * 5632 * 2 && actual.len() == captured.len(),
        "native control extent"
    );
    ensure!(
        actual == captured,
        "native replay did not reproduce captured block-08-fc1"
    );
    crate::metrics::compare(actual, captured)?;
    Ok(())
}

pub fn compare_repeats(mode: CandidateMode, repeats: &[Vec<u8>], captured: &[u8]) -> Result<Value> {
    ensure!(
        repeats.len() == 2,
        "exactly two reset-repeat outputs required"
    );
    ensure!(
        repeats[0] == repeats[1],
        "candidate reset-repeat bytes differ"
    );
    let metrics = crate::metrics::compare(&repeats[0], captured)?;
    Ok(json!({"mode":mode.label(),"status":"DIAGNOSTIC_COMPLETE",
        "comparison_kind":"same-input-operator-versus-native-control",
        "metrics_denominator":"captured native block-08-fc1","metrics":metrics,
        "exact_vs_native":repeats[0] == captured,"repeat_byte_equal":true,"repeat_count":2,
        "teacher_reference_available":false,"teacher_reference_sha256":null,
        "teacher_reference_payload":null,"full_encoder_qualified":false,"performance_qualified":false}))
}
