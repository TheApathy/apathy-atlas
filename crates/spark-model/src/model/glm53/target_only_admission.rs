// SPDX-License-Identifier: AGPL-3.0-only

//! Evidence-backed admission for NON-speculative GLM-5.3 EXL3 serving.
//!
//! [`super::kernels::Glm53KernelAdmission`] names the six seams of the
//! receipt-verified speculative B1/T1 executor (stage, verify, accept 0/1,
//! commit under device completion receipts). That executor is not built, and a
//! non-speculative walk never needs it: every step is accepted, so the commit is
//! unconditional. Admitting target-only serving through that gate would claim
//! receipts that do not exist; admitting it through the bring-up escape claims
//! nothing at all. This gate admits exactly the claim the evidence supports.
//!
//! # What the evidence is
//!
//! Teacher-forced logits from the served build, compared against an
//! independent implementation of the same checkpoint (ExLlamaV3's
//! `glm5_next`, whose KDA runs through fla's `chunk_kda` and whose MLA/DSA and
//! EXL3 GEMMs are its own), on the prefill rows of several prompts and on the
//! build's own greedy decode continuation. The bounds were fixed before the
//! first comparison ran. A deliberately broken build (the KDA commit skipped via
//! [`GLM53_NEGATIVE_CONTROL_ENV`]) must fail the same bounds, or the gate is
//! a check that cannot fail and admits nothing.
//!
//! The record lives in `bench/glm53_admission/evidence.json`; the unit test
//! below requires this constant to match it field for field.

use anyhow::{Result, bail, ensure};

/// Runtime switch that deliberately corrupts the EXL3 commit, for the gate's
/// negative control only. Admission refuses while it is set.
pub const GLM53_NEGATIVE_CONTROL_ENV: &str = "ATLAS_GLM53_NEGATIVE_CONTROL";
/// The one accepted value: skip the KDA convolution and recurrent-state commit.
pub const GLM53_NEGATIVE_CONTROL_SKIP_KDA_COMMIT: &str = "skip-kda-commit";

/// Pre-registered bounds (2026-09-23, before any reference comparison).
pub const MIN_ARGMAX_AGREEMENT_EXCL_TIES: f64 = 0.90;
pub const MAX_MEAN_KL: f64 = 0.05;
pub const MAX_TOP1_DELTA_ABS: f64 = 0.01;
pub const MAX_NLL_DELTA_REL: f64 = 0.02;
/// The claim needs more than one prompt to be a claim about the model.
pub const MIN_PROMPTS: u32 = 3;

/// One arm's aggregate against the reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glm53GateMetrics {
    pub prompts: u32,
    pub positions: u32,
    pub argmax_agreement_excl_ties: f64,
    pub mean_kl: f64,
    /// |top-1(atlas) - top-1(reference)| against the true next tokens. `None`
    /// where no true tokens exist: a free-running decode continuation.
    pub top1_delta_abs: Option<f64>,
    /// |NLL(atlas) - NLL(reference)| / NLL(reference) against the true tokens.
    pub nll_delta_rel: Option<f64>,
}

impl Glm53GateMetrics {
    pub fn within_bounds(&self) -> bool {
        self.prompts >= MIN_PROMPTS
            && self.positions > 0
            && self.argmax_agreement_excl_ties >= MIN_ARGMAX_AGREEMENT_EXCL_TIES
            && self.mean_kl <= MAX_MEAN_KL
            && self
                .top1_delta_abs
                .is_none_or(|delta| delta <= MAX_TOP1_DELTA_ABS)
            && self
                .nll_delta_rel
                .is_none_or(|delta| delta <= MAX_NLL_DELTA_REL)
    }
}

/// The recorded gate run for the served EXL3 target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glm53TargetOnlyEvidence {
    /// Short commit the evaluated binary was built from.
    pub build_commit: &'static str,
    pub binary_sha256: &'static str,
    pub reference: &'static str,
    pub prompts_sha256: &'static str,
    pub prefill: Glm53GateMetrics,
    pub decode: Glm53GateMetrics,
    /// The negative control, which must fail.
    pub control: Glm53GateMetrics,
}

/// Recorded evidence for the EXL3 target-only path. `None` means closed.
pub(crate) const GLM53_EXL3_TARGET_ONLY_EVIDENCE: Option<Glm53TargetOnlyEvidence> = None;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glm53TargetOnlyAdmission {
    evidence: Option<Glm53TargetOnlyEvidence>,
}

impl Glm53TargetOnlyAdmission {
    pub(crate) const fn current() -> Self {
        Self {
            evidence: GLM53_EXL3_TARGET_ONLY_EVIDENCE,
        }
    }

    pub fn evidence(self) -> Option<Glm53TargetOnlyEvidence> {
        self.evidence
    }

    pub fn validate(self) -> Result<()> {
        let Some(evidence) = self.evidence else {
            bail!("GLM-5.3 EXL3 target-only admission has no recorded reference evidence");
        };
        ensure!(
            evidence.prefill.top1_delta_abs.is_some() && evidence.prefill.nll_delta_rel.is_some(),
            "GLM-5.3 target-only prefill evidence must be scored against the true tokens"
        );
        ensure!(
            evidence.prefill.within_bounds(),
            "GLM-5.3 target-only prefill evidence is outside the pre-registered bounds: {:?}",
            evidence.prefill
        );
        ensure!(
            evidence.decode.within_bounds(),
            "GLM-5.3 target-only decode evidence is outside the pre-registered bounds: {:?}",
            evidence.decode
        );
        ensure!(
            !evidence.control.within_bounds(),
            "GLM-5.3 target-only negative control PASSED the bounds, so they cannot fail: {:?}",
            evidence.control
        );
        Ok(())
    }
}

/// The negative-control selector, parsed strictly.
pub fn negative_control_from(value: Option<&str>) -> Result<bool> {
    match value {
        None => Ok(false),
        Some(GLM53_NEGATIVE_CONTROL_SKIP_KDA_COMMIT) => Ok(true),
        Some(other) => bail!(
            "{GLM53_NEGATIVE_CONTROL_ENV} must be absent or exactly \
             {GLM53_NEGATIVE_CONTROL_SKIP_KDA_COMMIT:?}; got {other:?}"
        ),
    }
}

pub fn negative_control_active() -> Result<bool> {
    negative_control_from(std::env::var(GLM53_NEGATIVE_CONTROL_ENV).ok().as_deref())
}

#[cfg(test)]
#[path = "target_only_admission_tests.rs"]
mod tests;
